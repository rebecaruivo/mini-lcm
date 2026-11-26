"""Perfil do usuário: um vetor de gosto aprendido só com o que a pessoa fez.

Ideia central (a do LCM, em miniatura): em vez de perguntar o que a pessoa quer,
olhamos o que ela avaliou. Cada produto já tem vetores (texto e foto). O perfil é
uma MÉDIA PONDERADA dos vetores dos produtos que ela avaliou: produtos de que ela
gostou puxam o perfil na direção deles; os de que não gostou, empurram para longe.
Recomendar vira "achar os produtos mais parecidos com o perfil", a mesma busca por
vizinhos da fase 1, só que a "consulta" é a pessoa.

Peso de cada avaliação: peso = 0,5 + (nota - 3) / 2

| nota | peso | leitura |
|---|---|---|
| 5 | 1,5 | gostou muito |
| 4 | 1,0 | gostou |
| 3 | 0,5 | neutro, mas se interessou a ponto de comprar e avaliar |
| 2 | 0,0 | não puxa nem empurra |
| 1 | -0,5 | empurra o perfil para longe |

O 0,5 de base existe porque avaliar já é sinal de interesse: a pessoa comprou ou
considerou o produto. Sem a base, uma nota 3 valeria zero, e quem só dá notas medianas
ficaria sem perfil. É uma escolha de projeto (está em docs/decisoes.md), e a avaliação
temporal diz se ela ajuda.

Um perfil POR CATEGORIA e POR CANAL:
- por categoria: o gosto por bonecas não deve contaminar as recomendações de placas;
- por canal: texto e foto vivem em espaços diferentes (CLIP vs e5) e não podem ser somados.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .armazenamento import Armazenamento
from .embeddings import Embedder
from .modelos import CANAIS, Avaliacao, Canal

Perfil = dict[Canal, np.ndarray]  # canal -> vetor normalizado (canal ausente = sem dados naquele canal)


def peso_da_avaliacao(nota: float) -> float:
    return 0.5 + (float(nota) - 3.0) / 2.0


def construir_perfil(historico: list[Avaliacao], vetores: dict[Canal, dict[str, np.ndarray]]) -> Perfil:
    """Média ponderada (pela nota) dos vetores dos produtos avaliados, em cada canal.

    Se a pessoa avaliou o mesmo produto mais de uma vez, vale a avaliação mais recente.
    Se a soma dos pesos não for positiva (ex.: só notas 1 e 2), não há direção de gosto
    para seguir e o canal fica sem perfil.
    """
    ultima: dict[str, Avaliacao] = {}
    for a in sorted(historico, key=lambda a: a.momento):
        ultima[a.item_id] = a
    perfil: Perfil = {}
    for canal, por_id in vetores.items():
        pares = [(por_id[i], peso_da_avaliacao(a.nota)) for i, a in ultima.items() if i in por_id]
        if not pares or sum(p for _, p in pares) <= 0:
            continue
        v = np.sum([vetor * p for vetor, p in pares], axis=0)
        norma = np.linalg.norm(v)
        if norma > 0:
            perfil[canal] = (v / norma).astype(np.float32)
    return perfil


def perfis_do_usuario(armazenamento: Armazenamento, embedder: Embedder, usuario: str) -> dict[str, Perfil]:
    """{categoria: perfil} com todo o histórico do usuário (usado na busca personalizada)."""
    historico = armazenamento.avaliacoes_de(usuario=usuario)
    if not historico:
        return {}
    itens = armazenamento.obter_itens(sorted({a.item_id for a in historico}))
    por_categoria: dict[str, list[Avaliacao]] = defaultdict(list)
    for a in historico:
        if a.item_id in itens:
            por_categoria[itens[a.item_id].categoria].append(a)
    perfis = {}
    for categoria, avs in por_categoria.items():
        ids = sorted({a.item_id for a in avs})
        vetores = {canal: armazenamento.vetores(embedder.chave(canal), canal, ids) for canal in CANAIS}
        perfil = construir_perfil(avs, vetores)
        if perfil:
            perfis[categoria] = perfil
    return perfis
