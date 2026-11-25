"""Busca: consulta em linguagem natural + filtros estruturados.

Passos:
1. A consulta vira um vetor (codificador de texto do CLIP).
2. Para cada canal (texto do anúncio e foto do anúncio), o armazenamento devolve
   os `candidatos` itens mais parecidos que passam nos filtros (preço, condição,
   categoria).
3. As duas listas são combinadas com Reciprocal Rank Fusion (RRF).
4. Duplicatas (mesmo vendedor + mesmo título) são removidas, mantendo a melhor.

Por que RRF em vez de somar as similaridades?
As similaridades dos dois canais estão em escalas diferentes. No CLIP, texto
contra texto costuma dar valores bem mais altos que texto contra imagem (que
ficam, em geral, entre 0,2 e 0,35). Somando, o canal de texto sempre ganharia.
O RRF ignora o valor e usa só a POSIÇÃO em cada lista:
pontuação = soma de peso / (K + posição). Ficar em 1º numa lista vale o mesmo,
seja qual for a escala dela. K = 60 é o valor do artigo original (Cormack et
al., 2009) e suaviza a diferença entre posições próximas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .armazenamento import Armazenamento
from .custo import RegistroCusto, cronometro
from .embeddings import Embedder
from .modelos import Canal, Filtros, Resultado

Modo = Literal["texto", "imagem", "fusao"]
MODOS: tuple[Modo, ...] = ("texto", "imagem", "fusao")
K_RRF = 60


@dataclass
class ConfigBusca:
    modo: Modo = "fusao"
    candidatos: int = 100  # quantos itens buscar em cada canal antes de fundir
    peso_imagem: float = 0.5  # na fusão: 0 = só texto, 1 = só imagem

    def pesos(self) -> dict[Canal, float]:
        if self.modo == "texto":
            return {"texto": 1.0}
        if self.modo == "imagem":
            return {"imagem": 1.0}
        return {"texto": 1.0 - self.peso_imagem, "imagem": self.peso_imagem}


def fundir_rrf(listas: dict[str, list[tuple[str, float]]], pesos: dict[str, float], k: int = K_RRF) -> dict[str, float]:
    """{canal: [(id, sim), ...] em ordem} -> {id: pontuação RRF}."""
    pontos: dict[str, float] = {}
    for canal, lista in listas.items():
        peso = pesos.get(canal, 0.0)
        for posicao, (item_id, _) in enumerate(lista, start=1):
            pontos[item_id] = pontos.get(item_id, 0.0) + peso / (k + posicao)
    return pontos


def buscar(
    armazenamento: Armazenamento,
    embedder: Embedder,
    consulta: str,
    filtros: Filtros | None = None,
    k: int = 10,
    config: ConfigBusca | None = None,
    registro: RegistroCusto | None = None,
) -> list[Resultado]:
    filtros = filtros or Filtros()
    config = config or ConfigBusca()
    reg = registro or RegistroCusto(etapa="busca", modelo=embedder.nome)

    with cronometro(reg):
        vetor = embedder.textos([consulta])[0]
        listas = {
            canal: armazenamento.vizinhos(embedder.nome, canal, vetor, filtros, config.candidatos)
            for canal, peso in config.pesos().items() if peso > 0
        }
    reg.itens += 1

    pontos = fundir_rrf(listas, config.pesos())
    sims = {canal: dict(lista) for canal, lista in listas.items()}
    posicoes = {canal: {item_id: p for p, (item_id, _) in enumerate(lista, start=1)} for canal, lista in listas.items()}

    ordenados = sorted(pontos, key=lambda i: -pontos[i])
    # Busca só os itens que podem entrar no resultado (com folga para as duplicatas removidas)
    itens = armazenamento.obter_itens(ordenados[: k * 5])

    resultados: list[Resultado] = []
    chaves_vistas: set[str] = set()
    for item_id in ordenados:
        item = itens.get(item_id)
        if item is None:
            continue
        if item.chave_duplicata in chaves_vistas:
            continue
        chaves_vistas.add(item.chave_duplicata)
        resultados.append(Resultado(
            item=item,
            pontuacao=pontos[item_id],
            similaridades={c: sims[c][item_id] for c in sims if item_id in sims[c]},
            posicoes={c: posicoes[c][item_id] for c in posicoes if item_id in posicoes[c]},
        ))
        if len(resultados) == k:
            break
    return resultados
