"""Busca: consulta em linguagem natural + filtros estruturados.

Passos:
1. A consulta vira um vetor em cada canal (no espaço do modelo daquele canal).
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

Busca personalizada (fase 2): quando há um usuário, a afinidade de cada candidato com o
perfil dele (perfil.py) vira MAIS UMA lista na fusão, uma por canal ("perfil:texto",
"perfil:imagem"). Duas propriedades importantes:
- A personalização só REORDENA candidatos que a busca já trouxe. Ela nunca põe no topo um
  produto que não tem nada a ver com a consulta só porque combina com o gosto da pessoa.
- `peso_perfil` controla o quanto o gosto pesa contra a consulta. 0 = busca da fase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .armazenamento import Armazenamento
from .custo import RegistroCusto, cronometro
from .embeddings import Embedder
from .modelos import CANAIS, Canal, Filtros, Resultado

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
    perfis: dict | None = None,  # {categoria: {canal: vetor}} do usuário (perfil.perfis_do_usuario)
    peso_perfil: float = 0.5,
) -> list[Resultado]:
    filtros = filtros or Filtros()
    config = config or ConfigBusca()
    reg = registro or RegistroCusto(etapa="busca", modelo=embedder.nome)

    with cronometro(reg):
        listas = {
            canal: armazenamento.vizinhos(embedder.chave(canal), canal, embedder.consulta(consulta, canal),
                                          filtros, config.candidatos)
            for canal, peso in config.pesos().items() if peso > 0
        }
    reg.itens += 1

    pesos: dict[str, float] = dict(config.pesos())
    if perfis and peso_perfil > 0:
        listas_perfil = _listas_de_perfil(armazenamento, embedder, listas, perfis)
        for nome, lista in listas_perfil.items():
            listas[nome] = lista
            pesos[nome] = peso_perfil / len(listas_perfil)

    pontos = fundir_rrf(listas, pesos)
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


def _listas_de_perfil(armazenamento: Armazenamento, embedder: Embedder,
                      listas: dict[str, list[tuple[str, float]]], perfis: dict) -> dict[str, list[tuple[str, float]]]:
    """Candidatos já recuperados, ordenados pela afinidade com o perfil, em cada canal.

    Cada candidato é comparado com o perfil DA CATEGORIA DELE: o gosto em bonecas julga
    bonecas, o gosto em placas julga placas. Candidato de uma categoria em que a pessoa não
    tem histórico simplesmente não entra nessas listas (fica com a posição da consulta).
    """
    candidatos = sorted({item_id for lista in listas.values() for item_id, _ in lista})
    itens = armazenamento.obter_itens(candidatos)
    saida: dict[str, list[tuple[str, float]]] = {}
    for canal in CANAIS:
        if not any(canal in p for p in perfis.values()):
            continue
        vetores = armazenamento.vetores(embedder.chave(canal), canal, candidatos)
        afinidades = []
        for item_id, v in vetores.items():
            item = itens.get(item_id)
            perfil = perfis.get(item.categoria, {}) if item else {}
            if canal in perfil:
                afinidades.append((item_id, float(v @ perfil[canal])))
        if afinidades:
            saida[f"perfil:{canal}"] = sorted(afinidades, key=lambda x: -x[1])
    return saida
