"""Teste de manipulação: quanto o peso da imagem deixa a busca vulnerável a foto copiada.

Cenário: alguém publica um anúncio de OUTRO produto usando a foto de um produto popular,
para aparecer nas buscas dele (golpe, ou item que não deveria estar ali). Com peso 1 na
imagem, o título do anúncio não conta para a ordem: quem controla a foto controla em que
buscas o anúncio aparece. Com peso 0, só o título conta.

Como o teste simula isso, sem mexer no catálogo:
1. Alvos: os produtos mais avaliados de cada categoria (os mais "atraentes" para copiar).
2. Intrusos: para cada alvo, alguns anúncios falsos. Cada um é um produto real de OUTRA
   linha (sorteado entre os de título menos parecido com o do alvo), com o título e o
   vetor de texto dele, mas com o VETOR DE FOTO do alvo. Nenhuma foto é copiada de fato:
   só o vetor, e só na memória.
3. Buscas: o começo do título de cada alvo (o que alguém digitaria para achá-lo).
4. Para cada peso da imagem, mede-se a fração de buscas com algum intruso no top 10, o
   número médio de intrusos no top 10 e se o alvo continua no top 10.

Limitação: só buscas por nome. Buscas visuais ("boneca de cabelo rosa") tendem a ser mais
vulneráveis, porque nelas o título do alvo ajuda menos; o número medido aqui é um piso.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .armazenamento import Armazenamento
from .busca import ConfigBusca, buscar
from .modelos import Canal, Filtros, Item

PREFIXO = "intruso:"


@dataclass
class Intruso:
    item: Item
    alvo: str  # id do produto cuja foto foi copiada
    doador: str  # id do produto real que empresta título e vetor de texto
    vetores: dict[tuple[str, Canal], np.ndarray]


class ArmazenamentoComIntrusos:
    """Embrulha um armazenamento e acrescenta anúncios falsos só na memória.

    `vizinhos` e `obter_itens` enxergam os intrusos; todo o resto vai direto para o
    armazenamento original, que não é alterado.
    """

    def __init__(self, base: Armazenamento, intrusos: list[Intruso]):
        self.base = base
        self.intrusos = {i.item.id: i for i in intrusos}

    def vizinhos(self, modelo: str, canal: Canal, vetor: np.ndarray, filtros: Filtros, n: int):
        lista = self.base.vizinhos(modelo, canal, vetor, filtros, n)
        extras = [(iid, float(i.vetores[(modelo, canal)] @ np.asarray(vetor, dtype=np.float32)))
                  for iid, i in self.intrusos.items() if (modelo, canal) in i.vetores and filtros.aceita(i.item)]
        return sorted(lista + extras, key=lambda x: -x[1])[:n]

    def obter_itens(self, ids: list[str]) -> dict[str, Item]:
        reais = self.base.obter_itens([i for i in ids if not i.startswith(PREFIXO)])
        return reais | {i: self.intrusos[i].item for i in ids if i in self.intrusos}

    def __getattr__(self, nome):
        return getattr(self.base, nome)


class ConsultaEmCache:
    """Guarda o vetor de cada consulta: o mesmo texto é buscado com vários pesos."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def consulta(self, texto: str, canal: Canal) -> np.ndarray:
        chave = (texto, canal)
        if chave not in self._cache:
            self._cache[chave] = self.embedder.consulta(texto, canal)
        return self._cache[chave]

    def __getattr__(self, nome):
        return getattr(self.embedder, nome)


def consulta_do_titulo(titulo: str, palavras: int = 6) -> str:
    """O começo do título, que é o que alguém digitaria para achar o produto."""
    return " ".join(titulo.split()[:palavras]).lower()


def montar_intrusos(arm: Armazenamento, embedder, categoria: str, alvos: int = 30, por_alvo: int = 3,
                    semente: int = 0) -> tuple[list[Item], list[Intruso]]:
    """-> (produtos-alvo, intrusos). Alvos: os mais avaliados com vetor de texto e de foto."""
    chave_t, chave_i = embedder.chave("texto"), embedder.chave("imagem")
    ids = sorted(arm.ids_categoria(categoria))
    textos = arm.vetores(chave_t, "texto", ids)
    fotos = arm.vetores(chave_i, "imagem", ids)
    com_vetores = [i for i in ids if i in textos and i in fotos]
    if len(com_vetores) < 2:
        raise ValueError(f"{categoria}: poucos produtos com vetor de texto e de foto ({chave_t}, {chave_i}). "
                         "Rode `mini-lcm indexar` com este EMBEDDER.")
    contagem = Counter(a.item_id for a in arm.avaliacoes_de(categoria))
    escolhidos = sorted(com_vetores, key=lambda i: (-contagem[i], i))[:alvos]
    itens = arm.obter_itens(com_vetores)
    rng = random.Random(semente)
    matriz = np.stack([textos[i] for i in com_vetores])
    intrusos: list[Intruso] = []
    for alvo in escolhidos:
        sims = matriz @ textos[alvo]
        # doadores: a metade de títulos menos parecidos com o do alvo (outra linha de produto)
        ordem = np.argsort(sims)[: max(len(com_vetores) // 2, 1)]
        candidatos = [com_vetores[k] for k in ordem if com_vetores[k] != alvo]
        for j, doador in enumerate(rng.sample(candidatos, min(por_alvo, len(candidatos)))):
            falso = replace(itens[doador], id=f"{PREFIXO}{alvo}:{j}", vendedor_id=f"{PREFIXO}{j}")
            intrusos.append(Intruso(item=falso, alvo=alvo, doador=doador,
                                    vetores={(chave_t, "texto"): textos[doador], (chave_i, "imagem"): fotos[alvo]}))
    return [itens[a] for a in escolhidos], intrusos


def medir(arm: Armazenamento, embedder, categoria: str, pesos: list[float], alvos: int = 30, por_alvo: int = 3,
          k: int = 10, semente: int = 0) -> list[dict]:
    """Uma linha por peso: buscas com intruso, intrusos no top k e alvo no top k (médias)."""
    lista_alvos, intrusos = montar_intrusos(arm, embedder, categoria, alvos, por_alvo, semente)
    falso = ArmazenamentoComIntrusos(arm, intrusos)
    emb = ConsultaEmCache(embedder)
    filtros = Filtros(categoria=categoria)
    linhas = []
    for peso in pesos:
        com_intruso, n_intrusos, alvo_no_top = [], [], []
        for alvo in lista_alvos:
            resultados = buscar(falso, emb, consulta_do_titulo(alvo.titulo), filtros, k=k,
                                config=ConfigBusca(modo="fusao", peso_imagem=peso))
            ids = [r.item.id for r in resultados]
            meus = [i for i in ids if i.startswith(f"{PREFIXO}{alvo.id}:")]
            com_intruso.append(bool(meus))
            n_intrusos.append(len(meus))
            alvo_no_top.append(alvo.id in ids)
        linhas.append({"categoria": categoria, "buscas": len(lista_alvos), "peso": peso,
                       "com_intruso": float(np.mean(com_intruso)), "intrusos": float(np.mean(n_intrusos)),
                       "alvo_no_top": float(np.mean(alvo_no_top))})
    return linhas


def relatorio(linhas: list[dict], por_alvo: int, k: int = 10) -> str:
    saida = [f"Teste de manipulação · {por_alvo} anúncios com a foto copiada por alvo · top {k}", "",
             f"| Categoria | Buscas | Peso da imagem | buscas com intruso | intrusos no top {k} | alvo no top {k} |",
             "|---|---|---|---|---|---|"]
    for l in linhas:
        saida.append(f"| {l['categoria']} | {l['buscas']} | {l['peso']:g} | {l['com_intruso']:.3f} | "
                     f"{l['intrusos']:.2f} | {l['alvo_no_top']:.3f} |")
    saida += ["", "buscas com intruso: fração das buscas com pelo menos um anúncio de foto copiada no top "
              f"{k}. intrusos no top {k}: média por busca (máximo {por_alvo}). alvo no top {k}: fração das buscas "
              "em que o produto procurado aparece."]
    return "\n".join(saida)
