"""Copia os dados do armazenamento local (arquivos em dados/) para outro armazenamento.

Uso típico: depois de importar e indexar no modo local, levar tudo para o Postgres
(Supabase) sem baixar os dados de novo nem recalcular nenhum vetor:

    mini-lcm migrar-para-postgres

Copia produtos, avaliações, custos e os vetores. Por padrão, só os vetores das chaves em
uso (as do EMBEDDER e do TEXTO_ITEM atuais), para não ocupar espaço com variantes que
foram só testadas; `todos=True` copia todas. É idempotente: rodar de novo atualiza o que
já existe e não duplica nada.
"""

from __future__ import annotations

import logging
from typing import Callable

from .armazenamento import Armazenamento, ArmazenamentoLocal

log = logging.getLogger(__name__)


def _lotes(lista: list, tamanho: int):
    for inicio in range(0, len(lista), tamanho):
        yield lista[inicio : inicio + tamanho]


def migrar(origem: ArmazenamentoLocal, destino: Armazenamento, chaves: set[tuple[str, str]] | None = None,
           avisar: Callable[[str], None] = print, lote: int = 1000) -> dict:
    """chaves: {(modelo, canal)} a copiar; None = todas as que existem na origem."""
    resumo = {"itens": 0, "avaliacoes": 0, "custos": 0, "vetores": {}}

    itens = list(origem.itens.values())
    for parte in _lotes(itens, lote):
        destino.salvar_itens(parte)
    resumo["itens"] = len(itens)
    avisar(f"produtos: {len(itens)}")

    avaliacoes = list(origem.avaliacoes.values())
    for parte in _lotes(avaliacoes, 5000):
        destino.salvar_avaliacoes(parte)
    resumo["avaliacoes"] = len(avaliacoes)
    avisar(f"avaliações: {len(avaliacoes)}")

    for (modelo, canal), por_id in sorted(origem.vetores_.items()):
        if chaves is not None and (modelo, canal) not in chaves:
            continue
        pares = [(i, v) for i, v in por_id.items() if i in origem.itens]
        for parte in _lotes(pares, lote):
            destino.salvar_embeddings(modelo, canal, parte)
        resumo["vetores"][f"{modelo}/{canal}"] = len(pares)
        avisar(f"vetores {modelo} ({canal}): {len(pares)}")

    for r in origem.custos():
        destino.registrar_custo(r)
    resumo["custos"] = len(origem.custos())
    return resumo
