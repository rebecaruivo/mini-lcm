"""Gera os embeddings que faltam, em lotes, e registra o custo de cada rodada.

Só processa itens que ainda não têm vetor para aquele (modelo, canal). Rodar de
novo depois de uma coleta nova embeda só os anúncios novos ou alterados.
"""

from __future__ import annotations

import logging

from .armazenamento import Armazenamento
from .custo import RegistroCusto
from .embeddings import Embedder, texto_do_item
from .modelos import CANAIS, Canal

log = logging.getLogger(__name__)


def gerar_embeddings(
    armazenamento: Armazenamento,
    embedder: Embedder,
    canais: tuple[Canal, ...] = CANAIS,
    categoria: str | None = None,
    lote: int = 256,  # itens por rodada: cada rodada salva o progresso (dá para interromper e continuar)
) -> dict[str, RegistroCusto]:
    registros: dict[str, RegistroCusto] = {}
    for canal in canais:
        pendentes = armazenamento.itens_sem_embedding(embedder.nome, canal, categoria)
        reg = RegistroCusto(etapa=f"embeddings:{canal}", modelo=embedder.nome)
        log.info("%s: %d itens sem vetor", canal, len(pendentes))
        for i in range(0, len(pendentes), lote):
            parte = pendentes[i : i + lote]
            if canal == "texto":
                matriz = embedder.textos([texto_do_item(it) for it in parte], reg)
                vetores = list(zip([it.id for it in parte], matriz))
            else:
                saida = embedder.imagens([it.foto_url for it in parte], reg)
                vetores = [(it.id, v) for it, v in zip(parte, saida) if v is not None]
            armazenamento.salvar_embeddings(embedder.nome, canal, vetores)
            log.info("%s: %d/%d", canal, min(i + lote, len(pendentes)), len(pendentes))
        if pendentes:
            armazenamento.registrar_custo(reg)
        registros[canal] = reg
    return registros
