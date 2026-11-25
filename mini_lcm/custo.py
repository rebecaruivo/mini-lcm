"""Medição de custo e tempo.

Cada etapa que usa um modelo (gerar embeddings, embedar uma consulta) gera um
`RegistroCusto`. Os registros ficam no armazenamento, para o README mostrar
números reais: quanto custou embedar o catálogo e quanto custa cada busca.

Com o CLIP local, o custo em dinheiro é zero e o que importa é o tempo. Se um
dia trocarmos por uma API paga, o mesmo registro guarda tokens e US$.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterator


@dataclass
class RegistroCusto:
    etapa: str  # ex.: "embeddings:texto", "embeddings:imagem", "busca"
    modelo: str
    itens: int = 0  # quantos itens (ou consultas) foram processados
    chamadas: int = 0  # requisições externas (API do ML, downloads de foto, API de embedding)
    segundos: float = 0.0
    tokens: int = 0
    custo_usd: float = 0.0
    falhas: int = 0
    criado_em: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def como_dict(self) -> dict:
        return asdict(self)


@contextmanager
def cronometro(registro: RegistroCusto) -> Iterator[RegistroCusto]:
    inicio = time.perf_counter()
    try:
        yield registro
    finally:
        registro.segundos += time.perf_counter() - inicio


def resumir(registros: list[RegistroCusto]) -> list[dict]:
    """Agrupa por etapa e modelo: totais e médias por item."""
    grupos: dict[tuple[str, str], RegistroCusto] = {}
    for r in registros:
        g = grupos.setdefault((r.etapa, r.modelo), RegistroCusto(etapa=r.etapa, modelo=r.modelo))
        g.itens += r.itens
        g.chamadas += r.chamadas
        g.segundos += r.segundos
        g.tokens += r.tokens
        g.custo_usd += r.custo_usd
        g.falhas += r.falhas
    linhas = []
    for g in grupos.values():
        por_item = g.segundos / g.itens if g.itens else 0.0
        linhas.append({
            "etapa": g.etapa,
            "modelo": g.modelo,
            "itens": g.itens,
            "falhas": g.falhas,
            "segundos": round(g.segundos, 2),
            "ms_por_item": round(por_item * 1000, 1),
            "custo_usd": round(g.custo_usd, 6),
            "usd_por_1000_itens": round(g.custo_usd / g.itens * 1000, 6) if g.itens else 0.0,
        })
    return sorted(linhas, key=lambda l: (l["etapa"], l["modelo"]))
