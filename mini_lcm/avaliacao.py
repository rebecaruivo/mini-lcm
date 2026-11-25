"""Avaliação da busca com um gabarito rotulado à mão.

Gabarito (avaliacao/gabarito.json): para cada consulta, os anúncios que DEVERIAM
aparecer. Duas métricas:

- recall@k: dos anúncios relevantes, quantos apareceram entre os k primeiros.
  Ex.: 4 relevantes, 3 no top 10 -> recall@10 = 0,75. Mede "achou o que devia?".
- MRR (Mean Reciprocal Rank): 1 / posição do primeiro relevante, na média das
  consultas. Primeiro relevante em 1º -> 1; em 2º -> 0,5; em 4º -> 0,25.
  Mede "o que importa aparece logo no topo?".

Como montar o gabarito sem viés (pooling): para cada consulta, juntamos os
primeiros resultados de TODOS os modos (texto, imagem e fusão) e você marca o
que é relevante sem saber de qual modo veio cada item. Se o gabarito saísse só
dos resultados da fusão, a fusão ganharia a avaliação por construção.
Limitação que fica: um item relevante que nenhum modo encontrou nunca entra no
gabarito, então o recall medido é um teto otimista.

Duplicatas: a busca mostra só uma de cada grupo de republicações (mesmo
vendedor + mesmo título). Por isso a comparação é feita por grupo, não por id:
encontrar qualquer anúncio do grupo conta como encontrar o grupo.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .armazenamento import Armazenamento
from .busca import MODOS, ConfigBusca, Modo, buscar
from .embeddings import Embedder
from .modelos import Filtros, Item


@dataclass
class ConsultaRotulada:
    id: str
    texto: str
    filtros: dict = field(default_factory=dict)
    relevantes: list[str] = field(default_factory=list)  # ids marcados como relevantes
    avaliados: list[str] = field(default_factory=list)  # todos os ids que você viu (relevantes ou não)

    def filtros_obj(self) -> Filtros:
        return Filtros(**self.filtros)


def carregar_gabarito(caminho: Path) -> list[ConsultaRotulada]:
    if not caminho.exists():
        return []
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return [ConsultaRotulada(**c) for c in dados.get("consultas", [])]


def salvar_gabarito(caminho: Path, consultas: list[ConsultaRotulada]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps({"consultas": [asdict(c) for c in consultas]}, ensure_ascii=False, indent=2),
                       encoding="utf-8")


def recall_em_k(encontrados: list[str], relevantes: set[str], k: int) -> float:
    if not relevantes:
        return 0.0
    return len(set(encontrados[:k]) & relevantes) / len(relevantes)


def reciproco_do_rank(encontrados: list[str], relevantes: set[str]) -> float:
    for posicao, x in enumerate(encontrados, start=1):
        if x in relevantes:
            return 1.0 / posicao
    return 0.0


def _grupos(ids: list[str], itens: dict[str, Item]) -> list[str]:
    """ids -> chaves de duplicata, mantendo a ordem e sem repetir."""
    vistos, saida = set(), []
    for i in ids:
        chave = itens[i].chave_duplicata if i in itens else i
        if chave not in vistos:
            vistos.add(chave)
            saida.append(chave)
    return saida


def avaliar(
    armazenamento: Armazenamento,
    embedder: Embedder,
    gabarito: list[ConsultaRotulada],
    modos: tuple[Modo, ...] = MODOS,
    k: int = 10,
    peso_imagem: float = 0.5,
) -> dict:
    consultas = [c for c in gabarito if c.relevantes]
    if not consultas:
        raise ValueError("O gabarito não tem nenhuma consulta com itens relevantes. Rode `mini-lcm rotular` primeiro.")
    todos_relevantes = armazenamento.obter_itens(sorted({i for c in consultas for i in c.relevantes}))
    relatorio: dict = {"k": k, "consultas": len(consultas), "modos": {}}
    for modo in modos:
        cfg = ConfigBusca(modo=modo, peso_imagem=peso_imagem)
        por_consulta = []
        for c in consultas:
            resultados = buscar(armazenamento, embedder, c.texto, c.filtros_obj(), k=k, config=cfg)
            itens = {r.item.id: r.item for r in resultados} | todos_relevantes
            encontrados = _grupos([r.item.id for r in resultados], itens)
            relevantes = set(_grupos(c.relevantes, itens))
            por_consulta.append({
                "id": c.id,
                "texto": c.texto,
                "relevantes": len(relevantes),
                f"recall@{k}": round(recall_em_k(encontrados, relevantes, k), 3),
                "rr": round(reciproco_do_rank(encontrados, relevantes), 3),
            })
        n = len(por_consulta)
        relatorio["modos"][modo] = {
            f"recall@{k}": round(sum(p[f"recall@{k}"] for p in por_consulta) / n, 3),
            "mrr": round(sum(p["rr"] for p in por_consulta) / n, 3),
            "por_consulta": por_consulta,
        }
    return relatorio


def relatorio_markdown(rel: dict) -> str:
    k = rel["k"]
    linhas = [f"Consultas avaliadas: {rel['consultas']}", "", f"| Modo | recall@{k} | MRR |", "|---|---|---|"]
    for modo, r in rel["modos"].items():
        linhas.append(f"| {modo} | {r[f'recall@{k}']:.3f} | {r['mrr']:.3f} |")
    linhas += ["", f"| Consulta | Relevantes | " + " | ".join(f"recall@{k} ({m})" for m in rel["modos"]) + " |",
               "|---|---|" + "---|" * len(rel["modos"])]
    primeiro = next(iter(rel["modos"].values()))["por_consulta"]
    for i, c in enumerate(primeiro):
        valores = " | ".join(f"{rel['modos'][m]['por_consulta'][i][f'recall@{k}']:.2f}" for m in rel["modos"])
        linhas.append(f"| {c['texto']} | {c['relevantes']} | {valores} |")
    return "\n".join(linhas)


# ─────────────────────────── Rotulagem ───────────────────────────


def pool_para_rotular(
    armazenamento: Armazenamento,
    embedder: Embedder,
    texto: str,
    filtros: Filtros,
    profundidade: int = 20,
    semente: int | None = None,
) -> list[Item]:
    """Os `profundidade` primeiros de cada modo, juntos e embaralhados (para você não saber a origem)."""
    vistos: dict[str, Item] = {}
    for modo in MODOS:
        for r in buscar(armazenamento, embedder, texto, filtros, k=profundidade, config=ConfigBusca(modo=modo)):
            vistos.setdefault(r.item.id, r.item)
    itens = list(vistos.values())
    random.Random(semente).shuffle(itens)
    return itens


def rotular(
    consulta: ConsultaRotulada,
    itens: list[Item],
    perguntar: Callable[[str], str] = input,
    mostrar: Callable[[str], None] = print,
) -> ConsultaRotulada:
    """Mostra cada item e pergunta: s = relevante, n = não, q = parar (salva o que já foi marcado)."""
    ja_vistos = set(consulta.avaliados)
    pendentes = [i for i in itens if i.id not in ja_vistos]
    mostrar(f'\nConsulta: "{consulta.texto}"  filtros: {consulta.filtros or "nenhum"}')
    mostrar(f"{len(pendentes)} itens para avaliar (s = relevante, n = não, q = parar)\n")
    for n, item in enumerate(pendentes, start=1):
        mostrar(f"[{n}/{len(pendentes)}] {item.titulo}\n    {item.preco_formatado} · {item.condicao} · {item.link}\n    foto: {item.foto_url}")
        while True:
            resposta = perguntar("    relevante? [s/n/q] ").strip().lower()
            if resposta in ("s", "n", "q"):
                break
        if resposta == "q":
            break
        consulta.avaliados.append(item.id)
        if resposta == "s":
            consulta.relevantes.append(item.id)
    return consulta
