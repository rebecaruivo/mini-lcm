"""Avaliação da busca com um gabarito de notas de relevância (0 a 3).

O gabarito (avaliacao/gabarito.json) guarda, para cada consulta, as notas que
cada produto recebeu, de duas fontes:
- `notas_humanas`: dadas por você em `mini-lcm rotular` ou `auditar`;
- `notas_ia`: dadas pelo juiz (`mini-lcm julgar`), com justificativa.

A avaliação pode usar só as humanas, só as da IA ou as duas combinadas (a
humana vale quando existe; senão, a da IA). A IA só deve entrar depois de
passar no teste de concordância (`mini-lcm concordancia`).

Métricas (k = 10):
- recall@k: dos produtos relevantes (nota >= 2), quantos aparecem no top k.
- MRR: 1 / posição do primeiro relevante, na média das consultas.
- NDCG@k: usa a nota inteira, não só "relevante ou não". Cada resultado vale
  2^nota - 1 (nota 3 vale 7, nota 2 vale 3, nota 1 vale 1) dividido por
  log2(posição + 1), que desconta quem aparece mais embaixo. A soma é dividida
  pela melhor soma possível (a lista ideal, ordenada pelas notas), dando um
  valor de 0 a 1. Diferente do recall, ela distingue "o exato em 1º e o
  parecido em 5º" de "o parecido em 1º e o exato em 5º".

Pooling: as notas são dadas sobre o conjunto dos primeiros resultados de TODOS
os modos (texto, imagem e fusão), embaralhados, para nenhum modo ganhar por
construção. Produto que aparece no top k mas nunca recebeu nota é tratado como
nota 0, e o relatório informa quantos foram assim (se forem muitos, o
gabarito precisa ser ampliado).

Duplicatas: a busca mostra só um produto de cada grupo (mesma loja + mesmo
título). Por isso a comparação é por grupo: a nota do grupo é a maior nota
entre os seus produtos.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .armazenamento import Armazenamento
from .busca import MODOS, ConfigBusca, Modo, buscar
from .embeddings import Embedder
from .modelos import Filtros, Item

Fonte = Literal["humano", "ia", "combinado"]
NOTA_RELEVANTE = 2  # nota mínima para contar como relevante no recall e no MRR
NOTAS = (0, 1, 2, 3)


@dataclass
class ConsultaRotulada:
    id: str
    texto: str
    filtros: dict = field(default_factory=dict)
    perfil: str = ""  # quem busca e com que intenção; vai para a IA e aparece para você
    notas_humanas: dict[str, int] = field(default_factory=dict)
    # item_id -> {"nota": int, "justificativa": str, "modelo": str, "rubrica": hash da rubrica}
    notas_ia: dict[str, dict] = field(default_factory=dict)
    # ids cujas notas humanas vieram da auditoria (amostra estratificada), e não da rotulagem
    auditados: list[str] = field(default_factory=list)

    def filtros_obj(self) -> Filtros:
        return Filtros(**self.filtros)

    def notas(self, fonte: Fonte) -> dict[str, int]:
        ia = {i: int(j["nota"]) for i, j in self.notas_ia.items()}
        if fonte == "humano":
            return dict(self.notas_humanas)
        if fonte == "ia":
            return ia
        return ia | self.notas_humanas  # a humana sobrescreve a da IA


def carregar_gabarito(caminho: Path) -> list[ConsultaRotulada]:
    if not caminho.exists():
        return []
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return [ConsultaRotulada(**c) for c in dados.get("consultas", [])]


def salvar_gabarito(caminho: Path, consultas: list[ConsultaRotulada]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(".tmp")  # grava em outro arquivo e troca: um Ctrl+C no meio não corrompe o gabarito
    tmp.write_text(json.dumps({"consultas": [asdict(c) for c in consultas]}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(caminho)


def obter_ou_criar(gabarito: list[ConsultaRotulada], texto: str, filtros: dict, perfil: str = "") -> ConsultaRotulada:
    for c in gabarito:
        if c.texto == texto and c.filtros == filtros:
            if perfil and not c.perfil:
                c.perfil = perfil
            return c
    c = ConsultaRotulada(id=f"c{len(gabarito) + 1}", texto=texto, filtros=filtros, perfil=perfil)
    gabarito.append(c)
    return c


# ─────────────────────────── Métricas ───────────────────────────


def recall_em_k(encontrados: list[str], relevantes: set[str], k: int) -> float:
    if not relevantes:
        return 0.0
    return len(set(encontrados[:k]) & relevantes) / len(relevantes)


def reciproco_do_rank(encontrados: list[str], relevantes: set[str]) -> float:
    for posicao, x in enumerate(encontrados, start=1):
        if x in relevantes:
            return 1.0 / posicao
    return 0.0


def ndcg_em_k(encontrados: list[str], notas: dict[str, int], k: int) -> float:
    def dcg(valores: list[int]) -> float:
        return sum((2**n - 1) / math.log2(p + 1) for p, n in enumerate(valores, start=1))

    ideal = dcg(sorted(notas.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([notas.get(x, 0) for x in encontrados[:k]]) / ideal


def _por_grupo(notas: dict[str, int], itens: dict[str, Item]) -> dict[str, int]:
    """Notas por item -> notas por grupo de duplicatas (maior nota do grupo)."""
    saida: dict[str, int] = {}
    for item_id, nota in notas.items():
        chave = itens[item_id].chave_duplicata if item_id in itens else item_id
        saida[chave] = max(nota, saida.get(chave, 0))
    return saida


def avaliar(
    armazenamento: Armazenamento,
    embedder: Embedder,
    gabarito: list[ConsultaRotulada],
    modos: tuple[Modo, ...] = MODOS,
    k: int = 10,
    peso_imagem: float | dict[str, float] = 0.5,
    fonte: Fonte = "combinado",
) -> dict:
    """peso_imagem: um número para todas as consultas, ou {categoria: peso} (o peso de cada
    consulta vem da categoria do filtro dela; categoria ausente no dicionário usa 0,5)."""
    consultas = [c for c in gabarito if any(n >= NOTA_RELEVANTE for n in c.notas(fonte).values())]
    if not consultas:
        raise ValueError(f"Nenhuma consulta tem produto com nota >= {NOTA_RELEVANTE} na fonte '{fonte}'. "
                         "Rode `mini-lcm rotular` ou `mini-lcm julgar` primeiro.")
    todos = armazenamento.obter_itens(sorted({i for c in consultas for i in c.notas(fonte)}))
    relatorio: dict = {"k": k, "fonte": fonte, "consultas": len(consultas), "modos": {}}
    def peso_de(c: ConsultaRotulada) -> float:
        if isinstance(peso_imagem, dict):
            return float(peso_imagem.get(c.filtros.get("categoria", ""), 0.5))
        return float(peso_imagem)

    for modo in modos:
        por_consulta = []
        for c in consultas:
            cfg = ConfigBusca(modo=modo, peso_imagem=peso_de(c))
            resultados = buscar(armazenamento, embedder, c.texto, c.filtros_obj(), k=k, config=cfg)
            itens = todos | {r.item.id: r.item for r in resultados}
            notas = _por_grupo(c.notas(fonte), itens)
            encontrados = [r.item.chave_duplicata for r in resultados]
            relevantes = {g for g, n in notas.items() if n >= NOTA_RELEVANTE}
            por_consulta.append({
                "id": c.id,
                "texto": c.texto,
                "relevantes": len(relevantes),
                "sem_nota_no_top": sum(1 for g in encontrados if g not in notas),
                f"recall@{k}": round(recall_em_k(encontrados, relevantes, k), 3),
                "rr": round(reciproco_do_rank(encontrados, relevantes), 3),
                f"ndcg@{k}": round(ndcg_em_k(encontrados, notas, k), 3),
            })
        n = len(por_consulta)
        relatorio["modos"][modo] = {
            f"recall@{k}": round(sum(p[f"recall@{k}"] for p in por_consulta) / n, 3),
            "mrr": round(sum(p["rr"] for p in por_consulta) / n, 3),
            f"ndcg@{k}": round(sum(p[f"ndcg@{k}"] for p in por_consulta) / n, 3),
            "sem_nota_no_top": sum(p["sem_nota_no_top"] for p in por_consulta),
            "por_consulta": por_consulta,
        }
    return relatorio


def relatorio_markdown(rel: dict) -> str:
    k = rel["k"]
    linhas = [f"Consultas avaliadas: {rel['consultas']} · notas: {rel['fonte']}", "",
              f"| Modo | recall@{k} | MRR | NDCG@{k} | resultados sem nota |", "|---|---|---|---|---|"]
    for modo, r in rel["modos"].items():
        linhas.append(f"| {modo} | {r[f'recall@{k}']:.3f} | {r['mrr']:.3f} | {r[f'ndcg@{k}']:.3f} | {r['sem_nota_no_top']} |")
    modos = list(rel["modos"])
    linhas += ["", "| Consulta | Relevantes | " + " | ".join(f"NDCG@{k} ({m})" for m in modos) + " |",
               "|---|---|" + "---|" * len(modos)]
    for i, c in enumerate(rel["modos"][modos[0]]["por_consulta"]):
        valores = " | ".join(f"{rel['modos'][m]['por_consulta'][i][f'ndcg@{k}']:.2f}" for m in modos)
        linhas.append(f"| {c['texto']} | {c['relevantes']} | {valores} |")
    return "\n".join(linhas)


# ─────────────────────────── Pool e rotulagem humana ───────────────────────────


def pool_para_rotular(
    armazenamento: Armazenamento,
    embedder: Embedder,
    texto: str,
    filtros: Filtros,
    profundidade: int = 20,
    semente: int | None = None,
) -> list[Item]:
    """Os `profundidade` primeiros de cada modo, juntos e embaralhados (para não se saber a origem)."""
    vistos: dict[str, Item] = {}
    for modo in MODOS:
        for r in buscar(armazenamento, embedder, texto, filtros, k=profundidade, config=ConfigBusca(modo=modo)):
            vistos.setdefault(r.item.id, r.item)
    itens = list(vistos.values())
    random.Random(semente).shuffle(itens)
    return itens


LEGENDA = "0 = irrelevante · 1 = relacionado · 2 = relevante · 3 = exato · q = parar"


def _descrever(item: Item) -> str:
    return (f"{item.titulo}\n    {item.preco_formatado} · {item.condicao} · "
            f"{item.atributos.get('Categoria', item.categoria)}\n    {item.link}\n    foto: {item.foto_url}")


def perguntar_nota(perguntar: Callable[[str], str]) -> int | None:
    while True:
        resposta = perguntar("    nota? [0-3/q] ").strip().lower()
        if resposta == "q":
            return None
        if resposta in {"0", "1", "2", "3"}:
            return int(resposta)


def rotular(
    consulta: ConsultaRotulada,
    itens: list[Item],
    perguntar: Callable[[str], str] | None = None,
    mostrar: Callable[[str], None] | None = None,
) -> ConsultaRotulada:
    """Pede uma nota de 0 a 3 para cada item ainda sem nota humana. 'q' para e guarda o que foi feito."""
    perguntar, mostrar = perguntar or input, mostrar or print
    pendentes = [i for i in itens if i.id not in consulta.notas_humanas]
    mostrar(f'\nConsulta: "{consulta.texto}"  filtros: {consulta.filtros or "nenhum"}')
    if consulta.perfil:
        mostrar(f"Perfil: {consulta.perfil}")
    mostrar(f"{len(pendentes)} itens. Rubrica em avaliacao/rubrica.md. {LEGENDA}\n")
    for n, item in enumerate(pendentes, start=1):
        mostrar(f"[{n}/{len(pendentes)}] {_descrever(item)}")
        nota = perguntar_nota(perguntar)
        if nota is None:
            break
        consulta.notas_humanas[item.id] = nota
    return consulta
