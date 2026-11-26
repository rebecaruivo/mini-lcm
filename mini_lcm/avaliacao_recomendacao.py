"""Avaliação da recomendação com o histórico real, sem nenhum rótulo.

A pergunta é: olhando o passado de uma pessoa, o sistema adivinha o que ela fez depois?

Divisão temporal ("leave-last-out"): para cada usuário com histórico suficiente, as
avaliações são ordenadas no tempo; a ÚLTIMA vira o teste (o produto escondido) e as
anteriores, o treino. O sistema recebe só o treino e ordena o catálogo. Se o produto
escondido aparece entre os 10 primeiros, acertou.

Por que dividir no TEMPO e não sortear? Sorteando, o sistema poderia "aprender" com uma
avaliação de 2022 para prever uma de 2019: usaria o futuro para prever o passado, e o
número sairia otimista. Na vida real, só se conhece o passado.

Cuidados contra vazamento:
- Popularidade e coocorrência são aprendidas SÓ com as interações de treino. Se o
  produto escondido de alguém entrasse na contagem, o método "veria" a resposta.
- Produtos que a pessoa já avaliou no treino não podem ser recomendados (ela já tem).
- Se o produto escondido também aparece no treino (a pessoa avaliou duas vezes), o
  usuário fica de fora: não há o que adivinhar.

Limitação que fica: o corte é por usuário, não uma data única. As interações de treino
de OUTRAS pessoas podem ser posteriores ao teste de alguém, e a coocorrência aproveita
um pouco desse futuro. Um corte global por data elimina isso, mas deixa muitos usuários sem
caso de teste. Fica registrado como limitação.

Métricas (k = 10), com um único produto certo por usuário:
- hit@k: o produto escondido está no top k? (média = fração de usuários acertados)
- NDCG@k: 1 / log2(posição + 1) se estiver no top k, senão 0. Acertar em 1º vale 1;
  em 3º, 0,5; em 10º, 0,29.
- MRR: 1 / posição do produto escondido na lista inteira.
- cobertura: fração do catálogo que aparece em algum top k. Popularidade recomenda os
  mesmos 10 produtos para todos (cobertura mínima); a personalização distribui.

Curva de histórico (cold start): o mesmo teste usando só as últimas N avaliações de
treino de cada pessoa (N = 1, 2, 3, 5, todas). Mostra quanto histórico cada método
precisa para valer a pena, que é o problema do "usuário novo".

Incerteza: cada hit@k vem com um intervalo de confiança de 95% por bootstrap (os
usuários testados são reamostrados com reposição, 2.000 vezes). Para comparar dois
métodos, o bootstrap é PAREADO: reamostra os mesmos usuários para os dois e mede a
diferença dentro de cada reamostra. Isso desconta a variação entre pessoas (umas são
fáceis de prever para qualquer método) e dá intervalos bem mais estreitos do que
comparar os dois intervalos separados. Se o intervalo da diferença não contém zero, a
diferença não é explicada só pela amostra de usuários.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .modelos import Avaliacao
from .recomendacao import METODOS, Catalogo, Recomendador


@dataclass
class Caso:
    usuario: str
    treino: list[Avaliacao]
    teste: Avaliacao


def dividir(avaliacoes: list[Avaliacao], ids_catalogo: set[str], min_historico: int = 3) -> tuple[list[Caso], list[Avaliacao]]:
    """-> (casos de teste, interações de treino de todos os usuários).

    Usuários com menos de `min_historico` produtos distintos não viram caso de teste, mas
    as interações deles entram inteiras no treino (não há nada deles sendo escondido).
    """
    por_usuario: dict[str, list[Avaliacao]] = defaultdict(list)
    for a in avaliacoes:
        if a.item_id in ids_catalogo:
            por_usuario[a.usuario].append(a)
    casos, treino_global = [], []
    for usuario in sorted(por_usuario):
        hist = sorted(por_usuario[usuario], key=lambda a: (a.momento, a.item_id))
        teste, treino = hist[-1], hist[:-1]
        distintos = {a.item_id for a in hist}
        if len(distintos) >= min_historico and teste.item_id not in {a.item_id for a in treino}:
            casos.append(Caso(usuario, treino, teste))
            treino_global.extend(treino)
        else:
            treino_global.extend(hist)
    return casos, treino_global


def _posicao(recomendador: Recomendador, metodo: str, caso: Caso, historico: list[Avaliacao]) -> tuple[int, list[int]]:
    """Posição (1 = primeiro) do produto escondido e os índices do top 10."""
    pontos = recomendador.pontuar(metodo, historico, semente=caso.usuario).astype(float)
    idx = recomendador.catalogo.indice
    for a in caso.treino:  # o que a pessoa já tem não concorre, mesmo quando o histórico usado é parcial
        pontos[idx[a.item_id]] = -np.inf
    alvo = pontos[idx[caso.teste.item_id]]
    posicao = int(np.sum(pontos > alvo)) + 1  # quantos ficaram estritamente na frente, + 1
    top = np.argsort(-pontos, kind="stable")[:10]
    return posicao, top.tolist()


def intervalo_bootstrap(valores: np.ndarray, reamostras: int = 2000, semente: int = 0) -> tuple[float, float]:
    """IC 95% da média por bootstrap percentil. Para diferença pareada, passe a - b."""
    valores = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(semente)
    medias = np.empty(reamostras)
    for r in range(reamostras):  # em laço para não montar uma matriz reamostras x usuários
        medias[r] = valores[rng.integers(0, len(valores), len(valores))].mean()
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def _rotulo(chave: str, metodo: str) -> str:
    return f"{metodo} ({'todas' if chave == 'todas' else 'N=' + chave})"


def avaliar_recomendacao(
    catalogo: Catalogo,
    avaliacoes: list[Avaliacao],
    metodos: tuple[str, ...] = METODOS,
    k: int = 10,
    min_historico: int = 3,
    historicos: tuple[int | None, ...] = (None,),
) -> dict:
    """historicos: quantas avaliações de treino usar por pessoa (None = todas)."""
    casos, treino_global = dividir(avaliacoes, set(catalogo.ids), min_historico)
    if not casos:
        raise ValueError(f"Nenhum usuário com pelo menos {min_historico} produtos distintos nesta categoria. "
                         "A importação foi feita com selecao: usuarios?")
    recomendador = Recomendador(catalogo).ajustar(treino_global)
    relatorio: dict = {"k": k, "usuarios_testados": len(casos), "itens": len(catalogo.ids),
                       "interacoes_treino": len(treino_global),
                       "historico_medio": round(sum(len(c.treino) for c in casos) / len(casos), 1),
                       "resultados": {}}
    acertos: dict[tuple[str, str], np.ndarray] = {}
    for n in historicos:
        chave = "todas" if n is None else str(n)
        relatorio["resultados"][chave] = {}
        for metodo in metodos:
            hits, ndcgs, rrs, vistos = [], [], [], set()
            for caso in casos:
                historico = caso.treino if n is None else caso.treino[-n:]
                posicao, top = _posicao(recomendador, metodo, caso, historico)
                hits.append(posicao <= k)
                ndcgs.append(1 / math.log2(posicao + 1) if posicao <= k else 0.0)
                rrs.append(1 / posicao)
                vistos.update(top[:k])
            acertos[(chave, metodo)] = np.array(hits, dtype=float)
            ic = intervalo_bootstrap(acertos[(chave, metodo)])
            relatorio["resultados"][chave][metodo] = {
                f"hit@{k}": round(float(np.mean(hits)), 4),
                f"hit@{k}_ic95": [round(ic[0], 4), round(ic[1], 4)],
                f"ndcg@{k}": round(float(np.mean(ndcgs)), 4),
                "mrr": round(float(np.mean(rrs)), 4),
                "cobertura": round(len(vistos) / len(catalogo.ids), 4),
            }
    relatorio["comparacoes"] = _comparacoes(acertos)
    return relatorio


def _comparacoes(acertos: dict[tuple[str, str], np.ndarray]) -> list[dict]:
    """A melhor configuração (método + histórico) contra o melhor histórico de cada outro método."""
    melhor = max(acertos, key=lambda c: acertos[c].mean())
    saida = []
    for metodo in dict.fromkeys(m for _, m in acertos):
        if metodo == melhor[1]:
            continue
        rival = max((c for c in acertos if c[1] == metodo), key=lambda c: acertos[c].mean())
        dif = acertos[melhor] - acertos[rival]
        lo, hi = intervalo_bootstrap(dif)
        saida.append({"a": _rotulo(*melhor), "b": _rotulo(*rival), "diferenca": round(float(dif.mean()), 4),
                      "ic95": [round(lo, 4), round(hi, 4)]})
    return saida


def relatorio_recomendacao_markdown(rel: dict, categoria: str, modelo: str) -> str:
    k = rel["k"]
    linhas = [f"Categoria: {categoria} · vetores: {modelo}",
              f"Usuários testados: {rel['usuarios_testados']} · produtos: {rel['itens']} · "
              f"interações de treino: {rel['interacoes_treino']} · histórico médio de treino: {rel['historico_medio']}", ""]
    if "todas" in rel["resultados"]:
        linhas += [f"| Método | hit@{k} | IC 95% | NDCG@{k} | MRR | cobertura |", "|---|---|---|---|---|---|"]
        for metodo, r in rel["resultados"]["todas"].items():
            lo, hi = r.get(f"hit@{k}_ic95", (float("nan"), float("nan")))
            linhas.append(f"| {metodo} | {r[f'hit@{k}']:.3f} | {lo:.3f}–{hi:.3f} | {r[f'ndcg@{k}']:.3f} | "
                          f"{r['mrr']:.3f} | {r['cobertura']:.1%} |")
    curva = [c for c in rel["resultados"] if c != "todas"]
    if curva:
        metodos = list(next(iter(rel["resultados"].values())))
        linhas += ["", f"Curva de histórico (hit@{k} usando só as últimas N avaliações de cada pessoa):", "",
                   "| N | " + " | ".join(metodos) + " |", "|---|" + "---|" * len(metodos)]
        for c in curva + (["todas"] if "todas" in rel["resultados"] else []):
            linhas.append(f"| {c} | " + " | ".join(f"{rel['resultados'][c][m][f'hit@{k}']:.3f}" for m in metodos) + " |")
    if rel.get("comparacoes"):
        linhas += ["", f"Comparações pareadas (diferença de hit@{k}, bootstrap pareado, IC 95%):", "",
                   "| Melhor configuração | Comparada com | Diferença | IC 95% |", "|---|---|---|---|"]
        for c in rel["comparacoes"]:
            linhas.append(f"| {c['a']} | {c['b']} | {c['diferenca']:+.3f} | {c['ic95'][0]:+.3f} a {c['ic95'][1]:+.3f} |")
    return "\n".join(linhas)
