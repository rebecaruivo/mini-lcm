"""Avaliação da recomendação com o histórico real, sem nenhum rótulo.

A pergunta é: olhando o passado de uma pessoa, o sistema adivinha o que ela fez depois?

Corte por usuário (`corte="usuario"`, "leave-last-out"): para cada usuário com histórico suficiente, as
avaliações são ordenadas no tempo; a ÚLTIMA vira o alvo (o produto escondido) e as
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

Corte por data (`corte="data"`, o padrão do comando): o corte por usuário acima tem dois
problemas, medidos nos dados reais. (1) Vazamento: o treino de OUTRAS pessoas inclui
avaliações posteriores ao alvo de cada um, e popularidade e coocorrência aproveitam esse
futuro. (2) Compras em lote: metade dos alvos das bonecas foi avaliada no mesmo dia da
avaliação anterior (produtos comprados juntos), casos muito mais fáceis que "a próxima
compra". No corte por data, há uma data única para todos: o modelo aprende só com o que
veio antes dela, e o alvo de cada pessoa é a primeira avaliação nos seis meses seguintes.
Como uma data só deixa poucos casos, são usados cortes semestrais (1º de janeiro e 1º de
julho de cada ano, "rolling origin"), cada um com o seu modelo. A mesma pessoa pode
aparecer em vários cortes, e por isso o intervalo de confiança sorteia pessoas, não casos.
Em cada corte, só concorrem os produtos que já existiam naquele semestre.

Métricas (k = 10), com um único produto certo por caso:
- hit@k: o produto escondido está no top k? (média = fração de usuários acertados)
- NDCG@k: 1 / log2(posição + 1) se estiver no top k, senão 0. Acertar em 1º vale 1;
  em 3º, 0,5; em 10º, 0,29.
- MRR: 1 / posição do produto escondido na lista inteira.
- cobertura: fração do catálogo que aparece em algum top k. Popularidade recomenda os
  mesmos 10 produtos para todos (cobertura mínima); a personalização distribui.

Curva de histórico (cold start): a mesma avaliação usando só as últimas N avaliações de
treino de cada pessoa (N = 1, 2, 3, 5, todas). Mostra quanto histórico cada método
precisa para valer a pena, que é o problema do "usuário novo".

Alvos novos: produtos sem nenhuma interação antes do corte. Popularidade, em alta e
coocorrência não conseguem recomendá-los; só os métodos de conteúdo. O relatório separa o
hit@k dos alvos novos e dos já vistos.

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
    alvo: Avaliacao


def dividir(avaliacoes: list[Avaliacao], ids_catalogo: set[str], min_historico: int = 3) -> tuple[list[Caso], list[Avaliacao]]:
    """-> (casos de avaliação, interações de treino de todos os usuários).

    Usuários com menos de `min_historico` produtos distintos não viram caso de avaliação, mas
    as interações deles entram inteiras no treino (não há nada deles sendo escondido).
    """
    por_usuario: dict[str, list[Avaliacao]] = defaultdict(list)
    for a in avaliacoes:
        if a.item_id in ids_catalogo:
            por_usuario[a.usuario].append(a)
    casos, treino_global = [], []
    for usuario in sorted(por_usuario):
        hist = sorted(por_usuario[usuario], key=lambda a: (a.momento, a.item_id))
        alvo, treino = hist[-1], hist[:-1]
        distintos = {a.item_id for a in hist}
        if len(distintos) >= min_historico and alvo.item_id not in {a.item_id for a in treino}:
            casos.append(Caso(usuario, treino, alvo))
            treino_global.extend(treino)
        else:
            treino_global.extend(hist)
    return casos, treino_global


def sem_a_ultima(avaliacoes: list[Avaliacao], ids_catalogo: set[str]) -> list[Avaliacao]:
    """Tira a última avaliação de cada pessoa: o que sobra serve de base para o CONJUNTO DE ESCOLHA.

    Escolher um parâmetro (como a meia-vida da recência) olhando o resultado da conferência é
    ajustar o modelo à prova: o número reportado sai otimista. O caminho correto é
    esconder a conferência, repetir a mesma divisão no que sobrou (a penúltima avaliação vira o
    alvo) e escolher o parâmetro aí. A conferência só é usada uma vez, com o valor escolhido.
    """
    por_usuario: dict[str, list[Avaliacao]] = defaultdict(list)
    for a in avaliacoes:
        if a.item_id in ids_catalogo:
            por_usuario[a.usuario].append(a)
    saida = []
    for hist in por_usuario.values():
        saida.extend(sorted(hist, key=lambda a: (a.momento, a.item_id))[:-1])
    return saida


def _posicao(recomendador: Recomendador, metodo: str, caso: Caso, historico: list[Avaliacao],
             disponiveis: np.ndarray | None = None) -> tuple[int, list[int]]:
    """Posição (1 = primeiro) do produto escondido e os índices do top 10.

    disponiveis: máscara dos produtos que existiam no momento (None = catálogo inteiro).
    """
    pontos = recomendador.pontuar(metodo, historico, semente=caso.usuario).astype(float)
    if disponiveis is not None:
        pontos[~disponiveis] = -np.inf
    idx = recomendador.catalogo.indice
    for a in caso.treino:  # o que a pessoa já tem não concorre, mesmo quando o histórico usado é parcial
        pontos[idx[a.item_id]] = -np.inf
    alvo = pontos[idx[caso.alvo.item_id]]
    posicao = int(np.sum(pontos > alvo)) + 1  # quantos ficaram estritamente na frente, + 1
    top = np.argsort(-pontos, kind="stable")[:10]
    return posicao, top.tolist()


def intervalo_bootstrap(valores: np.ndarray, reamostras: int = 2000, semente: int = 0,
                        grupos: list[str] | None = None) -> tuple[float, float]:
    """IC 95% da média por bootstrap percentil. Para diferença pareada, passe a - b.

    grupos: a quem pertence cada valor. Com cortes por data, a mesma pessoa pode ser
    testada em vários cortes, e os casos dela não são independentes (quem é fácil de
    prever num corte tende a ser no outro). O bootstrap então sorteia PESSOAS, levando
    todos os casos de cada uma ("bootstrap por grupos"), e o intervalo não fica estreito
    demais por contar a mesma pessoa várias vezes como se fossem várias.
    """
    valores = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(semente)
    if grupos is None:
        medias = np.empty(reamostras)
        for r in range(reamostras):  # em laço para não montar uma matriz reamostras x usuários
            medias[r] = valores[rng.integers(0, len(valores), len(valores))].mean()
    else:
        _, codigo = np.unique(np.asarray(grupos), return_inverse=True)
        somas = np.bincount(codigo, weights=valores)
        contagens = np.bincount(codigo).astype(float)
        g = len(somas)
        sorteio = rng.integers(0, g, (reamostras, g))
        medias = somas[sorteio].sum(axis=1) / contagens[sorteio].sum(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def _rotulo(chave: str, metodo: str) -> str:
    return f"{metodo} ({'todas' if chave == 'todas' else 'N=' + chave})"


# ───── corte por data ─────

METODOS_NO_TEMPO = ("em_alta", "hibrido_alta")  # dependem de "agora": só fazem sentido com corte por data


def cortes_semestrais(avaliacoes: list[Avaliacao]) -> list[str]:
    """Datas de corte em 1º de janeiro e 1º de julho, do primeiro ao último ano com dados."""
    if not avaliacoes:
        return []
    anos = range(int(min(a.momento for a in avaliacoes)[:4]), int(max(a.momento for a in avaliacoes)[:4]) + 1)
    return [f"{ano}-{mes:02d}-01T00:00:00+00:00" for ano in anos for mes in (1, 7)]


def _proximo_semestre(corte: str) -> str:
    ano, mes = int(corte[:4]), int(corte[5:7])
    return f"{ano + 1}-01-01T00:00:00+00:00" if mes == 7 else f"{ano}-07-01T00:00:00+00:00"


def dividir_por_data(avaliacoes: list[Avaliacao], ids_catalogo: set[str], corte: str,
                     fim: str, min_historico: int = 3) -> tuple[list[Caso], list[Avaliacao]]:
    """Um corte global: -> (casos de avaliação, interações de treino).

    Treino: tudo antes de `corte`, de todas as pessoas. Caso de avaliação: cada pessoa com pelo
    menos `min_historico` produtos distintos antes do corte e alguma avaliação entre
    `corte` e `fim`; o alvo é a PRIMEIRA delas (a próxima compra depois do corte). Nada
    posterior ao corte entra no treino: nem a popularidade nem a coocorrência veem o futuro.
    """
    treino = [a for a in avaliacoes if a.item_id in ids_catalogo and a.momento < corte]
    antes: dict[str, list[Avaliacao]] = defaultdict(list)
    for a in treino:
        antes[a.usuario].append(a)
    janela: dict[str, list[Avaliacao]] = defaultdict(list)
    for a in avaliacoes:
        if a.item_id in ids_catalogo and corte <= a.momento < fim:
            janela[a.usuario].append(a)
    casos = []
    for usuario in sorted(janela):
        hist = sorted(antes.get(usuario, []), key=lambda a: (a.momento, a.item_id))
        alvo = min(janela[usuario], key=lambda a: (a.momento, a.item_id))
        vistos = {a.item_id for a in hist}
        if len(vistos) >= min_historico and alvo.item_id not in vistos:
            casos.append(Caso(usuario, hist, alvo))
    return casos, treino


def _conjuntos_por_data(catalogo: Catalogo, avaliacoes: list[Avaliacao], min_historico: int, escolha: bool,
                        **parametros) -> tuple[list[tuple[Recomendador, list[Caso]]], dict]:
    """Um recomendador por corte, ajustado só com o passado daquele corte.

    Catálogo do período: num corte de 2015, boa parte dos produtos importados ainda nem
    tinha sido lançada. Concorre um produto que (a) já tinha alguma avaliação antes do
    corte ou (b) tem data de lançamento (dos metadados) anterior ao fim da janela. A regra
    não usa nenhuma avaliação posterior ao corte: usar "primeira avaliação antes do fim da
    janela" faria os produtos novos candidatos serem exatamente os que alguém vai avaliar
    na janela, e os alvos estariam entre eles. Produto novo sem data de lançamento não
    concorre (é um erro para todos os métodos se for o alvo).

    Os cortes são divididos no tempo pela quantidade de casos: os primeiros, até somar
    metade dos casos, formam o CONJUNTO DE ESCOLHA (onde os parâmetros são escolhidos); os demais,
    o DE CONFERÊNCIA (validação e teste, nos termos usuais). Dividir pela quantidade de cortes
    daria um conjunto de escolha minúsculo, porque os primeiros anos do dataset têm poucas
    avaliações. Os alvos do conjunto de escolha são todos anteriores ao primeiro corte da
    conferência.
    """
    ids = set(catalogo.ids)
    primeira: dict[str, str] = {}
    for a in avaliacoes:
        if a.item_id in ids and a.momento < primeira.get(a.item_id, "9999"):
            primeira[a.item_id] = a.momento
    por_corte = []
    for corte in cortes_semestrais(avaliacoes):
        casos, treino = dividir_por_data(avaliacoes, ids, corte, _proximo_semestre(corte), min_historico)
        if casos:
            por_corte.append((corte, casos, treino))
    total, acumulado, meio = sum(len(c) for _, c, _ in por_corte), 0, len(por_corte)
    for i, (_, casos, _) in enumerate(por_corte):
        acumulado += len(casos)
        if acumulado >= total / 2:
            meio = i + 1
            break
    escolhidos = por_corte[:meio] if escolha else por_corte[meio:]
    lancamentos = catalogo.lancamentos or [None] * len(catalogo.ids)
    conjuntos = []
    for corte, casos, treino in escolhidos:
        fim = _proximo_semestre(corte)
        disponiveis = np.array([primeira.get(i, "9999") < corte or (lanc is not None and lanc < fim[:10])
                                for i, lanc in zip(catalogo.ids, lancamentos)])
        recomendador = Recomendador(catalogo, **parametros).ajustar(treino, agora=corte, disponiveis=disponiveis)
        conjuntos.append((recomendador, casos, disponiveis))
    info = {"cortes": len(escolhidos),
            "periodo": f"{escolhidos[0][0][:7]} a {_proximo_semestre(escolhidos[-1][0])[:7]}" if escolhidos else "",
            "interacoes_treino": len(escolhidos[-1][2]) if escolhidos else 0}
    return conjuntos, info


def avaliar_recomendacao(
    catalogo: Catalogo,
    avaliacoes: list[Avaliacao],
    metodos: tuple[str, ...] = METODOS,
    k: int = 10,
    min_historico: int = 3,
    historicos: tuple[int | None, ...] = (None,),
    meia_vida: float | None = None,
    escolha: bool = False,
    corte: str = "usuario",
    janela_dias: int = 365,
    peso_em_alta: float = 1.0,
    vagas_novos: int = 0,
) -> dict:
    """historicos: quantas avaliações de treino usar por pessoa (None = todas).
    meia_vida, janela_dias, peso_em_alta: parâmetros do Recomendador.
    escolha: avalia no conjunto de escolha, sem tocar no de conferência.
    corte: "usuario" (deixa a última de cada pessoa de fora) ou "data" (cortes semestrais globais)."""
    parametros = {"meia_vida": meia_vida, "janela_dias": janela_dias, "peso_em_alta": peso_em_alta,
                  "vagas_novos": vagas_novos}
    omitidos: list[str] = []
    if corte == "usuario":
        omitidos = [m for m in metodos if m in METODOS_NO_TEMPO]
        metodos = tuple(m for m in metodos if m not in METODOS_NO_TEMPO)
        if escolha:
            avaliacoes = sem_a_ultima(avaliacoes, set(catalogo.ids))
        casos, treino_global = dividir(avaliacoes, set(catalogo.ids), min_historico)
        conjuntos = [(Recomendador(catalogo, **parametros).ajustar(treino_global), casos, None)] if casos else []
        info = {"cortes": 1, "periodo": "", "interacoes_treino": len(treino_global)}
    elif corte == "data":
        conjuntos, info = _conjuntos_por_data(catalogo, avaliacoes, min_historico, escolha, **parametros)
    else:
        raise ValueError(f"corte deve ser 'usuario' ou 'data', não {corte!r}")
    todos = [c for _, cs, _ in conjuntos for c in cs]
    if not todos:
        raise ValueError(f"Nenhum usuário com pelo menos {min_historico} produtos distintos nesta categoria. "
                         "A importação foi feita com selecao: usuarios?")
    grupos = [c.usuario for c in todos]
    # alvo "novo": produto sem nenhuma interação no treino daquele corte (só o conteúdo pode acertá-lo)
    novo = np.array([rec.popularidade[rec.catalogo.indice[c.alvo.item_id]] == 0
                     for rec, cs, _ in conjuntos for c in cs])
    relatorio: dict = {"k": k, "usuarios_testados": len(set(grupos)), "casos": len(todos), "itens": len(catalogo.ids),
                       "interacoes_treino": info["interacoes_treino"],
                       "historico_medio": round(sum(len(c.treino) for c in todos) / len(todos), 1),
                       "meia_vida": meia_vida, "janela_dias": janela_dias, "peso_em_alta": peso_em_alta,
                       "vagas_novos": vagas_novos, "conjunto": "escolha" if escolha else "conferencia", "corte": corte,
                       "cortes": info["cortes"], "periodo": info["periodo"], "metodos_omitidos": omitidos,
                       "fracao_alvos_novos": round(float(novo.mean()), 4),
                       "resultados": {}}
    acertos: dict[tuple[str, str], np.ndarray] = {}
    for n in historicos:
        chave = "todas" if n is None else str(n)
        relatorio["resultados"][chave] = {}
        for metodo in metodos:
            hits, ndcgs, rrs, vistos = [], [], [], set()
            for recomendador, casos, disponiveis in conjuntos:
                for caso in casos:
                    historico = caso.treino if n is None else caso.treino[-n:]
                    posicao, top = _posicao(recomendador, metodo, caso, historico, disponiveis)
                    hits.append(posicao <= k)
                    ndcgs.append(1 / math.log2(posicao + 1) if posicao <= k else 0.0)
                    rrs.append(1 / posicao)
                    vistos.update(top[:k])
            acertos[(chave, metodo)] = np.array(hits, dtype=float)
            ic = intervalo_bootstrap(acertos[(chave, metodo)], grupos=grupos)
            relatorio["resultados"][chave][metodo] = {
                f"hit@{k}": round(float(np.mean(hits)), 4),
                f"hit@{k}_ic95": [round(ic[0], 4), round(ic[1], 4)],
                f"ndcg@{k}": round(float(np.mean(ndcgs)), 4),
                "mrr": round(float(np.mean(rrs)), 4),
                "cobertura": round(len(vistos) / len(catalogo.ids), 4),
                f"hit@{k}_novos": round(float(acertos[(chave, metodo)][novo].mean()), 4) if novo.any() else None,
                f"hit@{k}_vistos": round(float(acertos[(chave, metodo)][~novo].mean()), 4) if (~novo).any() else None,
            }
    relatorio["comparacoes"] = _comparacoes(acertos, grupos)
    relatorio["_acertos"] = acertos  # por caso, para comparações extras (não vai para o .md)
    relatorio["_grupos"] = grupos
    return relatorio


def _comparacoes(acertos: dict[tuple[str, str], np.ndarray], grupos: list[str]) -> list[dict]:
    """A melhor configuração (método + histórico) contra o melhor histórico de cada outro método."""
    melhor = max(acertos, key=lambda c: acertos[c].mean())
    saida = []
    for metodo in dict.fromkeys(m for _, m in acertos):
        if metodo == melhor[1]:
            continue
        rival = max((c for c in acertos if c[1] == metodo), key=lambda c: acertos[c].mean())
        dif = acertos[melhor] - acertos[rival]
        lo, hi = intervalo_bootstrap(dif, grupos=grupos)
        saida.append({"a": _rotulo(*melhor), "b": _rotulo(*rival), "diferenca": round(float(dif.mean()), 4),
                      "ic95": [round(lo, 4), round(hi, 4)]})
    return saida


def relatorio_recomendacao_markdown(rel: dict, categoria: str, modelo: str) -> str:
    k = rel["k"]
    if rel.get("corte") == "data":
        corte = f"cortes semestrais por data ({rel['cortes']} cortes, {rel['periodo']})"
    else:
        corte = "por usuário (última avaliação de cada pessoa)"
    linhas = [f"Categoria: {categoria} · vetores: {modelo}",
              f"Corte: {corte} · conjunto: {rel.get('conjunto', 'conferencia')}",
              f"Casos: {rel.get('casos', rel['usuarios_testados'])} ({rel['usuarios_testados']} usuários) · "
              f"produtos: {rel['itens']} · histórico médio: {rel['historico_medio']}",
              "Parâmetros: recência "
              + (f"com meia-vida de {rel['meia_vida']:g} avaliação(ões)" if rel.get("meia_vida") else "desligada")
              + f" · janela do em_alta: {rel.get('janela_dias', 365)} dias · peso do em_alta no híbrido: "
              + f"{rel.get('peso_em_alta', 1.0):g} · vagas para produtos novos: {rel.get('vagas_novos', 0)}", ""]
    if rel.get("metodos_omitidos"):
        linhas[-1:] = [f"Omitidos (dependem da data, use --corte data): {', '.join(rel['metodos_omitidos'])}", ""]
    if "todas" in rel["resultados"]:
        def f(v):
            return "—" if v is None else f"{v:.3f}"
        if "fracao_alvos_novos" in rel:
            linhas += [f"Alvos novos (produto sem nenhuma interação antes do corte): {rel['fracao_alvos_novos']:.1%}", ""]
        linhas += [f"| Método | hit@{k} | IC 95% | NDCG@{k} | MRR | cobertura | hit@{k} alvos novos | hit@{k} alvos já vistos |",
                   "|---|---|---|---|---|---|---|---|"]
        for metodo, r in rel["resultados"]["todas"].items():
            lo, hi = r.get(f"hit@{k}_ic95", (float("nan"), float("nan")))
            linhas.append(f"| {metodo} | {r[f'hit@{k}']:.3f} | {lo:.3f}–{hi:.3f} | {r[f'ndcg@{k}']:.3f} | "
                          f"{r['mrr']:.3f} | {r['cobertura']:.1%} | {f(r.get(f'hit@{k}_novos'))} | "
                          f"{f(r.get(f'hit@{k}_vistos'))} |")
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
