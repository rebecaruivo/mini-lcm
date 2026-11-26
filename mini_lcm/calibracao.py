"""Os 4 passos para confiar nas notas da IA antes de usá-las em escala.

1. Você rotula um conjunto pequeno (`rotular`): o padrão-ouro.
2. A IA julga os mesmos pares (`julgar --apenas-rotulados`) e depois o resto (`julgar`).
3. Mede-se a concordância (`concordancia`): kappa ponderado, com intervalo de confiança.
4. Você revisa onde discordam (`revisar`) e confere uma amostra do que só a IA
   julgou (`auditar`), para saber se a concordância se mantém fora do conjunto inicial.

Por que kappa e não só "% de acerto"?
Se 80% dos produtos são irrelevantes, um juiz que responde sempre 0 acerta 80%
sem entender nada. O kappa de Cohen desconta a concordância esperada por acaso
(calculada a partir de quantas vezes cada um usa cada nota): 0 = não melhor que
o acaso, 1 = concordância perfeita.

Por que PONDERADO?
As notas são ordenadas: discordar entre 2 e 3 é bem menos grave que entre 0 e 3.
No kappa ponderado quadrático, o peso de uma discordância é proporcional ao
quadrado da distância entre as notas (0 vs 3 pesa 9 vezes mais que 2 vs 3).

Intervalo de confiança: com poucos pares, o kappa varia muito de amostra para
amostra. O bootstrap sorteia, com reposição, 2.000 novas amostras do mesmo
tamanho a partir dos pares existentes e recalcula o kappa em cada uma. Os
percentis 2,5% e 97,5% dão a faixa plausível. Se a faixa for larga demais, faltam pares.

Critério usado aqui: kappa ponderado >= 0,6 no limite INFERIOR do intervalo
(na escala usual de Landis & Koch, 0,61 a 0,80 é concordância "substancial").
É uma convenção, não uma lei: o relatório mostra os números para você decidir.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .armazenamento import Armazenamento
from .avaliacao import (LEGENDA, NOTA_RELEVANTE, NOTAS, ConsultaRotulada, _descrever, perguntar_nota,
                        pool_para_rotular)
from .custo import RegistroCusto
from .embeddings import Embedder
from .juiz import ErroJuiz, Juiz
from .modelos import Item

log = logging.getLogger(__name__)

KAPPA_MINIMO = 0.6
MIN_PARES = 50


# ─────────────────────────── Passo 2: a IA julga ───────────────────────────


def julgar_pendentes(
    consultas: list[ConsultaRotulada],
    armazenamento: Armazenamento,
    embedder: Embedder,
    juiz: Juiz,
    versao_rubrica: str,
    apenas_rotulados: bool = False,
    profundidade: int = 20,
    limite: int | None = None,
    salvar: Callable[[], None] = lambda: None,
) -> RegistroCusto:
    """Julga os pares que ainda não têm nota da IA com este modelo e esta versão da rubrica.

    apenas_rotulados=True: só os pares que você já rotulou (passo de calibração).
    Caso contrário: todo o pool de cada consulta (passo de escala).
    `salvar` é chamado a cada 20 julgamentos, para não perder trabalho (e dinheiro) num erro.
    """
    reg = RegistroCusto(etapa="julgamento", modelo=juiz.modelo)
    feitos = 0
    for c in consultas:
        if apenas_rotulados:
            itens = list(armazenamento.obter_itens(list(c.notas_humanas)).values())
        else:
            itens = pool_para_rotular(armazenamento, embedder, c.texto, c.filtros_obj(), profundidade)
        for item in itens:
            atual = c.notas_ia.get(item.id)
            if atual and atual.get("rubrica") == versao_rubrica and atual.get("modelo") == juiz.modelo:
                continue  # já julgado com esta rubrica e este modelo
            if limite is not None and feitos >= limite:
                salvar()
                return reg
            try:
                c.notas_ia[item.id] = juiz.julgar(c.texto, c.perfil, item, reg).para_gabarito()
            except ErroJuiz as e:
                log.warning("Não julguei %s para '%s': %s", item.id, c.texto, e)
                continue
            feitos += 1
            if feitos % 20 == 0:
                salvar()
    salvar()
    return reg


# ─────────────────────────── Passo 3: concordância ───────────────────────────


def pares(consultas: list[ConsultaRotulada], origem: str = "rotulagem") -> list[tuple[str, str, int, int]]:
    """(consulta_id, item_id, nota_humana, nota_ia) dos itens com as duas notas.

    origem="rotulagem": notas que você deu no `rotular` (o conjunto de calibração).
    origem="auditoria": notas da amostra estratificada do `auditar`. Ficam separadas porque
    a amostra tem a mesma quantidade de cada nota da IA, de propósito, e misturá-las
    distorceria o kappa do conjunto de calibração.
    origem="todos": as duas.
    """
    saida = []
    for c in consultas:
        auditados = set(c.auditados)
        for item_id, humana in c.notas_humanas.items():
            veio_da_auditoria = item_id in auditados
            if origem == "rotulagem" and veio_da_auditoria or origem == "auditoria" and not veio_da_auditoria:
                continue
            if item_id in c.notas_ia:
                saida.append((c.id, item_id, int(humana), int(c.notas_ia[item_id]["nota"])))
    return saida


def kappa_ponderado(a: list[int], b: list[int], categorias: tuple[int, ...] = NOTAS) -> float | None:
    """Kappa de Cohen com pesos quadráticos. None quando não é calculável (ex.: todos deram a mesma nota)."""
    k = len(categorias)
    indice = {c: i for i, c in enumerate(categorias)}
    observada = np.zeros((k, k))
    for x, y in zip(a, b):
        observada[indice[x], indice[y]] += 1
    n = observada.sum()
    if n == 0:
        return None
    observada /= n
    esperada = np.outer(observada.sum(axis=1), observada.sum(axis=0))  # se as duas notas fossem independentes
    i, j = np.indices((k, k))
    pesos = (i - j) ** 2 / (k - 1) ** 2
    denominador = (pesos * esperada).sum()
    if denominador == 0:
        return None
    return float(1 - (pesos * observada).sum() / denominador)


def kappa_binario(a: list[int], b: list[int], corte: int = NOTA_RELEVANTE) -> float | None:
    """Kappa simples sobre 'relevante ou não' (nota >= corte), que é o que o recall usa."""
    return kappa_ponderado([int(x >= corte) for x in a], [int(y >= corte) for y in b], categorias=(0, 1))


def intervalo_bootstrap(a: list[int], b: list[int], reamostras: int = 2000, semente: int = 0) -> tuple[float, float] | None:
    if len(a) < 2:
        return None
    rng = np.random.default_rng(semente)
    a_, b_ = np.array(a), np.array(b)
    valores = []
    for _ in range(reamostras):
        idx = rng.integers(0, len(a), len(a))
        k = kappa_ponderado(a_[idx].tolist(), b_[idx].tolist())
        if k is not None:
            valores.append(k)
    if not valores:
        return None
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


@dataclass
class RelatorioConcordancia:
    n: int
    acordo_exato: float
    acordo_relevancia: float
    kappa_ponderado: float | None
    intervalo: tuple[float, float] | None
    kappa_relevancia: float | None
    matriz: list[list[int]]  # linhas = nota humana, colunas = nota da IA
    vies_medio: float  # média de (IA - humano): positivo = IA mais generosa
    aprovado: bool
    motivo: str
    discordancias: list[tuple[str, str, int, int]] = field(default_factory=list)


def concordancia(consultas: list[ConsultaRotulada], kappa_minimo: float = KAPPA_MINIMO,
                 min_pares: int = MIN_PARES, origem: str = "rotulagem") -> RelatorioConcordancia:
    p = pares(consultas, origem)
    humanas, ia = [x[2] for x in p], [x[3] for x in p]
    n = len(p)
    matriz = [[0] * 4 for _ in range(4)]
    for h, m in zip(humanas, ia):
        matriz[h][m] += 1
    kw = kappa_ponderado(humanas, ia)
    intervalo = intervalo_bootstrap(humanas, ia)
    if n < min_pares:
        aprovado, motivo = False, f"poucos pares ({n}); o mínimo combinado é {min_pares}"
    elif intervalo is None or kw is None:
        aprovado, motivo = False, "kappa não calculável (pouca variação nas notas)"
    elif intervalo[0] < kappa_minimo:
        aprovado, motivo = False, f"limite inferior do kappa ({intervalo[0]:.2f}) abaixo de {kappa_minimo}"
    else:
        aprovado, motivo = True, f"limite inferior do kappa ({intervalo[0]:.2f}) >= {kappa_minimo}"
    return RelatorioConcordancia(
        n=n,
        acordo_exato=sum(h == m for h, m in zip(humanas, ia)) / n if n else 0.0,
        acordo_relevancia=sum((h >= NOTA_RELEVANTE) == (m >= NOTA_RELEVANTE) for h, m in zip(humanas, ia)) / n if n else 0.0,
        kappa_ponderado=kw,
        intervalo=intervalo,
        kappa_relevancia=kappa_binario(humanas, ia),
        matriz=matriz,
        vies_medio=float(np.mean(np.array(ia) - np.array(humanas))) if n else 0.0,
        aprovado=aprovado,
        motivo=motivo,
        discordancias=sorted([x for x in p if x[2] != x[3]], key=lambda x: -abs(x[2] - x[3])),
    )


def relatorio_concordancia_texto(r: RelatorioConcordancia) -> str:
    fmt = lambda v: "n/d" if v is None else f"{v:.2f}"  # noqa: E731
    faixa = "n/d" if r.intervalo is None else f"{r.intervalo[0]:.2f} a {r.intervalo[1]:.2f}"
    linhas = [
        f"Pares com nota humana e da IA: {r.n}",
        f"Acordo exato (mesma nota): {r.acordo_exato:.0%}",
        f"Acordo em 'relevante ou não' (nota >= {NOTA_RELEVANTE}): {r.acordo_relevancia:.0%}",
        f"Kappa ponderado: {fmt(r.kappa_ponderado)} (intervalo de 95%: {faixa})",
        f"Kappa em 'relevante ou não': {fmt(r.kappa_relevancia)}",
        f"Viés médio (IA - você): {r.vies_medio:+.2f}  ({'IA mais generosa' if r.vies_medio > 0 else 'IA mais rigorosa' if r.vies_medio < 0 else 'sem viés'})",
        "",
        "Matriz (linhas = sua nota, colunas = nota da IA):",
        "        IA0  IA1  IA2  IA3",
    ]
    for nota, linha in enumerate(r.matriz):
        linhas.append(f"  você{nota} " + " ".join(f"{v:4}" for v in linha))
    linhas += ["", f"Resultado: {'APROVADO' if r.aprovado else 'NÃO APROVADO'} ({r.motivo})",
               f"Discordâncias: {len(r.discordancias)} (revise com `mini-lcm revisar`)"]
    return "\n".join(linhas)


# ─────────────────────────── Passo 4: revisão e auditoria ───────────────────────────


def revisar(
    consultas: list[ConsultaRotulada],
    itens: dict[str, Item],
    distancia_minima: int = 1,
    perguntar: Callable[[str], str] | None = None,
    mostrar: Callable[[str], None] | None = None,
) -> int:
    """Mostra cada discordância COM a nota e a justificativa da IA; você mantém ou corrige a sua nota.

    Aqui você vê a opinião da IA de propósito: é a etapa de arbitragem, não de
    rotulagem cega. Às vezes a IA notou algo na foto que passou despercebido;
    às vezes errou. Enter mantém a sua nota.
    """
    perguntar, mostrar = perguntar or input, mostrar or print
    alteradas = 0
    por_id = {c.id: c for c in consultas}
    discordantes = [x for x in pares(consultas, "todos") if abs(x[2] - x[3]) >= distancia_minima]
    discordantes.sort(key=lambda x: -abs(x[2] - x[3]))
    mostrar(f"{len(discordantes)} discordâncias. Enter = manter sua nota · {LEGENDA}\n")
    for n, (cid, item_id, humana, nota_ia) in enumerate(discordantes, start=1):
        c, item = por_id[cid], itens.get(item_id)
        if item is None:
            continue
        mostrar(f'[{n}/{len(discordantes)}] Busca: "{c.texto}"' + (f"  perfil: {c.perfil}" if c.perfil else ""))
        mostrar(f"    {_descrever(item)}")
        mostrar(f"    sua nota: {humana} · IA: {nota_ia} — {c.notas_ia[item_id].get('justificativa', '')}")
        while True:
            resposta = perguntar("    nova nota? [Enter/0-3/q] ").strip().lower()
            if resposta in {"", "q", "0", "1", "2", "3"}:
                break
        if resposta == "q":
            break
        if resposta and int(resposta) != humana:
            c.notas_humanas[item_id] = int(resposta)
            alteradas += 1
    return alteradas


def amostra_auditoria(consultas: list[ConsultaRotulada], tamanho: int, semente: int | None = None) -> list[tuple[str, str]]:
    """Sorteia pares que SÓ a IA julgou, estratificando pela nota da IA.

    Estratificar = sortear a mesma quantidade de cada nota (0, 1, 2, 3), em vez de
    sortear ao acaso. Sem isso, como a maioria dos produtos costuma ser irrelevante,
    a amostra viria quase toda de notas 0 e diria pouco sobre os erros nas notas altas.
    """
    rng = random.Random(semente)
    por_nota: dict[int, list[tuple[str, str]]] = {n: [] for n in NOTAS}
    for c in consultas:
        for item_id, j in c.notas_ia.items():
            if item_id not in c.notas_humanas:
                por_nota[int(j["nota"])].append((c.id, item_id))
    for lista in por_nota.values():
        rng.shuffle(lista)
    escolhidos: list[tuple[str, str]] = []
    while len(escolhidos) < tamanho and any(por_nota.values()):
        for nota in NOTAS:
            if por_nota[nota] and len(escolhidos) < tamanho:
                escolhidos.append(por_nota[nota].pop())
    rng.shuffle(escolhidos)
    return escolhidos


def amostra_calibracao(consultas: list[ConsultaRotulada], aleatoria: int, estratificada: int,
                       semente: int | None = 0) -> list[tuple[str, str, bool]]:
    """Sorteia pares julgados só pela IA para você rotular: (consulta_id, item_id, é_auditoria).

    Duas partes, com papéis diferentes:
    - `aleatoria` pares sorteados ao acaso: o conjunto de CALIBRAÇÃO. Representa a
      distribuição real das notas, e é nele que o kappa decide se o juiz é aprovado.
    - `estratificada` pares sorteados com a mesma quantidade de cada nota da IA: a
      AUDITORIA. Mostra onde o juiz erra nas notas raras (os 3, quase sempre), mas tem
      notas raras demais para representar a população, por isso é reportada à parte.
    Na rotulagem as duas partes aparecem misturadas e sem a nota da IA.
    """
    rng = random.Random(semente)
    candidatos = [(c.id, item_id) for c in consultas for item_id in c.notas_ia if item_id not in c.notas_humanas]
    rng.shuffle(candidatos)
    calibracao = candidatos[:aleatoria]
    usados = set(calibracao)
    por_nota: dict[int, list[tuple[str, str]]] = {n: [] for n in NOTAS}
    for c in consultas:
        for item_id, j in c.notas_ia.items():
            if item_id not in c.notas_humanas and (c.id, item_id) not in usados:
                por_nota[int(j["nota"])].append((c.id, item_id))
    for lista in por_nota.values():
        lista.sort()
        rng.shuffle(lista)
    auditoria: list[tuple[str, str]] = []
    while len(auditoria) < estratificada and any(por_nota.values()):
        for nota in NOTAS:
            if por_nota[nota] and len(auditoria) < estratificada:
                auditoria.append(por_nota[nota].pop())
    return [(cid, i, False) for cid, i in calibracao] + [(cid, i, True) for cid, i in auditoria]


def auditar(
    consultas: list[ConsultaRotulada],
    itens: dict[str, Item],
    amostra: list[tuple[str, str]],
    perguntar: Callable[[str], str] | None = None,
    mostrar: Callable[[str], None] | None = None,
) -> int:
    """Você dá a sua nota, SEM ver a da IA, para os pares sorteados. Eles passam a contar na concordância."""
    perguntar, mostrar = perguntar or input, mostrar or print
    por_id = {c.id: c for c in consultas}
    feitos = 0
    mostrar(f"{len(amostra)} pares para auditar (a nota da IA fica escondida). {LEGENDA}\n")
    for n, (cid, item_id) in enumerate(amostra, start=1):
        c, item = por_id[cid], itens.get(item_id)
        if item is None:
            continue
        mostrar(f'[{n}/{len(amostra)}] Busca: "{c.texto}"' + (f"  perfil: {c.perfil}" if c.perfil else ""))
        mostrar(f"    {_descrever(item)}")
        nota = perguntar_nota(perguntar)
        if nota is None:
            break
        c.notas_humanas[item_id] = nota
        if item_id not in c.auditados:
            c.auditados.append(item_id)
        feitos += 1
    return feitos
