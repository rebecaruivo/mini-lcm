import itertools

import pytest

from mini_lcm.avaliacao import ConsultaRotulada
from mini_lcm.calibracao import (amostra_auditoria, auditar, concordancia, intervalo_bootstrap, julgar_pendentes,
                                 kappa_binario, kappa_ponderado, pares, revisar)
from mini_lcm.juiz import ErroJuiz, Julgamento


def kappa_referencia(a, b, k=4):
    """Implementação direta da definição, com laços, para conferir a versão com numpy."""
    n = len(a)
    obs = [[sum(1 for x, y in zip(a, b) if x == i and y == j) / n for j in range(k)] for i in range(k)]
    lin = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2  # noqa: E731
    num = sum(w(i, j) * obs[i][j] for i, j in itertools.product(range(k), repeat=2))
    den = sum(w(i, j) * lin[i] * col[j] for i, j in itertools.product(range(k), repeat=2))
    return 1 - num / den


def test_kappa_ponderado():
    assert kappa_ponderado([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    a, b = [0, 1, 2, 3, 3, 0, 2, 1], [0, 1, 3, 3, 2, 1, 2, 0]
    assert kappa_ponderado(a, b) == pytest.approx(kappa_referencia(a, b))
    assert kappa_ponderado(a, b) == pytest.approx(kappa_ponderado(b, a))  # simétrico
    assert kappa_ponderado([0, 0, 3, 3], [3, 3, 0, 0]) < 0  # pior que o acaso
    assert kappa_ponderado([2, 2, 2], [2, 2, 2]) is None  # sem variação, não há o que medir
    assert kappa_ponderado([], []) is None


def test_discordar_por_3_pesa_mais_que_por_1():
    base = [0, 1, 2, 3] * 5
    perto = [min(x + 1, 3) if i == 0 else x for i, x in enumerate(base)]
    longe = [3 if i == 0 else x for i, x in enumerate(base)]
    assert kappa_ponderado(base, perto) > kappa_ponderado(base, longe)


def test_kappa_binario_usa_corte_de_relevancia():
    assert kappa_binario([2, 3, 0, 1], [3, 2, 1, 0]) == pytest.approx(1.0)  # discordam na nota, não na relevância


def test_intervalo_bootstrap_contem_o_valor_e_estreita_com_mais_dados():
    a, b = [0, 1, 2, 3, 3, 0, 2, 1], [0, 1, 3, 3, 2, 1, 2, 0]
    baixo, alto = intervalo_bootstrap(a, b)
    assert baixo <= kappa_ponderado(a, b) <= alto
    baixo10, alto10 = intervalo_bootstrap(a * 10, b * 10)
    assert (alto10 - baixo10) < (alto - baixo)


def consulta_com_notas(humanas, ia, auditados=()):
    return ConsultaRotulada(id="c1", texto="draculaura", notas_humanas=dict(humanas),
                            notas_ia={k: {"nota": v, "justificativa": f"just {k}", "modelo": "m", "rubrica": "r"}
                                      for k, v in ia.items()},
                            auditados=list(auditados))


def test_concordancia_aprova_so_com_pares_suficientes_e_kappa_alto():
    notas = {f"i{n}": n % 4 for n in range(60)}
    perfeita = concordancia([consulta_com_notas(notas, notas)])
    assert perfeita.aprovado and perfeita.n == 60 and perfeita.acordo_exato == 1.0
    poucos = {k: v for k, v in list(notas.items())[:10]}
    assert not concordancia([consulta_com_notas(poucos, poucos)]).aprovado
    generosa = {k: min(v + 2, 3) for k, v in notas.items()}
    r = concordancia([consulta_com_notas(notas, generosa)])
    assert not r.aprovado and r.vies_medio > 0
    assert r.matriz[0][2] == 15  # você deu 0, a IA deu 2
    assert abs(r.discordancias[0][2] - r.discordancias[0][3]) == 2  # as maiores discordâncias vêm primeiro


def test_pares_separam_rotulagem_e_auditoria():
    c = consulta_com_notas({"a": 1, "b": 2}, {"a": 1, "b": 3}, auditados=["b"])
    assert [p[1] for p in pares([c], "rotulagem")] == ["a"]
    assert [p[1] for p in pares([c], "auditoria")] == ["b"]
    assert len(pares([c], "todos")) == 2


class JuizFalso:
    modelo = "falso"

    def __init__(self, falhar=()):
        self.chamadas = []
        self.falhar = set(falhar)

    def julgar(self, consulta, perfil, item, registro=None):
        self.chamadas.append(item.id)
        if item.id in self.falhar:
            raise ErroJuiz("bloqueado")
        if registro:
            registro.itens += 1
        return Julgamento(nota=3 if "Draculaura" in item.titulo else 0, justificativa="j", modelo=self.modelo, rubrica="v1")


def test_julgar_pendentes_calibracao_escala_cache_e_limite(indexado):
    arm, emb = indexado
    c = ConsultaRotulada(id="c1", texto="draculaura", filtros={"categoria": "bonecas"}, notas_humanas={"B1": 3, "B4": 0})
    juiz = JuizFalso()
    julgar_pendentes([c], arm, emb, juiz, "v1", apenas_rotulados=True)
    assert sorted(juiz.chamadas) == ["B1", "B4"]  # calibração: só o que você rotulou

    juiz.chamadas.clear()
    salvos = []
    julgar_pendentes([c], arm, emb, juiz, "v1", salvar=lambda: salvos.append(1))
    assert "B1" not in juiz.chamadas and "B2" in juiz.chamadas  # cache: B1 já julgado com v1
    assert salvos  # o progresso é salvo

    juiz.chamadas.clear()
    julgar_pendentes([c], arm, emb, juiz, "v1")
    assert juiz.chamadas == []  # nada novo

    juiz.chamadas.clear()
    julgar_pendentes([c], arm, emb, juiz, "v2", limite=2)  # rubrica nova: refaz, mas respeita o limite
    assert len(juiz.chamadas) == 2


def test_julgar_pendentes_segue_quando_um_julgamento_falha(indexado):
    arm, emb = indexado
    c = ConsultaRotulada(id="c1", texto="draculaura", filtros={"categoria": "bonecas"}, notas_humanas={"B1": 3, "B4": 0})
    julgar_pendentes([c], arm, emb, JuizFalso(falhar={"B1"}), "v1", apenas_rotulados=True)
    assert list(c.notas_ia) == ["B4"]


def test_revisar_mostra_a_ia_e_permite_corrigir(indexado):
    arm, _ = indexado
    c = consulta_com_notas({"B1": 0, "B4": 1, "B3": 2}, {"B1": 3, "B4": 0, "B3": 2})
    vistos = []
    respostas = iter(["3", ""])  # corrige a maior discordância (B1), mantém a outra
    alteradas = revisar([c], arm.obter_itens(["B1", "B3", "B4"]), perguntar=lambda _: next(respostas),
                        mostrar=vistos.append)
    assert alteradas == 1 and c.notas_humanas == {"B1": 3, "B4": 1, "B3": 2}
    assert any("just B1" in v for v in vistos)  # a justificativa da IA aparece


def test_auditoria_estratifica_esconde_a_ia_e_marca_origem(indexado):
    arm, _ = indexado
    ia = {"B1": 3, "B2": 3, "B3": 2, "B4": 0, "B5": 0, "P1": 1}
    c = consulta_com_notas({}, ia)
    amostra = amostra_auditoria([c], 4, semente=1)
    notas_amostradas = sorted(ia[i] for _, i in amostra)
    assert notas_amostradas == [0, 1, 2, 3]  # uma de cada nota
    vistos = []
    auditar([c], arm.obter_itens(list(ia)), amostra, perguntar=lambda _: "2", mostrar=vistos.append)
    assert len(c.notas_humanas) == 4 and sorted(c.auditados) == sorted(i for _, i in amostra)
    assert not any("just" in v for v in vistos)  # a opinião da IA não aparece na auditoria
    restantes = {i for _, i in amostra_auditoria([c], 10)}
    assert restantes == set(ia) - set(c.notas_humanas)  # o que já foi auditado não é sorteado de novo
