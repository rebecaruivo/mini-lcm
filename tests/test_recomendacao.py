"""Fase 2: perfil, recomendação, busca personalizada e avaliação temporal.

Cenário sintético com resposta conhecida: dois grupos de bonecas (góticas e bebês) e dois
grupos de usuários (fãs de cada um). Um bom recomendador deve sugerir góticas para os
fãs de góticas e bebês para os fãs de bebês.
"""

import random

import numpy as np
import pytest

from mini_lcm.armazenamento import ArmazenamentoLocal
from mini_lcm.avaliacao_recomendacao import (avaliar_recomendacao, cortes_semestrais, dividir, dividir_por_data,
                                             intervalo_bootstrap, relatorio_recomendacao_markdown, sem_a_ultima)
from mini_lcm.busca import buscar
from mini_lcm.embeddings import EmbedderFalso
from mini_lcm.indexacao import gerar_embeddings
from mini_lcm.modelos import Avaliacao, Filtros
from mini_lcm.perfil import construir_perfil, fator_recencia, peso_da_avaliacao, perfis_do_usuario, pesos_do_historico
from mini_lcm.recomendacao import Catalogo, Recomendador, explicar, ranks

from .conftest import item

GOTICAS = [f"G{i}" for i in range(1, 7)]
BEBES = [f"B{i}" for i in range(1, 7)]


def av(usuario, item_id, nota=5, dia=1):
    return Avaliacao(usuario=usuario, item_id=item_id, nota=nota, momento=f"2020-01-{dia:02d}T00:00:00+00:00")


@pytest.fixture
def mundo(tmp_path):
    itens = ([item(g, f"boneca gotica vampira caveira {g}", preco=50, vendedor=g) for g in GOTICAS]
             + [item(b, f"boneca bebe chupeta mamadeira {b}", preco=30, vendedor=b) for b in BEBES]
             + [item("P1", "placa de video rtx", categoria="placas", preco=900, vendedor="p")])
    emb = EmbedderFalso(descricoes_fotos={f"https://foto/{i.id}.jpg": i.titulo for i in itens})
    arm = ArmazenamentoLocal(tmp_path / "dados")
    arm.salvar_itens(itens)
    gerar_embeddings(arm, emb)
    rng = random.Random(0)
    avaliacoes = []
    for n in range(20):
        for grupo, prefixo in ((GOTICAS, "fa_gotica"), (BEBES, "fa_bebe")):
            for dia, i in enumerate(rng.sample(grupo, 4), start=1):
                avaliacoes.append(av(f"{prefixo}{n}", i, nota=rng.choice([4, 5]), dia=dia))
    arm.salvar_avaliacoes(avaliacoes)
    return arm, emb


# ───── perfil ─────

def test_peso_da_avaliacao():
    assert [peso_da_avaliacao(n) for n in (5, 4, 3, 2, 1)] == [1.5, 1.0, 0.5, 0.0, -0.5]


def test_perfil_aproxima_do_que_gostou_e_afasta_do_que_nao_gostou():
    gostou, nao_gostou = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    vetores = {"texto": {"a": gostou, "b": nao_gostou}}
    perfil = construir_perfil([av("u", "a", 5), av("u", "b", 1)], vetores)["texto"]
    assert perfil @ gostou > 0 > perfil @ nao_gostou
    assert np.isclose(np.linalg.norm(perfil), 1.0)
    assert construir_perfil([av("u", "b", 1)], vetores) == {}  # só negativas: sem direção de gosto
    assert construir_perfil([av("u", "a", 5, dia=1), av("u", "a", 1, dia=2)], vetores) == {}  # vale a mais recente


def test_perfis_por_categoria(mundo):
    arm, emb = mundo
    arm.salvar_avaliacoes([av("misto", "G1"), av("misto", "P1")])
    perfis = perfis_do_usuario(arm, emb, "misto")
    assert set(perfis) == {"bonecas", "placas"} and set(perfis["bonecas"]) == {"texto", "imagem"}
    assert perfis_do_usuario(arm, emb, "ninguem") == {}


# ───── recomendação ─────

def test_ranks():
    assert ranks(np.array([0.1, 0.9, 0.5])).tolist() == [3, 1, 2]


def test_metodos_recomendam_o_grupo_certo(mundo):
    arm, emb = mundo
    cat = Catalogo.carregar(arm, emb, "bonecas")
    assert len(cat.ids) == 12 and set(cat.matrizes) == {"texto", "imagem"}
    rec = Recomendador(cat).ajustar(arm.avaliacoes_de(categoria="bonecas"))
    historico = [av("nova", "G1"), av("nova", "G2", dia=2)]
    for metodo in ("ultimo", "conteudo", "coocorrencia", "hibrido"):
        top = [i for i, _ in rec.recomendar(metodo, historico, k=4)]
        assert set(top) <= set(GOTICAS) - {"G1", "G2"}, metodo  # nunca repete o histórico
    assert len(rec.recomendar("popularidade", historico, k=12)) == 10  # 12 - 2 já avaliados


def test_aleatorio_e_reprodutivel_por_usuario(mundo):
    arm, emb = mundo
    rec = Recomendador(Catalogo.carregar(arm, emb, "bonecas"))
    a = rec.recomendar("aleatorio", [], k=5, semente="u1")
    assert a == rec.recomendar("aleatorio", [], k=5, semente="u1") != rec.recomendar("aleatorio", [], k=5, semente="u2")


def test_coocorrencia_desempata_pela_popularidade(mundo):
    arm, emb = mundo
    cat = Catalogo.carregar(arm, emb, "bonecas")
    # só B1 tem interações além do usuário-alvo: ninguém coocorre com G1
    rec = Recomendador(cat).ajustar([av("x", "B1"), av("y", "B1"), av("z", "B2")])
    top = [i for i, _ in rec.recomendar("coocorrencia", [av("alvo", "G1")], k=2)]
    assert top == ["B1", "B2"]


def test_explicar_aponta_o_item_parecido_do_historico(mundo):
    arm, emb = mundo
    cat = Catalogo.carregar(arm, emb, "bonecas")
    assert explicar(cat, "G5", [av("u", "B1"), av("u", "G2")]) == "G2"


# ───── avaliação temporal ─────

def test_divisao_temporal_esconde_a_ultima_e_nao_vaza():
    avs = [av("u", "a", dia=1), av("u", "b", dia=3), av("u", "c", dia=2), av("curto", "a", dia=1)]
    casos, treino = dividir(avs, {"a", "b", "c"}, min_historico=3)
    assert len(casos) == 1 and casos[0].alvo.item_id == "b"  # a mais recente, não a última da lista
    assert {a.item_id for a in casos[0].treino} == {"a", "c"}
    assert all(not (a.usuario == "u" and a.item_id == "b") for a in treino)  # o alvo não entra no treino
    assert any(a.usuario == "curto" for a in treino)  # histórico curto: inteiro no treino


def test_quem_reavaliou_o_mesmo_produto_fica_de_fora():
    avs = [av("u", "a", dia=1), av("u", "b", dia=2), av("u", "c", dia=3), av("u", "a", dia=4)]
    casos, _ = dividir(avs, {"a", "b", "c"}, min_historico=3)
    assert casos == []


def test_avaliacao_temporal_personalizacao_ganha_do_acaso(mundo):
    arm, emb = mundo
    cat = Catalogo.carregar(arm, emb, "bonecas")
    rel = avaliar_recomendacao(cat, arm.avaliacoes_de(categoria="bonecas"), k=3, historicos=(1, None))
    assert rel["usuarios_testados"] == 40
    r = rel["resultados"]["todas"]
    for metodo in ("conteudo", "coocorrencia", "hibrido"):
        assert r[metodo]["hit@3"] == 1.0, metodo  # sobram 3 do grupo certo: os 3 primeiros têm de ser eles
        assert r[metodo]["hit@3"] > r["aleatorio"]["hit@3"]
    assert r["popularidade"]["cobertura"] < r["conteudo"]["cobertura"]  # popularidade: mesma lista para todos
    assert set(rel["resultados"]) == {"1", "todas"}
    md = relatorio_recomendacao_markdown(rel, "bonecas", emb.nome)
    assert "| hibrido |" in md and "Curva de histórico" in md
    lo, hi = r["aleatorio"]["hit@3_ic95"]
    assert lo <= r["aleatorio"]["hit@3"] <= hi
    assert len(rel["comparacoes"]) == 5  # a melhor configuração contra cada um dos outros 5 métodos
    assert "Comparações pareadas" in md


def test_intervalo_bootstrap():
    lo, hi = intervalo_bootstrap(np.array([0.0, 1.0] * 500))
    assert lo < 0.5 < hi and hi - lo < 0.1  # média 0,5; com 1.000 valores o intervalo fica estreito
    assert intervalo_bootstrap(np.zeros(50)) == (0.0, 0.0)  # sem variação, sem incerteza


def test_avaliacao_sem_historico_suficiente_explica(mundo):
    arm, emb = mundo
    with pytest.raises(ValueError, match="selecao: usuarios"):
        avaliar_recomendacao(Catalogo.carregar(arm, emb, "bonecas"), [av("u", "G1")])


# ───── busca personalizada ─────

def test_busca_personalizada_reordena_pelo_gosto(mundo):
    arm, emb = mundo
    consulta, filtros = "boneca", Filtros(categoria="bonecas")
    gotica = perfis_do_usuario(arm, emb, "fa_gotica0")
    bebe = perfis_do_usuario(arm, emb, "fa_bebe0")
    top_gotica = [r.item.id for r in buscar(arm, emb, consulta, filtros, k=4, perfis=gotica, peso_perfil=2.0)]
    top_bebe = [r.item.id for r in buscar(arm, emb, consulta, filtros, k=4, perfis=bebe, peso_perfil=2.0)]
    assert sum(i in GOTICAS for i in top_gotica) > sum(i in GOTICAS for i in top_bebe)
    r = buscar(arm, emb, consulta, filtros, k=1, perfis=gotica)[0]
    assert {"perfil:texto", "perfil:imagem"} <= set(r.posicoes)  # o resultado explica o peso do perfil


def test_peso_zero_e_a_busca_sem_personalizacao(mundo):
    arm, emb = mundo
    perfis = perfis_do_usuario(arm, emb, "fa_gotica0")
    sem = [r.item.id for r in buscar(arm, emb, "boneca bebe", k=5)]
    zero = [r.item.id for r in buscar(arm, emb, "boneca bebe", k=5, perfis=perfis, peso_perfil=0)]
    assert sem == zero


def test_personalizacao_so_reordena_candidatos_da_busca(mundo):
    arm, emb = mundo
    perfis = perfis_do_usuario(arm, emb, "fa_gotica0")
    # filtro de categoria "placas": o gosto por bonecas não pode trazer bonecas para cá
    r = buscar(arm, emb, "placa", Filtros(categoria="placas"), k=5, perfis=perfis, peso_perfil=5)
    assert [x.item.id for x in r] == ["P1"]


# ───── armazenamento ─────

def test_remover_itens_leva_vetores_e_avaliacoes(mundo):
    arm, emb = mundo
    arm.remover_itens(["G1"])
    assert "G1" not in arm.ids_categoria("bonecas")
    assert "G1" not in arm.vetores(emb.chave("texto"), "texto", ["G1", "G2"])
    assert all(a.item_id != "G1" for a in arm.avaliacoes_de())
    reaberto = ArmazenamentoLocal(arm.pasta)
    assert "G1" not in reaberto.ids_categoria("bonecas") and len(reaberto.avaliacoes_de()) == len(arm.avaliacoes_de())


def test_matriz_e_avaliacoes_por_categoria_e_usuario(mundo):
    arm, emb = mundo
    ids, m = arm.matriz(emb.chave("texto"), "texto", "bonecas")
    assert ids == sorted(GOTICAS + BEBES) and m.shape == (12, emb.dimensoes)
    avs = arm.avaliacoes_de(categoria="bonecas", usuario="fa_gotica3")
    assert len(avs) == 4 and [a.momento for a in avs] == sorted(a.momento for a in avs)


# ───── linha de comando ─────

def test_cli_fase_2(mundo, monkeypatch, capsys):
    from mini_lcm import cli
    from mini_lcm.config import Config

    arm, _ = mundo
    cfg = Config(database_url=None, embedder="fake", pasta_dados=arm.pasta, pasta_brutos=arm.pasta / "b")
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    cli.main(["usuarios", "--categoria", "bonecas", "-n", "3"])
    assert "40 usuários, 0 com 5 ou mais" in capsys.readouterr().out
    cli.main(["perfil", "fa_gotica0", "--categoria", "bonecas", "-k", "2"])
    saida = capsys.readouterr().out
    assert "Histórico de fa_gotica0" in saida and "parecido com:" in saida
    cli.main(["buscar", "boneca", "--categoria", "bonecas", "--usuario", "fa_bebe0", "-k", "3"])
    assert "Personalizado para fa_bebe0" in capsys.readouterr().out
    cli.main(["avaliar-recomendacao", "--categoria", "bonecas", "-k", "3", "--curva", "--corte", "usuario",
              "--salvar", str(arm.pasta / "r.md")])
    saida = capsys.readouterr().out
    assert "| conteudo |" in saida and (arm.pasta / "r.md").exists()


def test_cli_importar_substitui_o_catalogo_da_categoria(tmp_path, monkeypatch, capsys):
    from mini_lcm import cli
    from mini_lcm.config import Config

    from .test_amazon import categoria, escrever, meta

    brutos = tmp_path / "brutos"
    brutos.mkdir()
    escrever(brutos / "meta_Toys_and_Games.jsonl.gz", [meta("A1", "Monster High Doll", ["Dolls"])])
    escrever(brutos / "Toys_and_Games.jsonl.gz", [])
    cfg = Config(database_url=None, embedder="fake", pasta_dados=tmp_path / "dados", pasta_brutos=brutos,
                 categorias=[categoria(selecao="produtos")])
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    arm = cli.abrir_armazenamento(cfg)
    arm.salvar_itens([item("VELHO", "boneca antiga", categoria="bonecas"), item("P1", "placa", categoria="placas")])
    cli.main(["importar"])
    assert "1 removidos" in capsys.readouterr().out
    depois = cli.abrir_armazenamento(cfg)
    assert depois.ids_categoria("bonecas") == {"A1"}
    assert depois.ids_categoria("placas") == {"P1"}  # outra categoria não é tocada



def test_cli_importar_substitui_tambem_as_avaliacoes(tmp_path, monkeypatch, capsys):
    """Produto que continua no catálogo não pode guardar avaliações de uma importação anterior."""
    from mini_lcm import cli
    from mini_lcm.config import Config

    from .test_amazon import categoria, escrever, meta

    brutos = tmp_path / "brutos"
    brutos.mkdir()
    escrever(brutos / "meta_Toys_and_Games.jsonl.gz", [meta("A1", "Monster High Doll", ["Dolls"])])
    escrever(brutos / "Toys_and_Games.jsonl.gz",
             [{"user_id": "NOVO", "parent_asin": "A1", "rating": 5, "timestamp": 1600000000000}])
    cfg = Config(database_url=None, embedder="fake", pasta_dados=tmp_path / "dados", pasta_brutos=brutos,
                 categorias=[categoria(selecao="produtos")])
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    arm = cli.abrir_armazenamento(cfg)
    arm.salvar_itens([item("A1", "boneca", categoria="bonecas"), item("P1", "placa", categoria="placas")])
    arm.salvar_avaliacoes([av("ANTIGO", "A1"), av("OUTRA_CATEGORIA", "P1")])
    cli.main(["importar"])
    depois = cli.abrir_armazenamento(cfg)
    assert {a.usuario for a in depois.avaliacoes_de(categoria="bonecas")} == {"NOVO"}
    assert {a.usuario for a in depois.avaliacoes_de(categoria="placas")} == {"OUTRA_CATEGORIA"}  # intocada
    cli.main(["importar"])  # rodar de novo dá o mesmo resultado
    assert len(cli.abrir_armazenamento(cfg).avaliacoes_de(categoria="bonecas")) == 1


# ───── recência ─────

def test_fator_recencia():
    assert fator_recencia(0, 2) == 1.0
    assert fator_recencia(2, 2) == pytest.approx(0.5)
    assert fator_recencia(4, 2) == pytest.approx(0.25)
    assert fator_recencia(10, None) == 1.0  # desligado: média simples


def test_pesos_do_historico_contam_a_idade_pela_ordem_no_tempo():
    hist = [av("u", "c", dia=3), av("u", "a", dia=1), av("u", "b", dia=2, nota=4)]
    pesos = pesos_do_historico(hist, meia_vida=1)
    assert pesos == pytest.approx({"a": 1.5 * 0.25, "b": 1.0 * 0.5, "c": 1.5})
    # reavaliar um produto antigo o torna recente de novo
    pesos = pesos_do_historico(hist + [av("u", "a", dia=4)], meia_vida=1)
    assert pesos["a"] == pytest.approx(1.5) and pesos["c"] == pytest.approx(0.75)


def test_perfil_com_recencia_puxa_para_o_mais_recente():
    vetores = {"texto": {"velho": np.array([1.0, 0.0]), "novo": np.array([0.0, 1.0])}}
    hist = [av("u", "velho", dia=1), av("u", "novo", dia=2)]
    sem = construir_perfil(hist, vetores)["texto"]
    com = construir_perfil(hist, vetores, meia_vida=1)["texto"]
    assert sem[0] == pytest.approx(sem[1])  # média simples: no meio
    assert com[1] > com[0]  # com recência: mais perto do novo


def test_sem_a_ultima_esconde_o_teste():
    avs = [av("u", "a", dia=1), av("u", "b", dia=2), av("u", "c", dia=3), av("v", "a", dia=5)]
    sobra = sem_a_ultima(avs, {"a", "b", "c"})
    assert {(a.usuario, a.item_id) for a in sobra} == {("u", "a"), ("u", "b")}


def test_conjunto_de_escolha_nao_usa_a_conferencia(mundo):
    arm, emb = mundo
    cat = Catalogo.carregar(arm, emb, "bonecas")
    avs = arm.avaliacoes_de(categoria="bonecas")
    rel = avaliar_recomendacao(cat, avs, metodos=("conteudo",), meia_vida=1, escolha=True)
    assert rel["conjunto"] == "escolha" and rel["meia_vida"] == 1
    conferencia = {(c.usuario, c.alvo.item_id) for c in dividir(avs, set(cat.ids))[0]}
    escolha = {(c.usuario, c.alvo.item_id) for c in dividir(sem_a_ultima(avs, set(cat.ids)), set(cat.ids))[0]}
    assert not conferencia & escolha  # o alvo do conjunto de escolha nunca é o da conferência


# ───── corte por data, em alta ─────

def avd(usuario, item_id, data, nota=5):
    return Avaliacao(usuario=usuario, item_id=item_id, nota=nota, momento=f"{data}T12:00:00+00:00")


def test_cortes_semestrais():
    avs = [avd("u", "a", "2019-03-05"), avd("u", "b", "2020-11-01")]
    assert cortes_semestrais(avs) == [f"{a}-{m}-01T00:00:00+00:00" for a in (2019, 2020) for m in ("01", "07")]


def test_dividir_por_data_nao_ve_o_futuro():
    avs = [avd("u", "a", "2019-01-10"), avd("u", "b", "2019-02-10"), avd("u", "c", "2019-03-10"),
           avd("u", "d", "2019-08-01"), avd("u", "e", "2019-09-01"),  # depois do corte: d é o alvo
           avd("v", "a", "2019-07-20"), avd("v", "d", "2020-02-01")]
    casos, treino = dividir_por_data(avs, {"a", "b", "c", "d", "e"}, "2019-07-01T00:00:00+00:00",
                                     "2020-01-01T00:00:00+00:00")
    assert [(c.usuario, c.alvo.item_id, [a.item_id for a in c.treino]) for c in casos] == [("u", "d", ["a", "b", "c"])]
    assert all(a.momento < "2019-07-01" for a in treino)  # nada posterior ao corte
    assert {a.item_id for a in treino} == {"a", "b", "c"}  # v só aparece depois do corte


def test_em_alta_conta_so_a_janela_antes_de_agora(mundo):
    arm, emb = mundo
    cat = Catalogo.carregar(arm, emb, "bonecas")
    avs = [avd("x", "G1", "2019-01-01"), avd("y", "G1", "2019-01-02"),  # G1: popular, mas antigo
           avd("z", "B1", "2019-06-20"),                                 # B1: em alta
           avd("w", "B2", "2019-07-05")]                                 # B2: depois de agora, não conta
    rec = Recomendador(cat, janela_dias=30).ajustar(avs, agora="2019-07-01T00:00:00+00:00")
    i = cat.indice
    assert rec.em_alta[i["B1"]] == 1 and rec.em_alta[i["G1"]] == 0 and rec.em_alta[i["B2"]] == 0
    assert rec.popularidade[i["G1"]] == 2 and rec.popularidade[i["B2"]] == 0
    top = [x for x, _ in rec.recomendar("em_alta", [], k=1)]
    assert top == ["B1"]


def test_bootstrap_por_grupos_alarga_o_intervalo():
    # 20 pessoas, cada uma repetida 10 vezes com o mesmo resultado: na prática, 20 observações, não 200
    valores = np.repeat(np.array([0.0, 1.0] * 10), 10)
    grupos = np.repeat(np.arange(20), 10).astype(str).tolist()
    lo_s, hi_s = intervalo_bootstrap(valores)
    lo_g, hi_g = intervalo_bootstrap(valores, grupos=grupos)
    assert (hi_g - lo_g) > 2 * (hi_s - lo_s)


@pytest.fixture
def mundo_no_tempo(mundo):
    """As mesmas fãs do `mundo`, agora comprando ao longo de 2018 a 2021."""
    arm, emb = mundo
    avs = []
    for u in range(30):
        grupo = GOTICAS if u % 2 == 0 else BEBES
        for n, item_id in enumerate(grupo[:5]):
            ano, mes = 2018 + (u % 4), 1 + n * 2
            avs.append(avd(f"t{u}", item_id, f"{ano}-{mes:02d}-1{u % 9}"))
    arm.salvar_avaliacoes(avs)
    return arm, emb, avs


def test_avaliacao_por_data(mundo_no_tempo):
    arm, emb, avs = mundo_no_tempo
    cat = Catalogo.carregar(arm, emb, "bonecas")
    rel = avaliar_recomendacao(cat, avs, corte="data", k=3)
    r = rel["resultados"]["todas"]
    assert rel["corte"] == "data" and rel["casos"] > 0 and "hibrido_alta" in r and "em_alta" in r
    assert r["conteudo"]["hit@3"] > r["aleatorio"]["hit@3"]
    val = avaliar_recomendacao(cat, avs, corte="data", escolha=True, k=3)
    assert val["periodo"] < rel["periodo"]  # o conjunto de escolha vem antes da conferência no tempo
    md = relatorio_recomendacao_markdown(rel, "bonecas", emb.nome)
    assert "cortes semestrais" in md and "alvos novos" in md


def test_corte_por_usuario_omite_metodos_que_dependem_da_data(mundo):
    arm, emb = mundo
    rel = avaliar_recomendacao(Catalogo.carregar(arm, emb, "bonecas"), arm.avaliacoes_de(categoria="bonecas"), k=3)
    assert rel["metodos_omitidos"] == ["em_alta", "hibrido_alta"]
    assert "em_alta" not in rel["resultados"]["todas"]



# ───── catálogo do período e vagas para produtos novos ─────

def _com_lancamento(arm, datas):
    itens = arm.obter_itens(list(datas))
    for item_id, data in datas.items():
        itens[item_id].lancamento = data
    arm.salvar_itens(list(itens.values()))


def test_lancamento_sobrevive_ao_salvar_e_carregar(mundo, tmp_path):
    arm, emb = mundo
    _com_lancamento(arm, {"G6": "2019-08-01"})
    assert ArmazenamentoLocal(arm.pasta).obter_itens(["G6"])["G6"].lancamento == "2019-08-01"
    assert Catalogo.carregar(arm, emb, "bonecas").lancamentos[Catalogo.carregar(arm, emb, "bonecas").indice["G6"]] == "2019-08-01"


def test_catalogo_do_periodo_nao_usa_avaliacoes_futuras(mundo):
    """Produto novo sem data de lançamento não concorre, mesmo que alguém o avalie na janela."""
    from mini_lcm.avaliacao_recomendacao import _conjuntos_por_data

    arm, emb = mundo
    _com_lancamento(arm, {"G5": "2019-08-01"})  # lançado durante a janela: concorre
    avs = [avd(f"u{n}", g, f"2019-0{1 + n}-10") for n in range(3) for g in ("G1", "G2", "G3")]
    avs += [avd("u0", "G4", "2019-09-01")]  # G4: primeira avaliação só na janela, sem data de lançamento
    cat = Catalogo.carregar(arm, emb, "bonecas")
    conjuntos, _ = _conjuntos_por_data(cat, avs, 3, escolha=True)  # um corte só (jul/2019): cai no conjunto de escolha
    rec, casos, disponiveis = conjuntos[-1]
    assert [c.alvo.item_id for c in casos] == ["G4"]
    i = cat.indice
    assert disponiveis[i["G1"]] and disponiveis[i["G5"]]
    assert not disponiveis[i["G4"]] and not disponiveis[i["B1"]]


def test_vagas_para_novos_trazem_produtos_sem_interacao(mundo):
    arm, emb = mundo
    cat = Catalogo.carregar(arm, emb, "bonecas")
    avs = [avd(f"x{n}", "B1", "2019-06-20") for n in range(5)] + [avd("x9", "G1", "2019-06-01")]
    historico = [avd("eu", "G1", "2019-06-02")]
    i = cat.indice
    for vagas in (0, 2):
        rec = Recomendador(cat, janela_dias=30, vagas_novos=vagas).ajustar(avs + historico, agora="2019-07-01T00:00:00+00:00")
        top = [x for x, _ in rec.recomendar("hibrido_alta", historico, k=10)]
        novos_no_topo = [x for x in top if rec.popularidade[i[x]] == 0]
        if vagas:
            assert top[-2:] == novos_no_topo[-2:] and set(top[-2:]) <= set(GOTICAS)  # góticas novas, pelo gosto
        assert len(top) == 10
