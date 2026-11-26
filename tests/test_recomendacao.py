"""Fase 2: perfil, recomendação, busca personalizada e avaliação temporal.

Cenário sintético com resposta conhecida: dois grupos de bonecas (góticas e bebês) e dois
grupos de usuários (fãs de cada um). Um bom recomendador deve sugerir góticas para os
fãs de góticas e bebês para os fãs de bebês.
"""

import random

import numpy as np
import pytest

from mini_lcm.armazenamento import ArmazenamentoLocal
from mini_lcm.avaliacao_recomendacao import avaliar_recomendacao, dividir, relatorio_recomendacao_markdown
from mini_lcm.busca import buscar
from mini_lcm.embeddings import EmbedderFalso
from mini_lcm.indexacao import gerar_embeddings
from mini_lcm.modelos import Avaliacao, Filtros
from mini_lcm.perfil import construir_perfil, peso_da_avaliacao, perfis_do_usuario
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
    assert len(casos) == 1 and casos[0].teste.item_id == "b"  # a mais recente, não a última da lista
    assert {a.item_id for a in casos[0].treino} == {"a", "c"}
    assert all(not (a.usuario == "u" and a.item_id == "b") for a in treino)  # o teste não entra no treino
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
    cli.main(["avaliar-recomendacao", "--categoria", "bonecas", "-k", "3", "--curva", "--salvar", str(arm.pasta / "r.md")])
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
