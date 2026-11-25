import pytest

from mini_lcm import cli
import math

from mini_lcm.avaliacao import (ConsultaRotulada, avaliar, carregar_gabarito, ndcg_em_k, obter_ou_criar,
                                pool_para_rotular, recall_em_k, reciproco_do_rank, relatorio_markdown, rotular,
                                salvar_gabarito)
from mini_lcm.config import Config
from mini_lcm.custo import RegistroCusto, resumir
from mini_lcm.modelos import Filtros


def test_recall_mrr_e_ndcg():
    assert recall_em_k(["a", "x", "b"], {"a", "b", "c", "d"}, 10) == 0.5
    assert recall_em_k(["a", "x", "b"], {"a", "b"}, 1) == 0.5
    assert recall_em_k(["a"], set(), 10) == 0.0
    assert reciproco_do_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciproco_do_rank(["x"], {"a"}) == 0.0
    notas = {"exato": 3, "parecido": 2, "relacionado": 1}
    assert ndcg_em_k(["exato", "parecido", "relacionado"], notas, 10) == pytest.approx(1.0)
    invertido = ndcg_em_k(["relacionado", "parecido", "exato"], notas, 10)
    assert 0 < invertido < 1  # mesmos itens, ordem pior: o recall não veria diferença, o NDCG vê
    # cálculo à mão: DCG = 1/log2(2) + 3/log2(3) + 7/log2(4); ideal = 7/1 + 3/log2(3) + 1/2
    assert invertido == pytest.approx((1 + 3 / math.log2(3) + 7 / 2) / (7 + 3 / math.log2(3) + 0.5))
    assert ndcg_em_k(["x"], {"x": 0}, 10) == 0.0


def gabarito_exemplo():
    return [
        # B1 e B5 são o mesmo produto cadastrado duas vezes: contam como um grupo só
        ConsultaRotulada(id="c1", texto="draculaura", filtros={"categoria": "bonecas"},
                         notas_humanas={"B1": 3, "B2": 3, "B5": 3, "B4": 0}),
        ConsultaRotulada(id="c2", texto="rtx 3060", filtros={"categoria": "placas"},
                         notas_ia={"P2": {"nota": 3, "justificativa": "", "modelo": "m", "rubrica": "r"},
                                   "P1": {"nota": 1, "justificativa": "", "modelo": "m", "rubrica": "r"}}),
        ConsultaRotulada(id="c3", texto="sem notas ainda"),
    ]


def test_avaliacao_por_fonte_e_por_grupo(indexado):
    arm, emb = indexado
    gab = gabarito_exemplo()
    rel = avaliar(arm, emb, gab, k=10, fonte="combinado")
    assert rel["consultas"] == 2
    c1 = rel["modos"]["fusao"]["por_consulta"][0]
    assert c1["relevantes"] == 2 and c1["recall@10"] == 1.0
    assert set(rel["modos"]) == {"texto", "imagem", "fusao"}
    assert avaliar(arm, emb, gab, fonte="humano")["consultas"] == 1
    assert avaliar(arm, emb, gab, fonte="ia")["consultas"] == 1
    md = relatorio_markdown(rel)
    assert "| fusao |" in md and "NDCG@10" in md and "draculaura" in md


def test_nota_humana_prevalece_sobre_a_da_ia():
    c = ConsultaRotulada(id="c", texto="x", notas_humanas={"a": 0},
                         notas_ia={"a": {"nota": 3}, "b": {"nota": 2}})
    assert c.notas("combinado") == {"a": 0, "b": 2}


def test_avaliacao_sem_notas_explica_o_que_fazer(indexado):
    arm, emb = indexado
    with pytest.raises(ValueError, match="rotular"):
        avaliar(arm, emb, [ConsultaRotulada(id="c1", texto="x")])


def test_pool_junta_os_modos_e_embaralha(indexado):
    arm, emb = indexado
    pool = pool_para_rotular(arm, emb, "draculaura", Filtros(categoria="bonecas"), profundidade=2, semente=1)
    ids = [i.id for i in pool]
    assert len(ids) == len(set(ids))
    assert "B2" in ids  # veio do modo imagem, mesmo não estando no top 2 do texto


def test_rotular_pede_nota_0_a_3_e_retoma_de_onde_parou(indexado, tmp_path):
    arm, emb = indexado
    itens = list(arm.obter_itens(["B1", "B2", "B4"]).values())
    respostas = iter(["3", "talvez", "7", "1", "q"])  # respostas inválidas são perguntadas de novo
    c = rotular(ConsultaRotulada(id="c1", texto="draculaura", perfil="colecionadora"), itens,
                perguntar=lambda _: next(respostas), mostrar=lambda _: None)
    assert c.notas_humanas == {"B1": 3, "B2": 1}

    caminho = tmp_path / "gabarito.json"
    salvar_gabarito(caminho, [c])
    c2 = carregar_gabarito(caminho)[0]
    assert c2.perfil == "colecionadora"
    perguntas = []
    rotular(c2, itens, perguntar=lambda p: perguntas.append(p) or "0", mostrar=lambda _: None)
    assert len(perguntas) == 1  # só B4 faltava
    assert c2.notas_humanas["B4"] == 0


def test_obter_ou_criar_reaproveita_consulta():
    gab = []
    a = obter_ou_criar(gab, "x", {"categoria": "c"})
    b = obter_ou_criar(gab, "x", {"categoria": "c"}, perfil="p")
    assert a is b and len(gab) == 1 and a.perfil == "p"
    assert obter_ou_criar(gab, "x", {}).id == "c2"  # filtros diferentes = outra consulta


def test_resumo_de_custos_agrupa_por_etapa():
    regs = [RegistroCusto(etapa="busca", modelo="m", itens=1, segundos=0.02),
            RegistroCusto(etapa="busca", modelo="m", itens=1, segundos=0.04),
            RegistroCusto(etapa="embeddings:texto", modelo="m", itens=100, segundos=5, custo_usd=0.002)]
    linhas = {l["etapa"]: l for l in resumir(regs)}
    assert linhas["busca"]["itens"] == 2 and linhas["busca"]["ms_por_item"] == 30.0
    assert linhas["embeddings:texto"]["usd_por_1000_itens"] == 0.02


def test_cli_ponta_a_ponta_com_embedder_falso(catalogo, tmp_path, monkeypatch, capsys):
    """indexar -> buscar -> custos -> status, com dados locais e o embedder falso (sem rede, sem modelos)."""
    itens, _ = catalogo
    cfg = Config(database_url=None, embedder="fake", pasta_dados=tmp_path / "dados",
                 pasta_brutos=tmp_path / "brutos", categorias=[])
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    cli.abrir_armazenamento(cfg).salvar_itens(itens)

    cli.main(["indexar"])
    assert "texto: 8 itens" in capsys.readouterr().out
    cli.main(["buscar", "placa rtx", "--categoria", "placas", "--preco-max", "1500", "-k", "3"])
    saida = capsys.readouterr().out
    assert "RTX 3060" in saida and "RTX 4060" not in saida  # a 4060 custa 1900
    cli.main(["custos"])
    assert "| busca |" in capsys.readouterr().out
    cli.main(["status"])
    assert '"itens": 8' in capsys.readouterr().out


def test_cli_importar_e_status(tmp_path, monkeypatch, capsys):
    from .test_amazon import categoria, escrever, meta

    brutos = tmp_path / "brutos"
    brutos.mkdir()
    escrever(brutos / "meta_Toys_and_Games.jsonl.gz", [meta("A1", "Monster High Doll", ["Dolls"])])
    escrever(brutos / "Toys_and_Games.jsonl.gz", [{"user_id": "U1", "parent_asin": "A1", "rating": 5, "timestamp": 1}])
    cfg = Config(database_url=None, embedder="fake", pasta_dados=tmp_path / "dados", pasta_brutos=brutos,
                 categorias=[categoria()])
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    cli.main(["importar"])
    assert "1 avaliações (1 novas)" in capsys.readouterr().out
    cli.main(["importar"])  # rodar de novo não duplica
    assert "1 avaliações (0 novas)" in capsys.readouterr().out
    cli.main(["status"])
    saida = capsys.readouterr().out
    assert '"avaliacoes": 1' in saida and '"usuarios": 1' in saida


def test_cli_fluxo_de_calibracao(catalogo, tmp_path, monkeypatch, capsys):
    """rotular -> julgar --apenas-rotulados -> concordancia -> julgar (escala) -> avaliar, com juiz falso."""
    from .test_calibracao import JuizFalso

    itens, _ = catalogo
    (tmp_path / "aval").mkdir()
    (tmp_path / "aval" / "rubrica.md").write_text("rubrica")
    cfg = Config(database_url=None, embedder="fake", pasta_dados=tmp_path / "dados", pasta_brutos=tmp_path / "b",
                 pasta_avaliacao=tmp_path / "aval")
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    juiz = JuizFalso()
    juiz.versao_rubrica = "v1"
    monkeypatch.setattr(cli, "_juiz", lambda c: juiz)
    arm = cli.abrir_armazenamento(cfg)
    arm.salvar_itens(itens)
    cli.main(["indexar"])

    respostas = iter(["3", "0", "3", "0"])  # o pool tem 4 bonecas (B1 e B5 são duplicatas)
    monkeypatch.setattr("builtins.input", lambda _: next(respostas))
    cli.main(["rotular", "draculaura", "--categoria", "bonecas", "--perfil", "colecionadora"])
    cli.main(["julgar", "--apenas-rotulados"])
    assert "4 julgamentos novos" in capsys.readouterr().out
    cli.main(["concordancia", "--min-pares", "3"])
    saida = capsys.readouterr().out
    assert "Pares com nota humana e da IA: 4" in saida
    cli.main(["julgar"])  # escala: o resto do pool
    assert "julgamentos novos" in capsys.readouterr().out
    cli.main(["avaliar", "--fonte", "combinado"])
    saida = capsys.readouterr().out
    assert "Atenção: o juiz ainda não passou" in saida  # 4 pares < 50
    assert "| fusao |" in saida
