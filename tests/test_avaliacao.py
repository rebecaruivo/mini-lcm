import pytest

from mini_lcm import cli
from mini_lcm.avaliacao import (ConsultaRotulada, avaliar, carregar_gabarito, pool_para_rotular, recall_em_k,
                                reciproco_do_rank, relatorio_markdown, rotular, salvar_gabarito)
from mini_lcm.config import Config
from mini_lcm.custo import RegistroCusto, resumir
from mini_lcm.modelos import Filtros


def test_recall_e_mrr():
    assert recall_em_k(["a", "x", "b"], {"a", "b", "c", "d"}, 10) == 0.5
    assert recall_em_k(["a", "x", "b"], {"a", "b"}, 1) == 0.5
    assert recall_em_k(["a"], set(), 10) == 0.0
    assert reciproco_do_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciproco_do_rank(["x"], {"a"}) == 0.0


def test_avaliacao_compara_modos_e_conta_duplicatas_como_um_grupo(indexado):
    arm, emb = indexado
    gabarito = [
        # B1 e B5 são o mesmo anúncio republicado: contam como 1 relevante, não 2
        ConsultaRotulada(id="c1", texto="draculaura", filtros={"categoria": "bonecas"}, relevantes=["B1", "B2", "B5"]),
        ConsultaRotulada(id="c2", texto="rtx 3060", filtros={"categoria": "placas"}, relevantes=["P2"]),
        ConsultaRotulada(id="c3", texto="sem rótulo ainda"),  # ignorada: nenhum relevante marcado
    ]
    rel = avaliar(arm, emb, gabarito, k=10)
    assert rel["consultas"] == 2
    c1 = rel["modos"]["fusao"]["por_consulta"][0]
    assert c1["relevantes"] == 2
    assert c1["recall@10"] == 1.0  # com k=10 e um catálogo de 8 itens, tudo aparece
    assert set(rel["modos"]) == {"texto", "imagem", "fusao"}
    md = relatorio_markdown(rel)
    assert "| fusao |" in md and "draculaura" in md


def test_avaliacao_sem_rotulos_explica_o_que_fazer(indexado):
    arm, emb = indexado
    with pytest.raises(ValueError, match="rotular"):
        avaliar(arm, emb, [ConsultaRotulada(id="c1", texto="x")])


def test_pool_junta_os_modos_e_embaralha(indexado):
    arm, emb = indexado
    pool = pool_para_rotular(arm, emb, "draculaura", Filtros(categoria="bonecas"), profundidade=2, semente=1)
    ids = [i.id for i in pool]
    assert len(ids) == len(set(ids))
    assert "B2" in ids  # veio do modo imagem, mesmo não estando no top 2 do texto


def test_rotular_registra_respostas_e_retoma_de_onde_parou(indexado, tmp_path):
    arm, emb = indexado
    itens = list(arm.obter_itens(["B1", "B2", "B4"]).values())
    respostas = iter(["s", "talvez", "n", "q"])  # resposta inválida é perguntada de novo
    c = rotular(ConsultaRotulada(id="c1", texto="draculaura"), itens, perguntar=lambda _: next(respostas), mostrar=lambda _: None)
    assert c.relevantes == ["B1"] and c.avaliados == ["B1", "B2"]

    caminho = tmp_path / "gabarito.json"
    salvar_gabarito(caminho, [c])
    c2 = carregar_gabarito(caminho)[0]
    perguntas = []
    rotular(c2, itens, perguntar=lambda p: perguntas.append(p) or "s", mostrar=lambda _: None)
    assert len(perguntas) == 1  # só B4 faltava
    assert c2.relevantes == ["B1", "B4"]


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
