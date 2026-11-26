import gzip
import json

import pytest

from mini_lcm.amazon import (importar_avaliacoes, importar_itens, importar_por_usuarios, inspecionar, item_de_meta,
                             lancamento_de_meta)
from mini_lcm.config import Categoria, FiltroCategoria


def escrever(caminho, linhas, ruins=0):
    with gzip.open(caminho, "wt", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l) + "\n")
        f.write("{isto não é json\n" * ruins)


def meta(asin, titulo, categorias, avaliacoes=10, preco=19.99, fotos=True, loja="Mattel"):
    return {
        "parent_asin": asin, "title": titulo, "categories": categorias, "rating_number": avaliacoes,
        "price": preco, "store": loja, "features": ["Articulada", "Acompanha acessórios"],
        "images": [{"variant": "PT01", "large": f"https://img/{asin}-pt.jpg"},
                   {"variant": "MAIN", "large": f"https://img/{asin}.jpg", "hi_res": f"https://img/{asin}-hd.jpg"}]
        if fotos else [],
    }


@pytest.fixture
def pasta(tmp_path):
    brinquedos = ["Toys & Games", "Dolls & Accessories", "Dolls"]
    escrever(tmp_path / "meta_Toys_and_Games.jsonl.gz", [
        meta("A1", "Monster High Draculaura Doll", brinquedos),
        meta("A2", "Fashion doll with pink hair", brinquedos, preco="$12.50"),
        meta("A3", "Monster High Lunch Box", ["Toys & Games", "Kids' Furniture"]),  # entra pelo título
        meta("A4", "Lego City Police Station", ["Toys & Games", "Building Toys"]),  # fora do filtro
        meta("A5", "Rare doll", brinquedos, avaliacoes=1),  # poucas avaliações
        meta("A6", "Doll without photo", brinquedos, fotos=False, preco=None),
        {"parent_asin": "A7", "title": "", "categories": brinquedos, "rating_number": 50},  # sem título
    ], ruins=1)
    escrever(tmp_path / "Toys_and_Games.jsonl.gz", [
        {"user_id": "U1", "parent_asin": "A1", "asin": "A1x", "rating": 5.0, "timestamp": 1588687728923, "verified_purchase": True},
        {"user_id": "U2", "parent_asin": "A1", "rating": 3.0, "timestamp": 1588687728, "verified_purchase": False},
        {"user_id": "U1", "parent_asin": "A4", "rating": 4.0, "timestamp": 1588687728923},  # item não importado
        {"user_id": "U3", "parent_asin": "A2", "rating": 4.0},  # sem data
    ])
    return tmp_path


def categoria(**kw):
    base = dict(slug="bonecas", nome="Bonecas", fonte_amazon="Toys_and_Games",
                filtro=FiltroCategoria(categorias_contem=["dolls"], titulo_contem=["Monster High"]),
                max_itens=100, min_avaliacoes=5)
    return Categoria(**(base | kw))


def test_importa_so_o_que_passa_no_filtro(pasta):
    itens, contagem = importar_itens(categoria(), pasta)
    assert sorted(i.id for i in itens) == ["A1", "A2", "A3", "A6"]
    assert contagem["lidos"] == 7  # a linha corrompida é ignorada
    assert contagem["poucas_avaliacoes"] == 1 and contagem["sem_titulo_ou_id"] == 1
    assert contagem["sem_foto"] == 1


def test_normaliza_produto(pasta):
    itens = {i.id: i for i in importar_itens(categoria(), pasta)[0]}
    a1 = itens["A1"]
    assert a1.foto_url == "https://img/A1.jpg"  # MAIN, tamanho large
    assert a1.link == "https://www.amazon.com/dp/A1"
    assert (a1.preco, a1.moeda, a1.condicao, a1.vendedor_id) == (19.99, "USD", "novo", "Mattel")
    assert a1.atributos == {"Loja": "Mattel", "Categoria": "Dolls", "Destaques": "Articulada ; Acompanha acessórios"}
    assert itens["A2"].preco == 12.5  # "$12.50"
    assert itens["A6"].preco is None and itens["A6"].foto_url is None


def test_limite_de_itens_usa_amostra_aleatoria_reprodutivel(pasta):
    a = sorted(i.id for i in importar_itens(categoria(max_itens=2), pasta, semente=1)[0])
    b = sorted(i.id for i in importar_itens(categoria(max_itens=2), pasta, semente=1)[0])
    assert len(a) == 2 and a == b


def test_importa_avaliacoes_dos_itens_importados(pasta):
    avs = list(importar_avaliacoes(categoria(), pasta, {"A1", "A2"}))
    assert [(a.usuario, a.item_id, a.nota) for a in avs] == [("U1", "A1", 5.0), ("U2", "A1", 3.0)]
    # milissegundos e segundos viram a mesma data
    assert avs[0].momento[:19] == avs[1].momento[:19] == "2020-05-05T14:08:48"
    assert avs[0].compra_verificada and not avs[1].compra_verificada


def test_arquivo_ausente_explica_como_baixar(tmp_path):
    with pytest.raises(FileNotFoundError, match="baixar_amazon"):
        importar_itens(categoria(), tmp_path)


def test_inspecionar_conta_categorias(pasta):
    info = inspecionar(pasta, "Toys_and_Games", amostra=100)
    nomes = dict(info["categorias_mais_comuns"])
    assert nomes["Dolls"] == 5 and info["produtos_lidos"] == 7


def test_item_sem_id_e_descartado():
    assert item_de_meta({"title": "x"}, "c") is None



@pytest.fixture
def pasta_usuarios(tmp_path):
    """Catálogo de bonecas com usuários de históricos diferentes.

    U1: 5 avaliações (elegível) · U2: 5 (elegível, 2 produtos em comum com U1)
    U3: 2 (não elegível) · U4: 6 produtos só dele (elegível, mas estoura o limite de itens)
    """
    bonecas = ["Toys & Games", "Dolls"]
    metas = [meta(f"D{i}", f"Fashion doll {i}", bonecas) for i in range(1, 15)]
    metas.append(meta("MH1", "Monster High Frankie Stein", ["Toys & Games", "Dolls"]))
    metas.append(meta("L1", "Lego set", ["Toys & Games", "Building Toys"]))  # fora do filtro
    escrever(tmp_path / "meta_Toys_and_Games.jsonl.gz", metas)
    t = 1600000000000
    revs = []
    for n, d in enumerate(["D1", "D2", "D3", "D4", "D5"]):
        revs.append({"user_id": "U1", "parent_asin": d, "rating": 5, "timestamp": t + n})
    for n, d in enumerate(["D2", "D3", "D6", "D7", "D8"]):
        revs.append({"user_id": "U2", "parent_asin": d, "rating": 4, "timestamp": t + 10 + n})
    revs += [{"user_id": "U3", "parent_asin": "D1", "rating": 3, "timestamp": t},
             {"user_id": "U3", "parent_asin": "D9", "rating": 3, "timestamp": t}]
    for n, d in enumerate(["D9", "D10", "D11", "D12", "D13", "D14"]):
        revs.append({"user_id": "U4", "parent_asin": d, "rating": 2, "timestamp": t + 20 + n})
    revs.append({"user_id": "U1", "parent_asin": "L1", "rating": 5, "timestamp": t})  # produto fora da categoria
    escrever(tmp_path / "Toys_and_Games.jsonl.gz", revs)
    return tmp_path


def test_selecao_por_usuarios_monta_catalogo_com_historicos_completos(pasta_usuarios):
    cat = categoria(max_itens=10, min_avaliacoes_usuario=5)
    itens, avaliacoes, c = importar_por_usuarios(cat, pasta_usuarios, semente=3)
    ids = {i.id for i in itens}
    assert "MH1" in ids  # garantido pelo título
    assert "L1" not in ids
    assert c["usuarios_elegiveis"] == 3  # U1, U2, U4 (U3 tem só 2)
    assert len(ids) <= 10
    por_usuario = {}
    for a in avaliacoes:
        por_usuario.setdefault(a.usuario, set()).add(a.item_id)
    assert "U3" not in por_usuario
    # quem entrou entra com o histórico inteiro na categoria
    for u, historico in {"U1": {"D1", "D2", "D3", "D4", "D5"}, "U2": {"D2", "D3", "D6", "D7", "D8"}}.items():
        if historico <= ids:
            assert por_usuario[u] == historico
    # U1 e U2 juntos cabem (8 produtos + MH1 = 9); U4 junto estouraria (15): não pode estar inteiro
    assert not {"D9", "D10", "D11", "D12", "D13", "D14"} <= ids or not {"D1", "D6"} <= ids
    assert all(a.item_id in ids for a in avaliacoes)


def test_selecao_por_usuarios_e_reprodutivel(pasta_usuarios):
    cat = categoria(max_itens=10, min_avaliacoes_usuario=5)
    a = importar_por_usuarios(cat, pasta_usuarios, semente=7)
    b = importar_por_usuarios(cat, pasta_usuarios, semente=7)
    assert [i.id for i in a[0]] == [i.id for i in b[0]] and len(a[1]) == len(b[1])


def test_garantidos_ocupam_no_maximo_um_terco(pasta_usuarios):
    cat = categoria(max_itens=3, min_avaliacoes_usuario=5,
                    filtro=FiltroCategoria(categorias_contem=["Dolls"], titulo_contem=["doll"]))
    itens, _, c = importar_por_usuarios(cat, pasta_usuarios)
    assert c["garantidos_pelo_titulo"] == 1  # 3 // 3


def test_catalogo_completo_inclui_todos_os_candidatos(pasta_usuarios):
    cat = categoria(max_itens=10, min_avaliacoes_usuario=5, catalogo_completo=True)
    itens, avaliacoes, c = importar_por_usuarios(cat, pasta_usuarios)
    ids = {i.id for i in itens}
    assert ids == {f"D{i}" for i in range(1, 15)} | {"MH1"}  # todos os que passam no filtro, acima de max_itens
    assert {a.usuario for a in avaliacoes} == {"U1", "U2", "U4"}  # os elegíveis, com o histórico inteiro
    assert c["itens"] == 15


def test_data_de_lancamento():
    assert lancamento_de_meta({"details": {"Date First Available": "March 5, 2019"}}) == "2019-03-05"
    # a mais antiga das duas
    assert lancamento_de_meta({"details": {"Date First Available": "January 1, 2012",
                                           "Release date": "May 5, 2009"}}) == "2009-05-05"
    assert lancamento_de_meta({"details": {"Date First Available": "em breve"}}) is None
    assert lancamento_de_meta({}) is None
    m = meta("A9", "Doll", ["Dolls"]) | {"details": {"Date First Available": "Aug 2, 2014"}}
    assert item_de_meta(m, "bonecas").lancamento == "2014-08-02"
