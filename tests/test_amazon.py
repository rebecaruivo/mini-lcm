import gzip
import json

import pytest

from mini_lcm.amazon import importar_avaliacoes, importar_itens, inspecionar, item_de_meta
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
