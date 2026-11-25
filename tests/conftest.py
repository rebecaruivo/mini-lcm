import pytest

from mini_lcm.armazenamento import ArmazenamentoLocal
from mini_lcm.embeddings import EmbedderFalso
from mini_lcm.indexacao import gerar_embeddings
from mini_lcm.modelos import Item


def item(id_, titulo, categoria="bonecas", preco=100.0, condicao="usado", vendedor="v1", foto=True, **atributos):
    return Item(id=id_, categoria=categoria, titulo=titulo, preco=preco, moeda="BRL", condicao=condicao,
                vendedor_id=vendedor, link=f"https://x/{id_}", foto_url=f"https://foto/{id_}.jpg" if foto else None,
                atributos=atributos)


@pytest.fixture
def catalogo():
    """Catálogo pequeno com dois anúncios-armadilha:
    - B2 tem título ruim ("boneca antiga"), mas a foto é de uma Draculaura;
    - B5 é uma republicação de B1 pelo mesmo vendedor (duplicata).
    """
    itens = [
        item("B1", "Boneca Monster High Draculaura original", preco=190, vendedor="v1"),
        item("B2", "Boneca antiga cabelo rosa", preco=60, vendedor="v2"),
        item("B3", "Monster High Clawdeen Wolf lacrada", preco=350, condicao="novo", vendedor="v1"),
        item("B4", "Boneca Barbie anos 90", preco=80, vendedor="v3"),
        item("B5", "Boneca  MONSTER HIGH Draculaura Original", preco=185, vendedor="v1"),
        item("P1", "Placa de vídeo RTX 4060 8GB", categoria="placas", preco=1900, condicao="novo", vendedor="v9"),
        item("P2", "Placa de video RTX 3060 12GB usada", categoria="placas", preco=1200, vendedor="v8"),
        item("P3", "Placa de vídeo sem foto", categoria="placas", preco=500, vendedor="v7", foto=False),
    ]
    fotos = {
        "https://foto/B1.jpg": "draculaura monster high boneca rosa preto",
        "https://foto/B2.jpg": "draculaura monster high boneca rosa preto",
        "https://foto/B3.jpg": "clawdeen monster high boneca lobo",
        "https://foto/B4.jpg": "barbie boneca loira",
        "https://foto/B5.jpg": "draculaura monster high boneca rosa preto",
        "https://foto/P1.jpg": "placa video rtx",
        "https://foto/P2.jpg": "placa video rtx",
    }
    return itens, EmbedderFalso(descricoes_fotos=fotos)


@pytest.fixture
def indexado(catalogo, tmp_path):
    itens, embedder = catalogo
    arm = ArmazenamentoLocal(tmp_path / "dados")
    arm.salvar_itens(itens)
    gerar_embeddings(arm, embedder)
    return arm, embedder
