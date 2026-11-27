"""EmbeddingGemma 2 com um SentenceTransformer falso: confere prefixos, chaves e o caminho das fotos."""

import sys
import types

import numpy as np

from mini_lcm.modelos import Item


def produto(id_, titulo, atributos=None):
    return Item(id=id_, categoria="bonecas", titulo=titulo, preco=None, moeda=None, condicao="novo",
                vendedor_id=None, link=None, foto_url=None, atributos=atributos or {})


class STFalso:
    chamadas: list = []

    def __init__(self, nome, config_kwargs=None, **kw):
        STFalso.chamadas.append(("init", nome, config_kwargs))

    def encode(self, entradas, prompt_name=None, **kw):
        lista = entradas if isinstance(entradas, list) else [entradas]
        STFalso.chamadas.append(("encode", prompt_name, lista))
        if any(isinstance(e, dict) for e in lista) and len(lista) > 1:
            raise TypeError("lote multimodal não suportado")  # força o caminho de uma foto por vez
        m = np.ones((len(lista), 4))
        return m[0] if not isinstance(entradas, list) else m


def test_embeddinggemma2_prefixos_chaves_e_fotos(monkeypatch):
    modulo = types.ModuleType("sentence_transformers")
    modulo.SentenceTransformer = STFalso
    monkeypatch.setitem(sys.modules, "sentence_transformers", modulo)
    from PIL import Image

    from mini_lcm.embeddings import EmbeddingGemma2

    STFalso.chamadas = []
    emb = EmbeddingGemma2("titulo")
    assert STFalso.chamadas[0] == ("init", "google/embeddinggemma-2", {"audio_config": None})  # sem o áudio
    assert emb.chave("texto") == "embeddinggemma-2@titulo" and emb.chave("imagem") == "embeddinggemma-2"

    item = produto("A", "Monster High Draculaura", {"Loja": "Mattel"})
    assert emb.texto_do_produto(item) == "title: Monster High Draculaura | text: Loja: Mattel"
    assert emb.texto_do_produto(produto("B", "Doll")) == "title: Doll | text: none"

    emb.consulta("boneca gótica", "imagem")
    assert STFalso.chamadas[-1][1] == "SearchQuery"

    monkeypatch.setattr(emb, "_baixar", lambda url: None if "quebrada" in url else Image.new("RGB", (8, 8)))
    vetores = emb.imagens(["http://a.jpg", "http://quebrada.jpg", "http://b.jpg"])
    assert vetores[1] is None and vetores[0] is not None and vetores[2] is not None
    fotos = [c for c in STFalso.chamadas if c[0] == "encode" and isinstance(c[2][0], dict)]
    assert all(f[2][0]["text"] == "<|image|>" and f[2][0]["image"][0].endswith(".jpg") for f in fotos)


def test_clip_com_gemma2_so_texto(monkeypatch):
    modulo = types.ModuleType("sentence_transformers")
    modulo.SentenceTransformer = STFalso
    monkeypatch.setitem(sys.modules, "sentence_transformers", modulo)
    from mini_lcm.embeddings import ClipComGemma2

    STFalso.chamadas = []
    emb = ClipComGemma2("titulo")
    gemma = [c for c in STFalso.chamadas if c[0] == "init" and c[1] == "google/embeddinggemma-2"]
    assert gemma and gemma[0][2] == {"vision_config": None, "audio_config": None}  # só a parte de texto
    assert emb.chave("texto") == "embeddinggemma-2@titulo"
    assert emb.chave("imagem") == "clip-vit-b-32-multilingual"  # fotos continuam com o CLIP
    emb.consulta("rtx 3060", "texto")
    assert STFalso.chamadas[-1][1] == "SearchQuery"
    emb.textos_itens([produto("A", "Monster High Draculaura")])
    assert STFalso.chamadas[-1][2] == ["title: Monster High Draculaura | text: none"]
