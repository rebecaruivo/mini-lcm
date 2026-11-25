"""Testa a ClipLocal sem baixar os modelos: o sentence-transformers e a rede são substituídos por falsos.
O que se verifica é a lógica em volta do modelo (downloads, falhas, normalização, custo)."""

import io
import sys
import types

import numpy as np
import pytest
from PIL import Image


class ModeloFalso:
    def __init__(self, nome):
        self.nome = nome

    def encode(self, entradas, **kw):
        # devolve vetores NÃO normalizados, para conferir que a classe normaliza
        return np.array([[3.0, 4.0] + [0.0] * 510 for _ in entradas])


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "pink").save(buf, format="PNG")
    return buf.getvalue()


class RespostaHttp:
    def __init__(self, status, conteudo=b""):
        self.status_code, self.content = status, conteudo

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def clip(monkeypatch):
    falso = types.ModuleType("sentence_transformers")
    falso.SentenceTransformer = ModeloFalso
    monkeypatch.setitem(sys.modules, "sentence_transformers", falso)
    from mini_lcm.embeddings import ClipLocal

    c = ClipLocal()
    respostas = {"https://ok/1.jpg": RespostaHttp(200, png()), "https://ok/2.jpg": RespostaHttp(200, png()),
                 "https://fora/3.jpg": RespostaHttp(404), "https://lixo/4.jpg": RespostaHttp(200, b"nao e imagem")}
    c._http.get = lambda url, timeout=None: respostas[url]
    return c


def test_imagens_baixa_em_paralelo_e_tolera_falhas(clip):
    from mini_lcm.custo import RegistroCusto

    reg = RegistroCusto(etapa="embeddings:imagem", modelo=clip.nome)
    saida = clip.imagens(["https://ok/1.jpg", "https://fora/3.jpg", "https://ok/2.jpg", "https://lixo/4.jpg"], reg)
    assert [v is not None for v in saida] == [True, False, True, False]  # a ordem é preservada
    assert np.allclose(np.linalg.norm(saida[0]), 1.0)  # normalizado
    assert (reg.itens, reg.chamadas, reg.falhas) == (4, 4, 2)


def test_textos_normaliza_e_registra(clip):
    m = clip.textos(["boneca", "placa"])
    assert m.shape == (2, 512) and np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert clip._st_texto.nome.endswith("multilingual-v1") and clip._st_imagem.nome == "clip-ViT-B-32"
