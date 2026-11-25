"""Embeddings: transformar texto e imagem em vetores comparáveis.

Um embedding é uma lista de números (aqui, 512) que representa o significado
de um texto ou de uma imagem. Coisas parecidas viram vetores que apontam para
direções parecidas, e a "similaridade de cosseno" mede isso: 1 = mesma direção,
0 = nada a ver.

Usamos o CLIP, um modelo treinado com centenas de milhões de pares (imagem,
legenda) para colocar imagens e textos NO MESMO ESPAÇO. Isso permite comparar
a consulta "draculaura antiga" diretamente com a FOTO de um anúncio, mesmo que
o título diga só "boneca". É o que resolve o anúncio mal descrito.

Dois modelos, um espaço:
- imagens: `clip-ViT-B-32` (o CLIP original, da OpenAI)
- textos: `clip-ViT-B-32-multilingual-v1`, treinado para colocar textos em
  50+ línguas (português incluído) no mesmo lugar onde o CLIP original põe o
  texto em inglês. Sem ele, as consultas em português funcionariam mal.

Todo vetor sai normalizado (comprimento 1). Assim, cosseno = produto escalar,
que é mais barato de calcular.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
import requests

from .custo import RegistroCusto, cronometro
from .modelos import Item

log = logging.getLogger(__name__)

MAX_PALAVRAS_TEXTO = 40  # o codificador de texto do CLIP lê textos curtos; o resto seria cortado


class Embedder(Protocol):
    nome: str  # identifica o modelo no banco; trocar de modelo = novo nome, vetores novos
    dimensoes: int

    def textos(self, textos: list[str], registro: RegistroCusto | None = None) -> np.ndarray:
        """Matriz (n, dimensoes) com uma linha normalizada por texto."""
        ...

    def imagens(self, urls: list[str], registro: RegistroCusto | None = None) -> list[np.ndarray | None]:
        """Um vetor por URL, ou None quando a foto não pôde ser baixada ou lida."""
        ...


def texto_do_item(item: Item) -> str:
    """O texto que representa o anúncio: título + atributos ("Marca: Mattel").

    Os atributos ajudam quando o título é pobre. Cortamos em MAX_PALAVRAS_TEXTO
    palavras porque o modelo ignora o que passa do limite dele, e é melhor
    escolhermos o que fica (o título vem primeiro) do que deixar o modelo cortar.
    """
    partes = [item.titulo] + [f"{nome}: {valor}" for nome, valor in item.atributos.items()]
    palavras = " . ".join(partes).split()
    return " ".join(palavras[:MAX_PALAVRAS_TEXTO])


def normalizar_linhas(m: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return (m / normas).astype(np.float32)


# ─────────────────────────── CLIP local ───────────────────────────


class ClipLocal:
    """CLIP rodando na sua máquina, via sentence-transformers. Custo em dinheiro: zero.

    Na primeira execução baixa os dois modelos (cerca de 1 GB no total) para o cache
    do Hugging Face. Roda em CPU; com GPU fica bem mais rápido, sem mudar nada aqui.
    """

    nome = "clip-vit-b-32-multilingual"
    dimensoes = 512
    MODELO_IMAGEM = "clip-ViT-B-32"
    MODELO_TEXTO = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    def __init__(self, lote: int = 32, timeout_foto_s: float = 15, downloads_paralelos: int = 8):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError('Instale os modelos locais com: pip install -e ".[clip]"') from e
        self._st_imagem = SentenceTransformer(self.MODELO_IMAGEM)
        self._st_texto = SentenceTransformer(self.MODELO_TEXTO)
        self.lote = lote
        self.timeout_foto_s = timeout_foto_s
        self.downloads_paralelos = downloads_paralelos
        self._http = requests.Session()
        self._http.headers["User-Agent"] = "mini-lcm/0.1 (projeto de estudo)"

    def textos(self, textos: list[str], registro: RegistroCusto | None = None) -> np.ndarray:
        reg = registro or RegistroCusto(etapa="embeddings:texto", modelo=self.nome)
        with cronometro(reg):
            m = self._st_texto.encode(textos, batch_size=self.lote, convert_to_numpy=True, show_progress_bar=False)
        reg.itens += len(textos)
        return normalizar_linhas(np.asarray(m))

    def _baixar(self, url: str):
        from PIL import Image

        r = self._http.get(url, timeout=self.timeout_foto_s)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    def imagens(self, urls: list[str], registro: RegistroCusto | None = None) -> list[np.ndarray | None]:
        reg = registro or RegistroCusto(etapa="embeddings:imagem", modelo=self.nome)
        saida: list[np.ndarray | None] = [None] * len(urls)
        with cronometro(reg):
            # Baixar foto é esperar a rede, não usar CPU: várias ao mesmo tempo aceleram muito.
            def tentar(url: str):
                try:
                    return self._baixar(url)
                except Exception as e:  # foto fora do ar, formato estranho, etc.: o item fica só com texto
                    log.warning("Foto ignorada (%s): %s", url, e)
                    return None

            with ThreadPoolExecutor(self.downloads_paralelos) as pool:
                baixadas = list(pool.map(tentar, urls))
            reg.chamadas += len(urls)
            reg.falhas += sum(1 for f in baixadas if f is None)
            fotos = [f for f in baixadas if f is not None]
            posicoes = [i for i, f in enumerate(baixadas) if f is not None]
            if fotos:
                m = normalizar_linhas(np.asarray(
                    self._st_imagem.encode(fotos, batch_size=self.lote, convert_to_numpy=True, show_progress_bar=False)
                ))
                for linha, i in zip(m, posicoes):
                    saida[i] = linha
        reg.itens += len(urls)
        return saida


# ─────────────────────────── Falso (testes) ───────────────────────────


class EmbedderFalso:
    """Embedder determinístico e sem dependências, para testes e para rodar o fluxo sem baixar modelos.

    Texto: cada palavra (sem acento, minúscula) soma um vetor fixo derivado do hash dela.
    Textos com palavras em comum ficam parecidos. Não entende sinônimos nem significado.

    Imagem: o "conteúdo" da foto é uma descrição em texto passada em `descricoes_fotos`
    ({url: "boneca de cabelo rosa e preto"}). Assim os testes simulam um anúncio cujo
    título é ruim mas a foto mostra o produto. URL sem descrição gera um vetor aleatório
    (fixo para aquela URL); URL em `fotos_quebradas` falha como uma foto fora do ar.
    """

    nome = "falso-hash"

    def __init__(self, dimensoes: int = 256, descricoes_fotos: dict[str, str] | None = None,
                 fotos_quebradas: set[str] | None = None):
        self.dimensoes = dimensoes
        self.descricoes_fotos = descricoes_fotos or {}
        self.fotos_quebradas = fotos_quebradas or set()

    def _semente(self, texto: str) -> int:
        return int.from_bytes(hashlib.sha256(texto.encode()).digest()[:8], "little")

    def _vetor_palavra(self, palavra: str) -> np.ndarray:
        return np.random.default_rng(self._semente(palavra)).standard_normal(self.dimensoes)

    def _vetor_texto(self, texto: str) -> np.ndarray:
        from .modelos import normalizar_texto

        palavras = [p for p in re.split(r"\s+", normalizar_texto(texto)) if len(p) > 1]
        if not palavras:
            return np.zeros(self.dimensoes)
        return np.sum([self._vetor_palavra(p) for p in palavras], axis=0)

    def textos(self, textos: list[str], registro: RegistroCusto | None = None) -> np.ndarray:
        if registro:
            registro.itens += len(textos)
        return normalizar_linhas(np.array([self._vetor_texto(t) for t in textos]))

    def imagens(self, urls: list[str], registro: RegistroCusto | None = None) -> list[np.ndarray | None]:
        saida: list[np.ndarray | None] = []
        for url in urls:
            if registro:
                registro.itens += 1
                registro.chamadas += 1
            if url in self.fotos_quebradas:
                if registro:
                    registro.falhas += 1
                saida.append(None)
                continue
            descricao = self.descricoes_fotos.get(url)
            v = self._vetor_texto(descricao) if descricao else self._vetor_palavra("url:" + url)
            saida.append(normalizar_linhas(v[None, :])[0])
        return saida


def criar_embedder(nome: str) -> Embedder:
    if nome == "clip":
        return ClipLocal()
    if nome == "fake":
        return EmbedderFalso()
    raise ValueError(f"EMBEDDER desconhecido: '{nome}'. Use 'clip' ou 'fake'.")
