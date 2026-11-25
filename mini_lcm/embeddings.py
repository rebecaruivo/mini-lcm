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
from .modelos import Canal, Item

log = logging.getLogger(__name__)

MAX_PALAVRAS_TEXTO = 40  # o codificador de texto do CLIP lê textos curtos; o resto seria cortado
MODOS_TEXTO = ("completo", "titulo")


class Embedder(Protocol):
    """Transforma consultas, textos de produto e fotos em vetores.

    Cada canal (texto, imagem) pode usar um modelo diferente. Por isso a consulta
    é codificada POR CANAL: para comparar com fotos, ela precisa estar no espaço
    do CLIP; para comparar com textos, no espaço do modelo de texto. Isso só
    funciona porque a fusão (RRF) usa posições, não similaridades: os dois
    espaços não precisam ser comparáveis entre si.
    """

    nome: str  # descrição geral (aparece nos custos)

    def chave(self, canal: Canal) -> str:
        """Identifica os vetores deste canal no armazenamento. Mudou o modelo ou o texto
        dos produtos, muda a chave: os vetores antigos ficam guardados e não se misturam."""
        ...

    def consulta(self, texto: str, canal: Canal) -> np.ndarray:
        """Vetor da consulta, no espaço do canal."""
        ...

    def textos_itens(self, itens: list[Item], registro: RegistroCusto | None = None) -> np.ndarray:
        """Matriz (n, d) com o vetor de texto de cada produto."""
        ...

    def imagens(self, urls: list[str], registro: RegistroCusto | None = None) -> list[np.ndarray | None]:
        """Um vetor por URL, ou None quando a foto não pôde ser baixada ou lida."""
        ...


def texto_do_item(item: Item, modo: str = "completo", max_palavras: int = MAX_PALAVRAS_TEXTO) -> str:
    """O texto que representa o produto.

    modo="completo": título + todos os atributos (loja, categoria, destaques).
    modo="titulo": só título + loja.

    Por que existe o modo "titulo": modelos como o CLIP fazem a MÉDIA dos vetores
    das palavras. Se os destaques (texto de marketing, parecido em todo produto)
    ocupam a maior parte das palavras, eles dominam a média, e todos os produtos
    ficam com cara de "descrição genérica". O título, que é o que distingue um
    produto do outro, se dilui.

    Cortamos em `max_palavras` porque o modelo ignora o que passa do limite dele;
    melhor escolhermos o que fica (o título vem primeiro).
    """
    if modo not in MODOS_TEXTO:
        raise ValueError(f"modo de texto desconhecido: {modo}")
    atributos = item.atributos.items() if modo == "completo" else (
        [("Loja", item.atributos["Loja"])] if "Loja" in item.atributos else [])
    partes = [item.titulo] + [f"{nome}: {valor}" for nome, valor in atributos]
    palavras = " . ".join(partes).split()
    return " ".join(palavras[:max_palavras])


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

    NOME_CLIP = "clip-vit-b-32-multilingual"
    MODELO_IMAGEM = "clip-ViT-B-32"
    MODELO_TEXTO = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    def __init__(self, texto_item: str = "completo", lote: int = 32, timeout_foto_s: float = 15,
                 downloads_paralelos: int = 8):
        if texto_item not in MODOS_TEXTO:
            raise ValueError(f"TEXTO_ITEM deve ser um de {MODOS_TEXTO}")
        self.texto_item = texto_item
        self.nome = self.NOME_CLIP if texto_item == "completo" else f"{self.NOME_CLIP}@{texto_item}"
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

    def chave(self, canal: Canal) -> str:
        # As fotos não dependem do modo de texto: a chave da imagem é sempre a mesma,
        # e os vetores de imagem já calculados são reaproveitados.
        return self.NOME_CLIP if canal == "imagem" else self.nome

    def textos(self, textos: list[str], registro: RegistroCusto | None = None) -> np.ndarray:
        """Codificador de texto do CLIP (multilíngue), para qualquer lista de textos."""
        reg = registro or RegistroCusto(etapa="embeddings:texto", modelo=self.nome)
        with cronometro(reg):
            m = self._st_texto.encode(textos, batch_size=self.lote, convert_to_numpy=True, show_progress_bar=False)
        reg.itens += len(textos)
        return normalizar_linhas(np.asarray(m))

    def consulta(self, texto: str, canal: Canal) -> np.ndarray:
        return self.textos([texto], RegistroCusto(etapa="_", modelo=self.nome))[0]

    def textos_itens(self, itens: list[Item], registro: RegistroCusto | None = None) -> np.ndarray:
        return self.textos([texto_do_item(i, self.texto_item) for i in itens], registro)

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


# ─────────────────────── CLIP + modelo de texto dedicado ───────────────────────


class ClipComE5(ClipLocal):
    """Fotos com o CLIP; textos com o multilingual-e5-small, um modelo treinado só para busca em texto.

    O e5 foi treinado para busca ASSIMÉTRICA: uma consulta curta ("draculaura antiga")
    contra um texto longo (título + descrição do produto). Por isso ele pede prefixos:
    "query: " na consulta e "passage: " no texto do produto. Sem eles, a qualidade cai.

    Ele aceita textos bem mais longos que o CLIP (até 512 tokens), então o limite de
    palavras do texto do produto sobe para 120.

    Vetores de 384 dimensões, num espaço diferente do CLIP. Não há problema: a fusão
    usa posições, e cada canal compara a consulta só com vetores do próprio espaço.
    """

    MODELO_TEXTO_DEDICADO = "intfloat/multilingual-e5-small"
    NOME_TEXTO = "e5-small-multilingual"
    MAX_PALAVRAS = 120

    def __init__(self, texto_item: str = "completo", **kw):
        super().__init__(texto_item, **kw)
        from sentence_transformers import SentenceTransformer

        self._st_e5 = SentenceTransformer(self.MODELO_TEXTO_DEDICADO)
        self.nome = f"{self.NOME_TEXTO}@{texto_item}+{self.NOME_CLIP}"

    def chave(self, canal: Canal) -> str:
        return self.NOME_CLIP if canal == "imagem" else f"{self.NOME_TEXTO}@{self.texto_item}"

    def _e5(self, textos: list[str], registro: RegistroCusto | None) -> np.ndarray:
        reg = registro or RegistroCusto(etapa="embeddings:texto", modelo=self.chave("texto"))
        with cronometro(reg):
            m = self._st_e5.encode(textos, batch_size=self.lote, convert_to_numpy=True, show_progress_bar=False)
        reg.itens += len(textos)
        return normalizar_linhas(np.asarray(m))

    def consulta(self, texto: str, canal: Canal) -> np.ndarray:
        if canal == "imagem":
            return super().consulta(texto, canal)  # para comparar com fotos, o espaço do CLIP
        return self._e5([f"query: {texto}"], RegistroCusto(etapa="_", modelo=self.nome))[0]

    def textos_itens(self, itens: list[Item], registro: RegistroCusto | None = None) -> np.ndarray:
        return self._e5([f"passage: {texto_do_item(i, self.texto_item, self.MAX_PALAVRAS)}" for i in itens], registro)


# ─────────────────────────── Falso (testes) ───────────────────────────


class EmbedderFalso:
    """Embedder determinístico e sem dependências, para testes e para rodar o fluxo sem baixar modelos.

    Texto: cada palavra (sem acento, minúscula) soma um vetor fixo derivado do hash dela.
    Textos com palavras em comum ficam parecidos. Não entende sinônimos nem significado.

    Imagem: o "conteúdo" da foto é uma descrição em texto passada em `descricoes_fotos`
    ({url: "boneca de cabelo rosa e preto"}). Assim os testes simulam um produto cujo
    título é ruim mas a foto mostra o produto. URL sem descrição gera um vetor aleatório
    (fixo para aquela URL); URL em `fotos_quebradas` falha como uma foto fora do ar.
    """

    nome = "falso-hash"

    def __init__(self, dimensoes: int = 256, descricoes_fotos: dict[str, str] | None = None,
                 fotos_quebradas: set[str] | None = None, texto_item: str = "completo"):
        self.dimensoes = dimensoes
        self.descricoes_fotos = descricoes_fotos or {}
        self.fotos_quebradas = fotos_quebradas or set()
        self.texto_item = texto_item

    def chave(self, canal: Canal) -> str:
        return self.nome if canal == "imagem" or self.texto_item == "completo" else f"{self.nome}@{self.texto_item}"

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

    def consulta(self, texto: str, canal: Canal) -> np.ndarray:
        return self.textos([texto])[0]

    def textos_itens(self, itens: list[Item], registro: RegistroCusto | None = None) -> np.ndarray:
        return self.textos([texto_do_item(i, self.texto_item) for i in itens], registro)

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


EMBEDDERS = ("clip", "clip+e5", "fake")


def criar_embedder(nome: str, texto_item: str = "completo") -> Embedder:
    if nome == "clip":
        return ClipLocal(texto_item)
    if nome == "clip+e5":
        return ClipComE5(texto_item)
    if nome == "fake":
        return EmbedderFalso(texto_item=texto_item)
    raise ValueError(f"EMBEDDER desconhecido: '{nome}'. Use um de {EMBEDDERS}.")
