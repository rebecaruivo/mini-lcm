"""Os objetos que circulam pelo sistema.

Um `Item` é um produto já normalizado: os campos têm o mesmo nome e o mesmo
formato qualquer que seja a fonte (hoje, o Amazon Reviews 2023) ou a categoria.
Campos que variam de categoria para categoria (memória da placa, personagem
da boneca) ficam em `atributos`, um dicionário livre.

Uma `Avaliacao` é uma interação real de um usuário com um item (nota de 1 a 5).
Na fase 1 elas só são importadas e contadas; na fase 2 viram o sinal com que o
sistema aprende o gosto de cada pessoa.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

Condicao = Literal["novo", "usado", "desconhecida"]
Canal = Literal["texto", "imagem"]
CANAIS: tuple[Canal, ...] = ("texto", "imagem")


@dataclass
class Item:
    id: str  # id na fonte (na Amazon, o parent_asin). Também é a chave primária.
    categoria: str  # slug de categorias.yaml
    titulo: str
    preco: float | None
    moeda: str | None
    condicao: Condicao
    vendedor_id: str | None
    link: str | None
    foto_url: str | None
    atributos: dict[str, str] = field(default_factory=dict)
    fonte: str = "amazon"
    # Data em que o produto passou a ser vendido (AAAA-MM-DD), quando a fonte informa.
    # Na avaliação por data, diz se o produto já existia num corte sem olhar o comportamento.
    lancamento: str | None = None
    coletado_em: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def preco_formatado(self) -> str:
        if self.preco is None:
            return "sem preço"
        simbolo = {"BRL": "R$", "USD": "US$"}.get(self.moeda or "", self.moeda or "")
        return f"{simbolo} {self.preco:,.2f}".strip()

    @property
    def chave_duplicata(self) -> str:
        """Dois anúncios com a mesma chave são tratados como o mesmo anúncio.

        Lojas costumam cadastrar o mesmo produto com ids diferentes (outra cor,
        outro pacote). Mesma loja + mesmo título (sem acento, caixa ou espaços
        extras) é um sinal simples e seguro disso. Produtos iguais de lojas
        diferentes NÃO são duplicatas: são ofertas diferentes, e a
        personalização deve poder escolher entre elas.
        """
        return f"{self.vendedor_id or '?'}|{normalizar_texto(self.titulo)}"


@dataclass
class Avaliacao:
    usuario: str  # id do usuário na fonte (anônimo)
    item_id: str
    nota: float  # 1 a 5
    momento: str  # ISO 8601, UTC
    compra_verificada: bool = False
    fonte: str = "amazon"


@dataclass
class Filtros:
    """Filtros estruturados da busca. Viram WHERE no SQL."""

    categoria: str | None = None
    preco_min: float | None = None
    preco_max: float | None = None
    condicao: Condicao | None = None

    def aceita(self, item: Item) -> bool:
        if self.categoria and item.categoria != self.categoria:
            return False
        if self.condicao and item.condicao != self.condicao:
            return False
        if self.preco_min is not None and (item.preco is None or item.preco < self.preco_min):
            return False
        if self.preco_max is not None and (item.preco is None or item.preco > self.preco_max):
            return False
        return True


@dataclass
class Resultado:
    item: Item
    pontuacao: float  # pontuação final da fusão (maior = melhor)
    similaridades: dict[str, float]  # similaridade de cosseno em cada canal, para explicar o resultado
    posicoes: dict[str, int]  # posição (1 = primeiro) em cada canal


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sem acentos e sem pontuação, com espaços simples."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", sem_acento.lower())).strip()
