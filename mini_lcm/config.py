"""Configuração: lê o .env e o categorias.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent


@dataclass
class FiltroCategoria:
    categorias_contem: list[str] = field(default_factory=list)
    titulo_contem: list[str] = field(default_factory=list)

    def aceita(self, titulo: str, categorias: list[str]) -> bool:
        """O item entra se casar com qualquer regra."""
        nomes = {c.strip().lower() for c in categorias}
        if any(c.lower() in nomes for c in self.categorias_contem):
            return True
        titulo = titulo.lower()
        return any(t.lower() in titulo for t in self.titulo_contem)


@dataclass
class Categoria:
    slug: str
    nome: str
    fonte_amazon: str
    filtro: FiltroCategoria = field(default_factory=FiltroCategoria)
    max_itens: int = 3000
    min_avaliacoes: int = 5


@dataclass
class ConfigJuiz:
    api_key: str | None = None
    modelo: str = "gemini-3.5-flash-lite"
    preco_entrada_1m: float = 0.0  # US$ por 1 milhão de tokens; 0 = custo não calculado
    preco_saida_1m: float = 0.0


@dataclass
class Config:
    database_url: str | None
    embedder: str  # "clip", "clip+e5" ou "fake"
    pasta_dados: Path  # dados processados (itens, vetores, custos)
    pasta_brutos: Path  # arquivos baixados da Amazon
    categorias: list[Categoria] = field(default_factory=list)
    juiz: ConfigJuiz = field(default_factory=ConfigJuiz)
    texto_item: str = "completo"  # "completo" ou "titulo": que texto do produto vira vetor
    pasta_avaliacao: Path = RAIZ / "avaliacao"  # rubrica.md e gabarito.json

    @property
    def gabarito(self) -> Path:
        return self.pasta_avaliacao / "gabarito.json"

    @property
    def rubrica(self) -> Path:
        return self.pasta_avaliacao / "rubrica.md"

    def categoria(self, slug: str) -> Categoria:
        for c in self.categorias:
            if c.slug == slug:
                return c
        validas = ", ".join(c.slug for c in self.categorias)
        raise ValueError(f"Categoria '{slug}' não existe em categorias.yaml. Válidas: {validas}")


def carregar_categorias(caminho: Path) -> list[Categoria]:
    dados = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    categorias = []
    for bruto in dados.get("categorias", []):
        bruto = dict(bruto)
        filtro = FiltroCategoria(**(bruto.pop("filtro", None) or {}))
        cat = Categoria(filtro=filtro, **bruto)
        if not (filtro.categorias_contem or filtro.titulo_contem):
            raise ValueError(f"Categoria '{cat.slug}' sem filtro em {caminho}: importaria o domínio inteiro")
        categorias.append(cat)
    slugs = [c.slug for c in categorias]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"Slugs repetidos em {caminho}")
    return categorias


def carregar_config(raiz: Path = RAIZ) -> Config:
    load_dotenv(raiz / ".env")
    return Config(
        database_url=os.getenv("DATABASE_URL") or None,
        embedder=os.getenv("EMBEDDER") or "clip",
        texto_item=os.getenv("TEXTO_ITEM") or "completo",
        pasta_dados=raiz / "dados",
        pasta_brutos=raiz / "dados" / "brutos",
        categorias=carregar_categorias(raiz / "categorias.yaml"),
        juiz=ConfigJuiz(
            api_key=os.getenv("GEMINI_API_KEY") or None,
            modelo=os.getenv("JUIZ_MODELO") or "gemini-3.5-flash-lite",
            preco_entrada_1m=float(os.getenv("JUIZ_PRECO_ENTRADA_1M") or 0),
            preco_saida_1m=float(os.getenv("JUIZ_PRECO_SAIDA_1M") or 0),
        ),
    )
