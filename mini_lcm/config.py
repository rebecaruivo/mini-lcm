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
        return self.casa_titulo(titulo)

    def casa_titulo(self, titulo: str) -> bool:
        titulo = titulo.lower()
        return any(t.lower() in titulo for t in self.titulo_contem)


@dataclass
class ParametrosRecomendacao:
    """Parâmetros escolhidos na validação de cada categoria (categorias.yaml).

    meia_vida: decaimento pela recência no perfil e na coocorrência, em avaliações (None = desligado).
    janela_dias: tamanho da janela do método em_alta.
    peso_em_alta: peso da lista em_alta dentro do hibrido_alta (conteúdo e coocorrência pesam 1).
    vagas_novos: posições do top 10 do hibrido_alta reservadas para produtos novos (sem
        nenhuma interação), escolhidos pelo conteúdo. 0 = sem reserva.
    """

    meia_vida: float | None = 1.0
    janela_dias: int = 365
    peso_em_alta: float = 1.0
    vagas_novos: int = 0


@dataclass
class Categoria:
    slug: str
    nome: str
    fonte_amazon: str
    filtro: FiltroCategoria = field(default_factory=FiltroCategoria)
    max_itens: int = 3000
    min_avaliacoes: int = 5
    # "usuarios": escolhe usuários com histórico e monta o catálogo com o que eles avaliaram (fase 2)
    # "produtos": sorteia produtos (fase 1; quase todo usuário fica com 1 avaliação só)
    selecao: str = "usuarios"
    min_avaliacoes_usuario: int = 5
    # true: o catálogo são todos os produtos que passam no filtro, e não só os avaliados pelos
    # usuários escolhidos (ver amazon.importar_por_usuarios). max_itens deixa de valer.
    catalogo_completo: bool = False
    # Peso da imagem na fusão da busca (0 = só texto, 1 = só imagem), escolhido por categoria
    # na avaliação com gabarito: ver docs/resultados-busca.md.
    peso_imagem: float = 0.5
    recomendacao: ParametrosRecomendacao = field(default_factory=ParametrosRecomendacao)


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
        recomendacao = ParametrosRecomendacao(**(bruto.pop("recomendacao", None) or {}))
        cat = Categoria(filtro=filtro, recomendacao=recomendacao, **bruto)
        if cat.selecao not in ("usuarios", "produtos"):
            raise ValueError(f"Categoria '{cat.slug}': selecao deve ser 'usuarios' ou 'produtos'")
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
        embedder=os.getenv("EMBEDDER") or "clip+gemma2",  # escolhido: docs/resultados-busca.md
        texto_item=os.getenv("TEXTO_ITEM") or "titulo",
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
