"""Importação do Amazon Reviews 2023 (McAuley Lab, UC San Diego).

O dataset vem em dois arquivos por domínio (ex.: Toys_and_Games), ambos em
JSON Lines comprimido (.jsonl.gz, um objeto JSON por linha):

- meta_<Domínio>.jsonl.gz: um produto por linha (título, preço, loja, árvore de
  categorias, características, fotos, número de avaliações).
- <Domínio>.jsonl.gz: uma avaliação por linha (usuário, produto, nota de 1 a 5,
  data, compra verificada). São usuários reais, anônimos.

Os arquivos são grandes (Toys_and_Games tem 16 milhões de avaliações), então
eles são lidos linha a linha (streaming): nunca carregamos o arquivo inteiro na
memória.

Download: veja `scripts/baixar_amazon.sh`. Citação: Hou et al. (2024),
"Bridging Language and Items for Retrieval and Recommendation", arXiv:2403.03952.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import Categoria
from .modelos import Avaliacao, Item

log = logging.getLogger(__name__)

URL_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
MAX_DESTAQUES = 200  # caracteres das "características" que entram nos atributos


def caminho_meta(pasta: Path, fonte: str) -> Path:
    return pasta / f"meta_{fonte}.jsonl.gz"


def caminho_avaliacoes(pasta: Path, fonte: str) -> Path:
    return pasta / f"{fonte}.jsonl.gz"


def ler_jsonl(caminho: Path) -> Iterator[dict]:
    """Lê .jsonl ou .jsonl.gz linha a linha. Linhas corrompidas são puladas e contadas no log."""
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Baixe com: bash scripts/baixar_amazon.sh")
    abrir = gzip.open if caminho.suffix == ".gz" else open
    ruins = 0
    with abrir(caminho, "rt", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            try:
                yield json.loads(linha)
            except json.JSONDecodeError:
                ruins += 1
    if ruins:
        log.warning("%s: %d linhas ilegíveis ignoradas", caminho.name, ruins)


# ─────────────────────────── Produtos ───────────────────────────


def _preco(valor) -> float | None:
    """O preço às vezes vem como número, às vezes como texto ("$12.99") ou vazio."""
    if valor is None:
        return None
    try:
        preco = float(str(valor).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None
    return preco if preco >= 0 else None


def _foto(imagens: list[dict] | None) -> str | None:
    """A foto principal (variante MAIN) no tamanho "large".

    Existe também "hi_res", mas o CLIP reduz toda imagem para 224x224 pixels antes
    de processar: baixar a versão gigante só gastaria banda e tempo.
    """
    if not imagens:
        return None
    principal = next((i for i in imagens if i.get("variant") == "MAIN"), imagens[0])
    return principal.get("large") or principal.get("hi_res") or principal.get("thumb")


def item_de_meta(meta: dict, categoria: str) -> Item | None:
    item_id, titulo = meta.get("parent_asin"), (meta.get("title") or "").strip()
    if not item_id or not titulo:
        return None
    atributos: dict[str, str] = {}
    if meta.get("store"):
        atributos["Loja"] = str(meta["store"])
    if meta.get("categories"):
        atributos["Categoria"] = str(meta["categories"][-1])  # o nível mais específico da árvore
    destaques = " ; ".join(str(f) for f in (meta.get("features") or [])[:3])
    if destaques:
        atributos["Destaques"] = destaques[:MAX_DESTAQUES]
    return Item(
        id=item_id,
        categoria=categoria,
        titulo=titulo,
        preco=_preco(meta.get("price")),
        moeda="USD",
        condicao="novo",  # a Amazon vende produtos novos; o campo existe para fontes de usados
        vendedor_id=str(meta["store"]) if meta.get("store") else None,
        link=f"https://www.amazon.com/dp/{item_id}",
        foto_url=_foto(meta.get("images")),
        atributos=atributos,
        fonte="amazon",
    )


def importar_itens(categoria: Categoria, pasta: Path, semente: int = 42) -> tuple[list[Item], dict]:
    """Lê o arquivo de metadados e devolve até `max_itens` produtos que passam no filtro.

    Seleção por amostragem de reservatório (reservoir sampling): se o filtro aceita
    mais produtos do que o limite, ficamos com uma amostra ALEATÓRIA de todos eles,
    e não com os primeiros do arquivo, que poderiam ter alguma ordem escondida
    (por data de cadastro, por exemplo). O algoritmo funciona em uma única passada,
    sem saber de antemão quantos produtos passarão: o produto número n entra no
    reservatório com probabilidade max_itens / n, substituindo um sorteado.
    A semente fixa faz a mesma importação dar sempre o mesmo resultado.
    """
    rng = random.Random(semente)
    reservatorio: list[Item] = []
    contagem = Counter()
    for meta in ler_jsonl(caminho_meta(pasta, categoria.fonte_amazon)):
        contagem["lidos"] += 1
        if not categoria.filtro.aceita(meta.get("title") or "", meta.get("categories") or []):
            continue
        contagem["passaram_no_filtro"] += 1
        if (meta.get("rating_number") or 0) < categoria.min_avaliacoes:
            contagem["poucas_avaliacoes"] += 1
            continue
        item = item_de_meta(meta, categoria.slug)
        if item is None:
            contagem["sem_titulo_ou_id"] += 1
            continue
        contagem["candidatos"] += 1
        if len(reservatorio) < categoria.max_itens:
            reservatorio.append(item)
        else:
            j = rng.randrange(contagem["candidatos"])
            if j < categoria.max_itens:
                reservatorio[j] = item
    contagem["importados"] = len(reservatorio)
    contagem["sem_foto"] = sum(1 for i in reservatorio if not i.foto_url)
    return reservatorio, dict(contagem)


# ─────────────────────────── Avaliações ───────────────────────────


def _momento(timestamp) -> str:
    """O timestamp vem em milissegundos desde 1970 (apesar de a documentação dizer só "Unix time")."""
    t = float(timestamp)
    if t > 1e11:  # grande demais para segundos: são milissegundos
        t /= 1000
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _avaliacao(r: dict) -> Avaliacao | None:
    item_id = r.get("parent_asin") or r.get("asin")
    momento = r.get("timestamp") or r.get("sort_timestamp")
    if not item_id or not r.get("user_id") or r.get("rating") is None or momento is None:
        return None
    return Avaliacao(usuario=r["user_id"], item_id=item_id, nota=float(r["rating"]), momento=_momento(momento),
                     compra_verificada=bool(r.get("verified_purchase")))


def importar_avaliacoes(categoria: Categoria, pasta: Path, ids_itens: set[str]) -> Iterator[Avaliacao]:
    """Avaliações dos produtos importados. As avaliações apontam para `parent_asin`, o mesmo id dos itens."""
    for r in ler_jsonl(caminho_avaliacoes(pasta, categoria.fonte_amazon)):
        if (r.get("parent_asin") or r.get("asin")) in ids_itens:
            a = _avaliacao(r)
            if a:
                yield a


# ─────────────────────────── Seleção centrada em usuários ───────────────────────────

_RE_ITEM = re.compile(r'"parent_asin":\s*"([^"]+)"')
_RE_USUARIO = re.compile(r'"user_id":\s*"([^"]+)"')


def _varrer(caminho: Path, ids: set[str], ler_json: bool) -> Iterator[tuple[str, str, dict | None]]:
    """Percorre o arquivo de avaliações devolvendo só as linhas de produtos em `ids`.

    Otimização: o arquivo tem dezenas de milhões de linhas, e interpretar JSON é a parte
    cara. Uma expressão regular extrai o produto direto do texto da linha; o JSON só é
    interpretado quando o produto interessa (e só se `ler_json`, na segunda passada).
    """
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Baixe com: bash scripts/baixar_amazon.sh")
    abrir = gzip.open if caminho.suffix == ".gz" else open
    with abrir(caminho, "rt", encoding="utf-8") as f:
        for n, linha in enumerate(f, start=1):
            if n % 5_000_000 == 0:
                log.info("%s: %d milhões de linhas lidas", caminho.name, n // 1_000_000)
            m = _RE_ITEM.search(linha)
            if not m or m.group(1) not in ids:
                continue
            if ler_json:
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                yield r.get("user_id") or "", m.group(1), r
            else:
                u = _RE_USUARIO.search(linha)
                if u:
                    yield u.group(1), m.group(1), None


def importar_por_usuarios(categoria: Categoria, pasta: Path, semente: int = 42) -> tuple[list[Item], list[Avaliacao], dict]:
    """Monta o catálogo a partir de USUÁRIOS com histórico, e não de produtos sorteados.

    Por quê: sorteando produtos, quase todo usuário aparece com uma avaliação só (medimos:
    110 mil de 120 mil), e com uma interação não dá para aprender o gosto de ninguém.

    Passos:
    1. Metadados: todos os produtos que passam no filtro da categoria são candidatos.
       Os que casam com `titulo_contem` (ex.: "monster high") são garantidos no catálogo,
       para a busca da fase 1 continuar tendo esses produtos.
    2. Avaliações, 1ª passada: conta quantas avaliações cada usuário fez em candidatos.
    3. Elegíveis: usuários com pelo menos `min_avaliacoes_usuario` avaliações na categoria.
    4. Avaliações, 2ª passada: guarda o histórico completo (na categoria) dos elegíveis.
    5. Seleção gulosa: em ordem aleatória (semente fixa), cada elegível entra com todos os
       produtos que avaliou, enquanto o catálogo couber em `max_itens`. Usuário que estouraria
       o limite é pulado, e não cortado: o histórico de quem entra fica completo.
    6. Avaliações importadas: as de TODOS os elegíveis sobre produtos do catálogo. Os que não
       foram "selecionados" ainda contribuem com sinal (popularidade, coocorrência).

    São duas passadas pelo arquivo de avaliações (o de Electronics tem dezenas de milhões de
    linhas): leva vários minutos por categoria.
    """
    rng = random.Random(semente)
    contagem: Counter = Counter()
    candidatos: dict[str, Item] = {}
    garantidos: list[str] = []
    for meta in ler_jsonl(caminho_meta(pasta, categoria.fonte_amazon)):
        contagem["produtos_lidos"] += 1
        if not categoria.filtro.aceita(meta.get("title") or "", meta.get("categories") or []):
            continue
        if (meta.get("rating_number") or 0) < categoria.min_avaliacoes:
            continue
        item = item_de_meta(meta, categoria.slug)
        if item is None:
            continue
        candidatos[item.id] = item
        if categoria.filtro.casa_titulo(item.titulo):
            garantidos.append(item.id)
    contagem["candidatos"] = len(candidatos)

    caminho = caminho_avaliacoes(pasta, categoria.fonte_amazon)
    por_usuario: Counter = Counter()
    for usuario, _, _ in _varrer(caminho, set(candidatos), ler_json=False):
        por_usuario[usuario] += 1
    contagem["usuarios_com_avaliacao"] = len(por_usuario)
    elegiveis = sorted(u for u, n in por_usuario.items() if n >= categoria.min_avaliacoes_usuario)
    del por_usuario
    rng.shuffle(elegiveis)
    contagem["usuarios_elegiveis"] = len(elegiveis)

    conjunto_elegiveis = set(elegiveis)
    historico: dict[str, list[Avaliacao]] = defaultdict(list)
    for usuario, _, r in _varrer(caminho, set(candidatos), ler_json=True):
        if usuario in conjunto_elegiveis:
            a = _avaliacao(r)
            if a:
                historico[usuario].append(a)

    # Os garantidos ocupam no máximo um terço do catálogo, para sobrar espaço para os usuários
    rng.shuffle(garantidos)
    catalogo = set(garantidos[: categoria.max_itens // 3])
    contagem["garantidos_pelo_titulo"] = len(catalogo)
    selecionados = 0
    for usuario in elegiveis:
        itens_usuario = {a.item_id for a in historico[usuario]}
        if len(catalogo | itens_usuario) > categoria.max_itens:
            continue
        catalogo |= itens_usuario
        selecionados += 1
    contagem["usuarios_selecionados"] = selecionados

    avaliacoes = [a for u in elegiveis for a in historico[u] if a.item_id in catalogo]
    itens = [candidatos[i] for i in sorted(catalogo)]
    contagem["itens"] = len(itens)
    contagem["avaliacoes"] = len(avaliacoes)
    usuarios_finais = Counter(a.usuario for a in avaliacoes)
    contagem["usuarios_importados"] = len(usuarios_finais)
    contagem["usuarios_com_5_ou_mais"] = sum(1 for n in usuarios_finais.values() if n >= 5)
    contagem["sem_foto"] = sum(1 for i in itens if not i.foto_url)
    return itens, avaliacoes, dict(contagem)


# ─────────────────────────── Inspeção ───────────────────────────


def inspecionar(pasta: Path, fonte: str, amostra: int = 50_000, top: int = 40) -> dict:
    """Quais nomes de categoria aparecem no domínio? Serve para escrever os filtros do categorias.yaml."""
    nomes, lidos, com_foto, com_preco = Counter(), 0, 0, 0
    for meta in ler_jsonl(caminho_meta(pasta, fonte)):
        lidos += 1
        nomes.update(meta.get("categories") or ["(sem categoria)"])
        com_foto += bool(meta.get("images"))
        com_preco += _preco(meta.get("price")) is not None
        if lidos >= amostra:
            break
    return {
        "produtos_lidos": lidos,
        "com_foto": com_foto,
        "com_preco": com_preco,
        "categorias_mais_comuns": nomes.most_common(top),
    }
