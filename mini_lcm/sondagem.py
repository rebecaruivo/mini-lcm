"""Sondagem: um teste rápido e SEM rótulos para comparar variantes de embedding.

Para cada termo (ex.: "monster high", "barbie"), busca o próprio termo e mede:

- **acerto@10:** quantos dos 10 primeiros têm o termo no título. Se a pessoa digita
  "monster high" e existem centenas de produtos com esse nome, uma busca saudável
  deveria trazer quase só eles. Não precisa de gabarito porque a resposta está no título.
- **faixa de similaridade (só texto):** similaridade do 1º menos a do 100º no canal de
  texto. Faixa estreita (ex.: 0,93 vs 0,88) significa que o modelo acha tudo quase
  igualmente parecido com a consulta, e qualquer ruído troca a ordem.

Limitação importante: só mede buscas pelo NOME. Não diz nada sobre buscas descritivas
("boneca gótica de cabelo rosa"), que são o ponto forte dos embeddings. Serve para
achar defeitos grosseiros e comparar variantes; a avaliação de verdade é o `avaliar`.
"""

from __future__ import annotations

from .armazenamento import Armazenamento
from .busca import MODOS, ConfigBusca, buscar
from .embeddings import Embedder
from .modelos import Filtros, normalizar_texto


def sondar(armazenamento: Armazenamento, embedder: Embedder, termos: list[str], filtros: Filtros, k: int = 10) -> list[dict]:
    linhas = []
    for termo in termos:
        alvo = normalizar_texto(termo)
        linha: dict = {"termo": termo}
        for modo in MODOS:
            resultados = buscar(armazenamento, embedder, termo, filtros, k=k, config=ConfigBusca(modo=modo))
            acertos = sum(1 for r in resultados if alvo in normalizar_texto(r.item.titulo))
            linha[modo] = acertos / k
        vizinhos = armazenamento.vizinhos(embedder.chave("texto"), "texto", embedder.consulta(termo, "texto"), filtros, 100)
        linha["faixa_texto"] = (vizinhos[0][1] - vizinhos[-1][1]) if len(vizinhos) > 1 else 0.0
        linhas.append(linha)
    return linhas


def sondagem_markdown(linhas: list[dict], nome_embedder: str, k: int = 10) -> str:
    saida = [f"Variante: {nome_embedder}", "",
             f"| Termo | acerto@{k} texto | imagem | fusão | faixa de sim. (texto, 1º−100º) |",
             "|---|---|---|---|---|"]
    for l in linhas:
        saida.append(f"| {l['termo']} | {l['texto']:.0%} | {l['imagem']:.0%} | {l['fusao']:.0%} | {l['faixa_texto']:.3f} |")
    if linhas:
        media = {m: sum(l[m] for l in linhas) / len(linhas) for m in MODOS}
        saida.append(f"| **média** | **{media['texto']:.0%}** | **{media['imagem']:.0%}** | **{media['fusao']:.0%}** | |")
    return "\n".join(saida)
