"""Sondagem: um teste rápido e SEM rótulos para comparar variantes de embedding.

Para cada termo (ex.: "monster high", "barbie"), busca o próprio termo e mede:

- **acerto@10:** quantos dos 10 primeiros têm o termo no título. Se a pessoa digita
  "monster high" e existem centenas de produtos com esse nome, uma busca saudável
  deveria trazer quase só eles. Não precisa de gabarito porque a resposta está no título.
- **faixa de similaridade (só texto):** similaridade do 1º menos a do 100º no canal de
  texto. Faixa estreita (ex.: 0,93 vs 0,88) significa que o modelo acha tudo quase
  igualmente parecido com a consulta, e qualquer ruído troca a ordem.

- **teto:** o maior acerto@10 possível, min(10, produtos com o termo no título) / 10.
  Um termo raro no catálogo (ex.: um modelo lançado no fim do período coberto) não pode
  passar do teto, por melhor que seja o modelo. Só é calculado com `--categoria`.

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
    titulos: list[str] | None = None
    if filtros.categoria:
        itens = armazenamento.obter_itens(sorted(armazenamento.ids_categoria(filtros.categoria)))
        titulos = [normalizar_texto(i.titulo) for i in itens.values() if filtros.aceita(i)]
    for termo in termos:
        alvo = normalizar_texto(termo)
        linha: dict = {"termo": termo}
        linha["teto"] = min(k, sum(1 for t in titulos if alvo in t)) / k if titulos is not None else None
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
             f"| Termo | acerto@{k} texto | imagem | fusão | teto | faixa de sim. (texto, 1º−100º) |",
             "|---|---|---|---|---|---|"]

    def pct(v):
        return "—" if v is None else f"{v:.0%}"
    for l in linhas:
        saida.append(f"| {l['termo']} | {l['texto']:.0%} | {l['imagem']:.0%} | {l['fusao']:.0%} | "
                     f"{pct(l.get('teto'))} | {l['faixa_texto']:.3f} |")
    if linhas:
        media = {m: sum(l[m] for l in linhas) / len(linhas) for m in MODOS}
        tetos = [l.get("teto") for l in linhas]
        teto = sum(tetos) / len(tetos) if None not in tetos else None
        saida.append(f"| **média** | **{media['texto']:.0%}** | **{media['imagem']:.0%}** | **{media['fusao']:.0%}** | "
                     f"**{pct(teto)}** | |")
    return "\n".join(saida)
