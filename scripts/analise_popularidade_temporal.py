"""Análise do vazamento temporal da popularidade e da popularidade recente.

Reproduz as tabelas "a popularidade se aproveita do futuro" e "popularidade recente"
de docs/resultados-recomendacao.md:

    python scripts/analise_popularidade_temporal.py --categoria placas-de-video

Para cada pessoa testada (mesma divisão leave-last-out de `avaliar-recomendacao`):
- popularidade global: contada com todo o treino, como no relatório (inclui avaliações
  de outras pessoas posteriores ao teste);
- popularidade do passado: só interações de treino ANTERIORES ao momento do teste;
- popularidade recente: só interações numa janela de dias antes do momento do teste.

É uma análise exploratória: olha o conjunto de teste diretamente. Para virar método do
sistema, a janela precisa ser escolhida na validação.
"""

from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timedelta

import numpy as np

from mini_lcm.avaliacao_recomendacao import _posicao, dividir, intervalo_bootstrap
from mini_lcm.cli import abrir_armazenamento
from mini_lcm.config import carregar_config
from mini_lcm.embeddings import criar_embedder
from mini_lcm.recomendacao import Catalogo, Recomendador


def acertou(pontos: np.ndarray, indice: dict[str, int], caso, k: int = 10) -> bool:
    pontos = pontos.copy()
    for a in caso.treino:
        pontos[indice[a.item_id]] = -np.inf
    return int(np.sum(pontos > pontos[indice[caso.teste.item_id]])) + 1 <= k


def contar(interacoes, n: int, indice: dict[str, int]) -> np.ndarray:
    """Popularidade = pessoas distintas por produto, com desempate fixo pela ordem do catálogo."""
    pop = np.zeros(n)
    for usuario, item_id in {(a.usuario, a.item_id) for a in interacoes}:
        pop[indice[item_id]] += 1
    return pop + 1e-9 * np.arange(n)[::-1] / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--categoria", required=True)
    p.add_argument("--janelas", default="90,180,365,730", help="janelas em dias, separadas por vírgula")
    args = p.parse_args()

    cfg = carregar_config()
    arm = abrir_armazenamento(cfg)
    catalogo = Catalogo.carregar(arm, criar_embedder(cfg.embedder, cfg.texto_item), args.categoria)
    indice, n = catalogo.indice, len(catalogo.ids)
    casos, treino = dividir(arm.avaliacoes_de(categoria=args.categoria), set(catalogo.ids))
    global_ = Recomendador(catalogo).ajustar(treino)
    treino = sorted(treino, key=lambda a: a.momento)
    tempos = [a.momento for a in treino]

    h_global, h_passado, fracao_futuro = [], [], []
    janelas = [int(j) for j in args.janelas.split(",")]
    h_janela: dict[int, list[bool]] = {j: [] for j in janelas}
    for caso in casos:
        fim = bisect.bisect_left(tempos, caso.teste.momento)
        fracao_futuro.append(1 - fim / len(tempos))
        h_global.append(_posicao(global_, "popularidade", caso, caso.treino)[0] <= 10)
        h_passado.append(acertou(contar(treino[:fim], n, indice), indice, caso))
        for j in janelas:
            inicio_t = (datetime.fromisoformat(caso.teste.momento) - timedelta(days=j)).isoformat()
            inicio = bisect.bisect_left(tempos, inicio_t)
            h_janela[j].append(acertou(contar(treino[inicio:fim], n, indice), indice, caso))

    g, ps = np.array(h_global, float), np.array(h_passado, float)
    lo, hi = intervalo_bootstrap(g - ps)
    print(f"{args.categoria}: {len(casos)} usuários testados")
    print(f"fração média do treino posterior ao teste de cada pessoa: {np.mean(fracao_futuro):.3f}")
    print(f"popularidade global:     {g.mean():.3f}")
    print(f"popularidade do passado: {ps.mean():.3f}  (diferença {g.mean() - ps.mean():+.3f}, IC 95% {lo:+.3f} a {hi:+.3f})")
    for j in janelas:
        h = np.array(h_janela[j], float)
        lo, hi = intervalo_bootstrap(h)
        print(f"popularidade dos últimos {j} dias: {h.mean():.3f}  (IC 95% {lo:.3f}–{hi:.3f})")


if __name__ == "__main__":
    main()
