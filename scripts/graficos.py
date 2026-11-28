#!/usr/bin/env python3
"""Gráficos de experimentos: uma métrica em função de um parâmetro, um sistema por linha.

Lê CSV/TSV ou tabelas em markdown (inclusive no meio de um log) e desenha um gráfico por painel:
- no terminal, com barras de texto (só biblioteca padrão);
- em PNG, com matplotlib (opcional: --png PASTA).

Exemplos:
  python graficos.py resultados.csv --x peso --y ndcg --serie modelo --painel categoria
  python graficos.py log.txt --x "Peso da imagem" --y NDCG@10 --painel Categoria
  cat log.txt | python graficos.py - --x lr --y acuracia --serie otimizador --png graficos/

Linhas de cabeçalho no texto no formato "### serie=valor · painel=valor" (ou "### chave=valor")
dão colunas extras às tabelas que vêm depois delas: útil quando cada tabela de um log é de um
sistema diferente e o nome do sistema não está dentro da tabela.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path


def _numero(texto: str) -> float | None:
    t = re.sub(r"[←*%]|maior.*$", "", texto).strip().replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def ler_linhas(texto: str) -> list[dict[str, str]]:
    """Lista de dicionários, uma entrada por linha de dados, de CSV/TSV ou de tabelas markdown."""
    if not texto.strip():
        return []
    if not any(l.strip().startswith("|") for l in texto.splitlines()):
        dialeto = csv.Sniffer().sniff(texto[:2048], delimiters=",;\t")
        return [dict(l) for l in csv.DictReader(io.StringIO(texto), dialect=dialeto)]
    linhas, cabecalho, extras = [], None, {}
    for bruta in texto.splitlines():
        s = bruta.strip()
        if s.startswith("#"):
            pares = re.findall(r"([\w@.+-]+)\s*=\s*([^·|]+)", s)
            if pares:
                extras = {k.strip(): v.strip() for k, v in pares}
                cabecalho = None
            continue
        if not s.startswith("|"):
            continue
        celulas = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in celulas):
            continue
        if cabecalho is None or (len(celulas) == len(cabecalho) and celulas == cabecalho):
            cabecalho = celulas
            continue
        if len(celulas) == len(cabecalho):
            linhas.append({**extras, **dict(zip(cabecalho, celulas))})
    return linhas


def _valor(linha, colunas):
    """Junta várias colunas ("Categoria,consultas") num rótulo só."""
    return " · ".join(linha.get(c, "?") for c in colunas.split(",")) if colunas else None


def agrupar(linhas, x, y, serie=None, painel=None, repetidos=None):
    """{painel: {serie: {x: y}}}; x numérico quando possível. Pontos repetidos vão para `repetidos`."""
    grupos: dict[str, dict[str, dict]] = {}
    for l in linhas:
        if x not in l or y not in l:
            continue
        vy = _numero(l[y])
        if vy is None:
            continue
        vx = _numero(l[x])
        chave_x = vx if vx is not None else l[x]
        p = _valor(l, painel) or "todos"
        s = _valor(l, serie) or y
        destino = grupos.setdefault(p, {}).setdefault(s, {})
        if chave_x in destino and destino[chave_x] != vy and repetidos is not None:
            repetidos.append((p, s, chave_x))
        destino[chave_x] = vy
    return grupos


def avisos(grupos) -> list[str]:
    saida = []
    for p, series in grupos.items():
        tamanhos = {len(v) for v in series.values()}
        if len(tamanhos) > 1:
            saida.append(f"Aviso [{p}]: as séries têm quantidades diferentes de pontos; compare só onde as duas existem.")
        xs = sorted({k for v in series.values() for k in v if isinstance(k, float)})
        passos = {round(b - a, 9) for a, b in zip(xs, xs[1:])}
        if len(passos) > 1:
            saida.append(f"Aviso [{p}]: os valores de x não são igualmente espaçados; no terminal a distância some.")
    return saida


def terminal(grupos, y, largura=24, menor_melhor=False) -> str:
    todos = [v for s in grupos.values() for serie in s.values() for v in serie.values()]
    topo = max(1.0, max(todos)) if todos else 1.0
    base = min(0.0, min(todos)) if todos else 0.0
    partes = []
    for p, series in grupos.items():
        xs = sorted({k for v in series.values() for k in v}, key=lambda k: (isinstance(k, str), k))
        col = largura + 12
        partes.append(f"\n{p} · {y} (barra cheia = {topo:g})")
        partes.append("x".ljust(8) + "".join(str(s)[:col - 2].ljust(col) for s in series))
        escolhe = min if menor_melhor else max
        melhor = {s: escolhe(v, key=v.get) for s, v in series.items()}
        for k in xs:
            celulas = []
            for s, v in series.items():
                if k not in v:
                    celulas.append("".ljust(col))
                    continue
                n = round((v[k] - base) / (topo - base) * largura)
                marca = " ←" if melhor[s] == k else ""
                celulas.append(("█" * n).ljust(largura) + f" {v[k]:.4g}{marca}".ljust(col - largura))
            partes.append(f"{k:g}".ljust(8) if isinstance(k, float) else str(k)[:7].ljust(8))
            partes[-1] += "".join(celulas)
    partes.append("\n← = " + ("menor" if menor_melhor else "maior") + " valor de cada série")
    return "\n".join(partes)


def png(grupos, x, y, pasta: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Para PNG: pip install matplotlib")
    pasta.mkdir(parents=True, exist_ok=True)
    saida = []
    for p, series in grupos.items():
        fig, ax = plt.subplots(figsize=(6.4, 4))
        numerico = all(isinstance(k, float) for v in series.values() for k in v)
        for s, v in series.items():
            ks = sorted(v, key=lambda k: (isinstance(k, str), k))
            if numerico:
                ax.plot(ks, [v[k] for k in ks], marker="o", label=str(s))
            else:
                ax.plot([str(k) for k in ks], [v[k] for k in ks], marker="o", label=str(s))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(str(p))
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        nome = re.sub(r"[^\w-]+", "-", f"{p}-{y}".lower()).strip("-") or "grafico"
        caminho = pasta / f"{nome}.png"
        fig.savefig(caminho, dpi=120)
        plt.close(fig)
        saida.append(caminho)
    return saida


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("arquivo", help="CSV, TSV, markdown ou log com tabelas; - lê da entrada padrão")
    a.add_argument("--x", required=True, help="coluna do parâmetro que varia (eixo x)")
    a.add_argument("--y", required=True, help="coluna da métrica (eixo y)")
    a.add_argument("--serie", help="coluna(s) que separam os sistemas comparados, uma linha por valor (várias: a,b)")
    a.add_argument("--painel", help="coluna(s) que separam os gráficos, ex.: categoria,conjunto (várias: a,b)")
    a.add_argument("--png", help="pasta para salvar um PNG por painel (exige matplotlib)")
    a.add_argument("--menor-melhor", action="store_true", help="a seta marca o menor valor (erro, custo, latência)")
    args = a.parse_args(argv)

    texto = sys.stdin.read() if args.arquivo == "-" else Path(args.arquivo).read_text(encoding="utf-8")
    linhas = ler_linhas(texto)
    if not linhas:
        sys.exit("Nenhuma tabela encontrada.")
    colunas = sorted({c for l in linhas for c in l})
    pedidas = [args.x, args.y] + [c for g in (args.serie, args.painel) if g for c in g.split(",")]
    for c in pedidas:
        if c not in colunas:
            sys.exit(f"Coluna '{c}' não encontrada. Colunas disponíveis: {', '.join(colunas)}")
    repetidos: list = []
    grupos = agrupar(linhas, args.x, args.y, args.serie, args.painel, repetidos)
    if repetidos:
        p, s, k = repetidos[0]
        sys.exit(f"Há {len(repetidos)} pontos repetidos com valores diferentes (ex.: painel '{p}', série '{s}', "
                 f"x={k}). Falta separar alguma coluna: acrescente-a em --painel ou --serie (várias com vírgula). "
                 f"Colunas disponíveis: {', '.join(colunas)}")
    for aviso in avisos(grupos):
        print(aviso)
    print(terminal(grupos, args.y, menor_melhor=args.menor_melhor))
    if args.png:
        for c in png(grupos, args.x, args.y, Path(args.png)):
            print(f"PNG: {c}")


if __name__ == "__main__":
    main()
