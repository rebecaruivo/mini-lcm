"""Linha de comando.

    mini-lcm inspecionar Toys_and_Games   # nomes de categoria, para escrever os filtros
    mini-lcm importar                 # todas as categorias de categorias.yaml
    mini-lcm importar --categoria bonecas
    mini-lcm indexar                  # gera os embeddings que faltam
    mini-lcm buscar "draculaura antiga" --categoria bonecas --preco-max 200 --condicao usado
    mini-lcm rotular "draculaura antiga" --categoria bonecas
    mini-lcm avaliar                  # recall@10 e MRR de cada modo, a partir do gabarito
    mini-lcm custos
    mini-lcm status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .armazenamento import Armazenamento, ArmazenamentoLocal, ArmazenamentoPostgres
from .avaliacao import (ConsultaRotulada, avaliar, carregar_gabarito, pool_para_rotular, relatorio_markdown,
                        rotular, salvar_gabarito)
from .busca import MODOS, ConfigBusca, buscar
from .amazon import importar_avaliacoes, importar_itens, inspecionar
from .config import RAIZ, Config, carregar_config
from .custo import RegistroCusto, cronometro, resumir
from .embeddings import criar_embedder
from .indexacao import gerar_embeddings
from .modelos import Filtros

DIMENSOES = {"clip": 512, "fake": 256}  # para abrir o banco sem carregar o modelo (o CLIP pesa ~1 GB)
GABARITO = RAIZ / "avaliacao" / "gabarito.json"


def abrir_armazenamento(cfg: Config) -> Armazenamento:
    if cfg.database_url:
        return ArmazenamentoPostgres(cfg.database_url, DIMENSOES[cfg.embedder],
                                     categorias=[(c.slug, c.nome) for c in cfg.categorias])
    return ArmazenamentoLocal(cfg.pasta_dados)


def _filtros(args) -> Filtros:
    return Filtros(categoria=args.categoria, preco_min=args.preco_min, preco_max=args.preco_max, condicao=args.condicao)


def _args_filtros(p: argparse.ArgumentParser) -> None:
    p.add_argument("--categoria")
    p.add_argument("--preco-min", type=float)
    p.add_argument("--preco-max", type=float)
    p.add_argument("--condicao", choices=["novo", "usado"])


# ─────────────────────────── Comandos ───────────────────────────


def cmd_inspecionar(cfg: Config, arm: Armazenamento, args) -> None:
    info = inspecionar(cfg.pasta_brutos, args.fonte, amostra=args.amostra)
    print(f"{info['produtos_lidos']} produtos lidos: {info['com_foto']} com foto, {info['com_preco']} com preço\n")
    print("Categorias mais comuns (use os nomes em categorias_contem):")
    for nome, n in info["categorias_mais_comuns"]:
        print(f"  {n:7}  {nome}")


def cmd_importar(cfg: Config, arm: Armazenamento, args) -> None:
    categorias = [cfg.categoria(args.categoria)] if args.categoria else cfg.categorias
    for cat in categorias:
        reg = RegistroCusto(etapa="importacao", modelo=f"amazon:{cat.fonte_amazon}")
        with cronometro(reg):
            itens, contagem = importar_itens(cat, cfg.pasta_brutos)
            novos = arm.salvar_itens(itens)
            reg.itens = len(itens)
            print(f"{cat.slug}: {len(itens)} produtos ({novos} novos). Contagem: {contagem}")
            if not args.sem_avaliacoes:
                ids = {i.id for i in itens}
                lote, total, novas = [], 0, 0
                for a in importar_avaliacoes(cat, cfg.pasta_brutos, ids):
                    lote.append(a)
                    if len(lote) == 50_000:
                        novas += arm.salvar_avaliacoes(lote)
                        total += len(lote)
                        lote = []
                novas += arm.salvar_avaliacoes(lote)
                total += len(lote)
                print(f"{cat.slug}: {total} avaliações ({novas} novas)")
        arm.registrar_custo(reg)


def cmd_indexar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder)
    canais = (args.canal,) if args.canal else ("texto", "imagem")
    for canal, reg in gerar_embeddings(arm, embedder, canais, args.categoria).items():
        print(f"{canal}: {reg.itens} itens em {reg.segundos:.1f}s ({reg.falhas} fotos com falha)")


def cmd_buscar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder)
    reg = RegistroCusto(etapa="busca", modelo=embedder.nome)
    config = ConfigBusca(modo=args.modo, peso_imagem=args.peso_imagem)
    resultados = buscar(arm, embedder, args.consulta, _filtros(args), k=args.k, config=config, registro=reg)
    arm.registrar_custo(reg)
    if not resultados:
        print("Nenhum resultado. Rodou `mini-lcm indexar`? Os filtros estão muito restritos?")
    for n, r in enumerate(resultados, start=1):
        it = r.item
        canais = "  ".join(f"{c}: #{r.posicoes[c]} (sim {r.similaridades[c]:.2f})" for c in r.posicoes)
        print(f"{n:2}. {it.titulo}\n    {it.preco_formatado} · {it.condicao} · {it.categoria} · {it.link}\n    {canais}")
    print(f"\n{reg.segundos * 1000:.0f} ms")


def cmd_rotular(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder)
    gabarito = carregar_gabarito(GABARITO)
    filtros = {k: v for k, v in vars(_filtros(args)).items() if v is not None}
    consulta = next((c for c in gabarito if c.texto == args.consulta and c.filtros == filtros), None)
    if consulta is None:
        consulta = ConsultaRotulada(id=f"c{len(gabarito) + 1}", texto=args.consulta, filtros=filtros)
        gabarito.append(consulta)
    itens = pool_para_rotular(arm, embedder, consulta.texto, consulta.filtros_obj(), args.profundidade)
    try:
        rotular(consulta, itens)
    finally:  # salva mesmo se você interromper com Ctrl+C
        salvar_gabarito(GABARITO, gabarito)
        print(f"\nSalvo em {GABARITO}: {len(consulta.relevantes)} relevantes de {len(consulta.avaliados)} avaliados.")


def cmd_avaliar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder)
    modos = tuple(m.strip() for m in args.modos.split(","))
    rel = avaliar(arm, embedder, carregar_gabarito(GABARITO), modos, k=args.k, peso_imagem=args.peso_imagem)
    texto = relatorio_markdown(rel)
    print(texto)
    if args.salvar:
        Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
        Path(args.salvar).write_text(f"# Avaliação da busca ({embedder.nome})\n\n{texto}\n", encoding="utf-8")
        print(f"\nSalvo em {args.salvar}")


def cmd_custos(cfg: Config, arm: Armazenamento, args) -> None:
    linhas = resumir(arm.custos())
    if not linhas:
        print("Nenhum custo registrado ainda.")
        return
    print("| Etapa | Modelo | Itens | Falhas | Segundos | ms/item | US$ | US$/1000 itens |")
    print("|---|---|---|---|---|---|---|---|")
    for l in linhas:
        print(f"| {l['etapa']} | {l['modelo']} | {l['itens']} | {l['falhas']} | {l['segundos']} | "
              f"{l['ms_por_item']} | {l['custo_usd']} | {l['usd_por_1000_itens']} |")


def cmd_status(cfg: Config, arm: Armazenamento, args) -> None:
    print(json.dumps(arm.estatisticas(), indent=2, ensure_ascii=False))


# ─────────────────────────── Entrada ───────────────────────────


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mini-lcm", description="Motor de personalização em miniatura (fase 1: busca)")
    p.add_argument("-v", "--verboso", action="store_true")
    sub = p.add_subparsers(dest="comando", required=True)

    s = sub.add_parser("inspecionar", help="mostra as categorias de um domínio da Amazon")
    s.add_argument("fonte", help="ex.: Toys_and_Games")
    s.add_argument("--amostra", type=int, default=50_000)
    s.set_defaults(func=cmd_inspecionar)

    s = sub.add_parser("importar", help="importa produtos e avaliações dos arquivos da Amazon")
    s.add_argument("--categoria")
    s.add_argument("--sem-avaliacoes", action="store_true", help="só produtos (mais rápido)")
    s.set_defaults(func=cmd_importar)

    s = sub.add_parser("indexar", help="gera os embeddings que faltam")
    s.add_argument("--categoria")
    s.add_argument("--canal", choices=["texto", "imagem"])
    s.set_defaults(func=cmd_indexar)

    s = sub.add_parser("buscar", help="busca em linguagem natural com filtros")
    s.add_argument("consulta")
    _args_filtros(s)
    s.add_argument("--modo", choices=MODOS, default="fusao")
    s.add_argument("--peso-imagem", type=float, default=0.5)
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(func=cmd_buscar)

    s = sub.add_parser("rotular", help="marca à mão os resultados relevantes de uma consulta (gabarito)")
    s.add_argument("consulta")
    _args_filtros(s)
    s.add_argument("--profundidade", type=int, default=20, help="quantos resultados de cada modo entram no pool")
    s.set_defaults(func=cmd_rotular)

    s = sub.add_parser("avaliar", help="recall@k e MRR de cada modo de busca")
    s.add_argument("--modos", default=",".join(MODOS))
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--peso-imagem", type=float, default=0.5)
    s.add_argument("--salvar", help="caminho de um .md para guardar o relatório")
    s.set_defaults(func=cmd_avaliar)

    sub.add_parser("custos", help="tempo e custo por etapa").set_defaults(func=cmd_custos)
    sub.add_parser("status", help="quantos itens e vetores existem").set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verboso else logging.WARNING, format="%(levelname)s %(message)s")
    cfg = carregar_config()
    arm = abrir_armazenamento(cfg)
    try:
        args.func(cfg, arm, args)
    finally:
        arm.fechar()


if __name__ == "__main__":
    main()
