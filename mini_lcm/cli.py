"""Linha de comando.

    mini-lcm inspecionar Toys_and_Games   # nomes de categoria, para escrever os filtros
    mini-lcm importar                 # todas as categorias de categorias.yaml
    mini-lcm importar --categoria bonecas
    mini-lcm indexar                  # gera os embeddings que faltam
    mini-lcm buscar "draculaura antiga" --categoria bonecas --preco-max 200 --condicao usado

    Avaliação (notas de 0 a 3; ver docs/decisoes.md):
    mini-lcm rotular "boneca gótica" --categoria bonecas --perfil "colecionadora adulta"   # passo 1: você
    mini-lcm julgar --apenas-rotulados    # passo 2: a IA julga os mesmos pares
    mini-lcm concordancia                 # passo 3: kappa ponderado + intervalo de confiança
    mini-lcm revisar                      # passo 4: você arbitra as discordâncias
    mini-lcm julgar "outra busca" --categoria bonecas   # escala: só a IA
    mini-lcm auditar --amostra 30         # confere uma amostra do que só a IA julgou
    mini-lcm avaliar --fonte combinado    # recall@10, MRR e NDCG@10 de cada modo

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
from .avaliacao import (avaliar, carregar_gabarito, obter_ou_criar, pool_para_rotular, relatorio_markdown, rotular,
                        salvar_gabarito)
from .calibracao import (amostra_auditoria, auditar, concordancia, julgar_pendentes, relatorio_concordancia_texto,
                         revisar)
from .busca import MODOS, ConfigBusca, buscar
from .amazon import importar_avaliacoes, importar_itens, inspecionar
from .config import Config, carregar_config
from .custo import RegistroCusto, cronometro, resumir
from .embeddings import criar_embedder
from .indexacao import gerar_embeddings
from .juiz import JuizGemini, carregar_rubrica, hash_rubrica
from .modelos import Filtros

DIMENSOES = {"clip": 512, "fake": 256}  # para abrir o banco sem carregar o modelo (o CLIP pesa ~1 GB)


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


def _filtros_dict(args) -> dict:
    return {k: v for k, v in vars(_filtros(args)).items() if v is not None}


def _itens_do_gabarito(arm: Armazenamento, gabarito) -> dict:
    ids = {i for c in gabarito for i in list(c.notas_humanas) + list(c.notas_ia)}
    return arm.obter_itens(sorted(ids))


def cmd_rotular(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder)
    gabarito = carregar_gabarito(cfg.gabarito)
    consulta = obter_ou_criar(gabarito, args.consulta, _filtros_dict(args), args.perfil or "")
    itens = pool_para_rotular(arm, embedder, consulta.texto, consulta.filtros_obj(), args.profundidade)
    try:
        rotular(consulta, itens)
    finally:  # salva mesmo se você interromper com Ctrl+C
        salvar_gabarito(cfg.gabarito, gabarito)
        relevantes = sum(1 for n in consulta.notas_humanas.values() if n >= 2)
        print(f"\nSalvo: {len(consulta.notas_humanas)} notas suas nesta consulta ({relevantes} com nota >= 2).")


def _juiz(cfg: Config) -> JuizGemini:
    j = cfg.juiz
    return JuizGemini(j.api_key, carregar_rubrica(cfg.rubrica), j.modelo, j.preco_entrada_1m, j.preco_saida_1m)


def cmd_julgar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder)
    juiz = _juiz(cfg)
    gabarito = carregar_gabarito(cfg.gabarito)
    if args.consulta:
        alvo = [obter_ou_criar(gabarito, args.consulta, _filtros_dict(args), args.perfil or "")]
    else:
        alvo = gabarito
    if not alvo:
        sys.exit("O gabarito está vazio. Comece com `mini-lcm rotular` (ou passe uma consulta para `julgar`).")
    reg = julgar_pendentes(alvo, arm, embedder, juiz, juiz.versao_rubrica, apenas_rotulados=args.apenas_rotulados,
                           profundidade=args.profundidade, limite=args.limite,
                           salvar=lambda: salvar_gabarito(cfg.gabarito, gabarito))
    if reg.itens or reg.falhas:
        arm.registrar_custo(reg)
    custo = f"US$ {reg.custo_usd:.4f}" if cfg.juiz.preco_entrada_1m else "preço não configurado no .env"
    print(f"{reg.itens} julgamentos novos com {juiz.modelo} ({reg.falhas} falhas), "
          f"{reg.tokens} tokens, {custo}, {reg.segundos:.0f}s")


def cmd_concordancia(cfg: Config, arm: Armazenamento, args) -> None:
    gabarito = carregar_gabarito(cfg.gabarito)
    rel = concordancia(gabarito, kappa_minimo=args.kappa_minimo, min_pares=args.min_pares, origem=args.origem)
    print(f"Conjunto: {args.origem}\n")
    print(relatorio_concordancia_texto(rel))
    try:
        versao = hash_rubrica(carregar_rubrica(cfg.rubrica))
        antigas = sum(1 for c in gabarito for j in c.notas_ia.values() if j.get("rubrica") != versao)
        if antigas:
            print(f"\nAtenção: {antigas} notas da IA são de uma versão antiga da rubrica. Rode `mini-lcm julgar`.")
    except FileNotFoundError:
        pass


def cmd_revisar(cfg: Config, arm: Armazenamento, args) -> None:
    gabarito = carregar_gabarito(cfg.gabarito)
    try:
        alteradas = revisar(gabarito, _itens_do_gabarito(arm, gabarito), distancia_minima=args.distancia)
    finally:
        salvar_gabarito(cfg.gabarito, gabarito)
    print(f"\n{alteradas} notas suas alteradas. Rode `mini-lcm concordancia` de novo.")


def cmd_auditar(cfg: Config, arm: Armazenamento, args) -> None:
    gabarito = carregar_gabarito(cfg.gabarito)
    amostra = amostra_auditoria(gabarito, args.amostra)
    if not amostra:
        sys.exit("Não há pares julgados só pela IA. Rode `mini-lcm julgar` antes.")
    try:
        feitos = auditar(gabarito, _itens_do_gabarito(arm, gabarito), amostra)
    finally:
        salvar_gabarito(cfg.gabarito, gabarito)
    print(f"\n{feitos} pares auditados. Veja: mini-lcm concordancia --origem auditoria --min-pares 20")


def cmd_avaliar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder)
    gabarito = carregar_gabarito(cfg.gabarito)
    if args.fonte != "humano":
        calib = concordancia(gabarito)
        if not calib.aprovado:
            print(f"Atenção: o juiz ainda não passou na calibração ({calib.motivo}). "
                  f"Use --fonte humano ou trate estes números como provisórios.\n")
    modos = tuple(m.strip() for m in args.modos.split(","))
    rel = avaliar(arm, embedder, gabarito, modos, k=args.k, peso_imagem=args.peso_imagem, fonte=args.fonte)
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

    s = sub.add_parser("rotular", help="passo 1: você dá notas de 0 a 3 aos resultados de uma consulta")
    s.add_argument("consulta")
    _args_filtros(s)
    s.add_argument("--perfil", help='quem busca e com que intenção, ex.: "colecionadora adulta"')
    s.add_argument("--profundidade", type=int, default=20, help="quantos resultados de cada modo entram no pool")
    s.set_defaults(func=cmd_rotular)

    s = sub.add_parser("julgar", help="passo 2 e escala: a IA dá notas de 0 a 3")
    s.add_argument("consulta", nargs="?", help="sem consulta: julga todas as do gabarito")
    _args_filtros(s)
    s.add_argument("--perfil")
    s.add_argument("--apenas-rotulados", action="store_true", help="só os pares que você já rotulou (calibração)")
    s.add_argument("--profundidade", type=int, default=20)
    s.add_argument("--limite", type=int, help="máximo de julgamentos nesta rodada (controle de custo)")
    s.set_defaults(func=cmd_julgar)

    s = sub.add_parser("concordancia", help="passo 3: o quanto a IA concorda com você")
    s.add_argument("--origem", choices=["rotulagem", "auditoria", "todos"], default="rotulagem")
    s.add_argument("--kappa-minimo", type=float, default=0.6)
    s.add_argument("--min-pares", type=int, default=50)
    s.set_defaults(func=cmd_concordancia)

    s = sub.add_parser("revisar", help="passo 4: você arbitra as discordâncias, vendo a justificativa da IA")
    s.add_argument("--distancia", type=int, default=1, help="só discordâncias de pelo menos N pontos")
    s.set_defaults(func=cmd_revisar)

    s = sub.add_parser("auditar", help="você dá nota, às cegas, a uma amostra do que só a IA julgou")
    s.add_argument("--amostra", type=int, default=30)
    s.set_defaults(func=cmd_auditar)

    s = sub.add_parser("avaliar", help="recall@k, MRR e NDCG@k de cada modo de busca")
    s.add_argument("--modos", default=",".join(MODOS))
    s.add_argument("--fonte", choices=["humano", "ia", "combinado"], default="combinado")
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
