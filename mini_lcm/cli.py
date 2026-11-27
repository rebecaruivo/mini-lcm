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

    mini-lcm sondar "monster high" barbie --categoria bonecas   # teste rápido sem rótulos

    Fase 2 (personalização):
    mini-lcm usuarios --categoria bonecas               # quem tem mais histórico
    mini-lcm perfil USUARIO --categoria bonecas         # histórico + recomendações com o motivo
    mini-lcm buscar "boneca" --categoria bonecas --usuario USUARIO   # busca reordenada pelo gosto
    mini-lcm rotular --lote avaliacao/consultas.yaml --html avaliacao/rotular.html   # página com fotos
    mini-lcm importar-notas ~/Downloads/notas-rotulagem.json
    mini-lcm avaliar-recomendacao --categoria bonecas --curva        # avaliação temporal
    mini-lcm avaliar-recomendacao --categoria bonecas --validacao --janela-dias 90   # escolher parâmetro
    mini-lcm custos
    mini-lcm status

    Variantes de embedding (no .env ou na frente do comando):
    TEXTO_ITEM=titulo mini-lcm indexar --canal texto
    EMBEDDER=clip+e5 mini-lcm indexar --canal texto
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from .armazenamento import Armazenamento, ArmazenamentoLocal, ArmazenamentoPostgres
from .avaliacao import (ConsultaRotulada, avaliar, carregar_gabarito, obter_ou_criar, pool_para_rotular,
                        relatorio_markdown, rotular, salvar_gabarito)
from .calibracao import (amostra_auditoria, amostra_calibracao, auditar, concordancia, julgar_pendentes,
                         relatorio_concordancia_texto, revisar)
from .busca import MODOS, ConfigBusca, buscar
from .amazon import importar_avaliacoes, importar_itens, importar_por_usuarios, inspecionar
from .config import Config, ParametrosRecomendacao, carregar_config
from .rotulagem_html import carregar_lote, gerar_pagina, importar_notas
from .custo import RegistroCusto, cronometro, resumir
from .embeddings import criar_embedder
from .indexacao import gerar_embeddings
from .juiz import JuizGemini, carregar_rubrica, hash_rubrica
from .modelos import Filtros
from .sondagem import sondagem_markdown, sondar
from .avaliacao_recomendacao import avaliar_recomendacao, relatorio_recomendacao_markdown
from .perfil import perfis_do_usuario
from .recomendacao import METODOS as METODOS_REC
from .recomendacao import Catalogo, Recomendador, explicar



def abrir_armazenamento(cfg: Config) -> Armazenamento:
    if cfg.database_url:
        return ArmazenamentoPostgres(cfg.database_url, categorias=[(c.slug, c.nome) for c in cfg.categorias])
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
    """Importa uma categoria e SUBSTITUI o catálogo e as avaliações dela pelo resultado.

    Produtos que saem levam junto vetores e avaliações; os que continuam mantêm os vetores
    (o `indexar` seguinte só processa os novos), mas as avaliações deles são trocadas pelas
    da importação atual. Rodar a mesma importação duas vezes dá sempre o mesmo resultado.
    """
    categorias = [cfg.categoria(args.categoria)] if args.categoria else cfg.categorias
    for cat in categorias:
        reg = RegistroCusto(etapa="importacao", modelo=f"amazon:{cat.fonte_amazon}")
        with cronometro(reg):
            if cat.selecao == "usuarios":
                print(f"{cat.slug}: selecionando usuários com >= {cat.min_avaliacoes_usuario} avaliações "
                      "(duas passadas pelo arquivo de avaliações; pode levar vários minutos)...")
                itens, avaliacoes, contagem = importar_por_usuarios(cat, cfg.pasta_brutos)
            else:
                itens, contagem = importar_itens(cat, cfg.pasta_brutos)
                avaliacoes = [] if args.sem_avaliacoes else list(
                    importar_avaliacoes(cat, cfg.pasta_brutos, {i.id for i in itens}))
            novos_ids = {i.id for i in itens}
            saem = sorted(arm.ids_categoria(cat.slug) - novos_ids)
            arm.remover_itens(saem)
            # As avaliações da categoria também são substituídas: sem isto, produtos que continuam
            # no catálogo manteriam avaliações de importações anteriores (feitas com outra regra de
            # seleção), e o resultado dependeria do histórico de importações.
            arm.remover_avaliacoes_dos_itens(sorted(novos_ids))
            novos = arm.salvar_itens(itens)
            for inicio in range(0, len(avaliacoes), 50_000):
                arm.salvar_avaliacoes(avaliacoes[inicio : inicio + 50_000])
            reg.itens = len(itens)
        arm.registrar_custo(reg)
        print(f"{cat.slug}: {len(itens)} produtos ({novos} novos, {len(saem)} removidos), "
              f"{len(avaliacoes)} avaliações")
        print(f"  detalhes: {json.dumps(contagem, ensure_ascii=False)}")


def cmd_indexar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder, cfg.texto_item)
    canais = (args.canal,) if args.canal else ("texto", "imagem")
    for canal, reg in gerar_embeddings(arm, embedder, canais, args.categoria).items():
        print(f"{canal}: {reg.itens} itens em {reg.segundos:.1f}s ({reg.falhas} fotos com falha)")


def cmd_buscar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder, cfg.texto_item)
    reg = RegistroCusto(etapa="busca", modelo=embedder.nome)
    config = ConfigBusca(modo=args.modo, peso_imagem=args.peso_imagem)
    perfis = None
    if args.usuario:
        perfis = perfis_do_usuario(arm, embedder, args.usuario, meia_vida=_meia_vida_comum(cfg))
        if not perfis:
            sys.exit(f"Usuário {args.usuario} sem histórico. Veja usuários com `mini-lcm usuarios --categoria ...`.")
        print(f"Personalizado para {args.usuario} (categorias com perfil: {', '.join(sorted(perfis))}; "
              f"peso do perfil: {args.peso_perfil})\n")
    resultados = buscar(arm, embedder, args.consulta, _filtros(args), k=args.k, config=config, registro=reg,
                        perfis=perfis, peso_perfil=args.peso_perfil)
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
    embedder = criar_embedder(cfg.embedder, cfg.texto_item)
    gabarito = carregar_gabarito(cfg.gabarito)
    if args.lote or args.html:
        if not (args.lote and args.html):
            sys.exit("Use --lote e --html juntos: --lote avaliacao/consultas.yaml --html avaliacao/rotular.html")
        return _rotular_em_pagina(cfg, arm, embedder, gabarito, args)
    if not args.consulta:
        sys.exit("Passe uma consulta, ou --lote e --html para rotular várias numa página.")
    consulta = obter_ou_criar(gabarito, args.consulta, _filtros_dict(args), args.perfil or "")
    itens = pool_para_rotular(arm, embedder, consulta.texto, consulta.filtros_obj(), args.profundidade)
    try:
        rotular(consulta, itens)
    finally:  # salva mesmo se você interromper com Ctrl+C
        salvar_gabarito(cfg.gabarito, gabarito)
        relevantes = sum(1 for n in consulta.notas_humanas.values() if n >= 2)
        print(f"\nSalvo: {len(consulta.notas_humanas)} notas suas nesta consulta ({relevantes} com nota >= 2).")


def _rotular_em_pagina(cfg: Config, arm: Armazenamento, embedder, gabarito, args) -> None:
    if args.amostra:
        return _rotular_amostra(cfg, arm, gabarito, args)
    lote = []
    for c in carregar_lote(Path(args.lote)):
        consulta = next((g for g in gabarito if g.texto == c["texto"] and g.filtros == c["filtros"]), None)
        consulta = consulta or ConsultaRotulada(id="", texto=c["texto"], filtros=c["filtros"], perfil=c["perfil"])
        itens = pool_para_rotular(arm, embedder, consulta.texto, Filtros(**c["filtros"]), args.profundidade,
                                  semente=0)
        pendentes = [i for i in itens if i.id not in consulta.notas_humanas]
        lote.append((consulta, c["tipo"], pendentes))
        print(f'"{c["texto"]}": {len(itens)} produtos no pool, {len(pendentes)} sem nota sua')
    Path(args.html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.html).write_text(gerar_pagina(lote, cfg.rubrica.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"\nPágina gravada em {args.html} ({sum(len(i) for _, _, i in lote)} produtos). Abra no navegador, "
          f"dê as notas e use 'Baixar notas'; depois: mini-lcm importar-notas <arquivo baixado>")


def _consultas_do_lote(gabarito, caminho: Path) -> list[tuple[ConsultaRotulada, str]]:
    """As consultas do YAML, criadas no gabarito se ainda não existirem."""
    return [(obter_ou_criar(gabarito, c["texto"], c["filtros"], c["perfil"]), c["tipo"]) for c in carregar_lote(caminho)]


def _rotular_amostra(cfg: Config, arm: Armazenamento, gabarito, args) -> None:
    consultas = _consultas_do_lote(gabarito, Path(args.lote))
    julgadas = [c for c, _ in consultas if c.notas_ia]
    if not julgadas:
        sys.exit("Nenhuma consulta do lote foi julgada pela IA ainda. Rode antes:\n"
                 f"  mini-lcm julgar --lote {args.lote} --profundidade {args.profundidade}")
    metade = args.amostra // 2
    amostra = amostra_calibracao(julgadas, aleatoria=args.amostra - metade, estratificada=metade)
    itens = arm.obter_itens(sorted({i for _, i, _ in amostra}))
    lote = []
    for c, tipo in consultas:
        escolhidos = [(i, aud) for cid, i, aud in amostra if cid == c.id and i in itens]
        if escolhidos:
            random.Random(c.id).shuffle(escolhidos)
            lote.append((c, tipo, [itens[i] for i, _ in escolhidos], {i for i, aud in escolhidos if aud}))
    salvar_gabarito(cfg.gabarito, gabarito)
    Path(args.html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.html).write_text(gerar_pagina(lote, cfg.rubrica.read_text(encoding="utf-8")), encoding="utf-8")
    n_aud = sum(1 for *_, aud in amostra if aud)
    print(f"Página gravada em {args.html}: {len(amostra)} produtos de {len(lote)} consultas "
          f"({len(amostra) - n_aud} sorteados ao acaso para a calibração, {n_aud} estratificados para a auditoria). "
          f"Abra no navegador, dê as notas e use 'Baixar notas'; depois: mini-lcm importar-notas <arquivo baixado>")


def cmd_importar_notas(cfg: Config, arm: Armazenamento, args) -> None:
    gabarito = carregar_gabarito(cfg.gabarito)
    notas, consultas = importar_notas(gabarito, Path(args.arquivo))
    salvar_gabarito(cfg.gabarito, gabarito)
    print(f"{notas} notas de {consultas} consultas juntadas a {cfg.gabarito}.")


def _juiz(cfg: Config) -> JuizGemini:
    j = cfg.juiz
    return JuizGemini(j.api_key, carregar_rubrica(cfg.rubrica), j.modelo, j.preco_entrada_1m, j.preco_saida_1m)


def cmd_julgar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder, cfg.texto_item)
    juiz = _juiz(cfg)
    gabarito = carregar_gabarito(cfg.gabarito)
    if args.lote:
        alvo = [c for c, _ in _consultas_do_lote(gabarito, Path(args.lote))]
    elif args.consulta:
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
    medir = gabarito
    if args.lote:
        chaves = {(c["texto"], json.dumps(c["filtros"], sort_keys=True)) for c in carregar_lote(Path(args.lote))}
        medir = [c for c in gabarito if (c.texto, json.dumps(c.filtros, sort_keys=True)) in chaves]
    rel = concordancia(medir, kappa_minimo=args.kappa_minimo, min_pares=args.min_pares, origem=args.origem)
    print(f"Conjunto: {args.origem}" + (f" · só as consultas de {args.lote}" if args.lote else "") + "\n")
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
    embedder = criar_embedder(cfg.embedder, cfg.texto_item)
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


def cmd_sondar(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder, cfg.texto_item)
    pendentes = arm.itens_sem_embedding(embedder.chave("texto"), "texto", args.categoria)
    if pendentes:
        sys.exit(f"{len(pendentes)} produtos ainda sem vetor de texto para esta variante "
                 f"({embedder.chave('texto')}). Rode `mini-lcm indexar --canal texto` antes.")
    linhas = sondar(arm, embedder, args.termos, _filtros(args), k=args.k)
    texto = sondagem_markdown(linhas, embedder.nome, args.k)
    print(texto)
    if args.salvar:
        Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
        with open(args.salvar, "a", encoding="utf-8") as f:
            tabela = texto.split("\n", 2)[2]  # sem a linha "Variante:", que vira o título
            f.write(f"#### {args.categoria or 'todas as categorias'} · {embedder.nome}\n\n{tabela}\n\n")
        print(f"\nAcrescentado a {args.salvar}")


def _titulo(arm: Armazenamento, item_id: str) -> str:
    item = arm.obter_itens([item_id]).get(item_id)
    return item.titulo[:90] if item else item_id


def cmd_usuarios(cfg: Config, arm: Armazenamento, args) -> None:
    avaliacoes = arm.avaliacoes_de(categoria=args.categoria)
    por_usuario: dict[str, list] = {}
    for a in avaliacoes:
        por_usuario.setdefault(a.usuario, []).append(a)
    ordenados = sorted(por_usuario.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    com_5 = sum(1 for _, avs in ordenados if len(avs) >= 5)
    print(f"{len(ordenados)} usuários, {com_5} com 5 ou mais avaliações em '{args.categoria}'. Os {args.n} com mais:\n")
    for usuario, avs in ordenados[: args.n]:
        media = sum(a.nota for a in avs) / len(avs)
        print(f"  {usuario}  {len(avs):3} avaliações · nota média {media:.1f} · {avs[0].momento[:10]} a {avs[-1].momento[:10]}")


def cmd_perfil(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder, cfg.texto_item)
    historico = arm.avaliacoes_de(categoria=args.categoria, usuario=args.usuario)
    if not historico:
        sys.exit(f"Usuário {args.usuario} sem avaliações em '{args.categoria}'.")
    print(f"Histórico de {args.usuario} em '{args.categoria}' ({len(historico)} avaliações):\n")
    for a in historico:
        print(f"  {a.momento[:10]}  nota {a.nota:.0f}  {_titulo(arm, a.item_id)}")
    catalogo = Catalogo.carregar(arm, embedder, args.categoria)
    par = _parametros(cfg, args.categoria)
    rec = Recomendador(catalogo, par.meia_vida, par.janela_dias, par.peso_em_alta, par.vagas_novos).ajustar(
        arm.avaliacoes_de(categoria=args.categoria))
    print(f"\nRecomendados ({args.metodo}):\n")
    for n, (item_id, _) in enumerate(rec.recomendar(args.metodo, historico, k=args.k), start=1):
        motivo = explicar(catalogo, item_id, historico)
        porque = f"\n      parecido com: {_titulo(arm, motivo)}" if motivo else ""
        print(f"  {n:2}. {_titulo(arm, item_id)}{porque}")


def _parametros(cfg: Config, slug: str) -> ParametrosRecomendacao:
    """Parâmetros de recomendação da categoria; os padrões, se ela não estiver em categorias.yaml."""
    return next((c.recomendacao for c in cfg.categorias if c.slug == slug), ParametrosRecomendacao())


def _meia_vida_comum(cfg: Config) -> float | None:
    """A busca personalizada monta perfis de todas as categorias do usuário de uma vez;
    usa a meia-vida da primeira categoria (hoje todas usam o mesmo valor)."""
    return cfg.categorias[0].recomendacao.meia_vida if cfg.categorias else ParametrosRecomendacao().meia_vida


def cmd_avaliar_recomendacao(cfg: Config, arm: Armazenamento, args) -> None:
    embedder = criar_embedder(cfg.embedder, cfg.texto_item)
    catalogo = Catalogo.carregar(arm, embedder, args.categoria)
    metodos = tuple(m.strip() for m in args.metodos.split(","))
    historicos = (1, 2, 3, 5, None) if args.curva else (None,)
    par = _parametros(cfg, args.categoria)  # valores escolhidos na validação; as opções sobrepõem
    meia_vida = None if args.sem_recencia else (args.meia_vida if args.meia_vida is not None else par.meia_vida)
    rel = avaliar_recomendacao(catalogo, arm.avaliacoes_de(categoria=args.categoria), metodos, k=args.k,
                               min_historico=args.min_historico, historicos=historicos,
                               meia_vida=meia_vida, validacao=args.validacao, corte=args.corte,
                               janela_dias=args.janela_dias or par.janela_dias,
                               peso_em_alta=args.peso_em_alta if args.peso_em_alta is not None else par.peso_em_alta,
                               vagas_novos=args.vagas_novos if args.vagas_novos is not None else par.vagas_novos)
    texto = relatorio_recomendacao_markdown(rel, args.categoria, embedder.nome)
    print(texto)
    if args.salvar:
        Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
        with open(args.salvar, "a", encoding="utf-8") as f:
            conjunto = f" · corte por {args.corte}" + (" · validação" if args.validacao else "")
            f.write(f"## Recomendação · {args.categoria}{conjunto}\n\n{texto}\n\n")
        print(f"\nAcrescentado a {args.salvar}")


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
    p = argparse.ArgumentParser(prog="mini-lcm", description="Motor de personalização em miniatura")
    p.add_argument("-v", "--verboso", action="store_true")
    sub = p.add_subparsers(dest="comando", required=True)

    s = sub.add_parser("inspecionar", help="mostra as categorias de um domínio da Amazon")
    s.add_argument("fonte", help="ex.: Toys_and_Games")
    s.add_argument("--amostra", type=int, default=50_000)
    s.set_defaults(func=cmd_inspecionar)

    s = sub.add_parser("importar", help="importa produtos e avaliações dos arquivos da Amazon")
    s.add_argument("--categoria")
    s.add_argument("--sem-avaliacoes", action="store_true", help="só produtos (só na seleção por produtos)")
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
    s.add_argument("--usuario", help="personaliza a ordem pelo histórico deste usuário")
    s.add_argument("--peso-perfil", type=float, default=0.5, help="quanto o gosto pesa contra a consulta (0 = sem)")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(func=cmd_buscar)

    s = sub.add_parser("rotular", help="passo 1: você dá notas de 0 a 3 aos resultados de uma consulta")
    s.add_argument("consulta", nargs="?")
    _args_filtros(s)
    s.add_argument("--perfil", help='quem busca e com que intenção, ex.: "colecionadora adulta"')
    s.add_argument("--profundidade", type=int, default=20, help="quantos resultados de cada modo entram no pool")
    s.add_argument("--lote", help="YAML com várias consultas (ex.: avaliacao/consultas.yaml); exige --html")
    s.add_argument("--html", help="grava uma página de rotulagem com fotos em vez de perguntar no terminal")
    s.add_argument("--amostra", type=int,
                   help="só N produtos já julgados pela IA: metade ao acaso (calibração), metade estratificada (auditoria)")
    s.set_defaults(func=cmd_rotular)

    s = sub.add_parser("importar-notas", help="junta ao gabarito as notas baixadas da página de rotulagem")
    s.add_argument("arquivo")
    s.set_defaults(func=cmd_importar_notas)

    s = sub.add_parser("julgar", help="passo 2 e escala: a IA dá notas de 0 a 3")
    s.add_argument("consulta", nargs="?", help="sem consulta: julga todas as do gabarito")
    s.add_argument("--lote", help="YAML de consultas (ex.: avaliacao/consultas.yaml): cria e julga todas")
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
    s.add_argument("--lote", help="mede só as consultas deste YAML (ex.: as de validação, que não guiaram a rubrica)")
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

    s = sub.add_parser("sondar", help="teste rápido sem rótulos: buscar nomes e ver se eles aparecem no topo")
    s.add_argument("termos", nargs="+", help='ex.: "monster high" barbie troll')
    _args_filtros(s)
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--salvar", help="acrescenta a tabela a um .md (para comparar variantes)")
    s.set_defaults(func=cmd_sondar)

    s = sub.add_parser("usuarios", help="fase 2: usuários com mais avaliações numa categoria")
    s.add_argument("--categoria", required=True)
    s.add_argument("-n", type=int, default=15)
    s.set_defaults(func=cmd_usuarios)

    s = sub.add_parser("perfil", help="fase 2: histórico de um usuário e o que o sistema recomenda, com o motivo")
    s.add_argument("usuario")
    s.add_argument("--categoria", required=True)
    s.add_argument("--metodo", choices=METODOS_REC, default="hibrido")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(func=cmd_perfil)

    s = sub.add_parser("avaliar-recomendacao", help="fase 2: adivinha a última avaliação de cada pessoa (sem rótulos)")
    s.add_argument("--categoria", required=True)
    s.add_argument("--metodos", default=",".join(METODOS_REC))
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--min-historico", type=int, default=3)
    s.add_argument("--curva", action="store_true", help="também mede com só 1, 2, 3 e 5 avaliações de histórico")
    s.add_argument("--corte", choices=["data", "usuario"], default="data",
                   help="data: cortes semestrais globais, sem vazamento (padrão); usuario: última avaliação de cada pessoa")
    s.add_argument("--meia-vida", type=float, help="decaimento pela recência, em avaliações (padrão: categorias.yaml)")
    s.add_argument("--janela-dias", type=int, help="janela do método em_alta (padrão: categorias.yaml)")
    s.add_argument("--peso-em-alta", type=float, help="peso da lista em_alta no hibrido_alta (padrão: categorias.yaml)")
    s.add_argument("--vagas-novos", type=int,
                   help="posições do top 10 do hibrido_alta reservadas para produtos novos (padrão: categorias.yaml)")
    s.add_argument("--sem-recencia", action="store_true", help="média simples, sem decaimento (linha de base)")
    s.add_argument("--validacao", action="store_true",
                   help="avalia na penúltima avaliação de cada pessoa, para escolher parâmetros sem tocar no teste")
    s.add_argument("--salvar", help="acrescenta o relatório a um .md")
    s.set_defaults(func=cmd_avaliar_recomendacao)

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
