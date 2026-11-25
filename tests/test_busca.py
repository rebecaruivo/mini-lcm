import numpy as np
import pytest

from mini_lcm.armazenamento import ArmazenamentoLocal
from mini_lcm.busca import ConfigBusca, buscar, fundir_rrf
from mini_lcm.embeddings import EmbedderFalso, texto_do_item
from mini_lcm.indexacao import gerar_embeddings
from mini_lcm.modelos import Filtros

from .conftest import item


def ids(resultados):
    return [r.item.id for r in resultados]


def test_rrf_usa_posicao_e_nao_escala():
    # canal A tem similaridades altas, canal B baixas; quem é 1º em B pesa como quem é 1º em A
    listas = {"a": [("x", 0.9), ("y", 0.8)], "b": [("y", 0.3), ("z", 0.2)]}
    pontos = fundir_rrf(listas, {"a": 0.5, "b": 0.5})
    assert pontos["y"] > pontos["x"] > pontos["z"]  # y aparece nas duas listas
    assert pontos["x"] == pytest.approx(pontos["y"] - 0.5 / 62)  # x: só 1º em a; y: 2º em a + 1º em b


def test_foto_salva_anuncio_com_titulo_ruim(indexado):
    arm, emb = indexado
    so_texto = ids(buscar(arm, emb, "draculaura", config=ConfigBusca(modo="texto"), k=10))
    so_imagem = ids(buscar(arm, emb, "draculaura", config=ConfigBusca(modo="imagem"), k=10))
    fusao = ids(buscar(arm, emb, "draculaura", config=ConfigBusca(modo="fusao"), k=10))
    assert so_texto.index("B2") >= 2  # o título "boneca antiga cabelo rosa" não fala em Draculaura
    assert so_imagem.index("B2") <= 1  # mas a foto mostra uma (B1 e B5 são duplicatas: só uma fica)
    assert fusao.index("B2") < so_texto.index("B2")  # e a fusão faz o anúncio subir


def test_filtros_de_preco_condicao_e_categoria(indexado):
    arm, emb = indexado
    r = buscar(arm, emb, "boneca monster high", Filtros(categoria="bonecas", preco_max=200, condicao="usado"), k=10)
    assert r and all(x.item.categoria == "bonecas" and x.item.preco <= 200 and x.item.condicao == "usado" for x in r)
    assert "B3" not in ids(r)  # nova e cara
    placas = buscar(arm, emb, "boneca", Filtros(categoria="placas"), k=10)
    assert {x.item.categoria for x in placas} == {"placas"}


def test_remove_duplicata_do_mesmo_vendedor(indexado):
    arm, emb = indexado
    r = ids(buscar(arm, emb, "draculaura original", k=10))
    assert ("B1" in r) != ("B5" in r)  # só uma das duas republicações aparece


def test_item_sem_foto_aparece_pelo_texto(indexado):
    arm, emb = indexado
    assert "P3" in ids(buscar(arm, emb, "placa de video sem foto", Filtros(categoria="placas"), k=3))


def test_resultado_explica_similaridade_e_posicao_por_canal(indexado):
    arm, emb = indexado
    r = buscar(arm, emb, "draculaura", k=1)[0]
    assert set(r.similaridades) <= {"texto", "imagem"}
    assert all(-1.0 <= s <= 1.0 for s in r.similaridades.values())
    assert all(p >= 1 for p in r.posicoes.values())


def test_busca_registra_custo(indexado):
    from mini_lcm.custo import RegistroCusto

    arm, emb = indexado
    reg = RegistroCusto(etapa="busca", modelo=emb.nome)
    buscar(arm, emb, "x", registro=reg)
    buscar(arm, emb, "y", registro=reg)
    assert reg.itens == 2 and reg.segundos > 0


def test_indexacao_e_incremental_e_ignora_foto_quebrada(catalogo, tmp_path):
    itens, _ = catalogo
    emb = EmbedderFalso(fotos_quebradas={"https://foto/B4.jpg"})
    arm = ArmazenamentoLocal(tmp_path)
    arm.salvar_itens(itens)
    regs = gerar_embeddings(arm, emb)
    assert regs["texto"].itens == 8
    assert regs["imagem"].itens == 7 and regs["imagem"].falhas == 1  # P3 não tem foto; B4 quebrou
    regs2 = gerar_embeddings(arm, emb)
    assert regs2["texto"].itens == 0  # nada novo
    assert regs2["imagem"].itens == 1  # só tenta de novo a foto que falhou


def test_titulo_alterado_gera_vetor_novo(indexado):
    arm, emb = indexado
    antigo = arm.vetores[(emb.nome, "texto")]["B4"].copy()
    arm.salvar_itens([item("B4", "Boneca Barbie Malibu 1990", preco=80, vendedor="v3")])
    assert arm.itens_sem_embedding(emb.nome, "texto") == [arm.itens["B4"]]
    gerar_embeddings(arm, emb)
    assert not np.allclose(antigo, arm.vetores[(emb.nome, "texto")]["B4"])


def test_dados_locais_sobrevivem_a_reabertura(indexado, tmp_path):
    arm, emb = indexado
    antes = ids(buscar(arm, emb, "rtx", k=3))
    reaberto = ArmazenamentoLocal(arm.pasta)
    assert reaberto.estatisticas() == arm.estatisticas()
    assert ids(buscar(reaberto, emb, "rtx", k=3)) == antes
    assert len(reaberto.custos()) == 2  # texto e imagem


def test_texto_do_item_limita_tamanho():
    longo = item("X", "palavra " * 100, Marca="Mattel")
    assert len(texto_do_item(longo).split()) == 40
    assert texto_do_item(item("Y", "Draculaura", Marca="Mattel")) == "Draculaura . Marca: Mattel"
