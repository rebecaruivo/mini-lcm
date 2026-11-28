import numpy as np

from mini_lcm.armazenamento import ArmazenamentoLocal
from mini_lcm.indexacao import gerar_embeddings
from mini_lcm.manipulacao import (PREFIXO, ArmazenamentoComIntrusos, consulta_do_titulo, medir, montar_intrusos)
from mini_lcm.modelos import Avaliacao, Filtros


def _indexado(catalogo, tmp_path):
    itens, emb = catalogo
    arm = ArmazenamentoLocal(tmp_path)
    arm.salvar_itens(itens)
    gerar_embeddings(arm, emb)
    # B3 é o mais avaliado das bonecas
    arm.salvar_avaliacoes([Avaliacao(f"u{i}", "B3", 5, f"2020-01-0{i}") for i in range(1, 4)]
                          + [Avaliacao("u9", "B4", 4, "2020-01-01")])
    return arm, emb


def test_alvos_sao_os_mais_avaliados_e_intrusos_copiam_so_a_foto(catalogo, tmp_path):
    arm, emb = _indexado(catalogo, tmp_path)
    alvos, intrusos = montar_intrusos(arm, emb, "bonecas", alvos=1, por_alvo=2)
    assert [a.id for a in alvos] == ["B3"]
    assert len(intrusos) == 2 and all(i.alvo == "B3" and i.doador != "B3" for i in intrusos)
    foto_alvo = arm.vetores(emb.chave("imagem"), "imagem", ["B3"])["B3"]
    for i in intrusos:
        assert i.item.id.startswith(PREFIXO) and i.item.titulo == arm.obter_itens([i.doador])[i.doador].titulo
        assert np.allclose(i.vetores[(emb.chave("imagem"), "imagem")], foto_alvo)
    assert arm.estatisticas()["itens"] == 8  # o catálogo não muda


def test_embrulho_mostra_intrusos_sem_tocar_no_original(catalogo, tmp_path):
    arm, emb = _indexado(catalogo, tmp_path)
    _, intrusos = montar_intrusos(arm, emb, "bonecas", alvos=1, por_alvo=1)
    falso = ArmazenamentoComIntrusos(arm, intrusos)
    vetor = emb.consulta("clawdeen lobo", "imagem")
    ids = [i for i, _ in falso.vizinhos(emb.chave("imagem"), "imagem", vetor, Filtros(categoria="bonecas"), 3)]
    assert intrusos[0].item.id in ids  # a foto copiada fica tão perto quanto a do alvo
    assert intrusos[0].item.id in falso.obter_itens(ids)
    assert not any(i.startswith(PREFIXO) for i, _ in arm.vizinhos(emb.chave("imagem"), "imagem", vetor,
                                                                   Filtros(categoria="bonecas"), 10))
    assert not falso.vizinhos(emb.chave("imagem"), "imagem", vetor, Filtros(categoria="placas"), 3)[0][0].startswith(PREFIXO)


def test_mais_peso_na_imagem_deixa_entrar_mais_intrusos(catalogo, tmp_path):
    arm, emb = _indexado(catalogo, tmp_path)
    linhas = medir(arm, emb, "bonecas", [0.0, 1.0], alvos=1, por_alvo=2, k=3)
    so_texto, so_foto = linhas
    assert so_foto["intrusos"] >= so_texto["intrusos"] and so_foto["intrusos"] > 0
    assert set(so_texto) >= {"com_intruso", "intrusos", "alvo_no_top", "buscas", "peso"}


def test_consulta_do_titulo():
    assert consulta_do_titulo("Monster High Draculaura Doll with Pet Bat and Accessories") == \
        "monster high draculaura doll with pet"
