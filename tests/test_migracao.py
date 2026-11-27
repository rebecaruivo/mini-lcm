from mini_lcm.armazenamento import ArmazenamentoLocal
from mini_lcm.migracao import migrar
from mini_lcm.modelos import Avaliacao


def test_migrar_copia_tudo_e_filtra_vetores(indexado, tmp_path):
    origem, emb = indexado
    origem.salvar_avaliacoes([Avaliacao(usuario="u", item_id=next(iter(origem.itens)), nota=5,
                                        momento="2020-01-01T00:00:00+00:00")])
    destino = ArmazenamentoLocal(tmp_path / "destino")
    chave_texto = (emb.chave("texto"), "texto")
    resumo = migrar(origem, destino, chaves={chave_texto}, avisar=lambda _: None, lote=2)
    assert resumo["itens"] == len(origem.itens) and set(destino.itens) == set(origem.itens)
    assert len(destino.avaliacoes) == 1
    assert set(destino.vetores_) == {chave_texto}  # a imagem ficou de fora
    ids, m = destino.matriz(*chave_texto)
    ids0, m0 = origem.matriz(*chave_texto)
    assert sorted(ids) == sorted(ids0) and m.shape == m0.shape
    # idempotente: rodar de novo não duplica
    migrar(origem, destino, chaves={chave_texto}, avisar=lambda _: None)
    assert len(destino.avaliacoes) == 1 and len(destino.itens) == len(origem.itens)


def test_migrar_todas_as_chaves(indexado, tmp_path):
    origem, _ = indexado
    destino = ArmazenamentoLocal(tmp_path / "d2")
    migrar(origem, destino, avisar=lambda _: None)
    assert set(destino.vetores_) == set(origem.vetores_)
