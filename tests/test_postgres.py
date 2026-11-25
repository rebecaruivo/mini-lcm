"""Testes do armazenamento Postgres. Só rodam com um banco de TESTE configurado:

    DATABASE_URL_TESTE=postgresql://... pytest tests/test_postgres.py

ATENÇÃO: o teste apaga e recria as tabelas. Nunca aponte para o banco de verdade.
"""

import os
from pathlib import Path

import pytest

from mini_lcm.busca import ConfigBusca, buscar
from mini_lcm.indexacao import gerar_embeddings
from mini_lcm.modelos import Filtros

URL = os.getenv("DATABASE_URL_TESTE")
psycopg = pytest.importorskip("psycopg") if URL else None
pytestmark = pytest.mark.skipif(not URL, reason="defina DATABASE_URL_TESTE para testar com Postgres")

SCHEMA = (Path(__file__).parent.parent / "sql" / "001_schema.sql").read_text()


@pytest.fixture
def pg(catalogo):
    from mini_lcm.armazenamento import ArmazenamentoPostgres

    itens, emb = catalogo
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute("drop table if exists eventos, usuarios, custos, avaliacoes, embeddings_itens, itens, categorias cascade")
        c.execute(SCHEMA)
    arm = ArmazenamentoPostgres(URL, categorias=[("bonecas", "Bonecas"), ("placas", "Placas")])
    arm.salvar_itens(itens)
    gerar_embeddings(arm, emb)
    yield arm, emb
    arm.fechar()


def test_postgres_busca_igual_ao_local(pg, indexado):
    arm_pg, emb = pg
    arm_local, _ = indexado
    for consulta, filtros in [("draculaura", Filtros()), ("rtx", Filtros(categoria="placas", preco_max=1500))]:
        for modo in ("texto", "imagem", "fusao"):
            cfg = ConfigBusca(modo=modo)
            a = [r.item.id for r in buscar(arm_pg, emb, consulta, filtros, k=5, config=cfg)]
            b = [r.item.id for r in buscar(arm_local, emb, consulta, filtros, k=5, config=cfg)]
            assert a == b, (consulta, modo)


def test_postgres_estatisticas_e_custos(pg):
    arm, emb = pg
    est = arm.estatisticas()
    assert est["itens"] == 8
    assert est["vetores"][f"{emb.nome}/imagem"] == 7
    assert {c.etapa for c in arm.custos()} == {"embeddings:texto", "embeddings:imagem"}


def test_postgres_avaliacoes_sem_repeticao(pg):
    from mini_lcm.modelos import Avaliacao

    arm, _ = pg
    a = Avaliacao(usuario="U1", item_id="B1", nota=5, momento="2020-05-05T14:08:48+00:00")
    assert arm.salvar_avaliacoes([a]) == 1
    assert arm.salvar_avaliacoes([a]) == 0
    assert arm.estatisticas()["usuarios"] == 1


def test_postgres_eventos_sao_append_only(pg):
    arm, _ = pg
    uid = arm.conn.execute("insert into usuarios (nome, tipo) values ('p1', 'persona') returning id").fetchone()[0]
    arm.conn.execute("insert into eventos (usuario_id, tipo, consulta) values (%s, 'busca', 'x')", (uid,))
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        arm.conn.execute("delete from eventos")

