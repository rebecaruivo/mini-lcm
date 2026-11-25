from pathlib import Path

import pytest

from mini_lcm.config import carregar_categorias


def test_categorias_yaml_do_projeto_e_valido():
    cats = carregar_categorias(Path(__file__).parent.parent / "categorias.yaml")
    assert {c.slug for c in cats} >= {"bonecas", "placas-de-video"}
    assert all(c.max_itens > 0 and c.fonte_amazon for c in cats)


def test_slug_repetido_e_recusado(tmp_path):
    arq = tmp_path / "c.yaml"
    arq.write_text("categorias:\n"
                   "  - {slug: a, nome: A, fonte_amazon: X, filtro: {titulo_contem: [x]}}\n"
                   "  - {slug: a, nome: B, fonte_amazon: X, filtro: {titulo_contem: [y]}}\n")
    with pytest.raises(ValueError, match="repetidos"):
        carregar_categorias(arq)


def test_categoria_sem_filtro_e_recusada(tmp_path):
    arq = tmp_path / "c.yaml"
    arq.write_text("categorias:\n  - {slug: a, nome: A, fonte_amazon: X}\n")
    with pytest.raises(ValueError, match="sem filtro"):
        carregar_categorias(arq)
