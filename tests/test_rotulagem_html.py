import json
import re

import pytest

from mini_lcm.avaliacao import ConsultaRotulada, carregar_gabarito
from mini_lcm.rotulagem_html import carregar_lote, gerar_pagina, importar_notas


def _dados_da_pagina(pagina: str) -> list[dict]:
    bruto = re.search(r"const DADOS = (.*?);\nconst CHAVE", pagina, re.S).group(1)
    return json.loads(bruto.replace("<\\/", "</"))


def test_pagina_traz_produtos_com_foto_e_escapa_o_script(indexado):
    arm, _ = indexado
    itens = list(arm.obter_itens(sorted(arm.ids_categoria("bonecas"))).values())[:3]
    itens[0].titulo = "Boneca </script><b>injetada</b>"  # texto do produto nunca pode fechar a tag
    consulta = ConsultaRotulada(id="", texto="boneca gótica", filtros={"categoria": "bonecas"}, perfil="colecionadora")
    pagina = gerar_pagina([(consulta, "visual", itens)], rubrica="3 · Exato <&>")
    assert pagina.count("</script>") == 1  # só o fechamento verdadeiro
    assert "3 · Exato &lt;&amp;&gt;" in pagina
    dados = _dados_da_pagina(pagina)
    assert dados[0]["texto"] == "boneca gótica" and dados[0]["tipo"] == "visual"
    assert [p["id"] for p in dados[0]["produtos"]] == [i.id for i in itens]
    assert dados[0]["produtos"][0]["titulo"].startswith("Boneca </script>")


def test_importar_notas_junta_ao_gabarito(tmp_path):
    gabarito = [ConsultaRotulada(id="c1", texto="rtx 3060 12gb", filtros={"categoria": "placas-de-video"},
                                 notas_humanas={"A": 1})]
    arquivo = tmp_path / "notas.json"
    arquivo.write_text(json.dumps({"versao": 1, "consultas": [
        {"texto": "rtx 3060 12gb", "filtros": {"categoria": "placas-de-video"}, "perfil": "p", "notas": {"A": 3, "B": 0}},
        {"texto": "placa branca", "filtros": {"categoria": "placas-de-video"}, "perfil": "q", "notas": {"C": 2, "D": 7}},
        {"texto": "sem notas", "filtros": {}, "perfil": "", "notas": {}},
    ]}), encoding="utf-8")
    notas, consultas = importar_notas(gabarito, arquivo)
    assert (notas, consultas) == (3, 2)  # a nota 7 é inválida e a consulta sem notas não cria nada
    assert gabarito[0].notas_humanas == {"A": 3, "B": 0}  # a nota nova substitui a antiga
    assert gabarito[1].texto == "placa branca" and gabarito[1].notas_humanas == {"C": 2} and gabarito[1].perfil == "q"


def test_importar_notas_recusa_versao_desconhecida(tmp_path):
    arquivo = tmp_path / "n.json"
    arquivo.write_text(json.dumps({"versao": 99, "consultas": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        importar_notas([], arquivo)


def test_lote_de_consultas_do_repositorio():
    from mini_lcm.config import RAIZ

    consultas = carregar_lote(RAIZ / "avaliacao" / "consultas.yaml")
    assert len(consultas) >= 10
    assert {c["filtros"]["categoria"] for c in consultas} == {"bonecas", "placas-de-video"}
    assert all(c["perfil"] and c["tipo"] for c in consultas)


def test_cli_rotular_em_pagina_e_importar(indexado, tmp_path, monkeypatch, capsys):
    from mini_lcm import cli
    from mini_lcm.config import Config

    arm, _ = indexado
    pasta_aval = tmp_path / "avaliacao"
    pasta_aval.mkdir()
    (pasta_aval / "rubrica.md").write_text("rubrica de teste", encoding="utf-8")
    cfg = Config(database_url=None, embedder="fake", pasta_dados=arm.pasta, pasta_brutos=tmp_path / "b",
                 pasta_avaliacao=pasta_aval)
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    lote = tmp_path / "consultas.yaml"
    lote.write_text("consultas:\n  - texto: monster high\n    tipo: nome\n    filtros: {categoria: bonecas}\n"
                    "    perfil: colecionadora\n", encoding="utf-8")
    pagina = tmp_path / "rotular.html"
    cli.main(["rotular", "--lote", str(lote), "--html", str(pagina), "--profundidade", "3"])
    assert "Página gravada" in capsys.readouterr().out
    produtos = _dados_da_pagina(pagina.read_text(encoding="utf-8"))[0]["produtos"]
    assert produtos
    baixado = tmp_path / "notas-rotulagem.json"
    baixado.write_text(json.dumps({"versao": 1, "consultas": [
        {"texto": "monster high", "filtros": {"categoria": "bonecas"}, "perfil": "colecionadora",
         "notas": {produtos[0]["id"]: 3}}]}), encoding="utf-8")
    cli.main(["importar-notas", str(baixado)])
    assert "1 notas de 1 consultas" in capsys.readouterr().out
    assert carregar_gabarito(cfg.gabarito)[0].notas_humanas == {produtos[0]["id"]: 3}
    # a página seguinte não repete o que já tem nota
    cli.main(["rotular", "--lote", str(lote), "--html", str(pagina), "--profundidade", "3"])
    restantes = _dados_da_pagina(pagina.read_text(encoding="utf-8"))[0]["produtos"]
    assert produtos[0]["id"] not in {p["id"] for p in restantes}


def test_amostra_calibracao_separa_aleatoria_e_estratificada():
    from mini_lcm.calibracao import amostra_calibracao

    notas_ia = {f"i{n}": {"nota": 0 if n < 80 else (1 if n < 90 else (2 if n < 96 else 3))} for n in range(100)}
    c = ConsultaRotulada(id="c1", texto="x", notas_ia=notas_ia, notas_humanas={"i0": 0})
    amostra = amostra_calibracao([c], aleatoria=20, estratificada=12)
    ids = [i for _, i, _ in amostra]
    assert len(ids) == len(set(ids)) == 32 and "i0" not in ids  # sem repetição e sem o que já tem nota sua
    estrat = [i for _, i, aud in amostra if aud]
    contagem = {n: sum(1 for i in estrat if notas_ia[i]["nota"] == n) for n in range(4)}
    assert contagem == {0: 3, 1: 3, 2: 3, 3: 3}  # a auditoria tem todas as notas da IA na mesma quantidade
    assert amostra == amostra_calibracao([c], aleatoria=20, estratificada=12)  # reprodutível


def test_importar_marca_os_auditados(tmp_path):
    arquivo = tmp_path / "n.json"
    arquivo.write_text(json.dumps({"versao": 1, "consultas": [
        {"texto": "x", "filtros": {}, "perfil": "", "notas": {"A": 1, "B": 3}, "auditados": ["B"]}]}), encoding="utf-8")
    gabarito = []
    importar_notas(gabarito, arquivo)
    assert gabarito[0].notas_humanas == {"A": 1, "B": 3} and gabarito[0].auditados == ["B"]


def test_cli_julgar_lote_e_rotular_amostra(indexado, tmp_path, monkeypatch, capsys):
    from mini_lcm import cli
    from mini_lcm.config import Config

    from .test_calibracao import JuizFalso

    arm, _ = indexado
    pasta_aval = tmp_path / "avaliacao"
    pasta_aval.mkdir()
    (pasta_aval / "rubrica.md").write_text("rubrica", encoding="utf-8")
    cfg = Config(database_url=None, embedder="fake", pasta_dados=arm.pasta, pasta_brutos=tmp_path / "b",
                 pasta_avaliacao=pasta_aval)
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    juiz = JuizFalso()
    juiz.versao_rubrica = "v1"
    monkeypatch.setattr(cli, "_juiz", lambda cfg: juiz)
    lote = tmp_path / "consultas.yaml"
    lote.write_text("consultas:\n  - texto: draculaura\n    tipo: nome\n    filtros: {categoria: bonecas}\n"
                    "    perfil: colecionadora\n", encoding="utf-8")
    pagina = tmp_path / "amostra.html"
    with pytest.raises(SystemExit):  # sem julgamento ainda: avisa o que rodar
        cli.main(["rotular", "--lote", str(lote), "--html", str(pagina), "--amostra", "4"])
    cli.main(["julgar", "--lote", str(lote), "--profundidade", "3"])
    assert juiz.chamadas  # julgou o pool da consulta do lote
    cli.main(["rotular", "--lote", str(lote), "--html", str(pagina), "--amostra", "4", "--profundidade", "3"])
    assert "sorteados ao acaso" in capsys.readouterr().out
    produtos = _dados_da_pagina(pagina.read_text(encoding="utf-8"))[0]["produtos"]
    assert 0 < len(produtos) <= 4 and any(p["auditoria"] for p in produtos) and not all(p["auditoria"] for p in produtos)
    cli.main(["rotular", "--lote", str(lote), "--html", str(pagina), "--amostra", "4", "--sem-auditoria"])
    produtos = _dados_da_pagina(pagina.read_text(encoding="utf-8"))[0]["produtos"]
    assert produtos and not any(p["auditoria"] for p in produtos)


def test_cli_concordancia_so_do_lote(tmp_path, monkeypatch, capsys):
    from mini_lcm import cli
    from mini_lcm.avaliacao import salvar_gabarito
    from mini_lcm.config import Config

    pasta_aval = tmp_path / "avaliacao"
    pasta_aval.mkdir()
    cfg = Config(database_url=None, embedder="fake", pasta_dados=tmp_path / "d", pasta_brutos=tmp_path / "b",
                 pasta_avaliacao=pasta_aval)
    monkeypatch.setattr(cli, "carregar_config", lambda: cfg)
    ia = lambda n: {"nota": n, "rubrica": "x", "modelo": "m"}  # noqa: E731
    salvar_gabarito(cfg.gabarito, [
        ConsultaRotulada(id="a", texto="usada", filtros={}, notas_humanas={"1": 3, "2": 0}, notas_ia={"1": ia(3), "2": ia(0)}),
        ConsultaRotulada(id="b", texto="nova", filtros={}, notas_humanas={"3": 3}, notas_ia={"3": ia(0)}),
    ])
    lote = tmp_path / "v.yaml"
    lote.write_text("consultas:\n  - texto: nova\n    perfil: p\n    tipo: nome\n", encoding="utf-8")
    cli.main(["concordancia", "--lote", str(lote), "--min-pares", "1"])
    saida = capsys.readouterr().out
    assert "Pares com nota humana e da IA: 1" in saida and "só as consultas de" in saida
