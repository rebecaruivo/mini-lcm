from mini_lcm.graficos import agrupar, avisos, ler_grades, texto_terminal

SAIDA = """### EMBEDDER=gemma2 · consultas=consultas-escolha
Loading weights: 100%|████| 624/624
Grade do peso da imagem na fusão · notas: ia · consultas de avaliacao/consultas-escolha.yaml

| Categoria | Consultas | Peso da imagem | NDCG@10 | recall@10 | MRR |
|---|---|---|---|---|---|
| bonecas | 6 | 0 | 0.594 | 0.203 | 0.736 |
| bonecas | 6 | 1 ← maior NDCG | 0.809 | 0.393 | 0.917 |
Atenção: o juiz não passou no critério de kappa.
### EMBEDDER=clip+gemma2 · consultas=consultas-escolha
Grade do peso da imagem na fusão · notas: ia · consultas de avaliacao/consultas-escolha.yaml

| Categoria | Consultas | Peso da imagem | NDCG@10 | recall@10 | MRR |
|---|---|---|---|---|---|
| bonecas | 6 | 0 | 0.594 | 0.203 | 0.736 |
| bonecas | 6 | 1 | 0.434 | 0.156 | 0.400 |
### EMBEDDER=gemma2 · consultas=consultas-conferencia
Grade do peso da imagem na fusão · notas: ia · consultas de avaliacao/consultas-conferencia.yaml

| Categoria | Consultas | Peso da imagem | NDCG@10 | recall@10 | MRR |
|---|---|---|---|---|---|
| bonecas | 2 | 0 | 0.504 | 0.522 | 0.375 |
"""


def test_le_blocos_com_modelo_consultas_e_fonte():
    blocos = ler_grades(SAIDA)
    assert [(b.modelo, b.consultas, b.fonte) for b in blocos] == [
        ("gemma2", "escolha", "ia"), ("clip+gemma2", "escolha", "ia"), ("gemma2", "conferência", "ia")]
    assert blocos[0].linhas[1] == ("bonecas", 6, 1.0, 0.809, 0.393, 0.917)  # tira a marca "← maior NDCG"


def test_agrupa_por_categoria_e_consultas_com_escolha_primeiro():
    grupos = agrupar(ler_grades(SAIDA))
    assert list(grupos) == [("bonecas", "escolha"), ("bonecas", "conferência")]
    assert grupos[("bonecas", "escolha")]["clip+gemma2"] == {0.0: 0.594, 1.0: 0.434}
    assert agrupar(ler_grades(SAIDA), "mrr")[("bonecas", "escolha")]["gemma2"][1.0] == 0.917


def test_terminal_marca_o_melhor_de_cada_modelo():
    texto = texto_terminal(ler_grades(SAIDA))
    assert "Gemma sozinho" in texto and "CLIP + Gemma" in texto
    linha_peso_1 = next(l for l in texto.splitlines() if l.startswith("1 "))
    assert "0.809 ←" in linha_peso_1 and "0.434 ←" not in linha_peso_1


def test_avisa_fontes_diferentes_e_peso_zero_divergente():
    misturado = SAIDA.replace("notas: ia · consultas de avaliacao/consultas-escolha.yaml\n\n| Categoria | Consultas | Peso da imagem"
                              " | NDCG@10 | recall@10 | MRR |\n|---|---|---|---|---|---|\n| bonecas | 6 | 0 | 0.594 | 0.203"
                              " | 0.736 |\n| bonecas | 6 | 1 | 0.434",
                              "notas: combinado · consultas de avaliacao/consultas-escolha.yaml\n\n| Categoria | Consultas | Peso da"
                              " imagem | NDCG@10 | recall@10 | MRR |\n|---|---|---|---|---|---|\n| bonecas | 6 | 0 | 0.690 |"
                              " 0.203 | 0.736 |\n| bonecas | 6 | 1 | 0.434")
    texto = " ".join(avisos(ler_grades(misturado)))
    assert "fontes de notas diferentes" in texto
    assert "peso 0" in texto
    assert avisos(ler_grades(SAIDA)) == []


def test_bloco_sem_cabecalho_vira_desconhecido():
    so_tabela = "| bonecas | 6 | 0 | 0.594 | 0.203 | 0.736 |\n"
    blocos = ler_grades(so_tabela)
    assert blocos[0].modelo == "desconhecido 1" and blocos[0].consultas == "?"
