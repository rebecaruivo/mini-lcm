"""Teste rápido do EmbeddingGemma 2 antes de indexar o catálogo inteiro.

    python scripts/testar_gemma.py

Baixa o modelo (na primeira vez), codifica 3 consultas, os títulos e as fotos de 6
produtos do catálogo e mostra as similaridades e o tempo por item. Serve para conferir
que o modelo carrega, que as fotos são aceitas e quanto tempo a indexação vai levar.
Se as consultas não ficarem mais parecidas com os produtos certos, algo está errado
antes de gastar horas indexando.
"""

from __future__ import annotations

import time

import numpy as np

from mini_lcm.cli import abrir_armazenamento
from mini_lcm.config import carregar_config
from mini_lcm.embeddings import EmbeddingGemma2

CONSULTAS = ["monster high draculaura", "boneca com vestido de noiva", "rtx 3060 12gb"]
PROCURAR = [("bonecas", "draculaura"), ("bonecas", "bride"), ("placas-de-video", "3060")]


def main() -> None:
    cfg = carregar_config()
    arm = abrir_armazenamento(cfg)
    itens = []
    for categoria, termo in PROCURAR:
        todos = arm.obter_itens(sorted(arm.ids_categoria(categoria))).values()
        achados = [i for i in todos if termo in i.titulo.lower() and i.foto_url][:2]
        itens += achados
    if not itens:
        raise SystemExit("Nenhum produto encontrado no catálogo. Rode `mini-lcm importar` antes.")

    t = time.time()
    emb = EmbeddingGemma2("titulo")
    print(f"modelo carregado em {time.time() - t:.0f}s")

    t = time.time()
    textos = emb.textos_itens(itens)
    ms_texto = (time.time() - t) * 1000 / len(itens)
    t = time.time()
    fotos = emb.imagens([i.foto_url for i in itens])
    ms_foto = (time.time() - t) * 1000 / len(itens)
    consultas = np.array([emb.consulta(c, "texto") for c in CONSULTAS])

    print(f"\ntempo por produto: texto {ms_texto:.0f} ms, foto {ms_foto:.0f} ms (com download)")
    print(f"fotos codificadas: {sum(f is not None for f in fotos)} de {len(fotos)}\n")
    print("Similaridade (linhas = consultas; colunas = produtos):")
    for n, i in enumerate(itens):
        print(f"  [{n}] {i.titulo[:70]}")
    for nome, matriz in [("texto (título)", textos), ("foto", np.array([f if f is not None else np.zeros(textos.shape[1]) for f in fotos]))]:
        print(f"\ncanal {nome}:")
        print("  " + " " * 30 + "".join(f"  [{n}] " for n in range(len(itens))))
        for c, v in zip(CONSULTAS, consultas):
            print(f"  {c[:30]:30}" + "".join(f" {s:5.2f}" for s in matriz @ v))
    print("\nEsperado: em cada linha, os maiores valores nas colunas dos produtos daquela consulta "
          "([0][1] Draculaura, [2][3] noiva, [4][5] RTX 3060).")


if __name__ == "__main__":
    main()
