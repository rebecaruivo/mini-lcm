#!/usr/bin/env bash
# Baixa os arquivos do Amazon Reviews 2023 usados em categorias.yaml para dados/brutos/.
#
#   bash scripts/baixar_amazon.sh                       # domínios padrão
#   bash scripts/baixar_amazon.sh Toys_and_Games         # só um domínio
#
# Tamanho aproximado: são arquivos de vários GB (Electronics é o maior).
# Retoma downloads interrompidos (curl -C -).
set -euo pipefail

BASE="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
DESTINO="$(dirname "$0")/../dados/brutos"
mkdir -p "$DESTINO"

DOMINIOS=("$@")
if [ ${#DOMINIOS[@]} -eq 0 ]; then
  DOMINIOS=(Toys_and_Games Electronics)
fi

for d in "${DOMINIOS[@]}"; do
  echo "== $d: metadados dos produtos"
  curl -L -C - -o "$DESTINO/meta_$d.jsonl.gz" "$BASE/meta_categories/meta_$d.jsonl.gz"
  echo "== $d: avaliações"
  curl -L -C - -o "$DESTINO/$d.jsonl.gz" "$BASE/review_categories/$d.jsonl.gz"
done

echo "Pronto. Arquivos em dados/brutos/ (fora do git)."
