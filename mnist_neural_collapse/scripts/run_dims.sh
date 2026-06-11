#!/usr/bin/env bash
set -euo pipefail

DIMS=(2 4 8 16 32 64)
for D in "${DIMS[@]}"; do
  python train_mnist_nc.py \
    --mode single \
    --feature-dim "$D" \
    --epochs 80 \
    --batch-size 256 \
    --lr 0.05 \
    --weight-decay 5e-4 \
    --eval-every 2 \
    --out-dir "outputs/single_d${D}"
done
