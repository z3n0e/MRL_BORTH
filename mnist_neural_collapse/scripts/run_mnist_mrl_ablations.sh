#!/usr/bin/env bash
set -euo pipefail

D="${D:-32}"
PREFIX_DIMS="${PREFIX_DIMS:-2,4,8,16,32}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
EVAL_EVERY="${EVAL_EVERY:-2}"
OUT_ROOT="${OUT_ROOT:-outputs/mnist_mrl_ablations}"
RUN_SINGLE="${RUN_SINGLE:-1}"

COMMON_ARGS=(
  --mode mrl
  --feature-dim "$D"
  --prefix-dims "$PREFIX_DIMS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --eval-every "$EVAL_EVERY"
)

if [[ "$RUN_SINGLE" == "1" ]]; then
  IFS=',' read -r -a SINGLE_DIMS <<< "$PREFIX_DIMS"
  for SINGLE_D in "${SINGLE_DIMS[@]}"; do
    python train_mnist_nc.py \
      --mode single \
      --feature-dim "$SINGLE_D" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --eval-every "$EVAL_EVERY" \
      --out-dir "${OUT_ROOT}/single_d${SINGLE_D}"
  done
fi

LOSS_ABLATIONS=(uniform large-heavy small-heavy only-large only-small+big)
for LOSS_WEIGHTS in "${LOSS_ABLATIONS[@]}"; do
  python train_mnist_nc.py \
    "${COMMON_ARGS[@]}" \
    --loss-weights "$LOSS_WEIGHTS" \
    --vicreg none \
    --out-dir "${OUT_ROOT}/${LOSS_WEIGHTS}"
done

VICREG_ABLATIONS=(var-only cov-only var-cov var-cov-cross-cov)
for VICREG in "${VICREG_ABLATIONS[@]}"; do
  python train_mnist_nc.py \
    "${COMMON_ARGS[@]}" \
    --loss-weights uniform \
    --vicreg "$VICREG" \
    --out-dir "${OUT_ROOT}/${VICREG}"
done
