#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail
shopt -s nullglob

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:-${RUN_DIR:-}}

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: $0 /path/to/cifar100_runs/<run_dir>"
    echo
    echo "Example:"
    echo "  $0 cifar100_runs/cifar100_seed_0_20260601_234841"
    exit 2
fi

RUN_DIR=$(cd "$RUN_DIR" && pwd)
CHECKPOINT_DIR="$RUN_DIR/checkpoints"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Missing checkpoint directory: $CHECKPOINT_DIR"
    exit 1
fi

PYTHON=${PYTHON:-python}
CIFAR100_DIR=${CIFAR100_DIR:-"$HOME/.cache/torchvision"}
RETRIEVAL_ROOT=${RETRIEVAL_ROOT:-"$RUN_DIR/retrieval"}
METRICS_DIR=${METRICS_DIR:-"$RUN_DIR/retrieval_metrics"}
SUMMARY_CSV=${SUMMARY_CSV:-"$RUN_DIR/cifar100_retrieval_summary.csv"}

SEED=${SEED:-0}
DETERMINISTIC=${DETERMINISTIC:-1}
EVAL_WORKERS=${EVAL_WORKERS:-4}
INDEX_TYPE=${INDEX_TYPE:-exactl2}
K=${K:-2048}
SHORTLIST=${SHORTLIST:-"1 5 10 25 50 100"}
NESTED_DIMS=${NESTED_DIMS:-"8 16 32 64 128 256 512 1024 2048"}
USE_GPU=${USE_GPU:-1}
REBUILD_INDEX=${REBUILD_INDEX:-0}
FORCE_ARRAYS=${FORCE_ARRAYS:-0}
BOR_USE_TRIVIALIZATION=${BOR_USE_TRIVIALIZATION:-1}

mkdir -p "$RETRIEVAL_ROOT" "$METRICS_DIR"

echo "CIFAR-100 run: $RUN_DIR"
echo "CIFAR-100 data root: $CIFAR100_DIR"
echo "Retrieval root: $RETRIEVAL_ROOT"
echo "Metrics dir: $METRICS_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "Index type: $INDEX_TYPE"
echo "Neighbor shortlist length: $K"
echo "Metric k values: $SHORTLIST"
echo

run_method() {
    local name=$1
    local checkpoint=$2
    local retrieval_model=$3
    local rep_size=$4
    local dims=$5
    local feature_config=$6
    shift 6

    if [[ ! -f "$checkpoint" ]]; then
        echo "Skipping $name: missing checkpoint $checkpoint"
        return
    fi

    local method_root="$RETRIEVAL_ROOT/$name"
    local metrics_json="$METRICS_DIR/${name}.json"
    local train_x="$method_root/CIFAR100_train_${feature_config}-X.npy"
    local val_x="$method_root/CIFAR100_val_${feature_config}-X.npy"

    mkdir -p "$method_root"

    echo "================================================================"
    echo "Retrieval metrics for $name"
    echo "Checkpoint: $checkpoint"
    echo "Feature config: $feature_config"
    echo "Dims: $dims"

    local deterministic_args=()
    if [[ "$DETERMINISTIC" == "1" ]]; then
        deterministic_args=(--deterministic)
    fi

    if [[ "$FORCE_ARRAYS" == "1" || ! -f "$train_x" || ! -f "$val_x" ]]; then
        echo "Dumping retrieval arrays..."
        (
            cd "$ROOT_DIR/inference"
            "$PYTHON" pytorch_inference.py \
                --retrieval \
                --path "$checkpoint" \
                --dataset CIFAR100 \
                --data_root "$CIFAR100_DIR" \
                --retrieval_array_path "$method_root" \
                --workers "$EVAL_WORKERS" \
                --seed "$SEED" \
                "${deterministic_args[@]}" \
                "$@"
        )
    else
        echo "Using existing retrieval arrays in $method_root"
    fi

    local faiss_args=(
        faiss_nn.py
        --root "$method_root"
        --dataset CIFAR100
        --model "$retrieval_model"
        --feature-config "$feature_config"
        --rep-size "$rep_size"
        --index-type "$INDEX_TYPE"
        --k "$K"
        --dims
    )
    local dim
    for dim in $dims; do
        faiss_args+=("$dim")
    done
    if [[ "$USE_GPU" == "1" ]]; then
        faiss_args+=(--gpu)
    fi
    if [[ "$REBUILD_INDEX" == "1" ]]; then
        faiss_args+=(--rebuild-index)
    fi

    echo "Building/searching FAISS neighbors..."
    (
        cd "$ROOT_DIR/retrieval"
        "$PYTHON" "${faiss_args[@]}"
    )

    local metric_args=(
        compute_metrics.py
        --root "$method_root"
        --dataset CIFAR100
        --model "$retrieval_model"
        --feature-config "$feature_config"
        --rep-size "$rep_size"
        --eval-config vanilla
        --index-type "$INDEX_TYPE"
        --neighbor-k "$K"
        --output-json "$metrics_json"
        --dims
    )
    for dim in $dims; do
        metric_args+=("$dim")
    done
    metric_args+=(--shortlist)
    local shortlist_k
    for shortlist_k in $SHORTLIST; do
        metric_args+=("$shortlist_k")
    done

    echo "Computing metrics..."
    (
        cd "$ROOT_DIR/retrieval"
        "$PYTHON" "${metric_args[@]}"
    )
}

run_method mrl \
    "$CHECKPOINT_DIR/mrl_final_weights.pt" \
    mrl 2048 "$NESTED_DIMS" mrl1_e0_ff2048 \
    --mrl

run_method mrle \
    "$CHECKPOINT_DIR/mrle_final_weights.pt" \
    mrl_e 2048 "$NESTED_DIMS" mrl1_e1_ff2048 \
    --mrl --efficient

run_method full_feature \
    "$CHECKPOINT_DIR/full_feature_final_weights.pt" \
    ff 2048 "2048" mrl0_e0_ff2048 \
    --rep_size 2048

for fixed_checkpoint in "$CHECKPOINT_DIR"/fixed_*_final_weights.pt; do
    fixed_name=$(basename "$fixed_checkpoint" _final_weights.pt)
    fixed_dim=${fixed_name#fixed_}
    run_method "$fixed_name" \
        "$fixed_checkpoint" \
        ff "$fixed_dim" "$fixed_dim" "mrl0_e0_ff${fixed_dim}" \
        --rep_size "$fixed_dim"
done

run_method bor_mrl \
    "$CHECKPOINT_DIR/bor_mrl_final_weights.pt" \
    bor_mrl 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_mrl \
    --bor_mode orthogonal \
    --bor_orthogonal_map matrix_exp \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION"

run_method bor_block_mrl \
    "$CHECKPOINT_DIR/bor_block_mrl_final_weights.pt" \
    bor_block_mrl 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_block_mrl \
    --bor_mode orthogonal \
    --bor_orthogonal_map matrix_exp \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION"

run_method bor_mrl_cayley \
    "$CHECKPOINT_DIR/bor_mrl_cayley_final_weights.pt" \
    bor_mrl_cayley 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_mrl \
    --bor_mode orthogonal \
    --bor_orthogonal_map cayley \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION"

run_method bor_mrl_householder \
    "$CHECKPOINT_DIR/bor_mrl_householder_final_weights.pt" \
    bor_mrl_householder 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_mrl \
    --bor_mode orthogonal \
    --bor_orthogonal_map householder \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION"

run_method bor_mrl_frozen \
    "$CHECKPOINT_DIR/bor_mrl_frozen_final_weights.pt" \
    bor_mrl_frozen 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_mrl \
    --bor_mode frozen

"$PYTHON" - "$METRICS_DIR" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

metrics_dir = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
rows = []

for path in sorted(metrics_dir.glob("*.json")):
    with open(path) as handle:
        data = json.load(handle)
    for metric in data.get("metrics", []):
        rows.append({
            "method": path.stem,
            "model": data.get("model", ""),
            "feature_config": data.get("feature_config", ""),
            "eval_config": data.get("eval_config", ""),
            "index_type": data.get("index_type", ""),
            "dim": metric.get("dim", ""),
            "k": metric.get("k", ""),
            "top1": metric.get("top1", ""),
            "mAP": metric.get("mAP", ""),
            "precision": metric.get("precision", ""),
            "recall": metric.get("recall", ""),
            "topk": metric.get("topk", ""),
            "neighbors_path": metric.get("neighbors_path", ""),
        })

fieldnames = [
    "method", "model", "feature_config", "eval_config", "index_type",
    "dim", "k", "top1", "mAP", "precision", "recall", "topk",
    "neighbors_path",
]

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with open(summary_csv, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {summary_csv}")
PY

echo
echo "Done."
echo "Per-method JSON metrics: $METRICS_DIR"
echo "CSV summary: $SUMMARY_CSV"
echo "Open cifar100_retrieval_results.ipynb and set EXPERIMENT_DIR to visualize this run."
