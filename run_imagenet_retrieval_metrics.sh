#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:-${RUN_DIR:-}}

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: $0 /path/to/imagenet_runs/<run_dir>"
    echo "Example: IMAGENET_DIR=/path/to/imagenet $0 imagenet_runs/imagenet_seed_0_..."
    exit 2
fi

IMAGENET_DIR=${IMAGENET_DIR:-}
if [[ -z "$IMAGENET_DIR" ]]; then
    echo "Set IMAGENET_DIR to the ImageNet root containing train/ and val/."
    exit 2
fi

if [[ ! -d "$IMAGENET_DIR/train" || ! -d "$IMAGENET_DIR/val" ]]; then
    echo "Expected ImageNet folders at:"
    echo "  $IMAGENET_DIR/train"
    echo "  $IMAGENET_DIR/val"
    exit 1
fi

RUN_DIR=$(cd "$RUN_DIR" && pwd)
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
MANIFEST="$RUN_DIR/manifest.txt"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Missing checkpoint directory: $CHECKPOINT_DIR"
    exit 1
fi

TRAINING_MRL_LOSS_MODE=${MRL_LOSS_MODE:-}
TRAINING_SAMPLED_PREFIX_DISTRIBUTION=${SAMPLED_PREFIX_DISTRIBUTION:-}
TRAINING_SAMPLED_PREFIX_LOG_INTERVAL=${SAMPLED_PREFIX_LOG_INTERVAL:-}
TRAINING_PREFIX_MASK_PROB=${PREFIX_MASK_PROB:-}
TRAINING_PREFIX_MASK_SCALE=${PREFIX_MASK_SCALE:-}

if [[ -f "$MANIFEST" ]]; then
    while IFS='=' read -r key value; do
        case "$key" in
            mrl_loss_mode)
                [[ -z "$TRAINING_MRL_LOSS_MODE" ]] && TRAINING_MRL_LOSS_MODE=$value
                ;;
            sampled_prefix_distribution)
                [[ -z "$TRAINING_SAMPLED_PREFIX_DISTRIBUTION" ]] && TRAINING_SAMPLED_PREFIX_DISTRIBUTION=$value
                ;;
            sampled_prefix_log_interval)
                [[ -z "$TRAINING_SAMPLED_PREFIX_LOG_INTERVAL" ]] && TRAINING_SAMPLED_PREFIX_LOG_INTERVAL=$value
                ;;
            prefix_mask_prob)
                [[ -z "$TRAINING_PREFIX_MASK_PROB" ]] && TRAINING_PREFIX_MASK_PROB=$value
                ;;
            prefix_mask_scale)
                [[ -z "$TRAINING_PREFIX_MASK_SCALE" ]] && TRAINING_PREFIX_MASK_SCALE=$value
                ;;
        esac
    done < "$MANIFEST"
fi

TRAINING_MRL_LOSS_MODE=${TRAINING_MRL_LOSS_MODE:-all}
TRAINING_SAMPLED_PREFIX_DISTRIBUTION=${TRAINING_SAMPLED_PREFIX_DISTRIBUTION:-uniform}
TRAINING_SAMPLED_PREFIX_LOG_INTERVAL=${TRAINING_SAMPLED_PREFIX_LOG_INTERVAL:-100}
TRAINING_PREFIX_MASK_PROB=${TRAINING_PREFIX_MASK_PROB:-0.0}
TRAINING_PREFIX_MASK_SCALE=${TRAINING_PREFIX_MASK_SCALE:-inverted}

PYTHON=${PYTHON:-python}
RETRIEVAL_ROOT=${RETRIEVAL_ROOT:-"$RUN_DIR/retrieval"}
METRICS_DIR=${METRICS_DIR:-"$RUN_DIR/retrieval_metrics"}
SUMMARY_CSV=${SUMMARY_CSV:-"$RUN_DIR/imagenet_retrieval_summary.csv"}

SEED=${SEED:-0}
DETERMINISTIC=${DETERMINISTIC:-1}
EVAL_WORKERS=${EVAL_WORKERS:-12}
INDEX_TYPE=${INDEX_TYPE:-exactl2}
K=${K:-2048}
SHORTLIST=${SHORTLIST:-"10 25 50 100"}
NESTED_DIMS=${NESTED_DIMS:-"8 16 32 64 128 256 512 1024 2048"}
USE_GPU=${USE_GPU:-1}
REBUILD_INDEX=${REBUILD_INDEX:-0}
FORCE_ARRAYS=${FORCE_ARRAYS:-0}

mkdir -p "$RETRIEVAL_ROOT" "$METRICS_DIR"

echo "ImageNet run: $RUN_DIR"
echo "ImageNet data root: $IMAGENET_DIR"
echo "Retrieval root: $RETRIEVAL_ROOT"
echo "Metrics dir: $METRICS_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "Index type: $INDEX_TYPE"
echo "Neighbor shortlist length: $K"
echo "Metric k values: $SHORTLIST"
echo "Training MRL loss mode: $TRAINING_MRL_LOSS_MODE"
echo "Training prefix mask: p=$TRAINING_PREFIX_MASK_PROB scale=$TRAINING_PREFIX_MASK_SCALE"
echo

run_mrl() {
    local checkpoint="$CHECKPOINT_DIR/mrl_final_weights.pt"
    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing MRL checkpoint: $checkpoint"
        exit 1
    fi

    local method_root="$RETRIEVAL_ROOT/mrl"
    local metrics_json="$METRICS_DIR/mrl.json"
    local train_x="$method_root/1K_train_mrl1_e0_ff2048-X.npy"
    local val_x="$method_root/1K_val_mrl1_e0_ff2048-X.npy"

    mkdir -p "$method_root"

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
                --dataset 1K \
                --data_root "$IMAGENET_DIR" \
                --retrieval_array_path "$method_root" \
                --workers "$EVAL_WORKERS" \
                --seed "$SEED" \
                "${deterministic_args[@]}" \
                --mrl
        )
    else
        echo "Using existing retrieval arrays in $method_root"
    fi

    local faiss_args=(
        faiss_nn.py
        --root "$method_root"
        --dataset 1K
        --model mrl
        --feature-config mrl1_e0_ff2048
        --rep-size 2048
        --index-type "$INDEX_TYPE"
        --k "$K"
        --dims
    )
    local dim
    for dim in $NESTED_DIMS; do
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
        --dataset 1K
        --model mrl
        --feature-config mrl1_e0_ff2048
        --rep-size 2048
        --eval-config vanilla
        --index-type "$INDEX_TYPE"
        --neighbor-k "$K"
        --output-json "$metrics_json"
        --dims
    )
    for dim in $NESTED_DIMS; do
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

run_mrl

"$PYTHON" - "$METRICS_DIR" "$SUMMARY_CSV" \
    "$TRAINING_MRL_LOSS_MODE" \
    "$TRAINING_SAMPLED_PREFIX_DISTRIBUTION" \
    "$TRAINING_SAMPLED_PREFIX_LOG_INTERVAL" \
    "$TRAINING_PREFIX_MASK_PROB" \
    "$TRAINING_PREFIX_MASK_SCALE" <<'PY'
import csv
import json
import sys
from pathlib import Path

metrics_dir = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
training_mrl_loss_mode = sys.argv[3]
training_sampled_prefix_distribution = sys.argv[4]
training_sampled_prefix_log_interval = sys.argv[5]
training_prefix_mask_prob = sys.argv[6]
training_prefix_mask_scale = sys.argv[7]
rows = []

for path in sorted(metrics_dir.glob("*.json")):
    with open(path) as handle:
        data = json.load(handle)
    for metric in data.get("metrics", []):
        rows.append({
            "method": path.stem,
            "model": data.get("model", ""),
            "training_mrl_loss_mode": training_mrl_loss_mode,
            "training_sampled_prefix_distribution": training_sampled_prefix_distribution,
            "training_sampled_prefix_log_interval": training_sampled_prefix_log_interval,
            "training_prefix_mask_prob": training_prefix_mask_prob,
            "training_prefix_mask_scale": training_prefix_mask_scale,
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
    "method", "model", "training_mrl_loss_mode",
    "training_sampled_prefix_distribution",
    "training_sampled_prefix_log_interval",
    "training_prefix_mask_prob",
    "training_prefix_mask_scale",
    "feature_config", "eval_config", "index_type",
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
