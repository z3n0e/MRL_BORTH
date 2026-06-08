#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

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
MANIFEST="$RUN_DIR/manifest.txt"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Missing checkpoint directory: $CHECKPOINT_DIR"
    exit 1
fi

TRAINING_MRL_LOSS_MODE=${MRL_LOSS_MODE:-}
TRAINING_SAMPLED_PREFIX_DISTRIBUTION=${SAMPLED_PREFIX_DISTRIBUTION:-}
TRAINING_SAMPLED_PREFIX_LOG_INTERVAL=${SAMPLED_PREFIX_LOG_INTERVAL:-}
TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL=${MRL_GRADIENT_CONFLICT_INTERVAL:-}
TRAINING_RESIDUAL_ALIGNMENT_LOG_INTERVAL=${RESIDUAL_ALIGNMENT_LOG_INTERVAL:-}
TRAINING_MRL_CONFLICT_GATING=${MRL_CONFLICT_GATING:-}
TRAINING_MRL_CONFLICT_MODE=${MRL_CONFLICT_MODE:-}
TRAINING_MRL_CONFLICT_ALPHA=${MRL_CONFLICT_ALPHA:-}
TRAINING_MRL_CONFLICT_EPS=${MRL_CONFLICT_EPS:-}
TRAINING_RESIDUAL_ALIGN_MODE=${RESIDUAL_ALIGN_MODE:-}
TRAINING_RESIDUAL_ALIGN_ORTHOGONAL_MAP=${RESIDUAL_ALIGN_ORTHOGONAL_MAP:-}
TRAINING_RESIDUAL_ALIGN_USE_TRIVIALIZATION=${RESIDUAL_ALIGN_USE_TRIVIALIZATION:-}
TRAINING_RESIDUAL_ALIGN_MSE_WEIGHT=${RESIDUAL_ALIGN_MSE_WEIGHT:-}
TRAINING_RESIDUAL_ALIGN_COSINE_WEIGHT=${RESIDUAL_ALIGN_COSINE_WEIGHT:-}
TRAINING_RESIDUAL_ALIGN_DETACH_PREFIX_TARGET=${RESIDUAL_ALIGN_DETACH_PREFIX_TARGET:-}
TRAINING_RESIDUAL_INTERPOLATION_ALPHA=${RESIDUAL_INTERPOLATION_ALPHA:-}

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
            mrl_gradient_conflict_interval)
                [[ -z "$TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL" ]] && TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL=$value
                ;;
            residual_alignment_log_interval)
                [[ -z "$TRAINING_RESIDUAL_ALIGNMENT_LOG_INTERVAL" ]] && TRAINING_RESIDUAL_ALIGNMENT_LOG_INTERVAL=$value
                ;;
            mrl_conflict_gating)
                [[ -z "$TRAINING_MRL_CONFLICT_GATING" ]] && TRAINING_MRL_CONFLICT_GATING=$value
                ;;
            mrl_conflict_mode)
                [[ -z "$TRAINING_MRL_CONFLICT_MODE" ]] && TRAINING_MRL_CONFLICT_MODE=$value
                ;;
            mrl_conflict_alpha)
                [[ -z "$TRAINING_MRL_CONFLICT_ALPHA" ]] && TRAINING_MRL_CONFLICT_ALPHA=$value
                ;;
            mrl_conflict_eps)
                [[ -z "$TRAINING_MRL_CONFLICT_EPS" ]] && TRAINING_MRL_CONFLICT_EPS=$value
                ;;
            residual_align_mode)
                [[ -z "$TRAINING_RESIDUAL_ALIGN_MODE" ]] && TRAINING_RESIDUAL_ALIGN_MODE=$value
                ;;
            residual_align_orthogonal_map)
                [[ -z "$TRAINING_RESIDUAL_ALIGN_ORTHOGONAL_MAP" ]] && TRAINING_RESIDUAL_ALIGN_ORTHOGONAL_MAP=$value
                ;;
            residual_align_use_trivialization)
                [[ -z "$TRAINING_RESIDUAL_ALIGN_USE_TRIVIALIZATION" ]] && TRAINING_RESIDUAL_ALIGN_USE_TRIVIALIZATION=$value
                ;;
            residual_align_mse_weight)
                [[ -z "$TRAINING_RESIDUAL_ALIGN_MSE_WEIGHT" ]] && TRAINING_RESIDUAL_ALIGN_MSE_WEIGHT=$value
                ;;
            residual_align_cosine_weight)
                [[ -z "$TRAINING_RESIDUAL_ALIGN_COSINE_WEIGHT" ]] && TRAINING_RESIDUAL_ALIGN_COSINE_WEIGHT=$value
                ;;
            residual_align_detach_prefix_target)
                [[ -z "$TRAINING_RESIDUAL_ALIGN_DETACH_PREFIX_TARGET" ]] && TRAINING_RESIDUAL_ALIGN_DETACH_PREFIX_TARGET=$value
                ;;
            residual_interpolation_alpha)
                [[ -z "$TRAINING_RESIDUAL_INTERPOLATION_ALPHA" ]] && TRAINING_RESIDUAL_INTERPOLATION_ALPHA=$value
                ;;
        esac
    done < "$MANIFEST"
fi

TRAINING_MRL_LOSS_MODE=${TRAINING_MRL_LOSS_MODE:-all}
TRAINING_SAMPLED_PREFIX_DISTRIBUTION=${TRAINING_SAMPLED_PREFIX_DISTRIBUTION:-uniform}
TRAINING_SAMPLED_PREFIX_LOG_INTERVAL=${TRAINING_SAMPLED_PREFIX_LOG_INTERVAL:-100}
TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL=${TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL:-0}
TRAINING_RESIDUAL_ALIGNMENT_LOG_INTERVAL=${TRAINING_RESIDUAL_ALIGNMENT_LOG_INTERVAL:-100}
TRAINING_MRL_CONFLICT_GATING=${TRAINING_MRL_CONFLICT_GATING:-0}
TRAINING_MRL_CONFLICT_MODE=${TRAINING_MRL_CONFLICT_MODE:-none}
TRAINING_MRL_CONFLICT_ALPHA=${TRAINING_MRL_CONFLICT_ALPHA:-0.5}
TRAINING_MRL_CONFLICT_EPS=${TRAINING_MRL_CONFLICT_EPS:-1e-8}
TRAINING_RESIDUAL_ALIGN_MODE=${TRAINING_RESIDUAL_ALIGN_MODE:-orthogonal}
TRAINING_RESIDUAL_ALIGN_ORTHOGONAL_MAP=${TRAINING_RESIDUAL_ALIGN_ORTHOGONAL_MAP:-matrix_exp}
TRAINING_RESIDUAL_ALIGN_USE_TRIVIALIZATION=${TRAINING_RESIDUAL_ALIGN_USE_TRIVIALIZATION:-1}
TRAINING_RESIDUAL_ALIGN_MSE_WEIGHT=${TRAINING_RESIDUAL_ALIGN_MSE_WEIGHT:-10.0}
TRAINING_RESIDUAL_ALIGN_COSINE_WEIGHT=${TRAINING_RESIDUAL_ALIGN_COSINE_WEIGHT:-10.0}
TRAINING_RESIDUAL_ALIGN_DETACH_PREFIX_TARGET=${TRAINING_RESIDUAL_ALIGN_DETACH_PREFIX_TARGET:-1}
TRAINING_RESIDUAL_INTERPOLATION_ALPHA=${TRAINING_RESIDUAL_INTERPOLATION_ALPHA:-0.5}

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

mkdir -p "$RETRIEVAL_ROOT" "$METRICS_DIR"

echo "CIFAR-100 run: $RUN_DIR"
echo "CIFAR-100 data root: $CIFAR100_DIR"
echo "Retrieval root: $RETRIEVAL_ROOT"
echo "Metrics dir: $METRICS_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "Index type: $INDEX_TYPE"
echo "Neighbor shortlist length: $K"
echo "Metric k values: $SHORTLIST"
echo "Training MRL loss mode: $TRAINING_MRL_LOSS_MODE"
echo "Training residual alignment log interval: $TRAINING_RESIDUAL_ALIGNMENT_LOG_INTERVAL"
echo "Residual-Aligned MRL orthogonal map: $TRAINING_RESIDUAL_ALIGN_ORTHOGONAL_MAP"
echo "Residual interpolation alpha: $TRAINING_RESIDUAL_INTERPOLATION_ALPHA"
echo

run_method() {
    local name=$1
    local checkpoint=$2
    local retrieval_model=$3
    local rep_size=$4
    local dims=$5
    local feature_config=$6
    local residual_alpha=$7
    shift 7

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
    echo "Residual interpolation alpha: $residual_alpha"

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
        --residual-interpolate-alpha "$residual_alpha"
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
        --residual-interpolate-alpha "$residual_alpha"
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
    mrl 2048 "$NESTED_DIMS" mrl1_e0_ff2048 0.0 \
    --mrl

run_method residual_aligned_mrl \
    "$CHECKPOINT_DIR/residual_aligned_mrl_final_weights.pt" \
    residual_aligned_mrl 2048 "$NESTED_DIMS" mrl0_e0_ff2048 "$TRAINING_RESIDUAL_INTERPOLATION_ALPHA" \
    --residual_aligned_mrl \
    --residual_align_mode "$TRAINING_RESIDUAL_ALIGN_MODE" \
    --residual_align_orthogonal_map "$TRAINING_RESIDUAL_ALIGN_ORTHOGONAL_MAP" \
    --residual_align_use_trivialization "$TRAINING_RESIDUAL_ALIGN_USE_TRIVIALIZATION" \
    --residual_align_mse_weight "$TRAINING_RESIDUAL_ALIGN_MSE_WEIGHT" \
    --residual_align_cosine_weight "$TRAINING_RESIDUAL_ALIGN_COSINE_WEIGHT" \
    --residual_align_detach_prefix_target "$TRAINING_RESIDUAL_ALIGN_DETACH_PREFIX_TARGET"

"$PYTHON" - "$METRICS_DIR" "$SUMMARY_CSV" \
    "$TRAINING_MRL_LOSS_MODE" \
    "$TRAINING_SAMPLED_PREFIX_DISTRIBUTION" \
    "$TRAINING_SAMPLED_PREFIX_LOG_INTERVAL" \
    "$TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL" \
    "$TRAINING_RESIDUAL_ALIGNMENT_LOG_INTERVAL" \
    "$TRAINING_MRL_CONFLICT_GATING" \
    "$TRAINING_MRL_CONFLICT_MODE" \
    "$TRAINING_MRL_CONFLICT_ALPHA" \
    "$TRAINING_MRL_CONFLICT_EPS" \
    "$TRAINING_RESIDUAL_ALIGN_MODE" \
    "$TRAINING_RESIDUAL_ALIGN_ORTHOGONAL_MAP" \
    "$TRAINING_RESIDUAL_ALIGN_USE_TRIVIALIZATION" \
    "$TRAINING_RESIDUAL_ALIGN_MSE_WEIGHT" \
    "$TRAINING_RESIDUAL_ALIGN_COSINE_WEIGHT" \
    "$TRAINING_RESIDUAL_ALIGN_DETACH_PREFIX_TARGET" \
    "$TRAINING_RESIDUAL_INTERPOLATION_ALPHA" <<'PY'
import csv
import json
import sys
from pathlib import Path

metrics_dir = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
training_mrl_loss_mode = sys.argv[3]
training_sampled_prefix_distribution = sys.argv[4]
training_sampled_prefix_log_interval = sys.argv[5]
training_mrl_gradient_conflict_interval = sys.argv[6]
training_residual_alignment_log_interval = sys.argv[7]
training_mrl_conflict_gating = sys.argv[8]
training_mrl_conflict_mode = sys.argv[9]
training_mrl_conflict_alpha = sys.argv[10]
training_mrl_conflict_eps = sys.argv[11]
training_residual_align_mode = sys.argv[12]
training_residual_align_orthogonal_map = sys.argv[13]
training_residual_align_use_trivialization = sys.argv[14]
training_residual_align_mse_weight = sys.argv[15]
training_residual_align_cosine_weight = sys.argv[16]
training_residual_align_detach_prefix_target = sys.argv[17]
training_residual_interpolation_alpha = sys.argv[18]
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
            "training_mrl_gradient_conflict_interval": training_mrl_gradient_conflict_interval,
            "training_residual_alignment_log_interval": training_residual_alignment_log_interval,
            "training_mrl_conflict_gating": training_mrl_conflict_gating,
            "training_mrl_conflict_mode": training_mrl_conflict_mode,
            "training_mrl_conflict_alpha": training_mrl_conflict_alpha,
            "training_mrl_conflict_eps": training_mrl_conflict_eps,
            "training_residual_align_mode": training_residual_align_mode,
            "training_residual_align_orthogonal_map": training_residual_align_orthogonal_map,
            "training_residual_align_use_trivialization": training_residual_align_use_trivialization,
            "training_residual_align_mse_weight": training_residual_align_mse_weight,
            "training_residual_align_cosine_weight": training_residual_align_cosine_weight,
            "training_residual_align_detach_prefix_target": training_residual_align_detach_prefix_target,
            "training_residual_interpolation_alpha": training_residual_interpolation_alpha,
            "feature_config": data.get("feature_config", ""),
            "eval_config": data.get("eval_config", ""),
            "index_type": data.get("index_type", ""),
            "residual_interpolate_alpha": data.get("residual_interpolate_alpha", ""),
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
    "training_mrl_gradient_conflict_interval",
    "training_residual_alignment_log_interval",
    "training_mrl_conflict_gating",
    "training_mrl_conflict_mode",
    "training_mrl_conflict_alpha",
    "training_mrl_conflict_eps",
    "training_residual_align_mode",
    "training_residual_align_orthogonal_map",
    "training_residual_align_use_trivialization",
    "training_residual_align_mse_weight",
    "training_residual_align_cosine_weight",
    "training_residual_align_detach_prefix_target",
    "training_residual_interpolation_alpha",
    "feature_config", "eval_config", "index_type",
    "residual_interpolate_alpha",
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
