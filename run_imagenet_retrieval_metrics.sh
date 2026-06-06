#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail
shopt -s nullglob

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:-${RUN_DIR:-}}

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: $0 /path/to/imagenet_runs/<run_dir>"
    echo
    echo "Example:"
    echo "  IMAGENET_DIR=/path/to/imagenet $0 imagenet_runs/imagenet_seed_0_20260601_234841"
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
TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL=${MRL_GRADIENT_CONFLICT_INTERVAL:-}
TRAINING_MRL_CONFLICT_GATING=${MRL_CONFLICT_GATING:-}
TRAINING_MRL_CONFLICT_MODE=${MRL_CONFLICT_MODE:-}
TRAINING_MRL_CONFLICT_ALPHA=${MRL_CONFLICT_ALPHA:-}
TRAINING_MRL_CONFLICT_EPS=${MRL_CONFLICT_EPS:-}
TRAINING_T_ORTHOGONAL_MAP=${T_ORTHOGONAL_MAP:-}
TRAINING_BOR_ORTHOGONAL_MAP=${BOR_ORTHOGONAL_MAP:-}
TRAINING_RECURSIVE_LINK_HIDDEN_RATIO=${RECURSIVE_LINK_HIDDEN_RATIO:-}
TRAINING_RECURSIVE_LINK_DROPOUT=${RECURSIVE_LINK_DROPOUT:-}
TRAINING_RECURSIVE_LINK_ALPHA_INIT=${RECURSIVE_LINK_ALPHA_INIT:-}
TRAINING_RECURSIVE_LINK_STOP_GRADIENT=${RECURSIVE_LINK_STOP_GRADIENT:-}

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
            t_orthogonal_map)
                [[ -z "$TRAINING_T_ORTHOGONAL_MAP" ]] && TRAINING_T_ORTHOGONAL_MAP=$value
                ;;
            bor_orthogonal_map)
                [[ -z "$TRAINING_BOR_ORTHOGONAL_MAP" ]] && TRAINING_BOR_ORTHOGONAL_MAP=$value
                ;;
            recursive_link_hidden_ratio)
                [[ -z "$TRAINING_RECURSIVE_LINK_HIDDEN_RATIO" ]] && TRAINING_RECURSIVE_LINK_HIDDEN_RATIO=$value
                ;;
            recursive_link_dropout)
                [[ -z "$TRAINING_RECURSIVE_LINK_DROPOUT" ]] && TRAINING_RECURSIVE_LINK_DROPOUT=$value
                ;;
            recursive_link_alpha_init)
                [[ -z "$TRAINING_RECURSIVE_LINK_ALPHA_INIT" ]] && TRAINING_RECURSIVE_LINK_ALPHA_INIT=$value
                ;;
            recursive_link_stop_gradient)
                [[ -z "$TRAINING_RECURSIVE_LINK_STOP_GRADIENT" ]] && TRAINING_RECURSIVE_LINK_STOP_GRADIENT=$value
                ;;
        esac
    done < "$MANIFEST"
fi

TRAINING_MRL_LOSS_MODE=${TRAINING_MRL_LOSS_MODE:-all}
TRAINING_SAMPLED_PREFIX_DISTRIBUTION=${TRAINING_SAMPLED_PREFIX_DISTRIBUTION:-uniform}
TRAINING_SAMPLED_PREFIX_LOG_INTERVAL=${TRAINING_SAMPLED_PREFIX_LOG_INTERVAL:-100}
TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL=${TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL:-0}
TRAINING_MRL_CONFLICT_GATING=${TRAINING_MRL_CONFLICT_GATING:-0}
TRAINING_MRL_CONFLICT_MODE=${TRAINING_MRL_CONFLICT_MODE:-none}
TRAINING_MRL_CONFLICT_ALPHA=${TRAINING_MRL_CONFLICT_ALPHA:-0.5}
TRAINING_MRL_CONFLICT_EPS=${TRAINING_MRL_CONFLICT_EPS:-1e-8}
TRAINING_T_ORTHOGONAL_MAP=${TRAINING_T_ORTHOGONAL_MAP:-matrix_exp}
TRAINING_BOR_ORTHOGONAL_MAP=${TRAINING_BOR_ORTHOGONAL_MAP:-matrix_exp}
TRAINING_RECURSIVE_LINK_HIDDEN_RATIO=${TRAINING_RECURSIVE_LINK_HIDDEN_RATIO:-0.5}
TRAINING_RECURSIVE_LINK_DROPOUT=${TRAINING_RECURSIVE_LINK_DROPOUT:-0.0}
TRAINING_RECURSIVE_LINK_ALPHA_INIT=${TRAINING_RECURSIVE_LINK_ALPHA_INIT:--4.0}
TRAINING_RECURSIVE_LINK_STOP_GRADIENT=${TRAINING_RECURSIVE_LINK_STOP_GRADIENT:-0}

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
T_ORTHOGONAL_MAP=${T_ORTHOGONAL_MAP:-$TRAINING_T_ORTHOGONAL_MAP}
BOR_ORTHOGONAL_MAP=${BOR_ORTHOGONAL_MAP:-$TRAINING_BOR_ORTHOGONAL_MAP}
BOR_USE_TRIVIALIZATION=${BOR_USE_TRIVIALIZATION:-1}
BOR_STOP_GRADIENT=${BOR_STOP_GRADIENT:-0}
BOR_RESIDUAL_ALPHA_INIT=${BOR_RESIDUAL_ALPHA_INIT:--3.0}
CASCADE_STOP_GRADIENT=${CASCADE_STOP_GRADIENT:-1}
RECURSIVE_LINK_HIDDEN_RATIO=${RECURSIVE_LINK_HIDDEN_RATIO:-$TRAINING_RECURSIVE_LINK_HIDDEN_RATIO}
RECURSIVE_LINK_DROPOUT=${RECURSIVE_LINK_DROPOUT:-$TRAINING_RECURSIVE_LINK_DROPOUT}
RECURSIVE_LINK_ALPHA_INIT=${RECURSIVE_LINK_ALPHA_INIT:-$TRAINING_RECURSIVE_LINK_ALPHA_INIT}
RECURSIVE_LINK_STOP_GRADIENT=${RECURSIVE_LINK_STOP_GRADIENT:-$TRAINING_RECURSIVE_LINK_STOP_GRADIENT}

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
echo "Training sampled-prefix distribution: $TRAINING_SAMPLED_PREFIX_DISTRIBUTION"
echo "Training sampled-prefix log interval: $TRAINING_SAMPLED_PREFIX_LOG_INTERVAL"
echo "Training MRL gradient conflict interval: $TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL"
echo "Training MRL conflict gating: $TRAINING_MRL_CONFLICT_GATING"
echo "Training MRL conflict mode: $TRAINING_MRL_CONFLICT_MODE"
echo "Training MRL conflict alpha: $TRAINING_MRL_CONFLICT_ALPHA"
echo "Training MRL conflict eps: $TRAINING_MRL_CONFLICT_EPS"
echo "T orthogonal map: $T_ORTHOGONAL_MAP"
echo "BOR orthogonal map: $BOR_ORTHOGONAL_MAP"
echo "BOR stop gradient: $BOR_STOP_GRADIENT"
echo "BOR residual alpha init: $BOR_RESIDUAL_ALPHA_INIT"
echo "Cascade stop gradient: $CASCADE_STOP_GRADIENT"
echo "RecursiveLink hidden ratio: $RECURSIVE_LINK_HIDDEN_RATIO"
echo "RecursiveLink dropout: $RECURSIVE_LINK_DROPOUT"
echo "RecursiveLink alpha init: $RECURSIVE_LINK_ALPHA_INIT"
echo "RecursiveLink stop gradient: $RECURSIVE_LINK_STOP_GRADIENT"
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
    local train_x="$method_root/1K_train_${feature_config}-X.npy"
    local val_x="$method_root/1K_val_${feature_config}-X.npy"

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
                --dataset 1K \
                --data_root "$IMAGENET_DIR" \
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
        --dataset 1K
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
        --dataset 1K
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

run_method mrl_pcd \
    "$CHECKPOINT_DIR/mrl_pcd_final_weights.pt" \
    mrl_pcd 2048 "$NESTED_DIMS" mrl1_e0_ff2048 \
    --mrl

run_method recursive_link_mrl \
    "$CHECKPOINT_DIR/recursive_link_mrl_final_weights.pt" \
    recursive_link_mrl 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --recursive_link_mrl \
    --recursive_link_hidden_ratio "$RECURSIVE_LINK_HIDDEN_RATIO" \
    --recursive_link_dropout "$RECURSIVE_LINK_DROPOUT" \
    --recursive_link_alpha_init "$RECURSIVE_LINK_ALPHA_INIT" \
    --recursive_link_stop_gradient "$RECURSIVE_LINK_STOP_GRADIENT"

run_method recursive_link_mrl_pcd \
    "$CHECKPOINT_DIR/recursive_link_mrl_pcd_final_weights.pt" \
    recursive_link_mrl_pcd 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --recursive_link_mrl \
    --recursive_link_hidden_ratio "$RECURSIVE_LINK_HIDDEN_RATIO" \
    --recursive_link_dropout "$RECURSIVE_LINK_DROPOUT" \
    --recursive_link_alpha_init "$RECURSIVE_LINK_ALPHA_INIT" \
    --recursive_link_stop_gradient "$RECURSIVE_LINK_STOP_GRADIENT"

run_method t_orthogonal_mrl \
    "$CHECKPOINT_DIR/t_orthogonal_mrl_final_weights.pt" \
    t_orthogonal_mrl 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --t_orthogonal_mrl \
    --t_orthogonal_map "$T_ORTHOGONAL_MAP" \
    --bor_mode orthogonal \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
    --bor_stop_gradient "$BOR_STOP_GRADIENT"

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
    --bor_orthogonal_map "$BOR_ORTHOGONAL_MAP" \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
    --bor_stop_gradient "$BOR_STOP_GRADIENT"

run_method bor_mrl_residual \
    "$CHECKPOINT_DIR/bor_mrl_residual_final_weights.pt" \
    bor_mrl_residual 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_mrl \
    --bor_mode orthogonal \
    --bor_orthogonal_map "$BOR_ORTHOGONAL_MAP" \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
    --bor_stop_gradient "$BOR_STOP_GRADIENT" \
    --bor_residual_orthogonal 1 \
    --bor_residual_alpha_init "$BOR_RESIDUAL_ALPHA_INIT"

run_method bor_block_mrl \
    "$CHECKPOINT_DIR/bor_block_mrl_final_weights.pt" \
    bor_block_mrl 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_block_mrl \
    --bor_mode orthogonal \
    --bor_orthogonal_map "$BOR_ORTHOGONAL_MAP" \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
    --bor_stop_gradient "$BOR_STOP_GRADIENT"

run_method cascade_stop_gradient_mrl \
    "$CHECKPOINT_DIR/cascade_stop_gradient_mrl_final_weights.pt" \
    cascade_stop_gradient_mrl 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --cascade_stop_gradient_mrl \
    --cascade_stop_gradient "$CASCADE_STOP_GRADIENT"

run_method bor_mrl_cayley \
    "$CHECKPOINT_DIR/bor_mrl_cayley_final_weights.pt" \
    bor_mrl_cayley 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_mrl \
    --bor_mode orthogonal \
    --bor_orthogonal_map cayley \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
    --bor_stop_gradient "$BOR_STOP_GRADIENT"

run_method bor_mrl_householder \
    "$CHECKPOINT_DIR/bor_mrl_householder_final_weights.pt" \
    bor_mrl_householder 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_mrl \
    --bor_mode orthogonal \
    --bor_orthogonal_map householder \
    --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
    --bor_stop_gradient "$BOR_STOP_GRADIENT"

run_method bor_mrl_frozen \
    "$CHECKPOINT_DIR/bor_mrl_frozen_final_weights.pt" \
    bor_mrl_frozen 2048 "$NESTED_DIMS" mrl0_e0_ff2048 \
    --bor_mrl \
    --bor_mode frozen \
    --bor_stop_gradient "$BOR_STOP_GRADIENT"

"$PYTHON" - "$METRICS_DIR" "$SUMMARY_CSV" \
    "$TRAINING_MRL_LOSS_MODE" \
    "$TRAINING_SAMPLED_PREFIX_DISTRIBUTION" \
    "$TRAINING_SAMPLED_PREFIX_LOG_INTERVAL" \
    "$TRAINING_MRL_GRADIENT_CONFLICT_INTERVAL" \
    "$TRAINING_MRL_CONFLICT_GATING" \
    "$TRAINING_MRL_CONFLICT_MODE" \
    "$TRAINING_MRL_CONFLICT_ALPHA" \
    "$TRAINING_MRL_CONFLICT_EPS" <<'PY'
import csv
import json
import sys
from pathlib import Path

metrics_dir = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
training_mrl_loss_mode = sys.argv[3] if len(sys.argv) > 3 else ""
training_sampled_prefix_distribution = sys.argv[4] if len(sys.argv) > 4 else ""
training_sampled_prefix_log_interval = sys.argv[5] if len(sys.argv) > 5 else ""
training_mrl_gradient_conflict_interval = sys.argv[6] if len(sys.argv) > 6 else ""
training_mrl_conflict_gating = sys.argv[7] if len(sys.argv) > 7 else ""
training_mrl_conflict_mode = sys.argv[8] if len(sys.argv) > 8 else ""
training_mrl_conflict_alpha = sys.argv[9] if len(sys.argv) > 9 else ""
training_mrl_conflict_eps = sys.argv[10] if len(sys.argv) > 10 else ""
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
            "training_mrl_conflict_gating": training_mrl_conflict_gating,
            "training_mrl_conflict_mode": training_mrl_conflict_mode,
            "training_mrl_conflict_alpha": training_mrl_conflict_alpha,
            "training_mrl_conflict_eps": training_mrl_conflict_eps,
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
    "training_mrl_gradient_conflict_interval",
    "training_mrl_conflict_gating",
    "training_mrl_conflict_mode",
    "training_mrl_conflict_alpha",
    "training_mrl_conflict_eps",
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
