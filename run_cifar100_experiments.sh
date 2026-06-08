#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Reproducibility knobs.
SEED=${SEED:-0}
DETERMINISTIC=${DETERMINISTIC:-1}
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

# Data/output locations.
CIFAR100_DIR=${CIFAR100_DIR:-"$HOME/.cache/torchvision"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"$ROOT_DIR/cifar100_runs/cifar100_seed_${SEED}_$(date +%Y%m%d_%H%M%S)"}

# Training knobs.
PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-40}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
EVAL_WORKERS=${EVAL_WORKERS:-4}

# MRL loss/probe knobs.
MRL_LOSS_MODE=${MRL_LOSS_MODE:-all}
SAMPLED_PREFIX_DISTRIBUTION=${SAMPLED_PREFIX_DISTRIBUTION:-uniform}
SAMPLED_PREFIX_LOG_INTERVAL=${SAMPLED_PREFIX_LOG_INTERVAL:-100}
MRL_GRADIENT_CONFLICT_INTERVAL=${MRL_GRADIENT_CONFLICT_INTERVAL:-0}
RESIDUAL_ALIGNMENT_LOG_INTERVAL=${RESIDUAL_ALIGNMENT_LOG_INTERVAL:-100}
MRL_CONFLICT_GATING=${MRL_CONFLICT_GATING:-0}
MRL_CONFLICT_MODE=${MRL_CONFLICT_MODE:-none}
MRL_CONFLICT_ALPHA=${MRL_CONFLICT_ALPHA:-0.5}
MRL_CONFLICT_EPS=${MRL_CONFLICT_EPS:-1e-8}

# Method toggles. These scripts compare only MRL and Residual-Aligned MRL.
RUN_MRL=${RUN_MRL:-1}
RUN_RESIDUAL_ALIGNED_MRL=${RUN_RESIDUAL_ALIGNED_MRL:-1}
RUN_RETRIEVAL_METRICS=${RUN_RETRIEVAL_METRICS:-1}

# Residual-Aligned MRL knobs.
RESIDUAL_ALIGN_MODE=${RESIDUAL_ALIGN_MODE:-orthogonal}
RESIDUAL_ALIGN_ORTHOGONAL_MAP=${RESIDUAL_ALIGN_ORTHOGONAL_MAP:-matrix_exp}
RESIDUAL_ALIGN_USE_TRIVIALIZATION=${RESIDUAL_ALIGN_USE_TRIVIALIZATION:-1}
RESIDUAL_ALIGN_MSE_WEIGHT=${RESIDUAL_ALIGN_MSE_WEIGHT:-1.0}
RESIDUAL_ALIGN_COSINE_WEIGHT=${RESIDUAL_ALIGN_COSINE_WEIGHT:-1.0}
RESIDUAL_ALIGN_DETACH_PREFIX_TARGET=${RESIDUAL_ALIGN_DETACH_PREFIX_TARGET:-1}
RESIDUAL_INTERPOLATION_ALPHA=${RESIDUAL_INTERPOLATION_ALPHA:-0.5}

TRAINLOG_DIR="$EXPERIMENT_DIR/trainlogs"
EVAL_DIR="$EXPERIMENT_DIR/eval"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"

mkdir -p "$CIFAR100_DIR" "$TRAINLOG_DIR" "$EVAL_DIR" "$CHECKPOINT_DIR"

echo "CIFAR-100 experiment directory: $EXPERIMENT_DIR"
echo "Seed: $SEED"
echo "Deterministic: $DETERMINISTIC"
echo "CIFAR-100 data root: $CIFAR100_DIR"
echo "Run MRL: $RUN_MRL"
echo "Run Residual-Aligned MRL: $RUN_RESIDUAL_ALIGNED_MRL"
echo "MRL loss mode: $MRL_LOSS_MODE"
echo "Residual alignment log interval: $RESIDUAL_ALIGNMENT_LOG_INTERVAL"
echo "Residual-Aligned MRL orthogonal map: $RESIDUAL_ALIGN_ORTHOGONAL_MAP"
echo "Residual-Aligned MRL MSE weight: $RESIDUAL_ALIGN_MSE_WEIGHT"
echo "Residual-Aligned MRL cosine weight: $RESIDUAL_ALIGN_COSINE_WEIGHT"
echo "Residual interpolation alpha: $RESIDUAL_INTERPOLATION_ALPHA"
echo "Run retrieval metrics: $RUN_RETRIEVAL_METRICS"

MRL_TRAINING_ARGS=(
    --training.mrl_loss_mode="$MRL_LOSS_MODE"
    --training.sampled_prefix_distribution="$SAMPLED_PREFIX_DISTRIBUTION"
    --training.sampled_prefix_log_interval="$SAMPLED_PREFIX_LOG_INTERVAL"
    --training.mrl_gradient_conflict_interval="$MRL_GRADIENT_CONFLICT_INTERVAL"
    --training.residual_alignment_log_interval="$RESIDUAL_ALIGNMENT_LOG_INTERVAL"
)

MRL_CONFLICT_TRAINING_ARGS=(
    --training.mrl_conflict_gating="$MRL_CONFLICT_GATING"
    --training.mrl_conflict_mode="$MRL_CONFLICT_MODE"
    --training.mrl_conflict_alpha="$MRL_CONFLICT_ALPHA"
    --training.mrl_conflict_eps="$MRL_CONFLICT_EPS"
)

RESIDUAL_ALIGNED_TRAINING_ARGS=(
    --model.residual_aligned_mrl=1
    --model.residual_align_mode="$RESIDUAL_ALIGN_MODE"
    --model.residual_align_orthogonal_map="$RESIDUAL_ALIGN_ORTHOGONAL_MAP"
    --model.residual_align_use_trivialization="$RESIDUAL_ALIGN_USE_TRIVIALIZATION"
    --model.residual_align_mse_weight="$RESIDUAL_ALIGN_MSE_WEIGHT"
    --model.residual_align_cosine_weight="$RESIDUAL_ALIGN_COSINE_WEIGHT"
    --model.residual_align_detach_prefix_target="$RESIDUAL_ALIGN_DETACH_PREFIX_TARGET"
)

RESIDUAL_ALIGNED_EVAL_ARGS=(
    --residual_aligned_mrl
    --residual_align_mode "$RESIDUAL_ALIGN_MODE"
    --residual_align_orthogonal_map "$RESIDUAL_ALIGN_ORTHOGONAL_MAP"
    --residual_align_use_trivialization "$RESIDUAL_ALIGN_USE_TRIVIALIZATION"
    --residual_align_mse_weight "$RESIDUAL_ALIGN_MSE_WEIGHT"
    --residual_align_cosine_weight "$RESIDUAL_ALIGN_COSINE_WEIGHT"
    --residual_align_detach_prefix_target "$RESIDUAL_ALIGN_DETACH_PREFIX_TARGET"
)

write_manifest() {
    {
        echo "experiment_dir=$EXPERIMENT_DIR"
        echo "seed=$SEED"
        echo "deterministic=$DETERMINISTIC"
        echo "cifar100_dir=$CIFAR100_DIR"
        echo "python=$PYTHON"
        echo "epochs=$EPOCHS"
        echo "train_batch_size=$TRAIN_BATCH_SIZE"
        echo "val_batch_size=$VAL_BATCH_SIZE"
        echo "num_workers=$NUM_WORKERS"
        echo "eval_workers=$EVAL_WORKERS"
        echo "mrl_loss_mode=$MRL_LOSS_MODE"
        echo "sampled_prefix_distribution=$SAMPLED_PREFIX_DISTRIBUTION"
        echo "sampled_prefix_log_interval=$SAMPLED_PREFIX_LOG_INTERVAL"
        echo "mrl_gradient_conflict_interval=$MRL_GRADIENT_CONFLICT_INTERVAL"
        echo "residual_alignment_log_interval=$RESIDUAL_ALIGNMENT_LOG_INTERVAL"
        echo "mrl_conflict_gating=$MRL_CONFLICT_GATING"
        echo "mrl_conflict_mode=$MRL_CONFLICT_MODE"
        echo "mrl_conflict_alpha=$MRL_CONFLICT_ALPHA"
        echo "mrl_conflict_eps=$MRL_CONFLICT_EPS"
        echo "run_mrl=$RUN_MRL"
        echo "run_residual_aligned_mrl=$RUN_RESIDUAL_ALIGNED_MRL"
        echo "run_retrieval_metrics=$RUN_RETRIEVAL_METRICS"
        echo "residual_align_mode=$RESIDUAL_ALIGN_MODE"
        echo "residual_align_orthogonal_map=$RESIDUAL_ALIGN_ORTHOGONAL_MAP"
        echo "residual_align_use_trivialization=$RESIDUAL_ALIGN_USE_TRIVIALIZATION"
        echo "residual_align_mse_weight=$RESIDUAL_ALIGN_MSE_WEIGHT"
        echo "residual_align_cosine_weight=$RESIDUAL_ALIGN_COSINE_WEIGHT"
        echo "residual_align_detach_prefix_target=$RESIDUAL_ALIGN_DETACH_PREFIX_TARGET"
        echo "residual_interpolation_alpha=$RESIDUAL_INTERPOLATION_ALPHA"
    } > "$EXPERIMENT_DIR/manifest.txt"
}

train_run() {
    local run_name=$1
    shift

    local run_dir="$TRAINLOG_DIR/$run_name"
    if [[ -e "$run_dir" ]]; then
        echo "Run directory already exists: $run_dir"
        echo "Use a new EXPERIMENT_DIR or remove the existing run directory."
        exit 1
    fi

    echo "Training $run_name..."
    (
        cd "$ROOT_DIR/train"
        "$PYTHON" train_imagenet.py \
            --config-file rn50_configs/rn50_cifar100.yaml \
            --data.root="$CIFAR100_DIR" \
            --data.num_workers="$NUM_WORKERS" \
            --training.batch_size="$TRAIN_BATCH_SIZE" \
            --training.epochs="$EPOCHS" \
            --training.seed="$SEED" \
            --training.deterministic="$DETERMINISTIC" \
            --validation.batch_size="$VAL_BATCH_SIZE" \
            --logging.folder="$TRAINLOG_DIR" \
            --logging.run_name="$run_name" \
            "$@"
    )

    local checkpoint="$run_dir/final_weights.pt"
    if [[ ! -f "$checkpoint" ]]; then
        echo "Training finished but did not produce checkpoint: $checkpoint"
        exit 1
    fi
    cp "$checkpoint" "$CHECKPOINT_DIR/${run_name}_final_weights.pt"
    if [[ -f "$run_dir/latest_weights.pt" ]]; then
        cp "$run_dir/latest_weights.pt" "$CHECKPOINT_DIR/${run_name}_latest_weights.pt"
    fi
}

eval_run() {
    local run_name=$1
    shift

    local checkpoint="$TRAINLOG_DIR/$run_name/final_weights.pt"
    local metrics_output="$EVAL_DIR/${run_name}.json"
    local deterministic_args=()
    if [[ "$DETERMINISTIC" == "1" ]]; then
        deterministic_args=(--deterministic)
    fi

    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing checkpoint for $run_name: $checkpoint"
        exit 1
    fi

    echo "Evaluating $run_name..."
    (
        cd "$ROOT_DIR/inference"
        "$PYTHON" pytorch_inference.py \
            --path "$checkpoint" \
            --dataset CIFAR100 \
            --data_root "$CIFAR100_DIR" \
            --workers "$EVAL_WORKERS" \
            --seed "$SEED" \
            "${deterministic_args[@]}" \
            --metrics_output "$metrics_output" \
            "$@"
    )
}

run_retrieval_metrics() {
    if [[ "$RUN_RETRIEVAL_METRICS" != "1" ]]; then
        echo "Skipping retrieval metrics because RUN_RETRIEVAL_METRICS=$RUN_RETRIEVAL_METRICS"
        return
    fi

    echo "Computing retrieval metrics for this CIFAR-100 run..."
    CIFAR100_DIR="$CIFAR100_DIR" \
    PYTHON="$PYTHON" \
    SEED="$SEED" \
    DETERMINISTIC="$DETERMINISTIC" \
    EVAL_WORKERS="$EVAL_WORKERS" \
    RESIDUAL_INTERPOLATION_ALPHA="$RESIDUAL_INTERPOLATION_ALPHA" \
        "$ROOT_DIR/run_cifar100_retrieval_metrics.sh" "$EXPERIMENT_DIR"
}

write_manifest

if [[ "$RUN_MRL" == "1" ]]; then
    train_run mrl \
        "${MRL_TRAINING_ARGS[@]}" \
        "${MRL_CONFLICT_TRAINING_ARGS[@]}" \
        --model.mrl=1
    eval_run mrl --mrl
fi

if [[ "$RUN_RESIDUAL_ALIGNED_MRL" == "1" ]]; then
    train_run residual_aligned_mrl \
        "${MRL_TRAINING_ARGS[@]}" \
        "${RESIDUAL_ALIGNED_TRAINING_ARGS[@]}"
    eval_run residual_aligned_mrl \
        "${RESIDUAL_ALIGNED_EVAL_ARGS[@]}"
fi

run_retrieval_metrics

echo "Done."
echo "Metrics JSON files are in: $EVAL_DIR"
echo "Model checkpoints are in: $CHECKPOINT_DIR"
if [[ "$RUN_RETRIEVAL_METRICS" == "1" ]]; then
    echo "Retrieval metrics JSON files are in: $EXPERIMENT_DIR/retrieval_metrics"
    echo "Retrieval CSV summary is in: $EXPERIMENT_DIR/cifar100_retrieval_summary.csv"
fi
echo "Open cifar100_results.ipynb and set EXPERIMENT_DIR to visualize this run."
