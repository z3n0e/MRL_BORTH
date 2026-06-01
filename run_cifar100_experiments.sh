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
EPOCHS=${EPOCHS:-40}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
EVAL_WORKERS=${EVAL_WORKERS:-4}

# Method toggles.
RUN_MRL=${RUN_MRL:-1}
RUN_MRLE=${RUN_MRLE:-1}
RUN_FULL_FEATURE=${RUN_FULL_FEATURE:-1}
RUN_FIXED_FEATURE=${RUN_FIXED_FEATURE:-1}
RUN_BOR_MRL=${RUN_BOR_MRL:-1}
RUN_BOR_BLOCK_MRL=${RUN_BOR_BLOCK_MRL:-1}
RUN_BOR_MRL_IDENTITY=${RUN_BOR_MRL_IDENTITY:-1}
RUN_BOR_MRL_CAYLEY=${RUN_BOR_MRL_CAYLEY:-1}
RUN_BOR_MRL_HOUSEHOLDER=${RUN_BOR_MRL_HOUSEHOLDER:-1}
FIXED_FEATURE_DIMS=${FIXED_FEATURE_DIMS:-512}

# BOR-MRL knobs.
BOR_ORTHOGONAL_MAP=${BOR_ORTHOGONAL_MAP:-matrix_exp}
BOR_USE_TRIVIALIZATION=${BOR_USE_TRIVIALIZATION:-1}

TRAINLOG_DIR="$EXPERIMENT_DIR/trainlogs"
EVAL_DIR="$EXPERIMENT_DIR/eval"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"

mkdir -p "$CIFAR100_DIR" "$TRAINLOG_DIR" "$EVAL_DIR" "$CHECKPOINT_DIR"

echo "CIFAR-100 experiment directory: $EXPERIMENT_DIR"
echo "Seed: $SEED"
echo "Deterministic: $DETERMINISTIC"
echo "CIFAR-100 data root: $CIFAR100_DIR"

write_manifest() {
    {
        echo "experiment_dir=$EXPERIMENT_DIR"
        echo "seed=$SEED"
        echo "deterministic=$DETERMINISTIC"
        echo "cifar100_dir=$CIFAR100_DIR"
        echo "epochs=$EPOCHS"
        echo "train_batch_size=$TRAIN_BATCH_SIZE"
        echo "val_batch_size=$VAL_BATCH_SIZE"
        echo "num_workers=$NUM_WORKERS"
        echo "eval_workers=$EVAL_WORKERS"
        echo "fixed_feature_dims=$FIXED_FEATURE_DIMS"
        echo "run_mrl=$RUN_MRL"
        echo "run_mrle=$RUN_MRLE"
        echo "run_full_feature=$RUN_FULL_FEATURE"
        echo "run_fixed_feature=$RUN_FIXED_FEATURE"
        echo "run_bor_mrl=$RUN_BOR_MRL"
        echo "run_bor_block_mrl=$RUN_BOR_BLOCK_MRL"
        echo "run_bor_mrl_identity=$RUN_BOR_MRL_IDENTITY"
        echo "run_bor_mrl_cayley=$RUN_BOR_MRL_CAYLEY"
        echo "run_bor_mrl_householder=$RUN_BOR_MRL_HOUSEHOLDER"
        echo "bor_orthogonal_map=$BOR_ORTHOGONAL_MAP"
        echo "bor_use_trivialization=$BOR_USE_TRIVIALIZATION"
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
        python train_imagenet.py \
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
        python pytorch_inference.py \
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

write_manifest

if [[ "$RUN_MRL" == "1" ]]; then
    train_run mrl --model.mrl=1
    eval_run mrl --mrl
fi

if [[ "$RUN_MRLE" == "1" ]]; then
    train_run mrle --model.efficient=1
    eval_run mrle --mrl --efficient
fi

if [[ "$RUN_BOR_MRL" == "1" ]]; then
    train_run bor_mrl \
        --model.bor_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map="$BOR_ORTHOGONAL_MAP" \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION"
    eval_run bor_mrl \
        --bor_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map "$BOR_ORTHOGONAL_MAP" \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION"
fi

if [[ "$RUN_BOR_BLOCK_MRL" == "1" ]]; then
    train_run bor_block_mrl \
        --model.bor_block_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map="$BOR_ORTHOGONAL_MAP" \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION"
    eval_run bor_block_mrl \
        --bor_block_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map "$BOR_ORTHOGONAL_MAP" \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION"
fi

if [[ "$RUN_BOR_MRL_IDENTITY" == "1" ]]; then
    train_run bor_mrl_identity \
        --model.bor_mrl=1 \
        --model.bor_mode=identity
    eval_run bor_mrl_identity \
        --bor_mrl \
        --bor_mode identity
fi

if [[ "$RUN_BOR_MRL_CAYLEY" == "1" ]]; then
    train_run bor_mrl_cayley \
        --model.bor_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map=cayley \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION"
    eval_run bor_mrl_cayley \
        --bor_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map cayley \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION"
fi

if [[ "$RUN_BOR_MRL_HOUSEHOLDER" == "1" ]]; then
    train_run bor_mrl_householder \
        --model.bor_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map=householder \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION"
    eval_run bor_mrl_householder \
        --bor_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map householder \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION"
fi

if [[ "$RUN_FULL_FEATURE" == "1" ]]; then
    train_run full_feature
    eval_run full_feature --rep_size 2048
fi

if [[ "$RUN_FIXED_FEATURE" == "1" ]]; then
    for dim in $FIXED_FEATURE_DIMS; do
        train_run "fixed_${dim}" --model.fixed_feature="$dim"
        eval_run "fixed_${dim}" --rep_size "$dim"
    done
fi

echo "Done."
echo "Metrics JSON files are in: $EVAL_DIR"
echo "Model checkpoints are in: $CHECKPOINT_DIR"
echo "Open cifar100_results.ipynb and set EXPERIMENT_DIR to visualize this run."
