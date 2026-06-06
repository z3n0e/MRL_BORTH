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
MRL_CONFLICT_GATING=${MRL_CONFLICT_GATING:-0}
MRL_CONFLICT_MODE=${MRL_CONFLICT_MODE:-none}
MRL_CONFLICT_ALPHA=${MRL_CONFLICT_ALPHA:-0.5}
MRL_CONFLICT_EPS=${MRL_CONFLICT_EPS:-1e-8}

# Method toggles.
RUN_MRL=${RUN_MRL:-1}
RUN_MRLE=${RUN_MRLE:-0}
RUN_FULL_FEATURE=${RUN_FULL_FEATURE:-0}
RUN_FIXED_FEATURE=${RUN_FIXED_FEATURE:-0}
RUN_T_ORTHOGONAL_MRL=${RUN_T_ORTHOGONAL_MRL:-0}
RUN_BOR_MRL=${RUN_BOR_MRL:-0}
RUN_BOR_MRL_RESIDUAL=${RUN_BOR_MRL_RESIDUAL:-0}
RUN_BOR_BLOCK_MRL=${RUN_BOR_BLOCK_MRL:-0}
RUN_CASCADE_STOP_GRADIENT_MRL=${RUN_CASCADE_STOP_GRADIENT_MRL:-1}
RUN_RECURSIVE_LINK_MRL=${RUN_RECURSIVE_LINK_MRL:-0}
RUN_MRL_PCD=${RUN_MRL_PCD:-0}
RUN_RECURSIVE_LINK_MRL_PCD=${RUN_RECURSIVE_LINK_MRL_PCD:-0}
RUN_BOR_MRL_FROZEN=${RUN_BOR_MRL_FROZEN:-0}
RUN_BOR_MRL_CAYLEY=${RUN_BOR_MRL_CAYLEY:-0}
RUN_BOR_MRL_HOUSEHOLDER=${RUN_BOR_MRL_HOUSEHOLDER:-1}
FIXED_FEATURE_DIMS=${FIXED_FEATURE_DIMS:-"8 16 32 64 128 256 512 1024"}

# BOR-MRL knobs.
T_ORTHOGONAL_MAP=${T_ORTHOGONAL_MAP:-matrix_exp}
BOR_ORTHOGONAL_MAP=${BOR_ORTHOGONAL_MAP:-matrix_exp}
BOR_USE_TRIVIALIZATION=${BOR_USE_TRIVIALIZATION:-1}
BOR_STOP_GRADIENT=${BOR_STOP_GRADIENT:-1}
BOR_RESIDUAL_ALPHA_INIT=${BOR_RESIDUAL_ALPHA_INIT:--3.0}
CASCADE_STOP_GRADIENT=${CASCADE_STOP_GRADIENT:-1}
RECURSIVE_LINK_HIDDEN_RATIO=${RECURSIVE_LINK_HIDDEN_RATIO:-0.5}
RECURSIVE_LINK_DROPOUT=${RECURSIVE_LINK_DROPOUT:-0.0}
RECURSIVE_LINK_ALPHA_INIT=${RECURSIVE_LINK_ALPHA_INIT:--4.0}
RECURSIVE_LINK_STOP_GRADIENT=${RECURSIVE_LINK_STOP_GRADIENT:-0}
PROCRUSTES_CASCADE_WEIGHT=${PROCRUSTES_CASCADE_WEIGHT:-0.05}
PROCRUSTES_CASCADE_MAX_SVD_DIM=${PROCRUSTES_CASCADE_MAX_SVD_DIM:-1024}

TRAINLOG_DIR="$EXPERIMENT_DIR/trainlogs"
EVAL_DIR="$EXPERIMENT_DIR/eval"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"

mkdir -p "$CIFAR100_DIR" "$TRAINLOG_DIR" "$EVAL_DIR" "$CHECKPOINT_DIR"

echo "CIFAR-100 experiment directory: $EXPERIMENT_DIR"
echo "Seed: $SEED"
echo "Deterministic: $DETERMINISTIC"
echo "CIFAR-100 data root: $CIFAR100_DIR"
echo "MRL loss mode: $MRL_LOSS_MODE"
echo "Sampled-prefix distribution: $SAMPLED_PREFIX_DISTRIBUTION"
echo "Sampled-prefix log interval: $SAMPLED_PREFIX_LOG_INTERVAL"
echo "MRL gradient conflict interval: $MRL_GRADIENT_CONFLICT_INTERVAL"
echo "MRL conflict gating: $MRL_CONFLICT_GATING"
echo "MRL conflict mode: $MRL_CONFLICT_MODE"
echo "MRL conflict alpha: $MRL_CONFLICT_ALPHA"
echo "MRL conflict eps: $MRL_CONFLICT_EPS"
echo "T orthogonal map: $T_ORTHOGONAL_MAP"
echo "RecursiveLink hidden ratio: $RECURSIVE_LINK_HIDDEN_RATIO"
echo "RecursiveLink alpha init: $RECURSIVE_LINK_ALPHA_INIT"
echo "PCD weight: $PROCRUSTES_CASCADE_WEIGHT"

MRL_TRAINING_ARGS=(
    --training.mrl_loss_mode="$MRL_LOSS_MODE"
    --training.sampled_prefix_distribution="$SAMPLED_PREFIX_DISTRIBUTION"
    --training.sampled_prefix_log_interval="$SAMPLED_PREFIX_LOG_INTERVAL"
    --training.mrl_gradient_conflict_interval="$MRL_GRADIENT_CONFLICT_INTERVAL"
)

MRL_CONFLICT_TRAINING_ARGS=(
    --training.mrl_conflict_gating="$MRL_CONFLICT_GATING"
    --training.mrl_conflict_mode="$MRL_CONFLICT_MODE"
    --training.mrl_conflict_alpha="$MRL_CONFLICT_ALPHA"
    --training.mrl_conflict_eps="$MRL_CONFLICT_EPS"
)

RECURSIVE_LINK_TRAINING_ARGS=(
    --model.recursive_link_hidden_ratio="$RECURSIVE_LINK_HIDDEN_RATIO"
    --model.recursive_link_dropout="$RECURSIVE_LINK_DROPOUT"
    --model.recursive_link_alpha_init="$RECURSIVE_LINK_ALPHA_INIT"
    --model.recursive_link_stop_gradient="$RECURSIVE_LINK_STOP_GRADIENT"
)

RECURSIVE_LINK_EVAL_ARGS=(
    --recursive_link_mrl
    --recursive_link_hidden_ratio "$RECURSIVE_LINK_HIDDEN_RATIO"
    --recursive_link_dropout "$RECURSIVE_LINK_DROPOUT"
    --recursive_link_alpha_init "$RECURSIVE_LINK_ALPHA_INIT"
    --recursive_link_stop_gradient "$RECURSIVE_LINK_STOP_GRADIENT"
)

PCD_TRAINING_ARGS=(
    --training.procrustes_cascade_distill=1
    --training.procrustes_cascade_weight="$PROCRUSTES_CASCADE_WEIGHT"
    --training.procrustes_cascade_max_svd_dim="$PROCRUSTES_CASCADE_MAX_SVD_DIM"
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
        echo "mrl_conflict_gating=$MRL_CONFLICT_GATING"
        echo "mrl_conflict_mode=$MRL_CONFLICT_MODE"
        echo "mrl_conflict_alpha=$MRL_CONFLICT_ALPHA"
        echo "mrl_conflict_eps=$MRL_CONFLICT_EPS"
        echo "fixed_feature_dims=$FIXED_FEATURE_DIMS"
        echo "run_mrl=$RUN_MRL"
        echo "run_mrle=$RUN_MRLE"
        echo "run_full_feature=$RUN_FULL_FEATURE"
        echo "run_fixed_feature=$RUN_FIXED_FEATURE"
        echo "run_t_orthogonal_mrl=$RUN_T_ORTHOGONAL_MRL"
        echo "run_bor_mrl=$RUN_BOR_MRL"
        echo "run_bor_mrl_residual=$RUN_BOR_MRL_RESIDUAL"
        echo "run_bor_block_mrl=$RUN_BOR_BLOCK_MRL"
        echo "run_cascade_stop_gradient_mrl=$RUN_CASCADE_STOP_GRADIENT_MRL"
        echo "run_recursive_link_mrl=$RUN_RECURSIVE_LINK_MRL"
        echo "run_mrl_pcd=$RUN_MRL_PCD"
        echo "run_recursive_link_mrl_pcd=$RUN_RECURSIVE_LINK_MRL_PCD"
        echo "run_bor_mrl_frozen=$RUN_BOR_MRL_FROZEN"
        echo "run_bor_mrl_cayley=$RUN_BOR_MRL_CAYLEY"
        echo "run_bor_mrl_householder=$RUN_BOR_MRL_HOUSEHOLDER"
        echo "t_orthogonal_map=$T_ORTHOGONAL_MAP"
        echo "bor_orthogonal_map=$BOR_ORTHOGONAL_MAP"
        echo "bor_use_trivialization=$BOR_USE_TRIVIALIZATION"
        echo "bor_stop_gradient=$BOR_STOP_GRADIENT"
        echo "bor_residual_alpha_init=$BOR_RESIDUAL_ALPHA_INIT"
        echo "cascade_stop_gradient=$CASCADE_STOP_GRADIENT"
        echo "recursive_link_hidden_ratio=$RECURSIVE_LINK_HIDDEN_RATIO"
        echo "recursive_link_dropout=$RECURSIVE_LINK_DROPOUT"
        echo "recursive_link_alpha_init=$RECURSIVE_LINK_ALPHA_INIT"
        echo "recursive_link_stop_gradient=$RECURSIVE_LINK_STOP_GRADIENT"
        echo "procrustes_cascade_weight=$PROCRUSTES_CASCADE_WEIGHT"
        echo "procrustes_cascade_max_svd_dim=$PROCRUSTES_CASCADE_MAX_SVD_DIM"
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

write_manifest

if [[ "$RUN_MRL" == "1" ]]; then
    train_run mrl \
        "${MRL_TRAINING_ARGS[@]}" \
        "${MRL_CONFLICT_TRAINING_ARGS[@]}" \
        --model.mrl=1
    eval_run mrl --mrl
fi

if [[ "$RUN_MRL_PCD" == "1" ]]; then
    train_run mrl_pcd \
        "${MRL_TRAINING_ARGS[@]}" \
        "${MRL_CONFLICT_TRAINING_ARGS[@]}" \
        "${PCD_TRAINING_ARGS[@]}" \
        --model.mrl=1
    eval_run mrl_pcd --mrl
fi

if [[ "$RUN_RECURSIVE_LINK_MRL" == "1" ]]; then
    train_run recursive_link_mrl \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.recursive_link_mrl=1 \
        "${RECURSIVE_LINK_TRAINING_ARGS[@]}"
    eval_run recursive_link_mrl \
        "${RECURSIVE_LINK_EVAL_ARGS[@]}"
fi

if [[ "$RUN_RECURSIVE_LINK_MRL_PCD" == "1" ]]; then
    train_run recursive_link_mrl_pcd \
        "${MRL_TRAINING_ARGS[@]}" \
        "${PCD_TRAINING_ARGS[@]}" \
        --model.recursive_link_mrl=1 \
        "${RECURSIVE_LINK_TRAINING_ARGS[@]}"
    eval_run recursive_link_mrl_pcd \
        "${RECURSIVE_LINK_EVAL_ARGS[@]}"
fi

if [[ "$RUN_MRLE" == "1" ]]; then
    train_run mrle \
        "${MRL_TRAINING_ARGS[@]}" \
        "${MRL_CONFLICT_TRAINING_ARGS[@]}" \
        --model.efficient=1
    eval_run mrle --mrl --efficient
fi

if [[ "$RUN_T_ORTHOGONAL_MRL" == "1" ]]; then
    train_run t_orthogonal_mrl \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.t_orthogonal_mrl=1 \
        --model.t_orthogonal_map="$T_ORTHOGONAL_MAP" \
        --model.bor_mode=orthogonal \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION" \
        --model.bor_stop_gradient="$BOR_STOP_GRADIENT"
    eval_run t_orthogonal_mrl \
        --t_orthogonal_mrl \
        --t_orthogonal_map "$T_ORTHOGONAL_MAP" \
        --bor_mode orthogonal \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
        --bor_stop_gradient "$BOR_STOP_GRADIENT"
fi

if [[ "$RUN_BOR_MRL" == "1" ]]; then
    train_run bor_mrl \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.bor_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map="$BOR_ORTHOGONAL_MAP" \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION" \
        --model.bor_stop_gradient="$BOR_STOP_GRADIENT"
    eval_run bor_mrl \
        --bor_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map "$BOR_ORTHOGONAL_MAP" \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
        --bor_stop_gradient "$BOR_STOP_GRADIENT"
fi

if [[ "$RUN_BOR_MRL_RESIDUAL" == "1" ]]; then
    train_run bor_mrl_residual \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.bor_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map="$BOR_ORTHOGONAL_MAP" \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION" \
        --model.bor_stop_gradient="$BOR_STOP_GRADIENT" \
        --model.bor_residual_orthogonal=1 \
        --model.bor_residual_alpha_init="$BOR_RESIDUAL_ALPHA_INIT"
    eval_run bor_mrl_residual \
        --bor_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map "$BOR_ORTHOGONAL_MAP" \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
        --bor_stop_gradient "$BOR_STOP_GRADIENT" \
        --bor_residual_orthogonal 1 \
        --bor_residual_alpha_init "$BOR_RESIDUAL_ALPHA_INIT"
fi

if [[ "$RUN_BOR_BLOCK_MRL" == "1" ]]; then
    train_run bor_block_mrl \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.bor_block_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map="$BOR_ORTHOGONAL_MAP" \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION" \
        --model.bor_stop_gradient="$BOR_STOP_GRADIENT"
    eval_run bor_block_mrl \
        --bor_block_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map "$BOR_ORTHOGONAL_MAP" \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
        --bor_stop_gradient "$BOR_STOP_GRADIENT"
fi

if [[ "$RUN_CASCADE_STOP_GRADIENT_MRL" == "1" ]]; then
    train_run cascade_stop_gradient_mrl \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.cascade_stop_gradient_mrl=1 \
        --model.cascade_stop_gradient="$CASCADE_STOP_GRADIENT"
    eval_run cascade_stop_gradient_mrl \
        --cascade_stop_gradient_mrl \
        --cascade_stop_gradient "$CASCADE_STOP_GRADIENT"
fi

if [[ "$RUN_BOR_MRL_FROZEN" == "1" ]]; then
    train_run bor_mrl_frozen \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.bor_mrl=1 \
        --model.bor_mode=frozen \
        --model.bor_stop_gradient="$BOR_STOP_GRADIENT"
    eval_run bor_mrl_frozen \
        --bor_mrl \
        --bor_mode frozen \
        --bor_stop_gradient "$BOR_STOP_GRADIENT"
fi

if [[ "$RUN_BOR_MRL_CAYLEY" == "1" ]]; then
    train_run bor_mrl_cayley \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.bor_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map=cayley \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION" \
        --model.bor_stop_gradient="$BOR_STOP_GRADIENT"
    eval_run bor_mrl_cayley \
        --bor_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map cayley \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
        --bor_stop_gradient "$BOR_STOP_GRADIENT"
fi

if [[ "$RUN_BOR_MRL_HOUSEHOLDER" == "1" ]]; then
    train_run bor_mrl_householder \
        "${MRL_TRAINING_ARGS[@]}" \
        --model.bor_mrl=1 \
        --model.bor_mode=orthogonal \
        --model.bor_orthogonal_map=householder \
        --model.bor_use_trivialization="$BOR_USE_TRIVIALIZATION" \
        --model.bor_stop_gradient="$BOR_STOP_GRADIENT"
    eval_run bor_mrl_householder \
        --bor_mrl \
        --bor_mode orthogonal \
        --bor_orthogonal_map householder \
        --bor_use_trivialization "$BOR_USE_TRIVIALIZATION" \
        --bor_stop_gradient "$BOR_STOP_GRADIENT"
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
