#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SEED=${SEED:-0}
DETERMINISTIC=1
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

IMAGENET_DIR=${IMAGENET_DIR:-}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"$ROOT_DIR/imagenet_runs/imagenet_seed_${SEED}_$(date +%Y%m%d_%H%M%S)"}
WANDB_ENABLED=${WANDB_ENABLED:-1}
WANDB_PROJECT=MRL_BORTH
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_GROUP=${WANDB_GROUP:-$(basename "$EXPERIMENT_DIR")}
WANDB_TAGS=${WANDB_TAGS:-imagenet,mrl}
WANDB_MODE=${WANDB_MODE:-}
WANDB_DIR=${WANDB_DIR:-"$EXPERIMENT_DIR/wandb"}
export WANDB_ENABLED WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_TAGS WANDB_MODE WANDB_DIR

if [[ -z "$IMAGENET_DIR" ]]; then
    echo "Set IMAGENET_DIR to the ImageNet root containing train/ and val/."
    echo "Example: IMAGENET_DIR=/path/to/imagenet $0"
    exit 2
fi

if [[ ! -d "$IMAGENET_DIR/train" || ! -d "$IMAGENET_DIR/val" ]]; then
    echo "Expected ImageNet folders at:"
    echo "  $IMAGENET_DIR/train"
    echo "  $IMAGENET_DIR/val"
    exit 1
fi

PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-40}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-12}
EVAL_WORKERS=${EVAL_WORKERS:-12}

MRL_LOSS_MODE=${MRL_LOSS_MODE:-all}
SAMPLED_PREFIX_DISTRIBUTION=${SAMPLED_PREFIX_DISTRIBUTION:-uniform}
SAMPLED_PREFIX_LOG_INTERVAL=${SAMPLED_PREFIX_LOG_INTERVAL:-100}
PREFIX_MASK_PROB=${PREFIX_MASK_PROB:-0.0}
PREFIX_MASK_SCALE=${PREFIX_MASK_SCALE:-none}
PREFIX_MASK_SCOPE=${PREFIX_MASK_SCOPE:-batch}
PREFIX_MASK_SKIP_PROB=${PREFIX_MASK_SKIP_PROB:-0.0}
RUN_RETRIEVAL_METRICS=${RUN_RETRIEVAL_METRICS:-1}

TRAINLOG_DIR="$EXPERIMENT_DIR/trainlogs"
EVAL_DIR="$EXPERIMENT_DIR/eval"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"

mkdir -p "$TRAINLOG_DIR" "$EVAL_DIR" "$CHECKPOINT_DIR" "$WANDB_DIR"

echo "ImageNet experiment directory: $EXPERIMENT_DIR"
echo "Seed: $SEED"
echo "Deterministic: $DETERMINISTIC"
echo "ImageNet data root: $IMAGENET_DIR"
echo "W&B enabled: $WANDB_ENABLED"
echo "W&B project: $WANDB_PROJECT"
echo "W&B group: $WANDB_GROUP"
echo "MRL loss mode: $MRL_LOSS_MODE"
echo "Prefix mask probability: $PREFIX_MASK_PROB"
echo "Prefix mask scale: $PREFIX_MASK_SCALE"
echo "Prefix mask scope: $PREFIX_MASK_SCOPE"
echo "Prefix mask skip probability: $PREFIX_MASK_SKIP_PROB"
echo "Run retrieval metrics: $RUN_RETRIEVAL_METRICS"

MRL_TRAINING_ARGS=(
    --training.mrl_loss_mode="$MRL_LOSS_MODE"
    --training.sampled_prefix_distribution="$SAMPLED_PREFIX_DISTRIBUTION"
    --training.sampled_prefix_log_interval="$SAMPLED_PREFIX_LOG_INTERVAL"
    --model.prefix_mask_prob="$PREFIX_MASK_PROB"
    --model.prefix_mask_scale="$PREFIX_MASK_SCALE"
    --model.prefix_mask_scope="$PREFIX_MASK_SCOPE"
    --model.prefix_mask_skip_prob="$PREFIX_MASK_SKIP_PROB"
)

write_manifest() {
    {
        echo "experiment_dir=$EXPERIMENT_DIR"
        echo "seed=$SEED"
        echo "deterministic=$DETERMINISTIC"
        echo "imagenet_dir=$IMAGENET_DIR"
        echo "python=$PYTHON"
        echo "epochs=$EPOCHS"
        echo "train_batch_size=$TRAIN_BATCH_SIZE"
        echo "val_batch_size=$VAL_BATCH_SIZE"
        echo "num_workers=$NUM_WORKERS"
        echo "eval_workers=$EVAL_WORKERS"
        echo "wandb_enabled=$WANDB_ENABLED"
        echo "wandb_project=$WANDB_PROJECT"
        echo "wandb_entity=$WANDB_ENTITY"
        echo "wandb_group=$WANDB_GROUP"
        echo "wandb_tags=$WANDB_TAGS"
        echo "wandb_mode=$WANDB_MODE"
        echo "wandb_dir=$WANDB_DIR"
        echo "mrl_loss_mode=$MRL_LOSS_MODE"
        echo "sampled_prefix_distribution=$SAMPLED_PREFIX_DISTRIBUTION"
        echo "sampled_prefix_log_interval=$SAMPLED_PREFIX_LOG_INTERVAL"
        echo "prefix_mask_prob=$PREFIX_MASK_PROB"
        echo "prefix_mask_scale=$PREFIX_MASK_SCALE"
        echo "prefix_mask_scope=$PREFIX_MASK_SCOPE"
        echo "prefix_mask_skip_prob=$PREFIX_MASK_SKIP_PROB"
        echo "run_retrieval_metrics=$RUN_RETRIEVAL_METRICS"
    } > "$EXPERIMENT_DIR/manifest.txt"
}

train_mrl() {
    local run_name=mrl
    local run_dir="$TRAINLOG_DIR/$run_name"
    if [[ -e "$run_dir" ]]; then
        echo "Run directory already exists: $run_dir"
        echo "Use a new EXPERIMENT_DIR or remove the existing run directory."
        exit 1
    fi

    echo "Training MRL..."
    (
        cd "$ROOT_DIR/train"
        "$PYTHON" train_imagenet.py \
            --config-file rn50_configs/rn50_40_epochs.yaml \
            --data.dataset=imagenet \
            --data.root="$IMAGENET_DIR" \
            --data.num_workers="$NUM_WORKERS" \
            --training.batch_size="$TRAIN_BATCH_SIZE" \
            --training.epochs="$EPOCHS" \
            --training.seed="$SEED" \
            --training.deterministic="$DETERMINISTIC" \
            --validation.batch_size="$VAL_BATCH_SIZE" \
            --logging.folder="$TRAINLOG_DIR" \
            --logging.run_name="$run_name" \
            --model.mrl=1 \
            "${MRL_TRAINING_ARGS[@]}"
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

eval_mrl() {
    local checkpoint="$TRAINLOG_DIR/mrl/final_weights.pt"
    local metrics_output="$EVAL_DIR/mrl.json"
    local deterministic_args=()
    if [[ "$DETERMINISTIC" == "1" ]]; then
        deterministic_args=(--deterministic)
    fi

    echo "Evaluating MRL..."
    (
        cd "$ROOT_DIR/inference"
        "$PYTHON" pytorch_inference.py \
            --path "$checkpoint" \
            --dataset 1K \
            --data_root "$IMAGENET_DIR" \
            --workers "$EVAL_WORKERS" \
            --seed "$SEED" \
            "${deterministic_args[@]}" \
            --metrics_output "$metrics_output" \
            --mrl
    )
}

run_retrieval_metrics() {
    if [[ "$RUN_RETRIEVAL_METRICS" != "1" ]]; then
        echo "Skipping retrieval metrics because RUN_RETRIEVAL_METRICS=$RUN_RETRIEVAL_METRICS"
        return
    fi

    echo "Computing retrieval metrics for this ImageNet run..."
    IMAGENET_DIR="$IMAGENET_DIR" \
    PYTHON="$PYTHON" \
    SEED="$SEED" \
    DETERMINISTIC="$DETERMINISTIC" \
    EVAL_WORKERS="$EVAL_WORKERS" \
        "$ROOT_DIR/run_imagenet_retrieval_metrics.sh" "$EXPERIMENT_DIR"
}

write_manifest
train_mrl
eval_mrl
run_retrieval_metrics

echo "Done."
echo "Metrics JSON files are in: $EVAL_DIR"
echo "Model checkpoints are in: $CHECKPOINT_DIR"
if [[ "$RUN_RETRIEVAL_METRICS" == "1" ]]; then
    echo "Retrieval metrics JSON files are in: $EXPERIMENT_DIR/retrieval_metrics"
    echo "Retrieval CSV summary is in: $EXPERIMENT_DIR/imagenet_retrieval_summary.csv"
fi
