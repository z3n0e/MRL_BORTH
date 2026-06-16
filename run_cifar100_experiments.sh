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

CIFAR100_DIR=${CIFAR100_DIR:-"$HOME/.cache/torchvision"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"$ROOT_DIR/cifar100_runs/cifar100_seed_${SEED}_$(date +%Y%m%d_%H%M%S)"}
WANDB_ENABLED=${WANDB_ENABLED:-1}
WANDB_PROJECT=${WANDB_PROJECT:-mrl-borth}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_GROUP=${WANDB_GROUP:-$(basename "$EXPERIMENT_DIR")}
WANDB_TAGS=${WANDB_TAGS:-cifar100,resnet18,mrl}
WANDB_MODE=${WANDB_MODE:-}
WANDB_DIR=${WANDB_DIR:-"$EXPERIMENT_DIR/wandb"}
export WANDB_ENABLED WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_TAGS WANDB_MODE WANDB_DIR

PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-120}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
EVAL_WORKERS=${EVAL_WORKERS:-4}
NC_WORKERS=${NC_WORKERS:-4}
NC_BATCH_SIZE=${NC_BATCH_SIZE:-128}
LR=${LR:-0.1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
MIN_LR=${MIN_LR:-0.00001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0005}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}

MRL_LOSS_MODE=${MRL_LOSS_MODE:-all}
SAMPLED_PREFIX_DISTRIBUTION=${SAMPLED_PREFIX_DISTRIBUTION:-uniform}
SAMPLED_PREFIX_LOG_INTERVAL=${SAMPLED_PREFIX_LOG_INTERVAL:-100}
PREFIX_MASK_PROB=${PREFIX_MASK_PROB:-0.0}
PREFIX_MASK_SCALE=${PREFIX_MASK_SCALE:-none}
PREFIX_DIMS=${PREFIX_DIMS:-"8,16,32,64,128,256,512"}
FEATURE_DIM=${FEATURE_DIM:-512}
RUN_RETRIEVAL_METRICS=${RUN_RETRIEVAL_METRICS:-1}
RUN_NC_METRICS=${RUN_NC_METRICS:-1}

TRAINLOG_DIR="$EXPERIMENT_DIR/trainlogs"
EVAL_DIR="$EXPERIMENT_DIR/eval"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"

mkdir -p "$CIFAR100_DIR" "$TRAINLOG_DIR" "$EVAL_DIR" "$CHECKPOINT_DIR" "$WANDB_DIR"

echo "CIFAR-100 experiment directory: $EXPERIMENT_DIR"
echo "Seed: $SEED"
echo "Deterministic: $DETERMINISTIC"
echo "CIFAR-100 data root: $CIFAR100_DIR"
echo "MRL loss mode: $MRL_LOSS_MODE"
echo "Prefix mask probability: $PREFIX_MASK_PROB"
echo "Prefix mask scale: $PREFIX_MASK_SCALE"
echo "Feature dim: $FEATURE_DIM"
echo "Prefix dims: $PREFIX_DIMS"
echo "LR: $LR"
echo "Warmup epochs: $WARMUP_EPOCHS"
echo "Min LR: $MIN_LR"
echo "Weight decay: $WEIGHT_DECAY"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "W&B enabled: $WANDB_ENABLED"
echo "W&B project: $WANDB_PROJECT"
echo "W&B group: $WANDB_GROUP"
echo "Run Neural Collapse metrics: $RUN_NC_METRICS"
echo "Run retrieval metrics: $RUN_RETRIEVAL_METRICS"

MRL_TRAINING_ARGS=(
    --training.mrl_loss_mode="$MRL_LOSS_MODE"
    --training.sampled_prefix_distribution="$SAMPLED_PREFIX_DISTRIBUTION"
    --training.sampled_prefix_log_interval="$SAMPLED_PREFIX_LOG_INTERVAL"
    --model.prefix_mask_prob="$PREFIX_MASK_PROB"
    --model.prefix_mask_scale="$PREFIX_MASK_SCALE"
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
        echo "nc_workers=$NC_WORKERS"
        echo "nc_batch_size=$NC_BATCH_SIZE"
        echo "lr=$LR"
        echo "warmup_epochs=$WARMUP_EPOCHS"
        echo "min_lr=$MIN_LR"
        echo "weight_decay=$WEIGHT_DECAY"
        echo "label_smoothing=$LABEL_SMOOTHING"
        echo "mrl_loss_mode=$MRL_LOSS_MODE"
        echo "sampled_prefix_distribution=$SAMPLED_PREFIX_DISTRIBUTION"
        echo "sampled_prefix_log_interval=$SAMPLED_PREFIX_LOG_INTERVAL"
        echo "prefix_mask_prob=$PREFIX_MASK_PROB"
        echo "prefix_mask_scale=$PREFIX_MASK_SCALE"
        echo "feature_dim=$FEATURE_DIM"
        echo "prefix_dims=$PREFIX_DIMS"
        echo "wandb_enabled=$WANDB_ENABLED"
        echo "wandb_project=$WANDB_PROJECT"
        echo "wandb_entity=$WANDB_ENTITY"
        echo "wandb_group=$WANDB_GROUP"
        echo "wandb_tags=$WANDB_TAGS"
        echo "wandb_mode=$WANDB_MODE"
        echo "wandb_dir=$WANDB_DIR"
        echo "run_nc_metrics=$RUN_NC_METRICS"
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
            --config-file rn50_configs/rn18_cifar100.yaml \
            --model.mrl=1 \
            --data.root="$CIFAR100_DIR" \
            --data.num_workers="$NUM_WORKERS" \
            --training.batch_size="$TRAIN_BATCH_SIZE" \
            --training.epochs="$EPOCHS" \
            --training.seed="$SEED" \
            --training.deterministic="$DETERMINISTIC" \
            --lr.lr="$LR" \
            --lr.warmup_epochs="$WARMUP_EPOCHS" \
            --lr.min_lr="$MIN_LR" \
            --training.weight_decay="$WEIGHT_DECAY" \
            --training.label_smoothing="$LABEL_SMOOTHING" \
            --validation.batch_size="$VAL_BATCH_SIZE" \
            --logging.folder="$TRAINLOG_DIR" \
            --logging.run_name="$run_name" \
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
            --dataset CIFAR100 \
            --data_root "$CIFAR100_DIR" \
            --arch resnet18 \
            --rep_size "$FEATURE_DIM" \
            --prefix-dims "$PREFIX_DIMS" \
            --use_blurpool 0 \
            --workers "$EVAL_WORKERS" \
            --seed "$SEED" \
            "${deterministic_args[@]}" \
            --metrics_output "$metrics_output" \
            --mrl
    )
}

run_nc_metrics() {
    if [[ "$RUN_NC_METRICS" != "1" ]]; then
        echo "Skipping Neural Collapse metrics because RUN_NC_METRICS=$RUN_NC_METRICS"
        return
    fi

    local checkpoint="$TRAINLOG_DIR/mrl/final_weights.pt"
    local nc_dir="$EXPERIMENT_DIR/neural_collapse"
    local deterministic_args=()
    if [[ "$DETERMINISTIC" == "1" ]]; then
        deterministic_args=(--deterministic)
    fi

    mkdir -p "$nc_dir"
    echo "Computing CIFAR-100 Neural Collapse metrics..."
    (
        cd "$ROOT_DIR"
        "$PYTHON" cifar100_neural_collapse.py \
            --path "$checkpoint" \
            --data-root "$CIFAR100_DIR" \
            --arch resnet18 \
            --rep-size "$FEATURE_DIM" \
            --prefix-dims "$PREFIX_DIMS" \
            --batch-size "$NC_BATCH_SIZE" \
            --workers "$NC_WORKERS" \
            --seed "$SEED" \
            "${deterministic_args[@]}" \
            --output-csv "$nc_dir/cifar100_nc_metrics.csv" \
            --output-json "$nc_dir/cifar100_nc_metrics.json" \
            --mrl
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
    FEATURE_DIM="$FEATURE_DIM" \
    PREFIX_DIMS="$PREFIX_DIMS" \
        "$ROOT_DIR/run_cifar100_retrieval_metrics.sh" "$EXPERIMENT_DIR"
}

write_manifest
train_mrl
eval_mrl
run_nc_metrics
run_retrieval_metrics

echo "Done."
echo "Metrics JSON files are in: $EVAL_DIR"
echo "Model checkpoints are in: $CHECKPOINT_DIR"
if [[ "$RUN_NC_METRICS" == "1" ]]; then
    echo "Neural Collapse metrics are in: $EXPERIMENT_DIR/neural_collapse"
fi
if [[ "$RUN_RETRIEVAL_METRICS" == "1" ]]; then
    echo "Retrieval metrics JSON files are in: $EXPERIMENT_DIR/retrieval_metrics"
    echo "Retrieval CSV summary is in: $EXPERIMENT_DIR/cifar100_retrieval_summary.csv"
fi
