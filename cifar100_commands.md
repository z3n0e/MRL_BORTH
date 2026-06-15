# CIFAR-100 Commands

Run these from the project root:

```bash
cd /home/sricci/Desktop/MRL_BORTH
```

## Full Workflow

Preferred unified CLI:

```bash
python main.py full
```

Run individual stages:

```bash
python main.py train
python main.py eval --run-dir cifar100_runs/<run_id>
python main.py nc --run-dir cifar100_runs/<run_id>
python main.py retrieval --run-dir cifar100_runs/<run_id>
```

The older wrapper is still supported:

```bash
./run_cifar100_experiments.sh
```

This trains one standard MRL ResNet-18-CIFAR model, evaluates classification accuracy at every prefix dimension, measures Neural Collapse metrics, and optionally computes retrieval metrics. CIFAR-100 data defaults to `$HOME/.cache/torchvision`.

```bash
CIFAR100_DIR=/path/to/cifar100/root ./run_cifar100_experiments.sh
```

W&B logging is enabled by default and is additive to the existing local logs:

```bash
WANDB_PROJECT=mrl-borth ./run_cifar100_experiments.sh
```

Disable W&B with `WANDB_ENABLED=0`.

Training, classification eval, Neural Collapse, and retrieval are separate W&B jobs in one group. Metrics are namespaced as `train/*`, `eval/*`, `classification/*`, `nc/*`, and `retrieval/*`.

## W&B Sweep

```bash
wandb sweep sweeps/cifar100_resnet18_sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

The sweep runs `python main.py full`: train, classification eval, Neural Collapse/GNC, and retrieval. The objective is `eval/top1/dim_512`. Update `data-root` in `sweeps/cifar100_resnet18_sweep.yaml` if needed.

## Prefix Masking

The CIFAR/ImageNet trainer supports MNIST-style inherited-prefix masking:

```bash
PREFIX_MASK_PROB=0.1 PREFIX_MASK_SCALE=inverted ./run_cifar100_experiments.sh
```

The default is `PREFIX_MASK_PROB=0.0`.

## Train MRL Directly

```bash
cd train

python train_imagenet.py \
  --config-file rn50_configs/rn18_cifar100.yaml \
  --model.mrl=1 \
  --model.prefix_mask_prob=0.1 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## Train MRL-E

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn18_cifar100.yaml \
  --model.efficient=1 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## Evaluate Classification

```bash
cd ../inference

python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --arch resnet18 \
  --rep_size 512 \
  --prefix-dims 8,16,32,64,128,256,512 \
  --use_blurpool 0 \
  --mrl
```

Add `--efficient` for MRL-E checkpoints.

## Retrieval Metrics

Dump train/test feature arrays:

```bash
cd ../inference

python pytorch_inference.py \
  --retrieval \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --retrieval_array_path ../retrieval_arrays \
  --arch resnet18 \
  --rep_size 512 \
  --prefix-dims 8,16,32,64,128,256,512 \
  --use_blurpool 0 \
  --mrl
```

Build nearest-neighbor shortlists and compute metrics:

```bash
cd ../retrieval

python faiss_nn.py \
  --root ../retrieval_arrays \
  --dataset CIFAR100 \
  --model mrl \
  --feature-config mrl1_e0_ff512 \
  --rep-size 512 \
  --index-type exactl2 \
  --k 2048 \
  --dims 8 16 32 64 128 256 512

python compute_metrics.py \
  --root ../retrieval_arrays \
  --dataset CIFAR100 \
  --model mrl \
  --feature-config mrl1_e0_ff512 \
  --rep-size 512 \
  --eval-config vanilla \
  --index-type exactl2 \
  --dims 8 16 32 64 128 256 512
```

The experiment runner writes retrieval summaries to `cifar100_runs/<run>/cifar100_retrieval_summary.csv`. Retrieval JSON and CSV summaries include `cmc@1` and `cmc@5` in addition to the existing mAP, precision, recall, and top-k fields.

## Neural Collapse Metrics

The compact NC/GNC output focuses on `nc1_within_to_between`, `nc2_etf_error`, `nc3_alignment`, `nc4_ncc_mismatch`, `gnc2_weight_margin`, `gnc2_class_mean_margin`, `gnc3_alignment_error`, and `effective_rank`. Compatibility aliases such as `nc1`, `nc3_align_mean`, and `nc4_ncc_mismatch` are retained in CSV/JSON rows.

```bash
python cifar100_neural_collapse.py \
  --path train/trainlogs/<run_id>/final_weights.pt \
  --data-root /path/to/cifar100/root \
  --arch resnet18 \
  --rep-size 512 \
  --prefix-dims 8,16,32,64,128,256,512 \
  --output-csv cifar100_nc_metrics.csv \
  --mrl
```

The full experiment runner writes these rows to `cifar100_runs/<run>/neural_collapse/cifar100_nc_metrics.csv`.
