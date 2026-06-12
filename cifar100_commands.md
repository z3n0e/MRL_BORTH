# CIFAR-100 Commands

Run these from the project root:

```bash
cd /home/sricci/Desktop/MRL_BORTH
```

## Full Workflow

```bash
./run_cifar100_experiments.sh
```

This trains one standard MRL model, evaluates classification accuracy at every prefix dimension, and optionally computes retrieval metrics. CIFAR-100 data defaults to `$HOME/.cache/torchvision`.

```bash
CIFAR100_DIR=/path/to/cifar100/root ./run_cifar100_experiments.sh
```

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
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.mrl=1 \
  --model.prefix_mask_prob=0.1 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## Train MRL-E

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
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
  --mrl
```

Build nearest-neighbor shortlists and compute metrics:

```bash
cd ../retrieval

python faiss_nn.py \
  --root ../retrieval_arrays \
  --dataset CIFAR100 \
  --model mrl \
  --index-type exactl2 \
  --k 2048

python compute_metrics.py \
  --root ../retrieval_arrays \
  --dataset CIFAR100 \
  --model mrl \
  --eval-config vanilla \
  --index-type exactl2
```

The experiment runner writes retrieval summaries to `cifar100_runs/<run>/cifar100_retrieval_summary.csv`.
