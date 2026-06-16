# Project Report: Matryoshka Representation Learning

## Overview

This repository implements standard Matryoshka Representation Learning (MRL) for image classification and retrieval with a ResNet-50 backbone. One model is trained so that feature prefixes such as 8, 16, 32, ..., 2048 dimensions are useful for both classification and nearest-neighbor retrieval.

CIFAR-100 and ImageNet share the same training code. CIFAR-100 changes the dataset, class count, image transforms, and normalization; the MRL objective remains the same.

## Core Code

`MRL.py` contains the active method components:

- `Matryoshka_CE_Loss` computes cross-entropy over all prefix classifier outputs or samples one prefix per batch.
- `MRL_Linear_Layer` replaces the final classifier with nested prefix classifiers. It also supports MRL-E through a shared classifier.
- `mask_previous_prefix_features` implements training-only inherited-prefix masking, matching the MNIST neural-collapse masking idea.
- `FixedFeatureLayer` remains for compatibility with older fixed-prefix checkpoints and evaluations.

## Prefix Masking

For a larger prefix `h[:d_i]`, masking is applied only to the inherited coordinates `h[:d_{i-1}]`. The newly introduced block `h[d_{i-1}:d_i]` remains visible. This encourages each larger head to make use of new coordinates without changing evaluation-time representations.

Example:

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.mrl=1 \
  --model.prefix_mask_prob=0.1 \
  --model.prefix_mask_scale=none
```

The default mask probability is `0.0`, and the default mask scale is `none`.

## Training

Training code lives in `train/`.

- `train/train_imagenet.py` builds TorchVision datasets, creates the ResNet model, swaps in the MRL head, trains with SGD and AMP, evaluates top-1/top-5 accuracy, and writes logs/checkpoints.
- `train/rn50_configs/rn50_40_epochs.yaml` is the ImageNet ResNet-50 configuration.
- `train/rn50_configs/rn50_cifar100.yaml` is the CIFAR-100 ResNet-50 configuration.
- `run_cifar100_experiments.sh` and `run_imagenet_experiments.sh` run training, classification evaluation, and optional retrieval metrics for one MRL checkpoint.

The trainer uses standard PyTorch speed tools: AMP on CUDA, channels-last tensors on CUDA, pinned and persistent dataloader workers, nonblocking transfers, cuDNN benchmarking when deterministic mode is off, TF32 when allowed, and SGD `foreach` when available. It does not use `torch.compile`.

## CIFAR-100 Usage

```bash
CIFAR100_DIR=/path/to/cifar100/root ./run_cifar100_experiments.sh
```

TorchVision downloads CIFAR-100 automatically when it is missing.

## ImageNet Usage

```bash
IMAGENET_DIR=/path/to/imagenet ./run_imagenet_experiments.sh
```

The ImageNet directory must contain `train/` and `val/` subdirectories.

## Inference

Classification evaluation:

```bash
cd inference

python pytorch_inference.py \
  --path <path-to-final_weights.pt> \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --mrl
```

The same script can save predictions, logits, softmax probabilities, and ground-truth labels for later analysis.

## Retrieval

Retrieval uses raw ResNet avgpool features. MRL retrieval metrics are computed by slicing those arrays to each prefix dimension, building nearest-neighbor shortlists, and evaluating mAP, precision, recall, and top-k accuracy.

CIFAR-100 uses train as the database and test as the query split:

```bash
cd inference
python pytorch_inference.py --retrieval --path <checkpoint.pt> \
  --dataset CIFAR100 --data_root /path/to/cifar100/root \
  --retrieval_array_path ../retrieval_arrays --mrl

cd ../retrieval
python faiss_nn.py --root ../retrieval_arrays --dataset CIFAR100 --model mrl \
  --index-type exactl2 --k 2048 --dims 8 16 32 64 128 256 512 1024 2048

python compute_metrics.py --root ../retrieval_arrays --dataset CIFAR100 --model mrl \
  --eval-config vanilla --index-type exactl2 --shortlist 10 25 50 100
```

The experiment scripts write CSV summaries under each run directory.

## Tests

`tests/test_MRL.py` validates MRL loss behavior, prefix masking, classifier output shapes, fixed-feature compatibility, and retrieval metrics.
