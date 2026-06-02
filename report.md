# Project Report: Matryoshka Representation Learning

## Overview

This repository implements Matryoshka Representation Learning (MRL) for image classification and retrieval, centered on a ResNet-50 backbone. The key idea is to train one representation so that multiple prefixes of the feature vector, such as 8, 16, 32, ..., 2048 dimensions, are independently useful. At inference time, the same model can be evaluated at different representation sizes to trade accuracy for compute.

The original project is ImageNet-oriented, with PyTorch inference scripts, analysis notebooks, and retrieval notebooks. This version also supports CIFAR-100 as a smaller 100-class dataset while keeping ImageNet as the default behavior. Training now uses standard PyTorch and TorchVision dataloaders.

## Core Method

`MRL.py` contains the method-specific components.

- `Matryoshka_CE_Loss` computes a sum of cross-entropy losses over all nested classifier outputs. Each nesting level receives the same target label, preserving the original classification objective at multiple feature prefixes.
- `MRL_Linear_Layer` replaces the standard final classifier with classifiers over nested feature dimensions. In normal MRL mode it creates one head per nesting size. In efficient MRL-E mode it uses one shared classifier and slices its weights for each feature prefix.
- `FixedFeatureLayer` is the fixed-representation baseline. It classifies using only the first `in_features` dimensions of the backbone representation.

The method itself is unchanged for CIFAR-100. The dataset change only affects class count, normalization, and dataset loading.

## Training Pipeline

Training code lives in `train/`.

- `train/train_imagenet.py` is the main training entry point. It builds TorchVision datasets and PyTorch dataloaders, creates the ResNet model, swaps in the MRL or fixed-feature head, trains with SGD and AMP, evaluates top-1/top-5 accuracy, and writes logs/checkpoints.
- `train/rn50_configs/rn50_40_epochs.yaml` is the original ImageNet ResNet-50 configuration.
- `train/rn50_configs/rn50_cifar100.yaml` is the CIFAR-100 ResNet-50 configuration.
- `run_cifar100_experiments.sh` is an all-in-one deterministic CIFAR-100 runner for training and evaluation.
- `cifar100_results.ipynb` visualizes CIFAR-100 training logs and evaluation metrics.

The new training flag is:

```bash
--data.dataset=cifar100
--data.root=/path/to/dataset/root
```

When this flag is used, training switches to:

- 100 output classes
- CIFAR-100 mean and standard deviation
- TorchVision CIFAR-100 train/test datasets

When the flag is omitted, the trainer still uses ImageNet defaults: 1000 classes, ImageNet normalization, and `ImageFolder` datasets under `data.root/train` and `data.root/val`.

## Training Variants

The project now supports these training variants:

- Standard MRL
- MRL-E
- Full-feature baseline
- Fixed-feature baseline
- BOR-MRL identity
- BOR-MRL matrix_exp
- BOR-MRL cayley
- BOR-MRL householder
- Independent-block BOR-MRL

BOR-MRL builds each larger representation by applying an orthogonal transform to the previously trained prefix and concatenating the next raw residual coordinate block. With `[8, 16, 32]`, the classifiers see `h[:8]`, then `[Q_8 h[:8], h[8:16]]`, then `[Q_16 z_16, h[16:32]]`. Training still uses the same Matryoshka cross-entropy over independent linear softmax classifiers. No results are claimed here until the corresponding experiments are run.

Independent-block BOR-MRL keeps the older behavior as a separate method: it transforms each residual block independently, concatenates the transformed blocks, and applies the same MRL softmax classifiers to transformed prefixes.

## CIFAR-100 Usage

From the project root, run the deterministic CIFAR-100 workflow:

```bash
CIFAR100_DIR=/path/to/cifar100/root ./run_cifar100_experiments.sh
```

TorchVision downloads CIFAR-100 automatically when it is missing.

Train an MRL model:

```bash
cd train

python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.mrl=1 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

Train MRL-E by replacing `--model.mrl=1` with:

```bash
--model.efficient=1
```

Train a fixed-feature baseline by omitting MRL flags and optionally setting:

```bash
--model.fixed_feature=512
```

## Inference

Inference code is in `inference/pytorch_inference.py`.

It supports standard ImageNet validation, ImageNetV2, ImageNet-A, ImageNet-R, ImageNet-Sketch, and now CIFAR-100. CIFAR-100 inference uses 100 output classes and CIFAR-100 normalization.

Example CIFAR-100 evaluation:

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

Retrieval utilities are split between `inference/pytorch_inference.py`, `utils.py`, and notebooks in `retrieval/`.

- `generate_retrieval_data` collects feature vectors from the model's average-pooling layer and writes NumPy arrays.
- `retrieval/faiss_nn.ipynb` and `retrieval/faiss_nn.py` build FAISS nearest-neighbor indexes.
- `retrieval/reranking.ipynb` performs adaptive reranking with larger representation sizes.
- `retrieval/compute_metrics.ipynb` and `retrieval/compute_metrics.py` compute mAP, precision, recall, and top-k retrieval accuracy. Recall now derives the number of relevant database images from the labels, so the same code supports ImageNet and CIFAR-100.

CIFAR-100 is now available for retrieval array generation through:

```bash
python pytorch_inference.py \
  --retrieval \
  --path <checkpoint.pt> \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --retrieval_array_path <output_dir>/ \
  --mrl
```

## Model Analysis

The `model_analysis/` directory contains notebooks for deeper inspection:

- `GradCAM.ipynb` visualizes class attribution and confusion at different representation sizes.
- `Model_Cascades.ipynb` studies adaptive classification by confidence thresholds.
- `Custom_SuperClass_Performance.ipynb` evaluates behavior over WordNet-derived superclasses.
- `Oracle_Upper_Bound_Performance.ipynb` estimates the best possible per-example representation-size selection.
- `mapping.py`, `imagenet_id.py`, and `imgnt_meta/` provide ImageNet label and hierarchy metadata.

Most analysis notebooks are ImageNet-specific because they rely on WordNet and ImageNet class mappings.

## Utilities

`utils.py` contains shared evaluation and retrieval helpers.

- `evaluate_model_ff` evaluates a fixed-feature classifier.
- `evaluate_model_nesting` evaluates all MRL nesting sizes.
- `apply_blurpool` replaces strided convolutions with blur-pool wrappers.
- `get_ckpt` now loads both DDP checkpoints with `module.` prefixes and regular single-process checkpoints.
- `load_from_old_ckpt` supports legacy MRL checkpoint head layouts and now accepts `num_classes`.

## Tests

`tests/test_MRL.py` validates the Matryoshka cross-entropy loss:

- scalar loss output
- optional relative importance weights
- custom classifier output class counts, including the 100-class CIFAR-100 case
- equality with the original loop-based implementation

The tests focus on method correctness rather than end-to-end training.

## Data And Behavior Notes

ImageNet remains the default throughout the project. Existing ImageNet commands, class counts, normalization, and checkpoint behavior are preserved unless `--data.dataset=cifar100` or `--dataset CIFAR100` is explicitly selected.

CIFAR-100 support is intentionally narrow and method-preserving:

- no change to `Matryoshka_CE_Loss`
- no change to nesting dimensions
- no change to MRL versus MRL-E semantics
- no change to fixed-feature semantics
- only dataset-specific metadata, normalization, and loaders are changed

This makes CIFAR-100 a smaller-scale experimental substitute for ImageNet without changing what the MRL method is optimizing.

## Reproducibility

CIFAR-100 experiments are configured for deterministic runs through `training.seed` and `training.deterministic`. The runner script exports `PYTHONHASHSEED`, sets CUDA deterministic workspace configuration, and passes the same seed to training and evaluation. The trainer seeds Python, NumPy, PyTorch, CUDA, and dataloader workers.

Example:

```bash
SEED=0 ./run_cifar100_experiments.sh
```
