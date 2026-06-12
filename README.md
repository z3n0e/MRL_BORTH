# Matryoshka Representation Learning

This repository trains and evaluates standard Matryoshka Representation Learning (MRL) models for ImageNet and CIFAR-100. The active code path is intentionally small: standard MRL, MRL-E compatibility, fixed-feature compatibility for older checkpoints/evaluations, classification metrics, and retrieval metrics.

## Setup

```bash
pip3 install -r requirements.txt
```

ImageNet training expects the standard TorchVision `ImageFolder` layout:

```text
imagenet/
  train/
    class_1/
    class_2/
  val/
    class_1/
    class_2/
```

CIFAR-100 uses `torchvision.datasets.CIFAR100` and downloads automatically when needed.

## MRL Head

The ResNet classifier is replaced with the MRL linear head in [MRL.py](/home/sricci/Desktop/MRL_BORTH/MRL.py):

```python
nesting_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
fc_layer = MRL_Linear_Layer(nesting_list, num_classes=1000, efficient=False)
```

Use `efficient=True` for MRL-E, which shares one classifier over all prefixes.

## Prefix Masking

CIFAR-100 and ImageNet training support the same training-only inherited-prefix masking used in the MNIST neural-collapse experiments. For each larger MRL head, the already-seen prefix coordinates are randomly masked while the newly added coordinate block remains visible.

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.mrl=1 \
  --model.prefix_mask_prob=0.1 \
  --model.prefix_mask_scale=inverted
```

The default is `--model.prefix_mask_prob=0.0` to preserve baseline accuracy unless you opt in.

## Training

### CIFAR-100

```bash
./run_cifar100_experiments.sh
```

Useful overrides:

```bash
CIFAR100_DIR=/path/to/cifar100 \
PREFIX_MASK_PROB=0.1 \
./run_cifar100_experiments.sh
```

### ImageNet

```bash
IMAGENET_DIR=/path/to/imagenet ./run_imagenet_experiments.sh
```

The training entry point uses standard PyTorch speed features: AMP on CUDA, channels-last tensors on CUDA, pinned/persistent dataloader workers, nonblocking transfers, cuDNN benchmarking when deterministic mode is off, TF32 when allowed, and SGD `foreach` when available. It does not use `torch.compile`.

## Inference

Evaluate classification accuracy:

```bash
cd inference

python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100 \
  --mrl
```

For ImageNet:

```bash
python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset 1K \
  --data_root /path/to/imagenet \
  --mrl
```

Add `--efficient` for MRL-E checkpoints. Fixed-feature compatibility remains available with `--rep_size <dim>`.

## Retrieval

Retrieval uses the raw ResNet avgpool features and evaluates nearest-neighbor performance at MRL prefix dimensions. CIFAR-100 uses train as the database and test as the query split; ImageNet uses train as the database and val as the query split.

After running one of the experiment scripts, retrieval summaries are written to:

```text
cifar100_runs/.../cifar100_retrieval_summary.csv
imagenet_runs/.../imagenet_retrieval_summary.csv
```

Manual retrieval flow:

```bash
cd inference

python pytorch_inference.py \
  --retrieval \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100 \
  --retrieval_array_path ../retrieval_arrays \
  --mrl

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

## Tests

```bash
python -m pytest tests/test_MRL.py
```

## Citation

```text
@inproceedings{kusupati2022matryoshka,
  title     = {Matryoshka Representation Learning},
  author    = {Kusupati, Aditya and Bhatt, Gantavya and Rege, Aniket and Wallingford, Matthew and Sinha, Aditya and Ramanujan, Vivek and Howard-Snyder, William and Chen, Kaifeng and Kakade, Sham and Jain, Prateek and others},
  booktitle = {Advances in Neural Information Processing Systems},
  month     = {December},
  year      = {2022},
}
```
