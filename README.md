# Matryoshka Representation Learning

This repository trains and evaluates standard Matryoshka Representation Learning (MRL) models for ImageNet and CIFAR-100. The active code path is intentionally small: standard MRL, MRL-E compatibility, fixed-feature compatibility for older checkpoints/evaluations, classification metrics, and retrieval metrics.

## Setup

```bash
pip3 install -r requirements.txt
```

W&B logging is enabled by default. Install `wandb` in the same environment, or set `WANDB_ENABLED=0` to disable the W&B side-channel.

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

The ResNet classifier is replaced with the MRL linear head in [MRL.py](/home/sricci/Desktop/MRL_BORTH/MRL.py). ImageNet/ResNet-50 uses:

```python
nesting_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
fc_layer = MRL_Linear_Layer(nesting_list, num_classes=1000, efficient=False)
```

For CIFAR-100/ResNet-18, the feature dimension is 512 and the default prefixes are:

```python
nesting_list = [8, 16, 32, 64, 128, 256, 512]
```

Use `efficient=True` for MRL-E, which shares one classifier over all prefixes.

## Prefix Masking

CIFAR-100 and ImageNet training support the same training-only inherited-prefix masking used in the MNIST neural-collapse experiments. For each larger MRL head, the already-seen prefix coordinates are randomly masked while the newly added coordinate block remains visible.

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn18_cifar100.yaml \
  --model.mrl=1 \
  --model.prefix_mask_prob=0.1 \
  --model.prefix_mask_scale=none
```

The defaults are `--model.prefix_mask_prob=0.0` and `--model.prefix_mask_scale=none`.

## Training

### CIFAR-100

Recommended unified CLI:

```bash
python main.py full
```

Run individual stages when needed:

```bash
python main.py train
python main.py eval --run-dir cifar100_runs/<run_id>
python main.py nc --run-dir cifar100_runs/<run_id>
python main.py retrieval --run-dir cifar100_runs/<run_id>
```

The older scripts remain supported:

```bash
./run_cifar100_experiments.sh
```

These commands use TorchVision ResNet-18 with the CIFAR stem (`3x3` stride-1 `conv1`, no initial maxpool), CIFAR-100 augmentations, 512-dimensional pooled features, prefixes `[8, 16, 32, 64, 128, 256, 512]`, SGD/Nesterov, cosine LR with 5 warmup epochs, and label smoothing `0.1`.

Useful overrides:

```bash
CIFAR100_DIR=/path/to/cifar100 \
PREFIX_MASK_PROB=0.1 \
./run_cifar100_experiments.sh
```

W&B logging is additive to the existing logs and is on by default:

```bash
WANDB_PROJECT=mrl-borth \
./run_cifar100_experiments.sh
```

Disable W&B with `WANDB_ENABLED=0`.

The full runner groups separate W&B runs for training, classification eval, Neural Collapse, and retrieval under one `WANDB_GROUP`. Metrics are namespaced as `train/*`, `eval/*`, `classification/*`, `nc/*`, and `retrieval/*`.

### W&B Sweep

```bash
wandb sweep sweeps/cifar100_resnet18_sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

W&B sweep configs are captured when the sweep is created. If you change this YAML, create a new sweep ID before starting another agent.

The sweep runs the full CIFAR-100 experiment wrapper: train, classification eval, training-time Neural Collapse/GNC every 10 completed epochs, and final retrieval after training. It optimizes `eval/top1/dim_512` and logs `train/*`, `eval/*`, `classification/*`, `nc/*`, `gnc/*`, `nc_history/*`, `gnc_history/*`, and `retrieval/*` into the same W&B sweep run. Edit `data-root` in [sweeps/cifar100_resnet18_sweep.yaml](/home/sricci/Desktop/MRL_BORTH/sweeps/cifar100_resnet18_sweep.yaml:1) if your CIFAR-100 cache lives somewhere else.
The sweep now calls `python main.py full`, so sweep parameters use the same hyphenated CLI names as the manual workflow.

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
  --arch resnet18 \
  --rep_size 512 \
  --prefix-dims 8,16,32,64,128,256,512 \
  --use_blurpool 0 \
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
  --arch resnet18 \
  --rep_size 512 \
  --prefix-dims 8,16,32,64,128,256,512 \
  --use_blurpool 0 \
  --mrl

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

Retrieval metrics include mAP@k, precision@k, recall@k, top-k accuracy, and CMC@1/CMC@5.

## Neural Collapse

CIFAR-100 runs measure a compact NC/GNC metric set from the local `neural_collapse` package:

- `nc1_within_to_between`
- `nc2_etf_error`
- `nc3_alignment`
- `nc4_ncc_mismatch`
- `gnc2_weight_margin`
- `gnc2_class_mean_margin`
- `gnc3_alignment_error`
- `effective_rank`
- `class_mean_effective_rank`

```bash
python cifar100_neural_collapse.py \
  --path train/trainlogs/<run_id>/final_weights.pt \
  --data-root /path/to/cifar100 \
  --arch resnet18 \
  --rep-size 512 \
  --prefix-dims 8,16,32,64,128,256,512 \
  --output-csv cifar100_nc_metrics.csv \
  --mrl
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
