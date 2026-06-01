# CIFAR-100 Commands

Run these commands from the project root:

```bash
cd /home/sricci/Desktop/MRL_BORTH
```

## 1. Run The Full CIFAR-100 Workflow

```bash
./run_cifar100_experiments.sh
```

The script trains and evaluates MRL, MRL-E, the full-feature baseline, and fixed-feature baselines on CIFAR-100. Results are written under `cifar100_runs/`, and `cifar100_results.ipynb` visualizes the outputs.

The default CIFAR-100 data root is `$HOME/.cache/torchvision`. Override it with:

```bash
CIFAR100_DIR=/path/to/cifar100/root ./run_cifar100_experiments.sh
```

TorchVision downloads CIFAR-100 automatically when it is missing.

## 2. Train MRL

```bash
cd train

python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.mrl=1 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## 3. Train MRL-E

MRL-E is the efficient version of MRL with one shared nested classifier.

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.efficient=1 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## 4. Train Full-Feature Baseline

This is the standard 2048-dimensional ResNet classifier without MRL.

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## 5. Train Fixed-Feature Baseline

Example with 512 feature dimensions:

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.fixed_feature=512 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

You can replace `512` with another representation size, for example:

```text
8, 16, 32, 64, 128, 256, 512, 1024, 2048
```

## 6. Evaluate MRL

Replace `<run_id>` with the folder created under `train/trainlogs/`.

```bash
cd ../inference

python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --mrl
```

## 7. Evaluate MRL-E

```bash
python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --mrl \
  --efficient
```

## 8. Evaluate Fixed-Feature Baseline

Example with 512 feature dimensions:

```bash
python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --rep_size 512
```

## 9. Evaluate Full-Feature Baseline

Use `--rep_size 2048` for the full 2048-dimensional classifier:

```bash
python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --rep_size 2048
```

## Notes

- ImageNet remains the default dataset. CIFAR-100 is selected by the config file during training and by `--dataset CIFAR100` during inference.
- The MRL method is unchanged; only dataset metadata, class count, normalization, and data loading change for CIFAR-100.
- Training checkpoints are saved as `final_weights.pt` inside a run-specific folder under `trainlogs`.
