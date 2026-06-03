# CIFAR-100 Commands

Run these commands from the project root:

```bash
cd /home/sricci/Desktop/MRL_BORTH
```

## 1. Run The Full CIFAR-100 Workflow

```bash
./run_cifar100_experiments.sh
```

The script trains and evaluates MRL, MRL-E, recursive-prefix BOR-MRL, independent-block BOR-MRL, BOR-MRL map ablations, the full-feature baseline, and fixed-feature baselines on CIFAR-100. Results are written under `cifar100_runs/`, and `cifar100_results.ipynb` visualizes the outputs.

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

## 6. Train BOR-MRL

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.bor_mrl=1 \
  --model.bor_mode=orthogonal \
  --model.bor_orthogonal_map=matrix_exp \
  --model.bor_use_trivialization=1 \
  --model.bor_stop_gradient=0 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## 7. Train BOR-MRL Frozen Ablation

```bash
  python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.bor_mrl=1 \
  --model.bor_mode=frozen \
  --model.bor_stop_gradient=0 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## 8. Train Independent-Block BOR-MRL

This is the earlier variant where each residual block is transformed independently before the MRL classifiers see transformed prefixes.

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.bor_block_mrl=1 \
  --model.bor_mode=orthogonal \
  --model.bor_orthogonal_map=matrix_exp \
  --model.bor_use_trivialization=1 \
  --model.bor_stop_gradient=0 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## 8a. Train Cascade Stop-Gradient MRL

This is the no-rotation ablation: each larger prefix is `[sg(z_old), h_new]` by default.

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.cascade_stop_gradient_mrl=1 \
  --model.cascade_stop_gradient=1 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## 9. Train BOR-MRL Map Ablations

Cayley:

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.bor_mrl=1 \
  --model.bor_mode=orthogonal \
  --model.bor_orthogonal_map=cayley \
  --model.bor_stop_gradient=0 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

Householder:

```bash
python train_imagenet.py \
  --config-file rn50_configs/rn50_cifar100.yaml \
  --model.bor_mrl=1 \
  --model.bor_mode=orthogonal \
  --model.bor_orthogonal_map=householder \
  --model.bor_stop_gradient=0 \
  --data.root=/path/to/cifar100/root \
  --logging.folder=trainlogs
```

## 10. Evaluate BOR-MRL

```bash
cd ../inference

python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --bor_mrl \
  --bor_mode orthogonal \
  --bor_orthogonal_map matrix_exp \
  --bor_stop_gradient 0
```

For frozen orthogonal transforms, pass `--bor_mode frozen`. For Cayley or Householder, pass the matching `--bor_orthogonal_map`.

To evaluate the independent-block variant, replace `--bor_mrl` with `--bor_block_mrl`.

To evaluate the cascade stop-gradient ablation, replace `--bor_mrl` with `--cascade_stop_gradient_mrl`.

For BOR stop-gradient ablations, use `--model.bor_stop_gradient=1` or `--model.bor_stop_gradient=0` during training. The inference flag is `--bor_stop_gradient`; the default is off.

## 11. Evaluate MRL

Replace `<run_id>` with the folder created under `train/trainlogs/`.

```bash
cd ../inference

python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --mrl
```

## 12. Evaluate MRL-E

```bash
python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --mrl \
  --efficient
```

## 13. Evaluate Fixed-Feature Baseline

Example with 512 feature dimensions:

```bash
python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --rep_size 512
```

## 14. Evaluate Full-Feature Baseline

Use `--rep_size 2048` for the full 2048-dimensional classifier:

```bash
python pytorch_inference.py \
  --path ../train/trainlogs/<run_id>/final_weights.pt \
  --dataset CIFAR100 \
  --data_root /path/to/cifar100/root \
  --rep_size 2048
```

## 15. Compute Retrieval Metrics

Dump CIFAR-100 train/test feature arrays for a checkpoint:

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

For MRL-E use `--model mrl_e`. For fixed-feature checkpoints, pass `--rep_size <dim>` while dumping arrays and use `--model ff --rep-size <dim>` for the retrieval scripts.

To visualize a completed retrieval run, open `cifar100_retrieval_results.ipynb` and set `EXPERIMENT_DIR` to the run folder if you do not want it to auto-select the latest retrieval run.

## Notes

- ImageNet remains the default dataset. CIFAR-100 is selected by the config file during training and by `--dataset CIFAR100` during inference.
- Standard MRL is unchanged; BOR-MRL variants are selected explicitly with `--model.bor_mrl=1` or `--model.bor_block_mrl=1`.
- During training, each run writes `latest_weights.pt`; after completion it writes `final_weights.pt`.
- The runner also copies every method's final checkpoint into `checkpoints/` as `<run_name>_final_weights.pt`.
