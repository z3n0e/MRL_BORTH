# MNIST Neural Collapse Minimal Codebase

This repo trains a small supervised classifier on MNIST and logs Neural Collapse metrics during training.
It is intended as a sanity-check codebase before moving to CIFAR-100 / MRL experiments.

## What it measures

For each checkpoint/evaluation epoch, it computes:

- **Accuracy** on train/test.
- **NC1**: within-class collapse, measured as `trace(Sigma_W @ pinv(Sigma_B))` following the reference notebook. Lower is stronger collapse.
- **NC2 ETF metrics**: simplex ETF diagnostics for class means and classifier weights, including equinorm coefficient-of-variation, equiangular coherence, and Gram ETF error. These are only geometrically meaningful when `feature_dim >= num_classes - 1`; otherwise they are written as `NaN`.
- **NC3**: self-duality between classifiers and class means, logged both as average cosine alignment and as the reference notebook's normalized Frobenius distance.
- **NC4**: mismatch between the network classifier and nearest-class-center predictions.
- **Feature effective rank**: diagnostic for dimensional usage.
- **GNC1 trace-pinv**: generalized within-class collapse, measured as `trace(Sigma_W @ pinv(Sigma_B)) / K`. Lower is stronger collapse.
- **GNC2 Softmax Code geometry**: one-vs-rest convex-hull margins for normalized centered class means and classifier weights. Raw margins are higher-is-better.
- **GNC2 NC-like errors**: `gnc2_*_margin_error` is the normalized gap to the known Softmax Code optimum when the paper gives a closed-form target (`d=2`, `K<=d+1`, or `d+1<K<=2d`). Lower is closer to the generalized collapse geometry. For other regimes, the code logs the paper's lower/upper margin bounds.
- **GNC2 norm/margin balance**: `gnc2_*_norm_cov` and `gnc2_*_margin_cov` are Softmax-Code analogues of NC2 equinorm/equigeometry diagnostics.
- **GNC3 self-duality error**: `1 - mean cosine(classifier weight, centered class mean)` and normalized Frobenius distance. Lower is stronger self-duality.
- **UFM-compatible terminal geometry**: MNIST eval also logs the class-mean analogue of `mrl_ufm_geometry.py`, including `ce`, `prototype_accuracy`, `nc2_etf_error_H`, `nc2_etf_error_W`, `offdiag_*_H/W`, `ufm_nc3_align_mean`, `self_duality_error`, `effective_rank_H`, `spherical_margin_H/W`, `logit_margin_*`, `H_norm_mean`, and `W_norm_mean`.

For MNIST, `K=10`, so a perfect Simplex ETF is geometrically feasible when feature dimension `d >= 9`.
Try `--feature-dim 16` or higher first. Try `--feature-dim 2/4/8` to inspect the overcrowded regime.
In that regime, classical ETF columns such as `nc2_etf_error`, `nc2_mean_coherence`, and `nc2_mean_norm_cov` are intentionally `NaN`; `gnc2_class_mean_margin`, `gnc2_weight_margin`, and the corresponding `*_margin_error` columns are the appropriate generalized-NC diagnostics.

The UFM-compatible columns are computed by the same helper used by `mrl_ufm_geometry.py`. A few names are deliberately kept separate from MNIST's sample-level metrics: `accuracy` remains dataset accuracy, while `prototype_accuracy` is the class-mean/prototype accuracy; `nc1` remains within-class collapse, while `prototype_nc1` is the UFM one-prototype-per-class value; and `ufm_nc3_align_mean` is the row-centered UFM alignment.
Class-geometry metrics are centered by the unweighted mean of class means, not by the sample-weighted dataset mean. This keeps NC2/NC3/GNC geometry invariant to imbalanced class counts in train/test splits.

## Installation

```bash
cd mnist_neural_collapse
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Standard Neural Collapse run

```bash
python train_mnist_nc.py \
  --mode single \
  --feature-dim 16 \
  --epochs 80 \
  --batch-size 256 \
  --lr 0.05 \
  --weight-decay 5e-4 \
  --eval-every 2 \
  --out-dir outputs/single_d16
```

Expected behavior on the **training set**:

- NC1 should decrease.
- NC3 should increase.
- NC2 ETF error should generally decrease for `feature_dim >= 9`.
- GNC2 margins can be inspected at every feature dimension, including `feature_dim < 9`.
- Test NC may be weaker than train NC.

## Compare several dimensions

```bash
bash scripts/run_dims.sh
```

This runs independent single-scale models for dimensions `2,4,8,16,32,64`.

## Optional: MRL-style prefix experiment

The repo also includes a minimal MRL mode where one embedding of dimension `D` is supervised at several prefixes.

```bash
python train_mnist_nc.py \
  --mode mrl \
  --feature-dim 64 \
  --prefix-dims 2,4,8,16,32,64 \
  --loss-weights uniform \
  --epochs 80 \
  --batch-size 256 \
  --lr 0.05 \
  --weight-decay 5e-4 \
  --eval-every 2 \
  --out-dir outputs/mrl_d64
```

This is not meant as the final MRL paper experiment. It is only a quick sanity check.
For a publishable study, move to CIFAR-100 where `K/m` is much larger.

The MNIST MRL trainer supports the same loss-weight ablation names as the UFM script:

- `uniform`
- `large-heavy`
- `small-heavy`
- `only-large`
- `only-small+big`

It also supports auxiliary geometry regularizers, including VICReg-style losses and supervised contrastive loss:

```bash
python train_mnist_nc.py \
  --mode mrl \
  --feature-dim 32 \
  --prefix-dims 2,4,8,16,32 \
  --loss-weights uniform \
  --vicreg var-cov \
  --epochs 80 \
  --out-dir outputs/mnist_mrl_var_cov
```

Valid `--vicreg` presets are `none`, `var-only`, `cov-only`, `var-cov`, and `var-cov-cross-cov`. The variance term now defaults to class-mean Block-Var: it computes the variance floor on mini-batch class means for each newly added MRL block. Use `--vicreg-var-target ema-class-means` for EMA class means, or `--vicreg-var-target full-class-means --full-class-means-every N` to refresh train-set class means every `N` epochs. To reproduce the old raw-feature variance term, use `--vicreg-var-target features --vicreg-var-scope prefix`.
`--vicreg-target` still controls the covariance and cross-covariance targets (`features` by default, or `class-means` for a UFM-prototype proxy ablation).
For these runs, `train_loss` is logged as weighted CE plus the active auxiliary losses. Weight regularization stays in the optimizer via `--weight-decay`; `loss` in `metrics.csv` remains the evaluated sample cross-entropy for that split/prefix.

To use supervised contrastive loss instead of VICReg:

```bash
python train_mnist_nc.py \
  --mode mrl \
  --feature-dim 32 \
  --prefix-dims 2,4,8,16,32 \
  --loss-weights uniform \
  --vicreg none \
  --supcon-weight 1.0 \
  --supcon-temperature 0.1 \
  --epochs 80 \
  --out-dir outputs/mnist_mrl_supcon
```

By default SupCon is applied to every full prefix. Use `--supcon-scope block` to apply it only to each newly added MRL block.

To run the standard MNIST MRL ablation sweep:

```bash
bash scripts/run_mnist_mrl_ablations.sh
```

The sweep also runs independent single-scale baselines for each prefix dimension by default. Set `RUN_SINGLE=0` to skip them.

## Outputs

Each run writes:

- `metrics.csv`: one row per epoch/split/prefix.
- `plots/*.png`: curves for accuracy, classical NC metrics, generalized NC metrics, and pairwise cosine statistics.
- `checkpoints/latest.pt`, `checkpoints/best.pt`.

## Visualize Generalized NC

Open `notebooks/generalized_nc_visualization.ipynb` after running an experiment.
The first code cell is a parameter cell where you can choose:

- runs or explicit `metrics.csv` files,
- train/test splits,
- prefix dimensions,
- epochs,
- metric groups or individual metrics,
- whether to overlay known Softmax Code targets.
- whether to hide infeasible classical ETF metrics via `ONLY_FEASIBLE_ETF`.

For MRL runs, the most useful first view is usually the final metric-vs-prefix plot for `gnc2_class_mean_margin_error`, `gnc2_weight_margin_error`, `gnc2_class_mean_margin`, `gnc2_weight_margin`, and `gnc3_self_duality_error`.

## Main files

- `train_mnist_nc.py`: training/evaluation entry point.
- `src/models.py`: CNN backbone, single classifier, MRL prefix classifier.
- `src/nc_metrics.py`: Neural Collapse metrics.
- `src/plotting.py`: automatic plotting from CSV.
