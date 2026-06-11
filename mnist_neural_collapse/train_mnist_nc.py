from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models import make_model
from src.nc_metrics import compute_class_means, nc_metrics, ufm_geometry_metrics
from src.plotting import plot_metrics


LOSS_WEIGHT_PRESET_NAMES = (
    "uniform",
    "large-heavy",
    "small-heavy",
    "only-large",
    "only-small+big",
)


VICREG_PRESETS: Dict[str, Tuple[bool, bool, bool]] = {
    "none": (False, False, False),
    "var-only": (True, False, False),
    "cov-only": (False, True, False),
    "var-cov": (True, True, False),
    "var-cov-cross-cov": (True, True, True),
}


TRAIN_COMPONENT_KEYS = [
    "mrl_ce_total",
    "vicreg_var_loss",
    "vicreg_cov_loss",
    "vicreg_cross_cov_loss",
    "vicreg_total_loss",
    "train_loss",
]


def parse_prefix_dims(s: str | None, feature_dim: int) -> List[int]:
    if s is None or s.strip() == "":
        return [feature_dim]
    dims = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
    if dims[-1] != feature_dim:
        dims.append(feature_dim)
    if any(d <= 0 or d > feature_dim for d in dims):
        raise ValueError(f"Invalid prefix dims {dims} for feature_dim={feature_dim}")
    return dims


def _loss_weight_preset_values(name: str, n: int) -> List[float]:
    if n <= 0:
        raise ValueError("prefix_dims cannot be empty")
    if name == "uniform":
        return [1.0] * n
    if name == "large-heavy":
        if n == 1:
            return [1.0]
        if n == 5:
            return [0.25, 0.25, 0.5, 1.0, 2.0]
        return [float(x) for x in np.geomspace(0.25, 2.0, n)]
    if name == "small-heavy":
        if n == 1:
            return [1.0]
        return list(reversed(_loss_weight_preset_values("large-heavy", n)))
    if name == "only-large":
        return [0.0] * (n - 1) + [1.0]
    if name == "only-small+big":
        if n == 1:
            return [1.0]
        return [1.0] + [0.0] * (n - 2) + [1.0]
    raise ValueError(f"Unknown loss-weight preset: {name}")


def parse_loss_weights(spec: str, prefix_dims: List[int]) -> Tuple[str, List[float]]:
    spec = spec.strip() or "uniform"
    if spec in LOSS_WEIGHT_PRESET_NAMES:
        name = spec
        values = _loss_weight_preset_values(spec, len(prefix_dims))
    else:
        name = "custom"
        values = [float(x.strip()) for x in spec.split(",") if x.strip()]

    if len(values) != len(prefix_dims):
        raise ValueError(
            f"loss-weights {spec!r} has {len(values)} values, "
            f"but prefix-dims has {len(prefix_dims)} entries"
        )
    if any(v < 0 for v in values):
        raise ValueError("loss-weights must be non-negative")
    return name, values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loaders(data_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    eval_train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, eval_train_loader, test_loader


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def center_rows(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def offdiag(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[0] == x.shape[1]
    k = x.shape[0]
    if k <= 1:
        return x.new_empty(0)
    return x[~torch.eye(k, dtype=torch.bool, device=x.device)]


def batch_class_means(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Differentiable class means for the classes present in a mini-batch."""
    d = features.shape[1]
    means = torch.zeros(num_classes, d, device=features.device, dtype=features.dtype)
    counts = torch.zeros(num_classes, device=features.device, dtype=features.dtype)
    means.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    present = counts > 0
    means = means[present] / counts[present].clamp_min(1.0).unsqueeze(1)
    return means


def vicreg_variance_loss(H: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    if H.shape[0] <= 1:
        return H.new_zeros(())
    Hc = center_rows(H)
    std = torch.sqrt(Hc.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(H: torch.Tensor) -> torch.Tensor:
    n, d = H.shape
    if n <= 1 or d <= 1:
        return H.new_zeros(())
    Hc = center_rows(H)
    cov = (Hc.T @ Hc) / (n - 1)
    return offdiag(cov).pow(2).sum() / d


def vicreg_cross_covariance_loss(H_left: torch.Tensor, H_right: torch.Tensor) -> torch.Tensor:
    n = H_left.shape[0]
    if n <= 1 or H_left.shape[1] == 0 or H_right.shape[1] == 0:
        return H_left.new_zeros(())
    left = center_rows(H_left)
    right = center_rows(H_right)
    cross_cov = (left.T @ right) / (n - 1)
    return cross_cov.pow(2).mean()


def mnist_loss_components(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    logits_by_dim: Dict[int, torch.Tensor],
    loss_weights: Dict[int, float],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ce_total = features.sum() * 0.0
    var_loss = features.new_zeros(())
    cov_loss = features.new_zeros(())
    cross_cov_loss = features.new_zeros(())

    for d, logits in logits_by_dim.items():
        ce_total = ce_total + loss_weights[d] * F.cross_entropy(logits, labels)

    if args.vicreg != "none":
        if args.vicreg_target == "features":
            vicreg_target = features
        else:
            vicreg_target = batch_class_means(features, labels, args.num_classes)

        for d in model.prefix_dims:
            weight = loss_weights[d]
            Hm = vicreg_target[:, :d]
            if args.vicreg_use_var:
                var_loss = var_loss + weight * vicreg_variance_loss(
                    Hm, args.vicreg_gamma, args.vicreg_eps
                )
            if args.vicreg_use_cov:
                cov_loss = cov_loss + weight * vicreg_covariance_loss(Hm)

        if args.vicreg_use_cross_cov:
            for m_prev, m in zip(model.prefix_dims[:-1], model.prefix_dims[1:]):
                H_left = vicreg_target[:, :m_prev]
                H_right = vicreg_target[:, m_prev:m]
                cross_cov_loss = cross_cov_loss + loss_weights[m] * vicreg_cross_covariance_loss(
                    H_left, H_right
                )

    vicreg_total = (
        args.vicreg_var_weight * var_loss
        + args.vicreg_cov_weight * cov_loss
        + args.vicreg_cross_cov_weight * cross_cov_loss
    )
    total = ce_total + vicreg_total
    components = {
        "mrl_ce_total": ce_total,
        "vicreg_var_loss": var_loss,
        "vicreg_cov_loss": cov_loss,
        "vicreg_cross_cov_loss": cross_cov_loss,
        "vicreg_total_loss": vicreg_total,
        "train_loss": total,
    }
    return total, components


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[int, float],
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = {d: 0 for d in model.prefix_dims}
    component_sums = {key: 0.0 for key in TRAIN_COMPONENT_KEYS}
    total_seen = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        z = model.encoder(x)
        logits_by_dim = model.logits_from_features(z)
        loss, components = mnist_loss_components(model, z, y, logits_by_dim, loss_weights, args)
        loss.backward()
        optimizer.step()

        bs = y.numel()
        total_seen += bs
        total_loss += float(loss.item()) * bs
        for key in TRAIN_COMPONENT_KEYS:
            component_sums[key] += float(components[key].detach().item()) * bs
        for d, logits in logits_by_dim.items():
            total_correct[d] += int((logits.argmax(dim=1) == y).sum().item())
        postfix = {"loss": total_loss / max(total_seen, 1)}
        if args.vicreg != "none":
            postfix["vicreg"] = component_sums["vicreg_total_loss"] / max(total_seen, 1)
        pbar.set_postfix(**postfix)

    out = {"loss": total_loss / total_seen}
    for key in TRAIN_COMPONENT_KEYS:
        out[key] = component_sums[key] / total_seen
    for d in model.prefix_dims:
        out[f"acc_d{d}"] = total_correct[d] / total_seen
    return out


@torch.no_grad()
def collect_features_and_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, float], Dict[int, float]]:
    model.eval()
    zs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    logits_by_dim: Dict[int, List[torch.Tensor]] = {d: [] for d in model.prefix_dims}
    loss_sum = {d: 0.0 for d in model.prefix_dims}
    correct_sum = {d: 0 for d in model.prefix_dims}
    seen = 0

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = model.features(x)
        logits = model.logits_from_features(z)

        zs.append(z.cpu())
        ys.append(y.cpu())
        bs = y.numel()
        seen += bs
        for d, out in logits.items():
            logits_by_dim[d].append(out.cpu())
            loss_sum[d] += float(F.cross_entropy(out, y, reduction="sum").item())
            correct_sum[d] += int((out.argmax(dim=1) == y).sum().item())

    features = torch.cat(zs, dim=0)
    labels = torch.cat(ys, dim=0)
    logits_cat = {d: torch.cat(parts, dim=0) for d, parts in logits_by_dim.items()}
    loss = {d: loss_sum[d] / seen for d in model.prefix_dims}
    acc = {d: correct_sum[d] / seen for d in model.prefix_dims}
    return features, labels, logits_cat, loss, acc


def evaluate_nc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    epoch: int,
    num_classes: int,
    model_mode: str,
    loss_weight_name: str,
    loss_weights: Dict[int, float],
    vicreg_name: str,
    vicreg_target: str,
    train_stats: Dict[str, float] | None = None,
) -> List[Dict[str, float | int | str]]:
    features, labels, logits_by_dim, loss_by_dim, acc_by_dim = collect_features_and_logits(model, loader, device)
    rows: List[Dict[str, float | int | str]] = []
    train_stats = train_stats or {}
    nan = float("nan")

    for d in model.prefix_dims:
        z_d = features[:, :d]
        W_d = model.classifier_weight(d).detach().cpu()
        class_means, _ = compute_class_means(z_d, labels, num_classes)
        metrics = nc_metrics(
            features=z_d,
            labels=labels,
            classifier_weight=W_d,
            num_classes=num_classes,
            logits=logits_by_dim[d],
        )
        ufm_metrics = ufm_geometry_metrics(class_means, W_d)
        row: Dict[str, float | int | str] = {
            "name": f"{model_mode}_{split}_d{d}",
            "mode": model_mode,
            "epoch": epoch,
            "split": split,
            "prefix_dim": d,
            "loss": loss_by_dim[d],
            "accuracy": acc_by_dim[d],
            "loss_weight_name": loss_weight_name,
            "mrl_loss_weight": loss_weights[d],
            "vicreg": vicreg_name,
            "vicreg_target": vicreg_target,
        }
        for key in TRAIN_COMPONENT_KEYS:
            row[key] = train_stats.get(key, nan)
        row.update(metrics)
        for key, value in ufm_metrics.items():
            if key == "accuracy":
                row["prototype_accuracy"] = value
            elif key == "nc1":
                row["prototype_nc1"] = value
            elif key == "nc3_align_mean":
                row["ufm_nc3_align_mean"] = value
            elif key == "nc3_align_std":
                row["ufm_nc3_align_std"] = value
            elif key == "ncm_acc":
                row["prototype_ncm_acc"] = value
            else:
                row[key] = value
        row["prototype_ce"] = row["ce"]
        rows.append(row)
    return rows


def append_rows(csv_path: Path, rows: List[Dict[str, float | int | str]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = list(rows[0].keys())
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST model and measure Neural Collapse.")
    parser.add_argument("--mode", choices=["single", "mrl"], default="single")
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--prefix-dims", type=str, default=None, help="Comma-separated list, e.g. 2,4,8,16,32,64")
    parser.add_argument(
        "--loss-weights",
        type=str,
        default="uniform",
        help=(
            "MRL loss weighting preset "
            f"({', '.join(LOSS_WEIGHT_PRESET_NAMES)}) "
            "or comma-separated lambda_m values"
        ),
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs/run")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-test-eval", action="store_true", help="Only compute NC metrics on train split.")
    parser.add_argument(
        "--vicreg",
        type=str,
        default="none",
        choices=tuple(VICREG_PRESETS.keys()),
        help="optional VICReg-style regularizer applied to MNIST embeddings",
    )
    parser.add_argument(
        "--vicreg-target",
        type=str,
        default="features",
        choices=("class-means", "features"),
        help=(
            "regularize raw batch features, matching original VICReg; "
            "class-means is only a UFM-prototype ablation"
        ),
    )
    parser.add_argument(
        "--vicreg-var-weight",
        type=float,
        default=1.0,
        help="coefficient for the active VICReg variance term",
    )
    parser.add_argument(
        "--vicreg-cov-weight",
        type=float,
        default=1.0,
        help="coefficient for the active VICReg covariance term",
    )
    parser.add_argument(
        "--vicreg-cross-cov-weight",
        type=float,
        default=1.0,
        help="coefficient for the active VICReg cross-covariance term",
    )
    parser.add_argument(
        "--vicreg-gamma",
        type=float,
        default=1.0,
        help="target minimum per-coordinate standard deviation for VICReg variance",
    )
    parser.add_argument(
        "--vicreg-eps",
        type=float,
        default=1e-4,
        help="epsilon inside the VICReg variance standard deviation",
    )
    args = parser.parse_args()

    prefix_dims = parse_prefix_dims(args.prefix_dims, args.feature_dim)
    if args.mode == "single":
        prefix_dims = [args.feature_dim]

    args.loss_weight_name, args.loss_weight_values = parse_loss_weights(
        args.loss_weights, prefix_dims
    )
    args.loss_weight_by_dim = dict(zip(prefix_dims, args.loss_weight_values))
    if not any(v > 0 for v in args.loss_weight_values):
        raise ValueError("at least one loss weight must be positive")
    if args.vicreg_var_weight < 0:
        raise ValueError("vicreg-var-weight must be non-negative")
    if args.vicreg_cov_weight < 0:
        raise ValueError("vicreg-cov-weight must be non-negative")
    if args.vicreg_cross_cov_weight < 0:
        raise ValueError("vicreg-cross-cov-weight must be non-negative")
    if args.vicreg_gamma < 0:
        raise ValueError("vicreg-gamma must be non-negative")
    if args.vicreg_eps <= 0:
        raise ValueError("vicreg-eps must be positive")
    args.vicreg_use_var, args.vicreg_use_cov, args.vicreg_use_cross_cov = VICREG_PRESETS[
        args.vicreg
    ]

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump({**vars(args), "resolved_prefix_dims": prefix_dims}, f, indent=2)

    device = torch.device(args.device)
    train_loader, eval_train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = make_model(
        mode=args.mode,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        prefix_dims=prefix_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_weights = {d: float(v) for d, v in zip(model.prefix_dims, args.loss_weight_values)}

    csv_path = out_dir / "metrics.csv"
    best_test_acc = -1.0

    # Evaluate epoch 0 before training.
    rows = evaluate_nc(
        model,
        eval_train_loader,
        device,
        "train",
        0,
        args.num_classes,
        args.mode,
        args.loss_weight_name,
        loss_weights,
        args.vicreg,
        args.vicreg_target,
    )
    if not args.no_test_eval:
        rows += evaluate_nc(
            model,
            test_loader,
            device,
            "test",
            0,
            args.num_classes,
            args.mode,
            args.loss_weight_name,
            loss_weights,
            args.vicreg,
            args.vicreg_target,
        )
    append_rows(csv_path, rows)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, loss_weights, args)
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train loss {train_stats['loss']:.4f} "
                f"| vicreg {train_stats['vicreg_total_loss']:.4f}"
            )
            rows = evaluate_nc(
                model,
                eval_train_loader,
                device,
                "train",
                epoch,
                args.num_classes,
                args.mode,
                args.loss_weight_name,
                loss_weights,
                args.vicreg,
                args.vicreg_target,
                train_stats,
            )
            if not args.no_test_eval:
                rows += evaluate_nc(
                    model,
                    test_loader,
                    device,
                    "test",
                    epoch,
                    args.num_classes,
                    args.mode,
                    args.loss_weight_name,
                    loss_weights,
                    args.vicreg,
                    args.vicreg_target,
                    train_stats,
                )
            append_rows(csv_path, rows)

            # Best checkpoint based on largest full-dim test accuracy, or train if test disabled.
            full_dim = max(model.prefix_dims)
            split_for_best = "train" if args.no_test_eval else "test"
            current = [r for r in rows if r["split"] == split_for_best and r["prefix_dim"] == full_dim][0]
            current_acc = float(current["accuracy"])
            if current_acc > best_test_acc:
                best_test_acc = current_acc
                save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, args)
            save_checkpoint(out_dir / "checkpoints" / "latest.pt", model, optimizer, epoch, args)

            plot_metrics(csv_path, out_dir / "plots")

    print(f"Done. Metrics: {csv_path}")
    print(f"Plots: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
