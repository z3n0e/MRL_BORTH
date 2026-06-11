#!/usr/bin/env python3
"""
MRL-UFM terminal geometry experiment.

This script removes the backbone and dataset and optimizes only class prototypes H and
prefix classifiers W_m. It is meant to reveal the idealized geometry preferred by
Matryoshka Representation Learning under an Unconstrained Feature Model (UFM).

It trains:
  1) one shared MRL-UFM model with H in R^{K x D} and heads W_m for each prefix m;
  2) independent single-scale UFM baselines for every m in prefix_dims.

It saves:
  - metrics_final.csv
  - metrics_history.csv
  - gram matrices as .npy
  - Gram heatmaps and summary plots

Example:
  python mrl_ufm_geometry.py \
    --K 10 \
    --D 32 \
    --prefix-dims 2,4,8,16,32 \
    --epochs 20000 \
    --lr 0.03 \
    --alpha 1e-3 \
    --beta 1e-3 \
    --out-dir outputs/ufm_mrl_mnistK10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nc_metrics import ufm_geometry_metrics as shared_ufm_geometry_metrics


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_dims(s: str) -> List[int]:
    dims = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(dims) == 0:
        raise ValueError("prefix-dims is empty")
    if sorted(dims) != dims:
        raise ValueError("prefix-dims must be sorted increasingly")
    if len(set(dims)) != len(dims):
        raise ValueError("prefix-dims contains duplicates")
    return dims


LOSS_WEIGHT_PRESET_NAMES = (
    "uniform",
    "large-heavy",
    "small-heavy",
    "only-large",
    "only-small+big",
)


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


VICREG_PRESETS: Dict[str, Tuple[bool, bool, bool]] = {
    "none": (False, False, False),
    "var-only": (True, False, False),
    "cov-only": (False, True, False),
    "var-cov": (True, True, False),
    "var-cov-cross-cov": (True, True, True),
}


def parse_loss_weights(spec: str, prefix_dims: List[int]) -> Tuple[str, List[float]]:
    spec = spec.strip() or "uniform"
    if spec in LOSS_WEIGHT_PRESET_NAMES:
        values = _loss_weight_preset_values(spec, len(prefix_dims))
        name = spec
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
    if not any(v > 0 for v in values):
        raise ValueError("at least one loss weight must be positive")
    return name, values


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def offdiag(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[0] == x.shape[1]
    k = x.shape[0]
    return x[~torch.eye(k, dtype=torch.bool, device=x.device)]


def center_rows(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def effective_rank_from_eigs(eigs: torch.Tensor, eps: float = 1e-12) -> float:
    eigs = eigs.clamp_min(0)
    total = eigs.sum()
    if total.item() <= eps:
        return 0.0
    p = eigs / total
    entropy = -(p[p > eps] * torch.log(p[p > eps])).sum()
    return float(torch.exp(entropy).item())


# -----------------------------
# MRL-UFM model
# -----------------------------


class MRLUFM(nn.Module):
    """Shared class prototypes H in R^{K x D} and one linear head per prefix."""

    def __init__(self, K: int, D: int, prefix_dims: List[int], init_scale: float = 0.05):
        super().__init__()
        self.K = K
        self.D = D
        self.prefix_dims = prefix_dims
        self.H = nn.Parameter(init_scale * torch.randn(K, D))
        self.heads = nn.ParameterDict()
        for m in prefix_dims:
            self.heads[str(m)] = nn.Parameter(init_scale * torch.randn(K, m))

    def logits(self, m: int) -> torch.Tensor:
        """Logits for K class prototypes. Row c is the sample of class c."""
        Hm = self.H[:, :m]
        Wm = self.heads[str(m)]
        return Hm @ Wm.T

    def l2_penalty(self) -> Tuple[torch.Tensor, torch.Tensor]:
        h2 = self.H.pow(2).sum()
        w2 = sum(W.pow(2).sum() for W in self.heads.values())
        return h2, w2


class SingleUFM(nn.Module):
    """Independent single-scale UFM baseline."""

    def __init__(self, K: int, D: int, init_scale: float = 0.05):
        super().__init__()
        self.K = K
        self.D = D
        self.H = nn.Parameter(init_scale * torch.randn(K, D))
        self.W = nn.Parameter(init_scale * torch.randn(K, D))

    def logits(self) -> torch.Tensor:
        return self.H @ self.W.T

    def l2_penalty(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.H.pow(2).sum(), self.W.pow(2).sum()


# -----------------------------
# Geometry metrics
# -----------------------------


@torch.no_grad()
def geometry_metrics(
    H: torch.Tensor,
    W: torch.Tensor,
    name: str,
    dim: int,
    epoch: int,
    mode: str,
) -> Dict[str, float | str | int]:
    """
    Compute NC/GNC-inspired metrics for one feature matrix H and classifier W.

    H: K x d class prototypes / class means.
    W: K x d classifier weights.
    """
    metrics = shared_ufm_geometry_metrics(H, W)
    return {
        "name": name,
        "mode": mode,
        "epoch": int(epoch),
        **metrics,
    }


@torch.no_grad()
def gram_numpy(H: torch.Tensor) -> np.ndarray:
    Hc = center_rows(H)
    Hn = normalize_rows(Hc)
    return (Hn @ Hn.T).detach().cpu().numpy()


# -----------------------------
# VICReg-style losses
# -----------------------------


def vicreg_variance_loss(H: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    """Encourage every coordinate to keep class-prototype standard deviation >= gamma."""
    Hc = center_rows(H)
    std = torch.sqrt(Hc.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(H: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal feature covariance within one prefix."""
    n, d = H.shape
    if n <= 1 or d <= 1:
        return H.new_zeros(())
    Hc = center_rows(H)
    cov = (Hc.T @ Hc) / (n - 1)
    return offdiag(cov).pow(2).sum() / d


def vicreg_cross_covariance_loss(H_left: torch.Tensor, H_right: torch.Tensor) -> torch.Tensor:
    """Penalize covariance between an inherited prefix and the newly added block."""
    n = H_left.shape[0]
    if n <= 1 or H_left.shape[1] == 0 or H_right.shape[1] == 0:
        return H_left.new_zeros(())
    left = center_rows(H_left)
    right = center_rows(H_right)
    cross_cov = (left.T @ right) / (n - 1)
    return cross_cov.pow(2).mean()


def mrl_loss_components(
    model: MRLUFM,
    labels: torch.Tensor,
    lambdas: Dict[int, float],
    args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ce_total = model.H.new_zeros(())
    var_loss = model.H.new_zeros(())
    cov_loss = model.H.new_zeros(())
    cross_cov_loss = model.H.new_zeros(())

    for m in args.prefix_dims:
        weight = lambdas[m]
        Hm = model.H[:, :m]
        ce_total = ce_total + weight * F.cross_entropy(model.logits(m), labels)
        if args.vicreg_use_var:
            var_loss = var_loss + weight * vicreg_variance_loss(
                Hm, args.vicreg_gamma, args.vicreg_eps
            )
        if args.vicreg_use_cov:
            cov_loss = cov_loss + weight * vicreg_covariance_loss(Hm)

    if args.vicreg_use_cross_cov:
        for m_prev, m in zip(args.prefix_dims[:-1], args.prefix_dims[1:]):
            H_left = model.H[:, :m_prev]
            H_right = model.H[:, m_prev:m]
            cross_cov_loss = cross_cov_loss + lambdas[m] * vicreg_cross_covariance_loss(
                H_left, H_right
            )

    h2, w2 = model.l2_penalty()
    l2_loss = 0.5 * args.alpha * h2 + 0.5 * args.beta * w2
    vicreg_total = (
        args.vicreg_var_weight * var_loss
        + args.vicreg_cov_weight * cov_loss
        + args.vicreg_cross_cov_weight * cross_cov_loss
    )
    total = ce_total + l2_loss + vicreg_total

    components = {
        "mrl_ce_total": ce_total,
        "l2_loss": l2_loss,
        "vicreg_var_loss": var_loss,
        "vicreg_cov_loss": cov_loss,
        "vicreg_cross_cov_loss": cross_cov_loss,
        "vicreg_total_loss": vicreg_total,
        "train_loss": total,
    }
    return total, components


# -----------------------------
# Training
# -----------------------------


def train_mrl_ufm(args, device: torch.device) -> Tuple[MRLUFM, pd.DataFrame]:
    model = MRLUFM(args.K, args.D, args.prefix_dims, init_scale=args.init_scale).to(device)
    labels = torch.arange(args.K, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: List[Dict] = []

    lambdas = {m: v for m, v in zip(args.prefix_dims, args.loss_weight_values)}

    for epoch in range(args.epochs + 1):
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = mrl_loss_components(model, labels, lambdas, args)
            loss.backward()
            optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                _, components = mrl_loss_components(model, labels, lambdas, args)
                component_values = {k: float(v.item()) for k, v in components.items()}
            for m in args.prefix_dims:
                Hm = model.H[:, :m].detach()
                Wm = model.heads[str(m)].detach()
                row = geometry_metrics(Hm, Wm, f"mrl_d{m}", m, epoch, "mrl")
                row.update(component_values)
                row["mrl_loss_weight"] = float(lambdas[m])
                row["vicreg"] = args.vicreg
                history.append(row)
            if args.verbose:
                last = history[-1]
                print(
                    f"[MRL] epoch={epoch:05d} "
                    f"d={args.prefix_dims[-1]} "
                    f"acc={last['accuracy']:.3f} "
                    f"ce={last['ce']:.4f} "
                    f"vicreg={last['vicreg_total_loss']:.4f} "
                    f"ETF_H={last['nc2_etf_error_H']:.4f} "
                    f"NC3={last['nc3_align_mean']:.4f}"
                )

    return model, pd.DataFrame(history)


def train_single_ufm(args, dim: int, device: torch.device) -> Tuple[SingleUFM, pd.DataFrame]:
    model = SingleUFM(args.K, dim, init_scale=args.init_scale).to(device)
    labels = torch.arange(args.K, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: List[Dict] = []

    for epoch in range(args.epochs + 1):
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model.logits(), labels)
            h2, w2 = model.l2_penalty()
            loss = ce + 0.5 * args.alpha * h2 + 0.5 * args.beta * w2
            loss.backward()
            optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row = geometry_metrics(
                model.H.detach(),
                model.W.detach(),
                f"single_d{dim}",
                dim,
                epoch,
                "single",
            )
            history.append(row)
            if args.verbose and epoch % max(args.eval_every * 10, 1) == 0:
                print(
                    f"[SINGLE d={dim}] epoch={epoch:05d} "
                    f"acc={row['accuracy']:.3f} "
                    f"ce={row['ce']:.4f} "
                    f"ETF_H={row['nc2_etf_error_H']:.4f} "
                    f"NC3={row['nc3_align_mean']:.4f}"
                )

    return model, pd.DataFrame(history)


# -----------------------------
# Comparison metrics
# -----------------------------


def frob(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro"))


def compare_final_geometries(
    mrl_model: MRLUFM,
    single_models: Dict[int, SingleUFM],
    prefix_dims: List[int],
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for m in prefix_dims:
        H_mrl = mrl_model.H[:, :m].detach().cpu()
        H_single = single_models[m].H.detach().cpu()
        G_mrl = gram_numpy(H_mrl)
        G_single = gram_numpy(H_single)

        rows.append(
            {
                "dim": m,
                "gram_fro_distance": frob(G_mrl, G_single),
                "gram_fro_distance_normalized": frob(G_mrl, G_single) / G_mrl.shape[0],
            }
        )

        np.save(out_dir / f"gram_mrl_d{m}.npy", G_mrl)
        np.save(out_dir / f"gram_single_d{m}.npy", G_single)

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------


def make_plots(history: pd.DataFrame, final: pd.DataFrame, compare: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(out_dir / "plots")

    # Final metric vs dimension: MRL vs single.
    metrics = [
        "accuracy",
        "nc2_etf_error_H",
        "nc3_align_mean",
        "self_duality_error",
        "effective_rank_H",
        "spherical_margin_H",
        "logit_margin_mean",
    ]

    for metric in metrics:
        plt.figure()
        for mode in ["single", "mrl"]:
            df = final[final["mode"] == mode].sort_values("dim")
            if len(df) > 0:
                plt.plot(df["dim"], df[metric], marker="o", label=mode)
        plt.xlabel("dimension")
        plt.ylabel(metric)
        plt.xscale("log", base=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"final_{metric}_vs_dim.png", dpi=200)
        plt.close()

    # Geometry gap vs dimension.
    if len(compare) > 0:
        plt.figure()
        plt.plot(compare["dim"], compare["gram_fro_distance_normalized"], marker="o")
        plt.xlabel("dimension")
        plt.ylabel("normalized Gram distance: MRL vs single")
        plt.xscale("log", base=2)
        plt.tight_layout()
        plt.savefig(plot_dir / "gram_gap_mrl_vs_single.png", dpi=200)
        plt.close()

    # History for MRL prefixes.
    mrl_hist = history[history["mode"] == "mrl"]
    for metric in ["ce", "nc2_etf_error_H", "nc3_align_mean", "effective_rank_H", "spherical_margin_H"]:
        plt.figure()
        for dim, df in mrl_hist.groupby("dim"):
            df = df.sort_values("epoch")
            plt.plot(df["epoch"], df[metric], label=f"d={dim}")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"mrl_history_{metric}.png", dpi=200)
        plt.close()

    # Gram heatmaps for final geometry.
    for npy in sorted(out_dir.glob("gram_*.npy")):
        G = np.load(npy)
        plt.figure()
        plt.imshow(G, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(npy.stem)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{npy.stem}.png", dpi=200)
        plt.close()


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10, help="number of classes")
    parser.add_argument("--D", type=int, default=32, help="full MRL feature dimension")
    parser.add_argument("--prefix-dims", type=str, default="2,4,8,16,32")
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
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=1e-3, help="feature L2 coefficient")
    parser.add_argument("--beta", type=float, default=1e-3, help="head L2 coefficient")
    parser.add_argument(
        "--vicreg",
        type=str,
        default="none",
        choices=tuple(VICREG_PRESETS.keys()),
        help="optional VICReg-style prototype regularizer for MRL training",
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
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="outputs/ufm_geometry")
    parser.add_argument("--no-single", action="store_true", help="skip independent single-scale baselines")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.prefix_dims = parse_dims(args.prefix_dims)
    if args.prefix_dims[-1] != args.D:
        raise ValueError("last prefix dim must equal D")
    if any(m > args.D for m in args.prefix_dims):
        raise ValueError("prefix dim cannot exceed D")
    args.loss_weight_name, args.loss_weight_values = parse_loss_weights(
        args.loss_weights, args.prefix_dims
    )
    args.loss_weight_by_dim = dict(zip(args.prefix_dims, args.loss_weight_values))
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
    device = torch.device(args.device)
    out_dir = ensure_dir(args.out_dir)

    # Save config.
    with open(out_dir / "config.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    print("Training MRL-UFM...")
    mrl_model, mrl_hist = train_mrl_ufm(args, device)

    single_models: Dict[int, SingleUFM] = {}
    single_hists = []
    if not args.no_single:
        for m in args.prefix_dims:
            print(f"Training single-scale UFM d={m}...")
            set_seed(args.seed + 1000 + m)
            smodel, shist = train_single_ufm(args, m, device)
            single_models[m] = smodel
            single_hists.append(shist)

    history = pd.concat([mrl_hist] + single_hists, ignore_index=True) if single_hists else mrl_hist
    history.to_csv(out_dir / "metrics_history.csv", index=False)

    final = history.sort_values("epoch").groupby(["mode", "dim"], as_index=False).tail(1)
    final = final.sort_values(["mode", "dim"])
    final.to_csv(out_dir / "metrics_final.csv", index=False)

    compare = pd.DataFrame()
    if single_models:
        compare = compare_final_geometries(mrl_model, single_models, args.prefix_dims, out_dir)
        compare.to_csv(out_dir / "geometry_gap_mrl_vs_single.csv", index=False)

    # Save raw learned geometry.
    torch.save(
        {
            "H_mrl": mrl_model.H.detach().cpu(),
            "W_mrl": {m: mrl_model.heads[str(m)].detach().cpu() for m in args.prefix_dims},
            "H_single": {m: single_models[m].H.detach().cpu() for m in single_models},
            "W_single": {m: single_models[m].W.detach().cpu() for m in single_models},
            "prefix_dims": args.prefix_dims,
            "K": args.K,
            "D": args.D,
            "vicreg": args.vicreg,
            "vicreg_var_weight": args.vicreg_var_weight,
            "vicreg_cov_weight": args.vicreg_cov_weight,
            "vicreg_cross_cov_weight": args.vicreg_cross_cov_weight,
            "vicreg_gamma": args.vicreg_gamma,
        },
        out_dir / "learned_geometry.pt",
    )

    make_plots(history, final, compare, out_dir)

    print("\nDone.")
    print(f"Saved outputs to: {out_dir}")
    print("\nFinal metrics:")
    cols = [
        "mode",
        "dim",
        "accuracy",
        "ce",
        "nc2_etf_error_H",
        "nc3_align_mean",
        "self_duality_error",
        "effective_rank_H",
        "spherical_margin_H",
        "logit_margin_mean",
    ]
    print(final[cols].to_string(index=False))

    if len(compare) > 0:
        print("\nMRL-vs-single geometry gap:")
        print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
