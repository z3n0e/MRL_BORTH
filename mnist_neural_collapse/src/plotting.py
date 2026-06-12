from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


ETF_METRICS = {
    "nc2_etf_error",
    "nc2_weight_etf_error",
    "nc2_mean_norm_cov",
    "nc2_weight_norm_cov",
    "nc2_mean_coherence",
    "nc2_weight_coherence",
    "nc2_offdiag_mean",
    "nc2_offdiag_std",
    "nc2_offdiag_min",
    "nc2_offdiag_max",
}

DEFAULT_METRICS = [
    "accuracy",
    "prototype_accuracy",
    "loss",
    "ce",
    "prototype_ce",
    "train_loss",
    "mrl_ce_total",
    "vicreg_total_loss",
    "vicreg_var_loss",
    "vicreg_cov_loss",
    "vicreg_cross_cov_loss",
    "supcon_total_loss",
    "supcon_loss",
    "nc1",
    "nc1_sw_inv_sb",
    "gnc1_trace_pinv",
    "nc2_etf_error",
    "nc2_weight_etf_error",
    "nc2_etf_error_H",
    "nc2_etf_error_W",
    "nc2_mean_norm_cov",
    "nc2_weight_norm_cov",
    "nc2_mean_coherence",
    "nc2_weight_coherence",
    "nc2_offdiag_mean",
    "nc2_offdiag_std",
    "offdiag_mean_H",
    "offdiag_std_H",
    "offdiag_max_H",
    "offdiag_mean_W",
    "offdiag_std_W",
    "offdiag_max_W",
    "nc3_align_mean",
    "ufm_nc3_align_mean",
    "nc3_self_duality_fro",
    "self_duality_error",
    "nc4_ncc_mismatch",
    "gnc2_class_mean_norm_cov",
    "gnc2_weight_norm_cov",
    "gnc2_class_mean_margin",
    "gnc2_weight_margin",
    "gnc2_class_mean_margin_error",
    "gnc2_weight_margin_error",
    "gnc2_class_mean_target_ratio",
    "gnc2_weight_target_ratio",
    "gnc2_class_mean_margin_cov",
    "gnc2_weight_margin_cov",
    "gnc3_self_duality_error",
    "gnc3_self_duality_fro",
    "effective_rank",
    "effective_rank_H",
    "spherical_margin_H",
    "spherical_margin_W",
    "logit_margin_mean",
    "ncm_acc",
    "prototype_ncm_acc",
]


def _infer_num_classes(csv_path: Path, df: pd.DataFrame) -> int | None:
    if "num_classes" in df.columns:
        values = pd.to_numeric(df["num_classes"], errors="coerce").dropna()
        if len(values) > 0:
            return int(values.iloc[0])

    config_path = csv_path.parent / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        if "num_classes" in config:
            return int(config["num_classes"])
    return None


def _mask_infeasible_etf_metrics(
    df: pd.DataFrame,
    *,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """Hide classical ETF metrics when d < K - 1.

    New CSVs already write these entries as NaN. This extra pass keeps plots
    honest for older CSVs that still contain infeasible ETF diagnostics.
    """
    if "nc2_etf_feasible" in df.columns:
        feasible = pd.to_numeric(df["nc2_etf_feasible"], errors="coerce").fillna(0.0) >= 0.5
    elif csv_path is not None and (num_classes := _infer_num_classes(csv_path, df)) is not None:
        dim_col = "prefix_dim" if "prefix_dim" in df.columns else "feature_dim"
        if dim_col not in df.columns:
            return df
        feasible = pd.to_numeric(df[dim_col], errors="coerce") >= (num_classes - 1)
    else:
        return df

    out = df.copy()
    for metric in ETF_METRICS.intersection(out.columns):
        out.loc[~feasible, metric] = pd.NA
    return out


def plot_metrics(csv_path: str | Path, out_dir: str | Path, metrics: Iterable[str] = DEFAULT_METRICS) -> None:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _mask_infeasible_etf_metrics(pd.read_csv(csv_path), csv_path=csv_path)
    if df.empty:
        return

    for metric in metrics:
        if metric not in df.columns:
            continue
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        if df[metric].notna().sum() == 0:
            continue
        plt.figure(figsize=(8, 5))
        for (split, prefix), g in df.groupby(["split", "prefix_dim"]):
            label = f"{split}-d{int(prefix)}"
            plt.plot(g["epoch"], g[metric], label=label)
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title(metric)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=160)
        plt.close()

    # Compact summary at final epoch: metric vs prefix dimension.
    final_epoch = int(df["epoch"].max())
    final_df = df[df["epoch"] == final_epoch].copy()
    for metric in [
        "accuracy",
        "prototype_accuracy",
        "ce",
        "prototype_ce",
        "train_loss",
        "mrl_ce_total",
        "vicreg_total_loss",
        "vicreg_var_loss",
        "vicreg_cov_loss",
        "vicreg_cross_cov_loss",
        "supcon_total_loss",
        "supcon_loss",
        "nc1",
        "nc1_sw_inv_sb",
        "gnc1_trace_pinv",
        "nc2_etf_error",
        "nc2_weight_etf_error",
        "nc2_etf_error_H",
        "nc2_etf_error_W",
        "nc2_mean_norm_cov",
        "nc2_weight_norm_cov",
        "nc2_mean_coherence",
        "nc2_weight_coherence",
        "offdiag_mean_H",
        "offdiag_std_H",
        "offdiag_max_H",
        "offdiag_mean_W",
        "offdiag_std_W",
        "offdiag_max_W",
        "gnc2_class_mean_norm_cov",
        "gnc2_weight_norm_cov",
        "gnc2_class_mean_margin",
        "gnc2_weight_margin",
        "gnc2_class_mean_margin_error",
        "gnc2_weight_margin_error",
        "gnc2_class_mean_target_ratio",
        "gnc2_weight_target_ratio",
        "gnc2_class_mean_margin_cov",
        "gnc2_weight_margin_cov",
        "nc3_align_mean",
        "ufm_nc3_align_mean",
        "nc3_self_duality_fro",
        "self_duality_error",
        "gnc3_self_duality_error",
        "gnc3_self_duality_fro",
        "effective_rank_H",
        "spherical_margin_H",
        "spherical_margin_W",
        "logit_margin_mean",
        "nc4_ncc_mismatch",
        "ncm_acc",
        "prototype_ncm_acc",
    ]:
        if metric not in final_df.columns:
            continue
        final_df[metric] = pd.to_numeric(final_df[metric], errors="coerce")
        if final_df[metric].notna().sum() == 0:
            continue
        plt.figure(figsize=(7, 5))
        for split, g in final_df.groupby("split"):
            g = g.sort_values("prefix_dim")
            plt.plot(g["prefix_dim"], g[metric], marker="o", label=split)
        plt.xlabel("prefix dimension")
        plt.ylabel(metric)
        plt.xscale("log", base=2)
        plt.title(f"Final {metric} vs prefix dimension")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"final_{metric}_vs_dim.png", dpi=160)
        plt.close()
