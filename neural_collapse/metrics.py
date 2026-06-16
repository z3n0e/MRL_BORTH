from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple

import torch

from .constants import CORE_NC_METRIC_KEYS

EPS = 1e-12
SOFTMAX_CODE_MAX_CLASSES = 256
SOFTMAX_CODE_HULL_STEPS = 200


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _coefficient_of_variation(values: torch.Tensor) -> torch.Tensor:
    return values.std(unbiased=False) / values.mean().clamp_min(EPS)


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("expected a square matrix")
    k = x.shape[0]
    if k <= 1:
        return x.new_empty(0)
    return x[~torch.eye(k, dtype=torch.bool, device=x.device)]


def effective_rank(features: torch.Tensor, eps: float = EPS) -> float:
    if features.ndim != 2 or features.shape[0] < 2:
        return float("nan")
    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum()
    if total.item() <= eps:
        return 0.0
    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float(torch.exp(entropy).item())


def compute_class_means(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = features.detach().float()
    labels = labels.detach().long()
    device = features.device
    means = torch.zeros(num_classes, features.shape[1], device=device, dtype=features.dtype)
    counts = torch.zeros(num_classes, device=device, dtype=features.dtype)
    means.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    means = means / counts.clamp_min(1.0).unsqueeze(1)
    return means, counts


def nc1_trace_pinv(
    features: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    centered_means: torch.Tensor,
    *,
    normalize_by_num_classes: bool = False,
) -> float:
    k = means.shape[0]
    if k == 0 or features.numel() == 0:
        return float("nan")

    residuals = features - means[labels]
    sigma_w = residuals.T @ residuals / max(int(features.shape[0]), 1)
    sigma_b = centered_means.T @ centered_means / max(k, 1)
    value = torch.trace(sigma_w @ torch.linalg.pinv(sigma_b))
    if normalize_by_num_classes:
        value = value / max(k, 1)
    return float(value.item())


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    if v.numel() == 1:
        return torch.ones_like(v)

    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    if not bool(cond.any().item()):
        return torch.full_like(v, 1.0 / v.numel())

    rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
    theta = cssv[rho] / (rho.to(dtype=v.dtype) + 1.0)
    return torch.clamp(v - theta, min=0.0)


def _point_to_convex_hull_distance(
    point: torch.Tensor,
    hull_points: torch.Tensor,
    *,
    max_iter: int = SOFTMAX_CODE_HULL_STEPS,
    tol: float = 1e-8,
) -> torch.Tensor:
    m = hull_points.shape[0]
    if m == 0:
        return torch.tensor(float("nan"), device=point.device, dtype=point.dtype)
    if m == 1:
        return torch.norm(point - hull_points[0])

    gram = hull_points @ hull_points.T
    rhs = hull_points @ point
    step = 1.0 / float(max(m, 1))
    alpha = torch.full((m,), 1.0 / m, device=point.device, dtype=point.dtype)

    for _ in range(max_iter):
        grad = gram @ alpha - rhs
        next_alpha = _project_simplex(alpha - step * grad)
        if torch.max(torch.abs(next_alpha - alpha)) <= tol:
            alpha = next_alpha
            break
        alpha = next_alpha

    projection = alpha @ hull_points
    return torch.norm(point - projection)


def one_vs_rest_convex_hull_distances(
    vectors: torch.Tensor,
    *,
    normalize: bool = True,
    max_classes: int = SOFTMAX_CODE_MAX_CLASSES,
) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [K, d]")

    k = vectors.shape[0]
    if k < 2 or k > max_classes:
        return torch.full((k,), float("nan"), device=vectors.device, dtype=vectors.dtype)

    vectors = vectors.float()
    if normalize:
        vectors = _safe_normalize(vectors, dim=1)

    distances = []
    for class_idx in range(k):
        mask = torch.ones(k, device=vectors.device, dtype=torch.bool)
        mask[class_idx] = False
        distances.append(_point_to_convex_hull_distance(vectors[class_idx], vectors[mask]))
    return torch.stack(distances)


@lru_cache(maxsize=256)
def softmax_code_reference_margin(feature_dim: int, num_classes: int) -> Tuple[float, str]:
    d = int(feature_dim)
    k = int(num_classes)
    if d <= 0 or k <= 1:
        return float("nan"), "undefined"
    if d == 1:
        if k <= 2:
            return 2.0, "antipodal"
        return 0.0, "dimension_1_repeated_points"
    if d == 2:
        return float(1.0 - math.cos(2.0 * math.pi / k)), "regular_polygon"
    if k <= d + 1:
        return float(k / (k - 1)), "simplex_etf"
    if d + 1 < k <= 2 * d:
        return 1.0, "cross_polytope_subset"
    return float("nan"), "unknown_general_case"


def softmax_code_margin_bounds(feature_dim: int, num_classes: int) -> Tuple[float, float]:
    d = int(feature_dim)
    k = int(num_classes)
    if d <= 1 or k <= 1:
        return float("nan"), float("nan")

    log_lower_term = (
        0.5 * math.log(math.pi)
        - math.log(k)
        + math.lgamma((d + 1) / 2)
        - math.lgamma(d / 2 + 1)
    )
    log_upper_term = (
        math.log(2.0 * math.sqrt(math.pi))
        - math.log(k)
        + math.lgamma((d + 1) / 2)
        - math.lgamma(d / 2)
    )
    lower = 0.5 * math.exp((2.0 / (d - 1)) * log_lower_term)
    upper = 2.0 * math.exp((1.0 / (d - 1)) * log_upper_term)
    return float(lower), float(upper)


def _finite_summary(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        nan = float("nan")
        return {
            prefix: nan,
            f"{prefix}_mean": nan,
            f"{prefix}_std": nan,
            f"{prefix}_max": nan,
        }
    return {
        prefix: float(finite.min().item()),
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_std": float(finite.std(unbiased=False).item()),
        f"{prefix}_max": float(finite.max().item()),
    }


def _target_comparison(value: float, target: float, prefix: str) -> Dict[str, float]:
    if math.isfinite(value) and math.isfinite(target) and target > 0:
        relative_error = float((target - value) / target)
        return {
            f"{prefix}_target_gap": float(target - value),
            f"{prefix}_target_ratio": float(value / target),
            f"{prefix}_target_error": relative_error,
            f"{prefix}_margin_error": relative_error,
        }
    return {
        f"{prefix}_target_gap": float("nan"),
        f"{prefix}_target_ratio": float("nan"),
        f"{prefix}_target_error": float("nan"),
        f"{prefix}_margin_error": float("nan"),
    }


def _margin_metrics(
    margins: torch.Tensor,
    *,
    target_margin: float,
    lower_bound: float,
    upper_bound: float,
    prefix: str,
) -> Dict[str, float]:
    out = _finite_summary(margins, f"{prefix}_margin")
    margin = out[f"{prefix}_margin"]
    margin_mean = out[f"{prefix}_margin_mean"]
    margin_std = out[f"{prefix}_margin_std"]

    if math.isfinite(margin_mean) and abs(margin_mean) > EPS:
        out[f"{prefix}_margin_cov"] = float(margin_std / margin_mean)
    else:
        out[f"{prefix}_margin_cov"] = float("nan")
    out.update(_target_comparison(margin, target_margin, prefix))

    out[f"{prefix}_bound_lower_gap"] = (
        float(margin - lower_bound)
        if math.isfinite(margin) and math.isfinite(lower_bound)
        else float("nan")
    )
    out[f"{prefix}_bound_upper_gap"] = (
        float(upper_bound - margin)
        if math.isfinite(margin) and math.isfinite(upper_bound)
        else float("nan")
    )
    if (
        math.isfinite(margin)
        and math.isfinite(lower_bound)
        and math.isfinite(upper_bound)
        and upper_bound > lower_bound
    ):
        out[f"{prefix}_bound_position"] = float((margin - lower_bound) / (upper_bound - lower_bound))
    else:
        out[f"{prefix}_bound_position"] = float("nan")
    return out


def _etf_metrics(centered_means: torch.Tensor) -> Dict[str, float]:
    k, d = centered_means.shape
    feasible = k > 1 and d >= k - 1
    nan = float("nan")
    if not feasible:
        return {
            "nc2_etf_error": nan,
            "nc2_etf_feasible": 0.0,
            "nc2_mean_norm_cov": nan,
            "nc2_mean_coherence": nan,
            "nc2_offdiag_mean": nan,
            "nc2_offdiag_std": nan,
            "nc2_offdiag_min": nan,
            "nc2_offdiag_max": nan,
        }

    means_n = _safe_normalize(centered_means, dim=1)
    gram = means_n @ means_n.T
    eye = torch.eye(k, device=gram.device, dtype=gram.dtype)
    target = eye - (1.0 / (k - 1)) * (1.0 - eye)
    off = _offdiag(gram)
    coherence = torch.sum(torch.abs(off + 1.0 / (k - 1))) / (k * (k - 1))
    return {
        "nc2_etf_error": float((torch.norm(gram - target, p="fro") / torch.norm(target, p="fro").clamp_min(EPS)).item()),
        "nc2_etf_feasible": 1.0,
        "nc2_mean_norm_cov": float(_coefficient_of_variation(centered_means.norm(dim=1)).item()),
        "nc2_mean_coherence": float(coherence.item()),
        "nc2_offdiag_mean": float(off.mean().item()),
        "nc2_offdiag_std": float(off.std(unbiased=False).item()),
        "nc2_offdiag_min": float(off.min().item()),
        "nc2_offdiag_max": float(off.max().item()),
    }


def nc_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor | None,
    num_classes: int,
    logits: torch.Tensor | None = None,
) -> Dict[str, float | str]:
    features = features.detach().float()
    labels = labels.detach().long()
    if classifier_weight is not None:
        classifier_weight = classifier_weight.detach().float()
    if logits is not None:
        logits = logits.detach().float()

    num_samples, dim = features.shape
    means, counts = compute_class_means(features, labels, num_classes)
    class_global_mean = means.mean(dim=0, keepdim=True)
    centered_means = means - class_global_mean
    mean_norms = centered_means.norm(dim=1)

    assigned_means = means[labels]
    within = ((features - assigned_means) ** 2).sum(dim=1).mean()
    between = (centered_means ** 2).sum(dim=1).mean()
    trace_ratio = float((within / between.clamp_min(EPS)).item())
    nc1 = nc1_trace_pinv(features, labels, means, centered_means)
    gnc1 = nc1_trace_pinv(
        features,
        labels,
        means,
        centered_means,
        normalize_by_num_classes=True,
    )

    out: Dict[str, float | str] = {
        "feature_dim": float(dim),
        "num_samples": float(num_samples),
        "min_class_count": float(counts.min().item()) if counts.numel() else float("nan"),
        "nc1_within_to_between": nc1,
        "nc1": nc1,
        "nc1_sw_inv_sb": nc1,
        "nc1_trace_ratio": trace_ratio,
        "gnc1_trace_pinv": gnc1,
        "class_mean_norm_mean": float(mean_norms.mean().item()),
        "class_mean_norm_std": float(mean_norms.std(unbiased=False).item()),
        "effective_rank": effective_rank(features),
        "class_mean_effective_rank": effective_rank(centered_means),
    }
    out.update(_etf_metrics(centered_means))

    target_margin, target_kind = softmax_code_reference_margin(dim, num_classes)
    lower_bound, upper_bound = softmax_code_margin_bounds(dim, num_classes)
    out.update({
        "gnc2_regime": "softmax_code" if num_classes > dim + 1 else "simplex_etf",
        "gnc2_target_margin": target_margin,
        "gnc2_target_kind": target_kind,
        "gnc2_margin_bound_lower": lower_bound,
        "gnc2_margin_bound_upper": upper_bound,
        "gnc2_class_mean_norm_cov": float(_coefficient_of_variation(mean_norms).item()),
    })
    class_mean_margins = one_vs_rest_convex_hull_distances(centered_means)
    out.update(_margin_metrics(
        class_mean_margins,
        target_margin=target_margin,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        prefix="gnc2_class_mean",
    ))

    nc3_alignment = torch.tensor(float("nan"), device=features.device)
    nc3_alignment_std = torch.tensor(float("nan"), device=features.device)
    nc3_self_duality_fro = torch.tensor(float("nan"), device=features.device)
    weight_norm_cov = torch.tensor(float("nan"), device=features.device)
    weight_margin_summary = _margin_metrics(
        torch.empty(0, device=features.device),
        target_margin=target_margin,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        prefix="gnc2_weight",
    )
    if classifier_weight is not None and classifier_weight.shape == means.shape:
        weight_n = _safe_normalize(classifier_weight, dim=1)
        means_n = _safe_normalize(centered_means, dim=1)
        align = (weight_n * means_n).sum(dim=1)
        nc3_alignment = align.mean()
        nc3_alignment_std = align.std(unbiased=False)
        weight_norm_cov = _coefficient_of_variation(classifier_weight.norm(dim=1))
        means_t = centered_means.T / torch.norm(centered_means.T, p="fro").clamp_min(EPS)
        weights_t = classifier_weight.T / torch.norm(classifier_weight.T, p="fro").clamp_min(EPS)
        nc3_self_duality_fro = torch.norm(weights_t - means_t, p="fro") ** 2
        weight_margins = one_vs_rest_convex_hull_distances(weight_n, normalize=False)
        weight_margin_summary = _margin_metrics(
            weight_margins,
            target_margin=target_margin,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            prefix="gnc2_weight",
        )

        if out["nc2_etf_feasible"] == 1.0:
            k = num_classes
            target = torch.eye(k, device=features.device) - (1.0 / (k - 1)) * (
                1.0 - torch.eye(k, device=features.device)
            )
            weight_gram = weight_n @ weight_n.T
            out["nc2_weight_etf_error"] = float(
                (torch.norm(weight_gram - target, p="fro") / torch.norm(target, p="fro").clamp_min(EPS)).item()
            )
            out["nc2_weight_norm_cov"] = float(weight_norm_cov.item())
        else:
            out["nc2_weight_etf_error"] = float("nan")
            out["nc2_weight_norm_cov"] = float("nan")
    else:
        out["nc2_weight_etf_error"] = float("nan")
        out["nc2_weight_norm_cov"] = float("nan")

    out.update(weight_margin_summary)
    out.update({
        "gnc2_weight_norm_cov": float(weight_norm_cov.item()),
        "nc3_alignment": float(nc3_alignment.item()),
        "nc3_align_mean": float(nc3_alignment.item()),
        "nc3_align_std": float(nc3_alignment_std.item()),
        "nc3_self_duality_fro": float(nc3_self_duality_fro.item()),
        "gnc3_alignment_error": float((1.0 - nc3_alignment).item()),
        "gnc3_self_duality_error": float((1.0 - nc3_alignment).item()),
        "gnc3_self_duality_fro": float(nc3_self_duality_fro.item()),
    })

    dist = torch.cdist(features, means)
    ncm_pred = dist.argmin(dim=1)
    ncm_acc = (ncm_pred == labels).float().mean()
    ncc_mismatch = torch.tensor(float("nan"), device=features.device)
    if logits is not None and logits.shape[0] == labels.shape[0]:
        net_pred = logits.argmax(dim=1).to(ncm_pred.device)
        ncc_mismatch = (ncm_pred != net_pred).float().mean()

    out.update({
        "ncm_acc": float(ncm_acc.item()),
        "nc4_ncc_mismatch": float(ncc_mismatch.item()),
    })
    return out
