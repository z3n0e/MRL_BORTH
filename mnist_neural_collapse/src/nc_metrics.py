from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-12
SOFTMAX_CODE_MAX_CLASSES = 256
SOFTMAX_CODE_HULL_STEPS = 200


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _center_rows(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[0] == x.shape[1]
    k = x.shape[0]
    if k <= 1:
        return x.new_empty(0)
    return x[~torch.eye(k, dtype=torch.bool, device=x.device)]


def effective_rank(x: torch.Tensor, eps: float = EPS) -> float:
    """Entropy effective rank of centered features."""
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    xc = x - x.mean(dim=0, keepdim=True)
    # Compute singular values on CPU-friendly tensor sizes.
    s = torch.linalg.svdvals(xc)
    p = s / s.sum().clamp_min(eps)
    entropy = -(p * torch.log(p.clamp_min(eps))).sum()
    return float(torch.exp(entropy).item())


def effective_rank_from_eigs(eigs: torch.Tensor, eps: float = EPS) -> float:
    eigs = eigs.clamp_min(0)
    total = eigs.sum()
    if total.item() <= eps:
        return 0.0
    p = eigs / total
    entropy = -(p[p > eps] * torch.log(p[p > eps])).sum()
    return float(torch.exp(entropy).item())


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex."""
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
    """Distance from one point to the convex hull of hull_points.

    The projection is a small simplex-constrained least-squares problem. A
    projected-gradient solver keeps this dependency-free and is accurate for the
    low-class-count regimes where these diagnostics are intended to run.
    """
    m = hull_points.shape[0]
    if m == 0:
        return torch.tensor(float("nan"), device=point.device, dtype=point.dtype)
    if m == 1:
        return torch.norm(point - hull_points[0])

    gram = hull_points @ hull_points.T
    rhs = hull_points @ point
    # For unit-normalized hull points, lambda_max(gram) <= trace(gram) = m.
    step = 1.0 / float(max(m, 1))
    alpha = torch.full((m,), 1.0 / m, device=point.device, dtype=point.dtype)

    for _ in range(max_iter):
        grad = gram @ alpha - rhs
        new_alpha = _project_simplex(alpha - step * grad)
        if torch.max(torch.abs(new_alpha - alpha)) <= tol:
            alpha = new_alpha
            break
        alpha = new_alpha

    projection = alpha @ hull_points
    return torch.norm(point - projection)


def one_vs_rest_convex_hull_distances(
    vectors: torch.Tensor,
    *,
    normalize: bool = True,
    max_classes: int = SOFTMAX_CODE_MAX_CLASSES,
) -> torch.Tensor:
    """Return per-class Softmax Code one-vs-rest convex-hull distances.

    For K vectors on the unit sphere, this returns
    dist(v_k, conv({v_j: j != k})) for each class k. The minimum over k is the
    GNC2 Softmax Code margin rho_one-vs-rest.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [K, d]")

    K = vectors.shape[0]
    device = vectors.device
    dtype = vectors.dtype
    if K < 2 or K > max_classes:
        return torch.full((K,), float("nan"), device=device, dtype=dtype)

    vectors = vectors.float()
    if normalize:
        vectors = _safe_normalize(vectors, dim=1)

    distances = []
    for k in range(K):
        mask = torch.ones(K, device=vectors.device, dtype=torch.bool)
        mask[k] = False
        distances.append(_point_to_convex_hull_distance(vectors[k], vectors[mask]))
    return torch.stack(distances).to(device=device, dtype=dtype)


@lru_cache(maxsize=256)
def softmax_code_reference_margin(feature_dim: int, num_classes: int) -> Tuple[float, str]:
    """Known Softmax Code optimum rho_one-vs-rest for common regimes.

    The generalized-NC paper gives closed forms for d=2, K<=d+1, and
    d+1<K<=2d. Outside those regimes the exact optimum is generally not known,
    so this returns NaN and the margin bounds below are the safer reference.
    """
    d = int(feature_dim)
    K = int(num_classes)
    if d <= 0 or K <= 1:
        return float("nan"), "undefined"

    if d == 1:
        if K <= 2:
            return 2.0, "antipodal"
        return 0.0, "dimension_1_repeated_points"

    if d == 2:
        return float(1.0 - math.cos(2.0 * math.pi / K)), "regular_polygon"

    if K <= d + 1:
        return float(K / (K - 1)), "simplex_etf"

    if d + 1 < K <= 2 * d:
        return 1.0, "cross_polytope_subset"

    return float("nan"), "unknown_general_case"


def softmax_code_margin_bounds(feature_dim: int, num_classes: int) -> Tuple[float, float]:
    """The paper's lower/upper bounds on the optimal one-vs-rest margin."""
    d = int(feature_dim)
    K = int(num_classes)
    if d <= 1 or K <= 1:
        return float("nan"), float("nan")

    log_lower_term = (
        0.5 * math.log(math.pi)
        - math.log(K)
        + math.lgamma((d + 1) / 2)
        - math.lgamma(d / 2 + 1)
    )
    log_upper_term = (
        math.log(2.0 * math.sqrt(math.pi))
        - math.log(K)
        + math.lgamma((d + 1) / 2)
        - math.lgamma(d / 2)
    )
    lower = 0.5 * math.exp((2.0 / (d - 1)) * log_lower_term)
    upper = 2.0 * math.exp((1.0 / (d - 1)) * log_upper_term)
    return float(lower), float(upper)


def _tensor_summary(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        nan = float("nan")
        return {
            f"{prefix}": nan,
            f"{prefix}_mean": nan,
            f"{prefix}_std": nan,
            f"{prefix}_max": nan,
        }
    return {
        f"{prefix}": float(finite.min().item()),
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_std": float(finite.std(unbiased=False).item()),
        f"{prefix}_max": float(finite.max().item()),
    }


def _target_comparison(margin: float, target: float, prefix: str) -> Dict[str, float]:
    if math.isfinite(margin) and math.isfinite(target) and target > 0:
        target_error = float((target - margin) / target)
        return {
            f"{prefix}_target_gap": float(target - margin),
            f"{prefix}_target_ratio": float(margin / target),
            f"{prefix}_target_error": target_error,
            f"{prefix}_margin_error": target_error,
        }
    return {
        f"{prefix}_target_gap": float("nan"),
        f"{prefix}_target_ratio": float("nan"),
        f"{prefix}_target_error": float("nan"),
        f"{prefix}_margin_error": float("nan"),
    }


def _softmax_code_geometry_metrics(
    margins: torch.Tensor,
    *,
    target_margin: float,
    lower_bound: float,
    upper_bound: float,
    prefix: str,
) -> Dict[str, float]:
    """Softmax Code geometry diagnostics analogous to NC2 ETF diagnostics.

    The raw GNC2 quantity is the minimum one-vs-rest margin. When the optimum
    margin is known, target_error is the direct lower-is-better analogue of
    nc2_etf_error. When the exact optimum is unknown, the bound columns compare
    the observed margin with the paper's bounds on the optimal margin.
    """
    out = _tensor_summary(margins, f"{prefix}_margin")
    margin = out[f"{prefix}_margin"]
    margin_mean = out[f"{prefix}_margin_mean"]
    margin_std = out[f"{prefix}_margin_std"]

    if math.isfinite(margin_mean) and abs(margin_mean) > EPS:
        out[f"{prefix}_margin_cov"] = float(margin_std / margin_mean)
    else:
        out[f"{prefix}_margin_cov"] = float("nan")

    out.update(_target_comparison(margin, target_margin, prefix))

    if math.isfinite(margin) and math.isfinite(lower_bound):
        out[f"{prefix}_bound_lower_gap"] = float(margin - lower_bound)
    else:
        out[f"{prefix}_bound_lower_gap"] = float("nan")

    if math.isfinite(margin) and math.isfinite(upper_bound):
        out[f"{prefix}_bound_upper_gap"] = float(upper_bound - margin)
    else:
        out[f"{prefix}_bound_upper_gap"] = float("nan")

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


def nc1_trace_pinv(
    features: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    centered_means: torch.Tensor,
    *,
    normalize_by_num_classes: bool = False,
) -> float:
    """Return trace(Sigma_W Sigma_B^dagger), optionally divided by K."""
    K = means.shape[0]
    if K == 0 or features.numel() == 0:
        return float("nan")

    residuals = features - means[labels]
    sigma_w = residuals.T @ residuals / max(int(features.shape[0]), 1)

    sigma_b = centered_means.T @ centered_means / K
    sigma_b_pinv = torch.linalg.pinv(sigma_b)
    value = torch.trace(sigma_w @ sigma_b_pinv)
    if normalize_by_num_classes:
        value = value / K
    return float(value.item())


def _coefficient_of_variation(values: torch.Tensor) -> torch.Tensor:
    return values.std(unbiased=False) / values.mean().clamp_min(EPS)


def _etf_coherence_error(normalized_vectors: torch.Tensor) -> torch.Tensor:
    """Average |cos(v_i,v_j) + 1/(K-1)| over off-diagonal entries."""
    K = normalized_vectors.shape[0]
    if K <= 1:
        return torch.tensor(float("nan"), device=normalized_vectors.device)
    gram = normalized_vectors @ normalized_vectors.T
    err = gram + 1.0 / (K - 1)
    err = err - torch.diag(torch.diag(err))
    return torch.sum(torch.abs(err)) / (K * (K - 1))


def _normalized_fro_distance_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = a / torch.norm(a, p="fro").clamp_min(EPS)
    b_n = b / torch.norm(b, p="fro").clamp_min(EPS)
    return torch.norm(a_n - b_n, p="fro") ** 2


def compute_class_means(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return class means [K,d] and class counts [K]."""
    device = features.device
    d = features.shape[1]
    means = torch.zeros(num_classes, d, device=device, dtype=features.dtype)
    counts = torch.zeros(num_classes, device=device, dtype=features.dtype)
    means.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    means = means / counts.clamp_min(1.0).unsqueeze(1)
    return means, counts


def ufm_geometry_metrics(
    class_means: torch.Tensor,
    classifier_weight: torch.Tensor,
) -> Dict[str, float | int | bool]:
    """Compute the UFM script's terminal geometry metrics on class means.

    `mrl_ufm_geometry.py` optimizes one prototype per class. For a real MNIST
    network, the closest analogue is the K x d matrix of class means together
    with the K x d classifier weight matrix for a prefix.
    """
    H = class_means.detach().float()
    W = classifier_weight.detach().float()
    if H.ndim != 2 or W.ndim != 2:
        raise ValueError("class_means and classifier_weight must be rank-2 tensors")
    if H.shape != W.shape:
        raise ValueError(
            f"class_means shape {tuple(H.shape)} must match "
            f"classifier_weight shape {tuple(W.shape)}"
        )

    device = H.device
    K, d = H.shape
    if K <= 1:
        nan = float("nan")
        return {
            "K": int(K),
            "dim": int(d),
            "etf_feasible": False,
            "ce": nan,
            "accuracy": nan,
            "nc1": 0.0,
            "nc2_etf_error_H": nan,
            "nc2_etf_error_W": nan,
            "offdiag_mean_H": nan,
            "offdiag_std_H": nan,
            "offdiag_min_H": nan,
            "offdiag_max_H": nan,
            "offdiag_mean_W": nan,
            "offdiag_std_W": nan,
            "offdiag_min_W": nan,
            "offdiag_max_W": nan,
            "nc3_align_mean": nan,
            "nc3_align_std": nan,
            "self_duality_error": nan,
            "effective_rank_H": nan,
            "numerical_rank_H": 0,
            "min_angle_H_rad": nan,
            "min_angle_W_rad": nan,
            "spherical_margin_H": nan,
            "spherical_margin_W": nan,
            "logit_margin_min": nan,
            "logit_margin_mean": nan,
            "logit_margin_std": nan,
            "ncm_acc": nan,
            "H_norm_mean": nan,
            "W_norm_mean": nan,
        }

    labels = torch.arange(K, device=device)
    logits = H @ W.T
    ce = F.cross_entropy(logits, labels)
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean()

    Hc = _center_rows(H)
    Wc = _center_rows(W)
    Hn = _safe_normalize(Hc, dim=1)
    Wn = _safe_normalize(Wc, dim=1)

    GH = Hn @ Hn.T
    GW = Wn @ Wn.T
    offH = _offdiag(GH)
    offW = _offdiag(GW)

    etf_target = -1.0 / (K - 1)
    etf_error_H = ((offH - etf_target) ** 2).mean()
    etf_error_W = ((offW - etf_target) ** 2).mean()

    nc3_align = (Hn * Wn).sum(dim=1)
    self_duality = torch.linalg.norm(GH - GW, ord="fro") / K

    eigs = torch.linalg.eigvalsh(Hc @ Hc.T).clamp_min(0)
    effective_rank_H = effective_rank_from_eigs(eigs)
    numerical_rank_H = int((eigs > 1e-6 * eigs.max().clamp_min(EPS)).sum().item())

    max_offdiag_cos_H = offH.max()
    min_offdiag_cos_H = offH.min()
    min_angle_H = torch.arccos(max_offdiag_cos_H.clamp(-1 + 1e-7, 1 - 1e-7))
    spherical_margin_H = 1.0 - max_offdiag_cos_H

    max_offdiag_cos_W = offW.max()
    min_offdiag_cos_W = offW.min()
    min_angle_W = torch.arccos(max_offdiag_cos_W.clamp(-1 + 1e-7, 1 - 1e-7))
    spherical_margin_W = 1.0 - max_offdiag_cos_W

    correct = logits.diag()
    masked = logits.masked_fill(torch.eye(K, dtype=torch.bool, device=device), -float("inf"))
    wrong = masked.max(dim=1).values
    margins = correct - wrong

    dist = torch.cdist(H, H, p=2)
    ncm_pred = (-dist).argmax(dim=1)
    ncm_acc = (ncm_pred == labels).float().mean()

    return {
        "K": int(K),
        "dim": int(d),
        "etf_feasible": bool(d >= K - 1),
        "ce": float(ce.item()),
        "accuracy": float(acc.item()),
        "nc1": 0.0,
        "nc2_etf_error_H": float(etf_error_H.item()),
        "nc2_etf_error_W": float(etf_error_W.item()),
        "offdiag_mean_H": float(offH.mean().item()),
        "offdiag_std_H": float(offH.std(unbiased=False).item()),
        "offdiag_min_H": float(min_offdiag_cos_H.item()),
        "offdiag_max_H": float(max_offdiag_cos_H.item()),
        "offdiag_mean_W": float(offW.mean().item()),
        "offdiag_std_W": float(offW.std(unbiased=False).item()),
        "offdiag_min_W": float(min_offdiag_cos_W.item()),
        "offdiag_max_W": float(max_offdiag_cos_W.item()),
        "nc3_align_mean": float(nc3_align.mean().item()),
        "nc3_align_std": float(nc3_align.std(unbiased=False).item()),
        "self_duality_error": float(self_duality.item()),
        "effective_rank_H": float(effective_rank_H),
        "numerical_rank_H": int(numerical_rank_H),
        "min_angle_H_rad": float(min_angle_H.item()),
        "min_angle_W_rad": float(min_angle_W.item()),
        "spherical_margin_H": float(spherical_margin_H.item()),
        "spherical_margin_W": float(spherical_margin_W.item()),
        "logit_margin_min": float(margins.min().item()),
        "logit_margin_mean": float(margins.mean().item()),
        "logit_margin_std": float(margins.std(unbiased=False).item()),
        "ncm_acc": float(ncm_acc.item()),
        "H_norm_mean": float(H.norm(dim=1).mean().item()),
        "W_norm_mean": float(W.norm(dim=1).mean().item()),
    }


def nc_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor | None,
    num_classes: int,
    logits: torch.Tensor | None = None,
) -> Dict[str, float | str]:
    """Compute classical and generalized Neural Collapse diagnostics.

    Args:
        features: Tensor [N,d]. Use train-set features for canonical NC measurement.
        labels: Tensor [N].
        classifier_weight: Tensor [K,d], or None if NC3 is not desired.
        num_classes: number of classes K.
        logits: Optional Tensor [N,K] for the NC4 NCC-vs-network mismatch.

    Returns:
        Dictionary of scalar metrics.
    """
    features = features.detach().float()
    labels = labels.detach().long()
    if classifier_weight is not None:
        classifier_weight = classifier_weight.detach().float()
    if logits is not None:
        logits = logits.detach().float()

    device = features.device
    K = num_classes
    N, d = features.shape

    means, counts = compute_class_means(features, labels, K)
    class_global_mean = means.mean(dim=0, keepdim=True)
    centered_means = means - class_global_mean

    # Legacy scalar NC1 proxy retained for continuity with earlier CSVs.
    assigned_means = means[labels]
    within = ((features - assigned_means) ** 2).sum(dim=1).mean()
    between = (centered_means ** 2).sum(dim=1).mean()
    nc1_trace_ratio = within / between.clamp_min(EPS)

    # Standard NC1 from Papyan et al. / the reference notebook: Tr{Sw Sb^-1}.
    nc1 = nc1_trace_pinv(
        features,
        labels,
        means,
        centered_means,
        normalize_by_num_classes=False,
    )
    gnc1 = nc1_trace_pinv(
        features,
        labels,
        means,
        centered_means,
        normalize_by_num_classes=True,
    )

    # Classical NC2: simplex ETF geometry. This is only geometrically feasible
    # when K vertices fit in d dimensions, i.e. d >= K - 1.
    etf_feasible = K > 1 and d >= K - 1
    nan = torch.tensor(float("nan"), device=device)
    mean_norms = centered_means.norm(dim=1)
    centered_means_n = _safe_normalize(centered_means, dim=1)
    gram = centered_means_n @ centered_means_n.T
    eye = torch.eye(K, device=device, dtype=gram.dtype)
    off_mask = ~eye.bool()
    off = gram[off_mask] if K > 1 else torch.empty(0, device=device, dtype=gram.dtype)

    nc2_mean_norm_cov = nan
    nc2_mean_coherence = nan
    nc2_weight_norm_cov = nan
    nc2_weight_coherence = nan
    nc2_weight_etf_error = nan
    if etf_feasible:
        target = eye - (1.0 / (K - 1)) * (1.0 - eye)
        etf_error = torch.norm(gram - target, p="fro") / torch.norm(target, p="fro").clamp_min(EPS)
        nc2_mean_norm_cov = _coefficient_of_variation(mean_norms)
        nc2_mean_coherence = _etf_coherence_error(centered_means_n)
        off_mean = off.mean()
        off_std = off.std(unbiased=False)
        off_min = off.min()
        off_max = off.max()
    else:
        etf_error = nan
        off_mean = nan
        off_std = nan
        off_min = nan
        off_max = nan

    # GNC2: Softmax Code one-vs-rest distance. The paper states this for
    # classifier weights; we also compute it on class means to inspect the
    # representation geometry directly.
    target_margin, target_kind = softmax_code_reference_margin(d, K)
    lower_bound, upper_bound = softmax_code_margin_bounds(d, K)
    class_mean_margins = one_vs_rest_convex_hull_distances(centered_means_n)
    class_mean_margin_summary = _softmax_code_geometry_metrics(
        class_mean_margins,
        target_margin=target_margin,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        prefix="gnc2_class_mean",
    )
    gnc2_class_mean_norm_cov = _coefficient_of_variation(mean_norms)

    # NC3: classifier-to-class-mean alignment.
    nc3_mean = torch.tensor(float("nan"), device=device)
    nc3_std = torch.tensor(float("nan"), device=device)
    nc3_self_duality_fro = torch.tensor(float("nan"), device=device)
    gnc3_error = torch.tensor(float("nan"), device=device)
    gnc3_self_duality_fro = torch.tensor(float("nan"), device=device)
    gnc2_weight_norm_cov = torch.tensor(float("nan"), device=device)
    weight_margin_summary = _softmax_code_geometry_metrics(
        torch.empty(0, device=device),
        target_margin=target_margin,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        prefix="gnc2_weight",
    )
    if classifier_weight is not None:
        W = classifier_weight
        if W.shape == means.shape:
            Wn = _safe_normalize(W, dim=1)
            Mn = _safe_normalize(centered_means, dim=1)
            align = (Wn * Mn).sum(dim=1)
            nc3_mean = align.mean()
            nc3_std = align.std(unbiased=False)
            nc3_self_duality_fro = _normalized_fro_distance_squared(W.T, centered_means.T)
            gnc3_self_duality_fro = nc3_self_duality_fro
            gnc3_error = 1.0 - nc3_mean
            gnc2_weight_norm_cov = _coefficient_of_variation(W.norm(dim=1))
            if etf_feasible:
                weight_gram = Wn @ Wn.T
                target = eye - (1.0 / (K - 1)) * (1.0 - eye)
                nc2_weight_etf_error = (
                    torch.norm(weight_gram - target, p="fro")
                    / torch.norm(target, p="fro").clamp_min(EPS)
                )
                nc2_weight_norm_cov = _coefficient_of_variation(W.norm(dim=1))
                nc2_weight_coherence = _etf_coherence_error(Wn)
            weight_margins = one_vs_rest_convex_hull_distances(Wn, normalize=False)
            weight_margin_summary = _softmax_code_geometry_metrics(
                weight_margins,
                target_margin=target_margin,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                prefix="gnc2_weight",
            )

    # Nearest class-center train/test diagnostic.
    # Uses centered means only for geometry, but nearest center on raw means.
    dist = torch.cdist(features, means)
    ncm_pred = dist.argmin(dim=1)
    ncm_acc = (ncm_pred == labels).float().mean()
    ncc_mismatch = torch.tensor(float("nan"), device=device)
    if logits is not None and logits.shape[0] == labels.shape[0]:
        net_pred = logits.argmax(dim=1).to(ncm_pred.device)
        ncc_mismatch = (ncm_pred != net_pred).float().mean()

    out = {
        "feature_dim": float(d),
        "num_samples": float(N),
        "nc1": nc1,
        "nc1_sw_inv_sb": nc1,
        "nc1_trace_ratio": float(nc1_trace_ratio.item()),
        "gnc1_trace_pinv": gnc1,
        "nc2_etf_error": float(etf_error.item()),
        "nc2_weight_etf_error": float(nc2_weight_etf_error.item()),
        "nc2_etf_feasible": float(etf_feasible),
        "nc2_mean_norm_cov": float(nc2_mean_norm_cov.item()),
        "nc2_weight_norm_cov": float(nc2_weight_norm_cov.item()),
        "nc2_mean_coherence": float(nc2_mean_coherence.item()),
        "nc2_weight_coherence": float(nc2_weight_coherence.item()),
        "nc2_offdiag_mean": float(off_mean.item()),
        "nc2_offdiag_std": float(off_std.item()),
        "nc2_offdiag_min": float(off_min.item()),
        "nc2_offdiag_max": float(off_max.item()),
        "gnc2_regime": "softmax_code" if K > d + 1 else "simplex_etf",
        "gnc2_target_margin": target_margin,
        "gnc2_target_kind": target_kind,
        "gnc2_margin_bound_lower": lower_bound,
        "gnc2_margin_bound_upper": upper_bound,
        "gnc2_class_mean_norm_cov": float(gnc2_class_mean_norm_cov.item()),
        "gnc2_weight_norm_cov": float(gnc2_weight_norm_cov.item()),
        "class_mean_norm_mean": float(mean_norms.mean().item()),
        "class_mean_norm_std": float(mean_norms.std(unbiased=False).item()),
        "nc3_align_mean": float(nc3_mean.item()),
        "nc3_align_std": float(nc3_std.item()),
        "nc3_self_duality_fro": float(nc3_self_duality_fro.item()),
        "gnc3_self_duality_error": float(gnc3_error.item()),
        "gnc3_self_duality_fro": float(gnc3_self_duality_fro.item()),
        "ncm_acc": float(ncm_acc.item()),
        "nc4_ncc_mismatch": float(ncc_mismatch.item()),
        "effective_rank": effective_rank(features),
    }
    out.update(class_mean_margin_summary)
    out.update(weight_margin_summary)
    return out
