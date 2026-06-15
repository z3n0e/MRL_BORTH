from .constants import CORE_NC_METRIC_KEYS

__all__ = [
    "base_model",
    "classifier_weight",
    "collect_features_and_logits",
    "compute_class_means",
    "CORE_NC_METRIC_KEYS",
    "evaluate_nc_rows",
    "nc_metrics",
    "topk_correct",
]


def __getattr__(name):
    if name in {
        "base_model",
        "classifier_weight",
        "collect_features_and_logits",
        "evaluate_nc_rows",
        "topk_correct",
    }:
        from . import evaluator

        return getattr(evaluator, name)
    if name in {"compute_class_means", "nc_metrics"}:
        from . import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
