from pathlib import Path


DEFAULT_NESTING_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

MODEL_DIRS = {
    "mrl": "mrl",
    "mrl_e": "mrl_e",
    "ff": "ff",
    "t_orthogonal_mrl": "t_orthogonal_mrl",
    "bor_mrl": "bor_mrl",
    "bor_block_mrl": "bor_block_mrl",
    "cascade_stop_gradient_mrl": "cascade_stop_gradient_mrl",
    "bor_mrl_frozen": "bor_mrl_frozen",
    "bor_mrl_cayley": "bor_mrl_cayley",
    "bor_mrl_householder": "bor_mrl_householder",
    "bor_mrl_residual": "bor_mrl_residual",
}

MODEL_ALIASES = {
    "mrl": "mrl",
    "mrl_e": "mrl_e",
    "mrle": "mrl_e",
    "mrl-e": "mrl_e",
    "ff": "ff",
    "fixed": "ff",
    "fixed_feature": "ff",
    "full_feature": "ff",
    "t": "t_orthogonal_mrl",
    "t_orthogonal": "t_orthogonal_mrl",
    "t_orthogonal_mrl": "t_orthogonal_mrl",
    "t-orthogonal-mrl": "t_orthogonal_mrl",
    "bor": "bor_mrl",
    "bor_mrl": "bor_mrl",
    "bor-mrl": "bor_mrl",
    "bor_block_mrl": "bor_block_mrl",
    "bor-block-mrl": "bor_block_mrl",
    "cascade_stop_gradient_mrl": "cascade_stop_gradient_mrl",
    "cascade-stop-gradient-mrl": "cascade_stop_gradient_mrl",
    "cascade_sg_mrl": "cascade_stop_gradient_mrl",
    "cascade-sg-mrl": "cascade_stop_gradient_mrl",
    "bor_mrl_frozen": "bor_mrl_frozen",
    "bor-mrl-frozen": "bor_mrl_frozen",
    "bor_mrl_cayley": "bor_mrl_cayley",
    "bor-mrl-cayley": "bor_mrl_cayley",
    "bor_mrl_householder": "bor_mrl_householder",
    "bor-mrl-householder": "bor_mrl_householder",
    "bor_mrl_residual": "bor_mrl_residual",
    "bor-mrl-residual": "bor_mrl_residual",
    "bor_residual_mrl": "bor_mrl_residual",
    "bor-residual-mrl": "bor_mrl_residual",
}


def normalize_model_name(model):
    key = model.strip().strip("/").lower().replace("-", "_")
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model {model!r}. Expected one of {sorted(MODEL_ALIASES)}")
    return MODEL_ALIASES[key]


def model_subdir(model):
    return MODEL_DIRS[normalize_model_name(model)]


def default_feature_config_candidates(model, rep_size):
    model = normalize_model_name(model)
    rep_size = int(rep_size)
    if model == "mrl":
        return [f"mrl1_e0_ff{rep_size}"]
    if model == "mrl_e":
        return [
            f"mrl1_e1_ff{rep_size}",
            f"mrl0_e1_ff{rep_size}",  # Legacy notebook naming.
        ]
    if model == "t_orthogonal_mrl" or model.startswith("bor_"):
        return [f"mrl0_e0_ff{rep_size}"]
    return [f"mrl0_e0_ff{rep_size}"]


def dataset_name_candidates(dataset):
    candidates = [str(dataset)]
    lower = str(dataset).lower()
    if lower == "cifar100":
        candidates.extend(["CIFAR100", "cifar100"])
    elif lower in {"1k", "imagenet1k", "imagenet-1k"}:
        candidates.extend(["1K", "1k"])
    elif lower == "v2":
        candidates.extend(["V2", "v2"])
    elif lower == "4k":
        candidates.extend(["4K", "4k"])
    return list(dict.fromkeys(candidates))


def resolve_feature_path(root, dataset, split, config_candidates, suffix):
    root = Path(root)
    tried = []
    for dataset_name in dataset_name_candidates(dataset):
        for config in config_candidates:
            path = root / f"{dataset_name}_{split}_{config}-{suffix}.npy"
            tried.append(path)
            if path.exists():
                return path
    tried_str = "\n  ".join(str(path) for path in tried)
    raise FileNotFoundError(f"Could not find feature file. Tried:\n  {tried_str}")


def resolve_feature_pair(root, db_dataset, query_dataset, config_candidates, suffix):
    last_error = None
    for config in config_candidates:
        try:
            db_path = resolve_feature_path(root, db_dataset, "train", [config], suffix)
            query_path = resolve_feature_path(root, query_dataset, "val", [config], suffix)
            return config, db_path, query_path
        except FileNotFoundError as exc:
            last_error = exc
    raise last_error


def format_list_for_filename(values):
    return str([int(value) for value in values])
