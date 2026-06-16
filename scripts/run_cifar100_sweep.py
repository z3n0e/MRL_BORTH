#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from main import main


SUBCOMMANDS = {"train", "eval", "nc", "retrieval", "full"}
LEGACY_ARG_MAP = {
    "--data.root": "--data-root",
    "--data.num_workers": "--workers",
    "--output_root": "--output-root",
    "--model.feature_dim": "--feature-dim",
    "--model.prefix_dims": "--prefix-dims",
    "--model.prefix_mask_prob": "--prefix-mask-prob",
    "--model.prefix_mask_scale": "--prefix-mask-scale",
    "--model.prefix_mask_scope": "--prefix-mask-scope",
    "--training.epochs": "--epochs",
    "--training.batch_size": "--train-batch-size",
    "--training.seed": "--seed",
    "--training.weight_decay": "--weight-decay",
    "--training.label_smoothing": "--label-smoothing",
    "--training.mrl_loss_mode": "--mrl-loss-mode",
    "--training.sampled_prefix_distribution": "--sampled-prefix-distribution",
    "--training.sampled_prefix_log_interval": "--sampled-prefix-log-interval",
    "--validation.batch_size": "--val-batch-size",
    "--lr.lr": "--lr",
    "--lr.warmup_epochs": "--warmup-epochs",
    "--lr.min_lr": "--min-lr",
    "--eval.workers": "--eval-workers",
    "--nc.enabled": "--nc-enabled",
    "--nc.interval": "--nc-interval",
    "--nc.splits": "--nc-splits",
    "--nc.workers": "--nc-workers",
    "--nc.batch_size": "--nc-batch-size",
    "--retrieval.index_type": "--index-type",
    "--retrieval.k": "--neighbor-k",
    "--retrieval.shortlist": "--shortlist",
    "--retrieval.use_gpu": "--use-gpu",
    "--retrieval.rebuild_index": "--rebuild-index",
    "--retrieval.force_arrays": "--force-arrays",
    "--wandb.enabled": "--wandb-enabled",
    "--wandb.project": "--wandb-project",
    "--wandb.entity": "--wandb-entity",
    "--wandb.group": "--wandb-group",
    "--wandb.name": "--wandb-name",
    "--wandb.tags": "--wandb-tags",
    "--wandb.mode": "--wandb-mode",
    "--wandb.dir": "--wandb-dir",
}
IGNORED_LEGACY_ARGS = {
    "--model.mrl",
    "--model.arch",
    "--model.nesting_start",
    "--training.deterministic",
}


def translate_arg(arg: str) -> str | None:
    for old, new in LEGACY_ARG_MAP.items():
        if arg == old:
            return new
        if arg.startswith(old + "="):
            return new + arg[len(old):]
    for old in IGNORED_LEGACY_ARGS:
        if arg == old or arg.startswith(old + "="):
            return None
    return arg


def normalize_argv(argv: list[str]) -> list[str]:
    args = []
    skip_next = False
    for index, arg in enumerate(argv[1:]):
        if skip_next:
            skip_next = False
            continue
        translated = translate_arg(arg)
        if translated is None:
            next_index = index + 2
            if next_index < len(argv) and not argv[next_index].startswith("--"):
                skip_next = True
            continue
        args.append(translated)
    if not args or args[0] not in SUBCOMMANDS:
        args.insert(0, "full")
    return [argv[0], *args]


if __name__ == "__main__":
    sys.argv = normalize_argv(sys.argv)
    raise SystemExit(main())
