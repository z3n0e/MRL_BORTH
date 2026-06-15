#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from wandb_utils import init_wandb_run, wandb_finish, wandb_log


def add_arg(parser: argparse.ArgumentParser, name: str, **kwargs) -> None:
	parser.add_argument(f"--{name}", dest=name.replace(".", "_"), **kwargs)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the full CIFAR-100 experiment inside a W&B sweep.")
	add_arg(parser, "data.root", default=str(Path.home() / ".cache/torchvision"))
	parser.add_argument("--output_root", default="./cifar100_sweep_runs")
	add_arg(parser, "model.mrl", type=int, default=1)
	add_arg(parser, "training.epochs", type=int, default=120)
	add_arg(parser, "training.batch_size", type=int, default=128)
	add_arg(parser, "validation.batch_size", type=int, default=128)
	add_arg(parser, "training.seed", type=int, default=0)
	add_arg(parser, "lr.lr", type=float, default=0.1)
	add_arg(parser, "lr.warmup_epochs", type=int, default=5)
	add_arg(parser, "lr.min_lr", type=float, default=0.00001)
	add_arg(parser, "training.weight_decay", type=float, default=0.0005)
	add_arg(parser, "training.label_smoothing", type=float, default=0.1)
	add_arg(parser, "model.prefix_mask_prob", type=float, default=0.0)
	add_arg(parser, "training.mrl_loss_mode", default="all")
	add_arg(parser, "training.sampled_prefix_distribution", default="uniform")
	add_arg(parser, "wandb.enabled", type=int, default=1)
	add_arg(parser, "wandb.project", default=os.environ.get("WANDB_PROJECT", "mrl-borth"))
	add_arg(parser, "wandb.group", default=os.environ.get("WANDB_GROUP", "cifar100-resnet18-mrl-sweep"))
	add_arg(parser, "wandb.tags", default="cifar100,resnet18,mrl,sweep")
	return parser.parse_args()


def args_config(args: argparse.Namespace) -> dict:
	return vars(args)


def experiment_dir(args: argparse.Namespace) -> Path:
	run_id = os.environ.get("WANDB_RUN_ID") or f"seed_{args.training_seed}_{uuid4().hex[:8]}"
	return (ROOT_DIR / args.output_root / run_id).resolve()


def run_experiment(args: argparse.Namespace, out_dir: Path) -> None:
	env = os.environ.copy()
	env.update({
		"PYTHON": sys.executable,
		"CIFAR100_DIR": args.data_root,
		"EXPERIMENT_DIR": str(out_dir),
		"SEED": str(args.training_seed),
		"EPOCHS": str(args.training_epochs),
		"TRAIN_BATCH_SIZE": str(args.training_batch_size),
		"VAL_BATCH_SIZE": str(args.validation_batch_size),
		"LR": str(args.lr_lr),
		"WARMUP_EPOCHS": str(args.lr_warmup_epochs),
		"MIN_LR": str(args.lr_min_lr),
		"WEIGHT_DECAY": str(args.training_weight_decay),
		"LABEL_SMOOTHING": str(args.training_label_smoothing),
		"PREFIX_MASK_PROB": str(args.model_prefix_mask_prob),
		"MRL_LOSS_MODE": args.training_mrl_loss_mode,
		"SAMPLED_PREFIX_DISTRIBUTION": args.training_sampled_prefix_distribution,
		"RUN_NC_METRICS": "1",
		"RUN_RETRIEVAL_METRICS": "1",
		"WANDB_ENABLED": "0",
	})
	subprocess.run([str(ROOT_DIR / "run_cifar100_experiments.sh")], cwd=ROOT_DIR, env=env, check=True)


def log_train_eval(run, out_dir: Path) -> None:
	log_path = out_dir / "trainlogs" / "mrl" / "log"
	if not log_path.exists():
		print(f"Missing train log for W&B replay: {log_path}")
		return
	with log_path.open() as handle:
		for line in handle:
			record = json.loads(line)
			payload = {}
			if "epoch" in record:
				payload["epoch"] = record["epoch"]
			if "train_loss" in record:
				payload["train/loss"] = record["train_loss"]
			if "current_lr" in record:
				payload["train/lr"] = record["current_lr"]
			if "val_time" in record:
				payload["eval/time_sec"] = record["val_time"]
			for key, value in record.items():
				if key.startswith("top_1_"):
					payload[f"eval/top1/dim_{key.removeprefix('top_1_')}"] = value
				elif key.startswith("top_5_"):
					payload[f"eval/top5/dim_{key.removeprefix('top_5_')}"] = value
			if payload:
				wandb_log(run, payload)


def log_classification(run, out_dir: Path) -> None:
	path = out_dir / "eval" / "mrl.json"
	if not path.exists():
		print(f"Missing classification metrics for W&B replay: {path}")
		return
	with path.open() as handle:
		data = json.load(handle)
	wandb_log(run, {
		"classification/num_images": data.get("num_images"),
		"classification/total_time_sec": data.get("total_time"),
	})
	for row in data.get("metrics", []):
		dim = int(row["rep_size"])
		wandb_log(run, {
			"dim": dim,
			"classification/top1": row.get("top1"),
			"classification/top5": row.get("top5"),
			f"classification/top1/dim_{dim}": row.get("top1"),
			f"classification/top5/dim_{dim}": row.get("top5"),
		})


def log_nc(run, out_dir: Path) -> None:
	path = out_dir / "neural_collapse" / "cifar100_nc_metrics.json"
	if not path.exists():
		print(f"Missing NC metrics for W&B replay: {path}")
		return
	with path.open() as handle:
		rows = json.load(handle)
	for row in rows:
		split = str(row["split"])
		dim = int(row["prefix_dim"])
		payload = {"dim": dim}
		for key, value in row.items():
			if key in {"name", "dataset", "arch", "mode", "split"}:
				continue
			if isinstance(value, bool):
				value = int(value)
			if isinstance(value, (int, float)):
				payload[f"nc/{split}/{key}"] = value
				payload[f"nc/{split}/{key}/dim_{dim}"] = value
		wandb_log(run, payload)


def log_retrieval(run, out_dir: Path) -> None:
	path = out_dir / "retrieval_metrics" / "mrl.json"
	if not path.exists():
		print(f"Missing retrieval metrics for W&B replay: {path}")
		return
	with path.open() as handle:
		data = json.load(handle)
	seen_top1_dims = set()
	for row in data.get("metrics", []):
		dim = int(row["dim"])
		k = int(row["k"])
		if dim not in seen_top1_dims:
			wandb_log(run, {
				"dim": dim,
				"retrieval/top1": row.get("top1"),
				f"retrieval/top1/dim_{dim}": row.get("top1"),
			})
			seen_top1_dims.add(dim)
		wandb_log(run, {
			"dim": dim,
			"k": k,
			f"retrieval/mAP_at_{k}": row.get("mAP"),
			f"retrieval/precision_at_{k}": row.get("precision"),
			f"retrieval/recall_at_{k}": row.get("recall"),
			f"retrieval/top{k}": row.get("topk"),
			f"retrieval/mAP_at_{k}/dim_{dim}": row.get("mAP"),
			f"retrieval/precision_at_{k}/dim_{dim}": row.get("precision"),
			f"retrieval/recall_at_{k}/dim_{dim}": row.get("recall"),
			f"retrieval/top{k}/dim_{dim}": row.get("topk"),
		})


def main() -> int:
	args = parse_args()
	out_dir = experiment_dir(args)
	run = init_wandb_run(
		bool(args.wandb_enabled),
		project=args.wandb_project,
		group=args.wandb_group,
		name=out_dir.name,
		job_type="cifar100_full_sweep",
		tags=args.wandb_tags,
		config={**args_config(args), "experiment_dir": str(out_dir)},
	)
	try:
		run_experiment(args, out_dir)
		log_train_eval(run, out_dir)
		log_classification(run, out_dir)
		log_nc(run, out_dir)
		log_retrieval(run, out_dir)
	finally:
		wandb_finish(run)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
