from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "inference"))

from cifar_resnet import make_torchvision_model, maybe_apply_cifar_stem, parse_prefix_dims
from MRL import FixedFeatureLayer, MRL_Linear_Layer
from neural_collapse import CORE_NC_METRIC_KEYS, evaluate_nc_rows
from utils import apply_blurpool
from wandb_utils import env_default, env_flag, init_wandb_run, wandb_finish, wandb_log


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR100_NUM_CLASSES = 100
LOGGER = logging.getLogger("cifar100_neural_collapse")


def configure_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
		datefmt="%H:%M:%S",
	)


def set_reproducibility(seed: int, deterministic: bool) -> None:
	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	if deterministic:
		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
			torch.backends.cuda.matmul.allow_tf32 = False
		if hasattr(torch.backends.cudnn, "allow_tf32"):
			torch.backends.cudnn.allow_tf32 = False
		if hasattr(torch, "use_deterministic_algorithms"):
			try:
				torch.use_deterministic_algorithms(True, warn_only=False)
			except TypeError:
				torch.use_deterministic_algorithms(True)
	else:
		torch.backends.cudnn.benchmark = True
		if hasattr(torch, "set_float32_matmul_precision"):
			torch.set_float32_matmul_precision("high")


def seed_worker(worker_id: int) -> None:
	worker_seed = torch.initial_seed() % 2**32
	random.seed(worker_seed + worker_id)
	np.random.seed(worker_seed + worker_id)


def make_loader(dataset, batch_size: int, workers: int, seed: int) -> DataLoader:
	generator = torch.Generator()
	generator.manual_seed(seed)
	kwargs = {
		"dataset": dataset,
		"batch_size": batch_size,
		"shuffle": False,
		"num_workers": workers,
		"pin_memory": torch.cuda.is_available(),
		"persistent_workers": workers > 0,
		"worker_init_fn": seed_worker,
		"generator": generator,
	}
	return DataLoader(**kwargs)


def make_cifar100_loaders(
	data_root: Path,
	batch_size: int,
	workers: int,
	seed: int,
) -> Dict[str, DataLoader]:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
	])
	train_dataset = datasets.CIFAR100(
		root=str(data_root),
		train=True,
		download=True,
		transform=transform,
	)
	test_dataset = datasets.CIFAR100(
		root=str(data_root),
		train=False,
		download=True,
		transform=transform,
	)
	return {
		"train": make_loader(train_dataset, batch_size, workers, seed),
		"test": make_loader(test_dataset, batch_size, workers, seed),
	}


def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
	checkpoint = torch.load(path, map_location="cpu")
	if isinstance(checkpoint, dict) and "model" in checkpoint:
		checkpoint = checkpoint["model"]
	if any(key.startswith("module.") for key in checkpoint.keys()):
		checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
	return checkpoint


def method_name(args: argparse.Namespace) -> str:
	if args.efficient:
		return "mrl_e"
	if args.mrl:
		return "mrl"
	return "fixed_feature"


def build_model(
	args: argparse.Namespace,
	device: torch.device,
) -> Tuple[nn.Module, List[int], int]:
	model = make_torchvision_model(args.arch, pretrained=False)
	model = maybe_apply_cifar_stem(model, "cifar100", args.arch)
	feature_dim = int(model.fc.in_features)
	if args.rep_size > feature_dim:
		raise ValueError(f"rep_size={args.rep_size} exceeds model feature dimension {feature_dim}")

	is_nested = args.mrl or args.efficient
	prefix_dims = parse_prefix_dims(args.prefix_dims, feature_dim, args.nesting_start)
	if not is_nested:
		prefix_dims = [args.rep_size]

	if is_nested:
		model.fc = MRL_Linear_Layer(prefix_dims, num_classes=CIFAR100_NUM_CLASSES, efficient=args.efficient)
	else:
		model.fc = FixedFeatureLayer(args.rep_size, CIFAR100_NUM_CLASSES)

	if args.use_blurpool:
		apply_blurpool(model)

	model.load_state_dict(load_checkpoint(Path(args.path)))
	model = model.to(device)
	if device.type == "cuda":
		model = model.to(memory_format=torch.channels_last)
	model.eval()
	return model, prefix_dims, feature_dim


def evaluate_nc(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	prefix_dims: List[int],
	is_nested: bool,
	split: str,
	args: argparse.Namespace,
	feature_dim: int,
) -> List[Dict[str, float | int | str | bool]]:
	return evaluate_nc_rows(
		model=model,
		loader=loader,
		device=device,
		prefix_dims=prefix_dims,
		is_nested=is_nested,
		split=split,
		num_classes=CIFAR100_NUM_CLASSES,
		dataset="cifar100",
		arch=args.arch,
		mode=method_name(args),
		feature_dim=feature_dim,
		cifar_stem=args.arch == "resnet18",
	)


def write_csv(path: Path, rows: List[Dict[str, float | int | str | bool]]) -> None:
	if not rows:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames: List[str] = []
	for row in rows:
		for key in row.keys():
			if key not in fieldnames:
				fieldnames.append(key)
	with path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def write_json(path: Path, rows: List[Dict[str, float | int | str | bool]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w") as handle:
		json.dump(rows, handle, indent=2)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Measure classical and generalized Neural Collapse metrics for CIFAR-100 ResNet checkpoints."
	)
	parser.add_argument("--path", required=True, help="checkpoint path")
	parser.add_argument("--data-root", default=str(Path.home() / ".cache/torchvision"))
	parser.add_argument("--arch", default="resnet18", help="TorchVision architecture")
	parser.add_argument("--rep-size", type=int, default=512)
	parser.add_argument("--prefix-dims", default="8,16,32,64,128,256,512")
	parser.add_argument("--nesting-start", type=int, default=3)
	parser.add_argument("--mrl", action="store_true")
	parser.add_argument("--efficient", action="store_true")
	parser.add_argument("--use-blurpool", action="store_true")
	parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--deterministic", action="store_true")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output-csv", required=True)
	parser.add_argument("--output-json", default="")
	parser.add_argument("--wandb-enabled", type=int, default=env_flag("WANDB_ENABLED", 1), help="enable W&B logging? (1/0)")
	parser.add_argument("--wandb-project", default=env_default("WANDB_PROJECT", "mrl-borth"))
	parser.add_argument("--wandb-entity", default=env_default("WANDB_ENTITY", ""))
	parser.add_argument("--wandb-group", default=env_default("WANDB_GROUP", ""))
	parser.add_argument("--wandb-name", default=env_default("WANDB_NAME", ""))
	parser.add_argument("--wandb-tags", default=env_default("WANDB_TAGS", ""))
	parser.add_argument("--wandb-mode", default=env_default("WANDB_MODE", ""))
	parser.add_argument("--wandb-dir", default=env_default("WANDB_DIR", ""))
	return parser.parse_args()


def log_nc_rows_to_wandb(wandb_run, rows: List[Dict[str, float | int | str | bool]]) -> None:
	for row in rows:
		split = str(row["split"])
		dim = int(row["prefix_dim"])
		payload = {"dim": dim}
		for key, value in row.items():
			if key in {"name", "dataset", "arch", "mode", "split"}:
				continue
			if key not in CORE_NC_METRIC_KEYS:
				continue
			if isinstance(value, bool):
				payload[f"nc/{split}/{key}"] = int(value)
				payload[f"nc/{split}/{key}/dim_{dim}"] = int(value)
			elif isinstance(value, (int, float)):
				payload[f"nc/{split}/{key}"] = value
				payload[f"nc/{split}/{key}/dim_{dim}"] = value
		wandb_log(wandb_run, payload)


def main() -> int:
	configure_logging()
	args = parse_args()
	set_reproducibility(args.seed, args.deterministic)
	device = torch.device(args.device)
	model, prefix_dims, feature_dim = build_model(args, device)

	LOGGER.info("Model: %s", args.arch)
	LOGGER.info("CIFAR stem: %s", "yes" if args.arch == "resnet18" else "no")
	LOGGER.info("Feature dim: %s", feature_dim)
	LOGGER.info("Prefix dims: %s", prefix_dims)
	LOGGER.info("Mode: %s", method_name(args))
	wandb_run = init_wandb_run(
		bool(args.wandb_enabled),
		project=args.wandb_project,
		entity=args.wandb_entity,
		group=args.wandb_group or f"cifar100_{args.arch}_seed_{args.seed}",
		name=args.wandb_name or f"{method_name(args)}_cifar100_neural_collapse",
		job_type="neural_collapse",
		tags=args.wandb_tags,
		mode=args.wandb_mode,
		dir=args.wandb_dir,
		config={
			**vars(args),
			"feature_dim": feature_dim,
			"resolved_prefix_dims": prefix_dims,
			"num_classes": CIFAR100_NUM_CLASSES,
		},
	)

	loaders = make_cifar100_loaders(
		Path(args.data_root).expanduser(),
		args.batch_size,
		args.workers,
		args.seed,
	)

	all_rows: List[Dict[str, float | int | str | bool]] = []
	for split in args.splits:
		LOGGER.info("Evaluating Neural Collapse metrics on CIFAR-100 %s split", split)
		all_rows.extend(
			evaluate_nc(
				model,
				loaders[split],
				device,
				prefix_dims,
				args.mrl or args.efficient,
				split,
				args,
				feature_dim,
			)
		)

	write_csv(Path(args.output_csv), all_rows)
	if args.output_json:
		write_json(Path(args.output_json), all_rows)
	log_nc_rows_to_wandb(wandb_run, all_rows)
	wandb_finish(wandb_run)
	LOGGER.info("Wrote %d rows to %s", len(all_rows), args.output_csv)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
