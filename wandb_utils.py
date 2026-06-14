import math
import os
from typing import Any, Dict, Iterable, Optional


def env_flag(name: str, default: int = 0) -> int:
	value = os.environ.get(name)
	if value is None:
		return int(default)
	return int(str(value).strip().lower() in {"1", "true", "yes", "on"})


def env_default(name: str, default: str = "") -> str:
	return os.environ.get(name, default)


def parse_tags(tags: str | Iterable[str] | None):
	if tags is None:
		return None
	if isinstance(tags, str):
		return [tag.strip() for tag in tags.split(",") if tag.strip()]
	return list(tags)


def _to_wandb_value(value: Any):
	try:
		import numpy as np
	except ImportError:
		np = None
	try:
		import torch
	except ImportError:
		torch = None

	if torch is not None and isinstance(value, torch.Tensor):
		if value.numel() == 1:
			return value.detach().cpu().item()
		return value.detach().cpu().tolist()
	if np is not None and isinstance(value, np.generic):
		return value.item()
	if np is not None and isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, os.PathLike):
		return str(value)
	return value


def clean_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
	cleaned = {}
	for key, value in metrics.items():
		value = _to_wandb_value(value)
		if isinstance(value, float) and not math.isfinite(value):
			continue
		cleaned[key] = value
	return cleaned


def init_wandb_run(
	enabled: bool,
	*,
	project: str = "",
	entity: str = "",
	group: str = "",
	name: str = "",
	job_type: str = "",
	tags: str = "",
	mode: str = "",
	dir: str = "",
	config: Optional[Dict[str, Any]] = None,
):
	if not enabled:
		return None
	try:
		import wandb
	except ImportError:
		print("W&B logging requested but wandb is not installed; continuing without W&B.")
		return None

	init_kwargs = {
		"config": clean_metrics(config or {}),
		"tags": parse_tags(tags),
	}
	if project:
		init_kwargs["project"] = project
	if entity:
		init_kwargs["entity"] = entity
	if group:
		init_kwargs["group"] = group
	if name:
		init_kwargs["name"] = name
	if job_type:
		init_kwargs["job_type"] = job_type
	if mode:
		init_kwargs["mode"] = mode
	if dir:
		init_kwargs["dir"] = dir

	run = wandb.init(**init_kwargs)
	define_common_metrics(run)
	return run


def define_common_metrics(run):
	if run is None:
		return
	try:
		run.define_metric("epoch")
		run.define_metric("dim")
		run.define_metric("k")
		run.define_metric("train/*", step_metric="epoch")
		run.define_metric("eval/*", step_metric="epoch")
		run.define_metric("classification/*", step_metric="dim")
		run.define_metric("retrieval/*", step_metric="dim")
		run.define_metric("nc/*", step_metric="dim")
	except Exception as exc:
		print(f"W&B metric definition failed; continuing without custom axes: {exc}")


def wandb_log(run, metrics: Dict[str, Any], step: int | None = None):
	if run is None:
		return
	payload = clean_metrics(metrics)
	if not payload:
		return
	try:
		run.log(payload, step=step)
	except Exception as exc:
		print(f"W&B log failed; continuing without this W&B record: {exc}")


def wandb_finish(run):
	if run is None:
		return
	try:
		run.finish()
	except Exception as exc:
		print(f"W&B finish failed: {exc}")
