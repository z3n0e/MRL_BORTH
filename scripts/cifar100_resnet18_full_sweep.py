#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from neural_collapse.constants import CORE_NC_METRIC_KEYS
from wandb_utils import init_wandb_run, wandb_finish, wandb_log

POLL_SECONDS = 10.0
CHILD_SHUTDOWN_TIMEOUT = 30.0
PREFIX_DIMS = "8,16,32,64,128,256,512"
SHORTLIST = "1,5,10,25,50,100"
DEFAULT_WANDB_PROJECT = "MRL_BORTH"
DEFAULT_PREFIX_MASK_SCALE = "none"
DEFAULT_PREFIX_MASK_SCOPE = "batch"
ALLOW_INVERTED_PREFIX_MASK_ENV = "MRL_ALLOW_PREFIX_MASK_SCALE_INVERTED"


class RunInterrupted(BaseException):
    def __init__(self, signum: int):
        self.signum = int(signum)
        self.returncode = 128 + self.signum
        super().__init__(f"interrupted by signal {self.signum}")


def _interrupt_handler(signum, _frame):
    raise RunInterrupted(signum)


def install_interrupt_handlers() -> None:
    signal.signal(signal.SIGINT, _interrupt_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupt_handler)


def env_flag(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    return int(str(value).strip().lower() in {"1", "true", "yes", "on"})


def normalized_exit_code(returncode: int) -> int:
    return 128 + abs(int(returncode)) if int(returncode) < 0 else int(returncode)


def in_wandb_sweep(env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("WANDB_SWEEP_ID"))


def enforce_prefix_mask_scale(args: argparse.Namespace) -> None:
    if str(args.model_prefix_mask_scale).lower() == "inverted" and env_flag(ALLOW_INVERTED_PREFIX_MASK_ENV, 0) != 1:
        print(
            "model.prefix_mask_scale=inverted was supplied; forcing model.prefix_mask_scale=none. "
            f"Set {ALLOW_INVERTED_PREFIX_MASK_ENV}=1 to run an explicit inverted-scaling ablation.",
            file=sys.stderr,
        )
        args.model_prefix_mask_scale = DEFAULT_PREFIX_MASK_SCALE


def enforce_wandb_project(args: argparse.Namespace) -> None:
    if str(args.wandb_project) != DEFAULT_WANDB_PROJECT:
        print(
            f"wandb.project={args.wandb_project!r} was supplied; forcing "
            f"wandb.project={DEFAULT_WANDB_PROJECT!r}.",
            file=sys.stderr,
        )
        args.wandb_project = DEFAULT_WANDB_PROJECT


def child_popen_kwargs() -> dict:
    if os.name == "nt":
        return {}
    return {"start_new_session": True}


def signal_child(process: subprocess.Popen, signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signum)
        else:
            os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def kill_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop_child(process: subprocess.Popen, signum: int, stage: str) -> None:
    if process.poll() is not None:
        return
    print(f"\n[{stage}] interrupt received; forwarding signal {signum} to child process.")
    signal_child(process, signum)
    try:
        process.wait(timeout=CHILD_SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"[{stage}] child did not stop after {CHILD_SHUTDOWN_TIMEOUT:.0f}s; killing it.")
        kill_child(process)
        process.wait()


def add_arg(parser: argparse.ArgumentParser, name: str, **kwargs) -> None:
    flags = [f"--{name}"]
    hyphen_name = name.replace(".", "-").replace("_", "-")
    hyphen_flag = f"--{hyphen_name}"
    if hyphen_flag not in flags:
        flags.append(hyphen_flag)
    parser.add_argument(*flags, dest=name.replace(".", "_"), **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train, classification, NC/GNC, and retrieval for the CIFAR-100 ResNet-18 W&B sweep."
    )
    add_arg(parser, "data.root", default="auto")
    add_arg(parser, "data.num_workers", type=int, default=4)
    parser.add_argument("--output_root", "--output-root", default="./cifar100_sweep_runs")

    add_arg(parser, "model.mrl", type=int, default=1)
    add_arg(parser, "model.arch", default="resnet18")
    add_arg(parser, "model.feature_dim", type=int, default=512)
    add_arg(parser, "model.prefix_dims", default=PREFIX_DIMS)
    add_arg(parser, "model.nesting_start", type=int, default=3)
    add_arg(parser, "model.prefix_mask_prob", type=float, default=0.0)
    add_arg(parser, "model.prefix_mask_scale", default=DEFAULT_PREFIX_MASK_SCALE, choices=("none", "inverted"))
    add_arg(parser, "model.prefix_mask_scope", default=DEFAULT_PREFIX_MASK_SCOPE, choices=("sample", "batch"))
    add_arg(parser, "model.prefix_mask_skip_prob", type=float, default=0.0)

    add_arg(parser, "training.epochs", type=int, default=120)
    add_arg(parser, "training.batch_size", type=int, default=128)
    add_arg(parser, "training.seed", type=int, default=0)
    add_arg(parser, "training.deterministic", type=int, default=1)
    add_arg(parser, "training.weight_decay", type=float, default=0.0005)
    add_arg(parser, "training.label_smoothing", type=float, default=0.1)
    add_arg(parser, "training.mrl_loss_mode", default="all")
    add_arg(parser, "training.sampled_prefix_distribution", default="uniform")
    add_arg(parser, "training.sampled_prefix_log_interval", type=int, default=100)
    add_arg(parser, "validation.batch_size", type=int, default=128)

    add_arg(parser, "lr.lr", type=float, default=0.1)
    add_arg(parser, "lr.warmup_epochs", type=int, default=5)
    add_arg(parser, "lr.min_lr", type=float, default=0.00001)

    add_arg(parser, "eval.workers", type=int, default=4)
    add_arg(parser, "nc.enabled", type=int, default=1)
    add_arg(parser, "nc.interval", type=int, default=10)
    add_arg(parser, "nc.splits", default="train,test")
    add_arg(parser, "nc.workers", type=int, default=4)
    add_arg(parser, "nc.batch_size", type=int, default=128)

    add_arg(parser, "retrieval.index_type", default="exactl2")
    add_arg(parser, "retrieval.k", type=int, default=2048)
    add_arg(parser, "retrieval.shortlist", default=SHORTLIST)
    add_arg(parser, "retrieval.use_gpu", type=int, default=1)
    add_arg(parser, "retrieval.rebuild_index", type=int, default=0)
    add_arg(parser, "retrieval.force_arrays", type=int, default=0)

    add_arg(parser, "wandb.enabled", type=int, default=1)
    add_arg(parser, "wandb.project", default=DEFAULT_WANDB_PROJECT)
    add_arg(parser, "wandb.entity", default=os.environ.get("WANDB_ENTITY", ""))
    add_arg(parser, "wandb.group", default=os.environ.get("WANDB_GROUP", "cifar100-resnet18-mrl-sweep"))
    add_arg(parser, "wandb.name", default=os.environ.get("WANDB_NAME", ""))
    add_arg(parser, "wandb.tags", default="cifar100,resnet18,mrl,sweep")
    add_arg(parser, "wandb.mode", default=os.environ.get("WANDB_MODE", ""))
    add_arg(parser, "wandb.dir", default=os.environ.get("WANDB_DIR", ""))
    parser.add_argument("--wandb-metrics-file", default="wandb_metrics.jsonl")
    return parser.parse_args()


def force_deterministic(args: argparse.Namespace) -> None:
    if int(args.training_deterministic) != 1:
        print("training.deterministic was not 1; forcing deterministic execution.")
    args.training_deterministic = 1
    os.environ["PYTHONHASHSEED"] = str(args.training_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def default_data_root() -> Path:
    for env_name in ("CIFAR100_DIR", "TORCHVISION_DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser()

    username = getpass.getuser()
    for base in (Path("/scratch/a100/users") / username, Path("/scratch/users") / username):
        if base.exists() and os.access(base, os.W_OK):
            return base / ".cache" / "torchvision"

    return Path.home() / ".cache" / "torchvision"


def resolve_data_root(args: argparse.Namespace) -> None:
    raw_value = str(args.data_root).strip()
    if raw_value.lower() in {"", "auto", "default", "user"}:
        path = default_data_root()
    else:
        path = Path(raw_value).expanduser()

    try:
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise PermissionError(f"data.root is not writable: {path}")
    except PermissionError:
        fallback = default_data_root()
        print(f"Cannot write data.root={path}; falling back to {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        if not os.access(fallback, os.W_OK):
            raise PermissionError(f"fallback data.root is not writable: {fallback}")
        path = fallback

    args.data_root = str(path.resolve())


def parse_int_list(value: str) -> list[int]:
    parts = str(value).replace(",", " ").split()
    return [int(part) for part in parts]


def run_id(args: argparse.Namespace) -> str:
    return os.environ.get("WANDB_RUN_ID") or f"seed_{args.training_seed}_{uuid4().hex[:8]}"


def make_paths(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = (ROOT_DIR / args.output_root / run_id(args)).resolve()
    paths = {
        "root": out_dir,
        "trainlog": out_dir / "trainlogs",
        "train_run": out_dir / "trainlogs" / "mrl",
        "eval": out_dir / "eval",
        "checkpoints": out_dir / "checkpoints",
        "nc": out_dir / "neural_collapse",
        "retrieval": out_dir / "retrieval",
        "retrieval_method": out_dir / "retrieval" / "mrl",
        "retrieval_metrics": out_dir / "retrieval_metrics",
    }
    for key, path in paths.items():
        if key != "train_run":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def metrics_path(args: argparse.Namespace, out_dir: Path) -> Path:
    path = Path(args.wandb_metrics_file)
    if not path.is_absolute():
        path = out_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def child_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": str(args.training_seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "WANDB_ENABLED": "0",
    })
    return env


def clean_payload(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if value is not None}


def numeric_value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return None


def append_metric(run, metrics_file: Path, source: str, payload: dict) -> None:
    payload = clean_payload(payload)
    if not payload:
        return
    with metrics_file.open("a") as handle:
        handle.write(json.dumps({"source": source, **payload}) + "\n")
    wandb_log(run, payload)


def command_text(command: list[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stage: str,
    run,
    metrics_file: Path,
) -> None:
    print(f"\n[{stage}] {command_text(command)}")
    start = time.time()
    append_metric(run, metrics_file, f"{stage}_status", {f"stage/{stage}/started": 1})
    process = subprocess.Popen(command, cwd=cwd, env=env, **child_popen_kwargs())
    interrupted = None
    try:
        returncode = process.wait()
    except RunInterrupted as exc:
        interrupted = exc
        stop_child(process, exc.signum, stage)
        returncode = process.returncode if process.returncode is not None else exc.returncode
    except KeyboardInterrupt:
        interrupted = RunInterrupted(signal.SIGINT)
        stop_child(process, signal.SIGINT, stage)
        returncode = process.returncode if process.returncode is not None else interrupted.returncode
    elapsed = time.time() - start
    payload = {
        f"stage/{stage}/exit_code": normalized_exit_code(returncode),
        f"stage/{stage}/time_sec": float(elapsed),
    }
    if int(returncode) < 0:
        payload[f"stage/{stage}/signal"] = abs(int(returncode))
    if interrupted is not None:
        payload[f"stage/{stage}/interrupted"] = 1
        payload[f"stage/{stage}/signal"] = int(interrupted.signum)
    append_metric(run, metrics_file, f"{stage}_status", payload)
    if interrupted is not None:
        raise interrupted
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def train_eval_payload(record: dict) -> dict:
    payload = {}
    handled_keys = {
        "epoch",
        "train_loss",
        "current_lr",
        "loss",
        "val_time",
        "sampled_prefix_counts",
        "sampled_prefix_counts_by_dim",
    }
    if "epoch" in record:
        payload["epoch"] = record["epoch"]
    if "train_loss" in record:
        payload["train/loss"] = record["train_loss"]
    if "current_lr" in record:
        payload["train/lr"] = record["current_lr"]
    if "loss" in record:
        payload["eval/loss"] = record["loss"]
    if "val_time" in record:
        payload["eval/time_sec"] = record["val_time"]
    for key, value in record.items():
        if key.startswith("top_1_"):
            payload[f"eval/top1/dim_{key.removeprefix('top_1_')}"] = value
        elif key.startswith("top_5_"):
            payload[f"eval/top5/dim_{key.removeprefix('top_5_')}"] = value
    counts_by_dim = record.get("sampled_prefix_counts_by_dim", {})
    if isinstance(counts_by_dim, dict):
        for dim, count in counts_by_dim.items():
            payload[f"mrl/sampled_prefix/count_dim_{dim}"] = count
    for key, value in record.items():
        value = numeric_value(value)
        if value is None or key in handled_keys or key.startswith(("top_1_", "top_5_")):
            continue
        payload[f"train/raw/{key}"] = value
    return payload


def train_nc_payload(record: dict) -> dict:
    split = str(record["split"])
    dim = int(record["prefix_dim"])
    payload = {
        "epoch": int(record["epoch"]),
        "dim": dim,
    }
    skip_keys = {
        "event",
        "name",
        "dataset",
        "arch",
        "mode",
        "epoch",
        "split",
        "prefix_dim",
        "timestamp",
        "relative_time",
    }
    for key, value in record.items():
        if key in skip_keys:
            continue
        if key not in CORE_NC_METRIC_KEYS:
            continue
        value = numeric_value(value)
        if value is None:
            continue
        payload[f"nc/{split}/{key}"] = value
        payload[f"nc/{split}/{key}/dim_{dim}"] = value
        payload[f"nc_history/{split}/{key}/dim_{dim}"] = value
        if key.startswith("gnc"):
            payload[f"gnc/{split}/{key}"] = value
            payload[f"gnc/{split}/{key}/dim_{dim}"] = value
            payload[f"gnc_history/{split}/{key}/dim_{dim}"] = value
    return payload


def drain_train_log(run, train_log: Path, metrics_file: Path, offset: int) -> int:
    if not train_log.exists():
        return offset
    if offset > train_log.stat().st_size:
        offset = 0
    with train_log.open() as handle:
        handle.seek(offset)
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                return handle.tell()
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return line_start
            if record.get("event") == "neural_collapse":
                append_metric(run, metrics_file, "train_neural_collapse", train_nc_payload(record))
            else:
                append_metric(run, metrics_file, "train_eval", train_eval_payload(record))


def run_training(args: argparse.Namespace, paths: dict[str, Path], run, metrics_file: Path) -> None:
    if paths["train_run"].exists():
        raise FileExistsError(f"Training run directory already exists: {paths['train_run']}")

    command = [
        sys.executable,
        "train_imagenet.py",
        "--config-file", "rn50_configs/rn18_cifar100.yaml",
        "--model.arch=resnet18",
        "--model.mrl=1",
        f"--model.fixed_feature={args.model_feature_dim}",
        f"--model.nesting_start={args.model_nesting_start}",
        f"--model.prefix_mask_prob={args.model_prefix_mask_prob}",
        f"--model.prefix_mask_scale={args.model_prefix_mask_scale}",
        f"--model.prefix_mask_scope={args.model_prefix_mask_scope}",
        f"--model.prefix_mask_skip_prob={args.model_prefix_mask_skip_prob}",
        f"--data.root={args.data_root}",
        f"--data.num_workers={args.data_num_workers}",
        f"--training.batch_size={args.training_batch_size}",
        f"--training.epochs={args.training_epochs}",
        f"--training.seed={args.training_seed}",
        "--training.deterministic=1",
        f"--training.weight_decay={args.training_weight_decay}",
        f"--training.label_smoothing={args.training_label_smoothing}",
        f"--training.mrl_loss_mode={args.training_mrl_loss_mode}",
        f"--training.sampled_prefix_distribution={args.training_sampled_prefix_distribution}",
        f"--training.sampled_prefix_log_interval={args.training_sampled_prefix_log_interval}",
        f"--lr.lr={args.lr_lr}",
        f"--lr.warmup_epochs={args.lr_warmup_epochs}",
        f"--lr.min_lr={args.lr_min_lr}",
        f"--validation.batch_size={args.validation_batch_size}",
        f"--nc.enabled={args.nc_enabled}",
        f"--nc.interval={args.nc_interval}",
        f"--nc.splits={args.nc_splits}",
        f"--nc.batch_size={args.nc_batch_size}",
        f"--nc.workers={args.nc_workers}",
        f"--logging.folder={paths['trainlog']}",
        "--logging.run_name=mrl",
    ]

    print(f"\n[train] {command_text(command)}")
    start = time.time()
    append_metric(run, metrics_file, "train_status", {"stage/train/started": 1})
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR / "train",
        env=child_env(args),
        **child_popen_kwargs(),
    )
    offset = 0
    train_log = paths["train_run"] / "log"
    interrupted = None
    try:
        while process.poll() is None:
            offset = drain_train_log(run, train_log, metrics_file, offset)
            time.sleep(POLL_SECONDS)
    except RunInterrupted as exc:
        interrupted = exc
        stop_child(process, exc.signum, "train")
    except KeyboardInterrupt:
        interrupted = RunInterrupted(signal.SIGINT)
        stop_child(process, signal.SIGINT, "train")
    finally:
        offset = drain_train_log(run, train_log, metrics_file, offset)

    elapsed = time.time() - start
    returncode = process.returncode if process.returncode is not None else (
        interrupted.returncode if interrupted is not None else 1
    )
    payload = {
        "stage/train/exit_code": normalized_exit_code(returncode),
        "stage/train/time_sec": float(elapsed),
    }
    if int(returncode) < 0:
        payload["stage/train/signal"] = abs(int(returncode))
    if interrupted is not None:
        payload["stage/train/interrupted"] = 1
        payload["stage/train/signal"] = int(interrupted.signum)
    append_metric(run, metrics_file, "train_status", payload)
    if interrupted is not None:
        raise interrupted
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)

    checkpoint = paths["train_run"] / "final_weights.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training finished without checkpoint: {checkpoint}")
    shutil.copy2(checkpoint, paths["checkpoints"] / "mrl_final_weights.pt")
    latest = paths["train_run"] / "latest_weights.pt"
    if latest.exists():
        shutil.copy2(latest, paths["checkpoints"] / "mrl_latest_weights.pt")


def run_classification(args: argparse.Namespace, paths: dict[str, Path], run, metrics_file: Path) -> None:
    output_json = paths["eval"] / "mrl.json"
    command = [
        sys.executable,
        "pytorch_inference.py",
        "--path", str(paths["train_run"] / "final_weights.pt"),
        "--dataset", "CIFAR100",
        "--data_root", str(args.data_root),
        "--arch", "resnet18",
        "--rep_size", str(args.model_feature_dim),
        "--prefix-dims", str(args.model_prefix_dims),
        "--use_blurpool", "0",
        "--workers", str(args.eval_workers),
        "--seed", str(args.training_seed),
        "--deterministic",
        "--metrics_output", str(output_json),
        "--wandb-enabled", "0",
        "--mrl",
    ]
    run_command(command, cwd=ROOT_DIR / "inference", env=child_env(args), stage="classification", run=run, metrics_file=metrics_file)
    log_classification(output_json, run, metrics_file)


def log_classification(path: Path, run, metrics_file: Path) -> None:
    with path.open() as handle:
        data = json.load(handle)
    summary_payload = {}
    for key, value in data.items():
        value = numeric_value(value)
        if value is not None:
            summary_payload[f"classification/{key}"] = value
    if "total_time" in data:
        summary_payload["classification/total_time_sec"] = data.get("total_time")
    append_metric(run, metrics_file, "classification_summary", summary_payload)
    for row in data.get("metrics", []):
        dim = int(row["rep_size"])
        payload = {"dim": dim}
        for key, value in row.items():
            value = numeric_value(value)
            if value is None or key == "rep_size":
                continue
            payload[f"classification/{key}"] = value
            payload[f"classification/{key}/dim_{dim}"] = value
        append_metric(run, metrics_file, "classification", payload)


def run_neural_collapse(args: argparse.Namespace, paths: dict[str, Path], run, metrics_file: Path) -> None:
    output_csv = paths["nc"] / "cifar100_nc_metrics.csv"
    output_json = paths["nc"] / "cifar100_nc_metrics.json"
    command = [
        sys.executable,
        "cifar100_neural_collapse.py",
        "--path", str(paths["train_run"] / "final_weights.pt"),
        "--data-root", str(args.data_root),
        "--arch", "resnet18",
        "--rep-size", str(args.model_feature_dim),
        "--prefix-dims", str(args.model_prefix_dims),
        "--batch-size", str(args.nc_batch_size),
        "--workers", str(args.nc_workers),
        "--seed", str(args.training_seed),
        "--deterministic",
        "--output-csv", str(output_csv),
        "--output-json", str(output_json),
        "--wandb-enabled", "0",
        "--mrl",
    ]
    run_command(command, cwd=ROOT_DIR, env=child_env(args), stage="neural_collapse", run=run, metrics_file=metrics_file)
    log_neural_collapse(output_json, run, metrics_file)


def log_neural_collapse(path: Path, run, metrics_file: Path) -> None:
    with path.open() as handle:
        rows = json.load(handle)
    for row in rows:
        split = str(row["split"])
        dim = int(row["prefix_dim"])
        payload = {"dim": dim}
        for key, value in row.items():
            if key in {"name", "dataset", "arch", "mode", "split"}:
                continue
            if key not in CORE_NC_METRIC_KEYS:
                continue
            value = numeric_value(value)
            if value is None:
                continue
            payload[f"nc/{split}/{key}"] = value
            payload[f"nc/{split}/{key}/dim_{dim}"] = value
            if key.startswith("gnc"):
                payload[f"gnc/{split}/{key}"] = value
                payload[f"gnc/{split}/{key}/dim_{dim}"] = value
        append_metric(run, metrics_file, f"nc_{split}", payload)


def run_retrieval(args: argparse.Namespace, paths: dict[str, Path], run, metrics_file: Path) -> None:
    feature_config = f"mrl1_e0_ff{args.model_feature_dim}"
    train_x = paths["retrieval_method"] / f"CIFAR100_train_{feature_config}-X.npy"
    val_x = paths["retrieval_method"] / f"CIFAR100_val_{feature_config}-X.npy"
    if int(args.retrieval_force_arrays) == 1 or not train_x.exists() or not val_x.exists():
        dump_retrieval_arrays(args, paths, run, metrics_file)
    else:
        print(f"\n[retrieval_arrays] using existing arrays in {paths['retrieval_method']}")

    dims = parse_int_list(args.model_prefix_dims)
    faiss_command = [
        sys.executable,
        "faiss_nn.py",
        "--root", str(paths["retrieval_method"]),
        "--dataset", "CIFAR100",
        "--model", "mrl",
        "--feature-config", feature_config,
        "--rep-size", str(args.model_feature_dim),
        "--index-type", str(args.retrieval_index_type),
        "--k", str(args.retrieval_k),
        "--dims",
        *[str(dim) for dim in dims],
    ]
    if int(args.retrieval_use_gpu) == 1:
        faiss_command.append("--gpu")
    if int(args.retrieval_rebuild_index) == 1:
        faiss_command.append("--rebuild-index")
    run_command(faiss_command, cwd=ROOT_DIR / "retrieval", env=child_env(args), stage="retrieval_faiss", run=run, metrics_file=metrics_file)

    output_json = paths["retrieval_metrics"] / "mrl.json"
    metric_command = [
        sys.executable,
        "compute_metrics.py",
        "--root", str(paths["retrieval_method"]),
        "--dataset", "CIFAR100",
        "--model", "mrl",
        "--feature-config", feature_config,
        "--rep-size", str(args.model_feature_dim),
        "--eval-config", "vanilla",
        "--index-type", str(args.retrieval_index_type),
        "--neighbor-k", str(args.retrieval_k),
        "--output-json", str(output_json),
        "--wandb-enabled", "0",
        "--dims",
        *[str(dim) for dim in dims],
        "--shortlist",
        *[str(k) for k in parse_int_list(args.retrieval_shortlist)],
    ]
    run_command(metric_command, cwd=ROOT_DIR / "retrieval", env=child_env(args), stage="retrieval_metrics", run=run, metrics_file=metrics_file)
    log_retrieval(output_json, run, metrics_file)
    write_retrieval_summary(output_json, paths["root"] / "cifar100_retrieval_summary.csv", args)


def dump_retrieval_arrays(args: argparse.Namespace, paths: dict[str, Path], run, metrics_file: Path) -> None:
    command = [
        sys.executable,
        "pytorch_inference.py",
        "--retrieval",
        "--path", str(paths["train_run"] / "final_weights.pt"),
        "--dataset", "CIFAR100",
        "--data_root", str(args.data_root),
        "--retrieval_array_path", str(paths["retrieval_method"]),
        "--arch", "resnet18",
        "--rep_size", str(args.model_feature_dim),
        "--prefix-dims", str(args.model_prefix_dims),
        "--use_blurpool", "0",
        "--workers", str(args.eval_workers),
        "--seed", str(args.training_seed),
        "--deterministic",
        "--wandb-enabled", "0",
        "--mrl",
    ]
    run_command(command, cwd=ROOT_DIR / "inference", env=child_env(args), stage="retrieval_arrays", run=run, metrics_file=metrics_file)


def log_retrieval(path: Path, run, metrics_file: Path) -> None:
    with path.open() as handle:
        data = json.load(handle)
    seen_top1_dims = set()
    for row in data.get("metrics", []):
        dim = int(row["dim"])
        k = int(row["k"])
        if dim not in seen_top1_dims:
            cmc_payload = {}
            for rank in (1, 5):
                key = f"cmc@{rank}"
                if key in row:
                    cmc_payload[f"retrieval/cmc_at_{rank}"] = row.get(key)
                    cmc_payload[f"retrieval/cmc_at_{rank}/dim_{dim}"] = row.get(key)
            append_metric(run, metrics_file, "retrieval_top1", {
                "dim": dim,
                "retrieval/top1": row.get("top1"),
                f"retrieval/top1/dim_{dim}": row.get("top1"),
                **cmc_payload,
            })
            seen_top1_dims.add(dim)
        payload = {"dim": dim, "k": k}
        for key, value in row.items():
            value = numeric_value(value)
            if value is None or key in {"dim", "k", "cmc@1", "cmc@5"}:
                continue
            payload[f"retrieval/{key}_at_{k}"] = value
            payload[f"retrieval/{key}_at_{k}/dim_{dim}"] = value
        if "mAP" in row:
            payload[f"retrieval/mAP_at_{k}"] = row.get("mAP")
            payload[f"retrieval/mAP_at_{k}/dim_{dim}"] = row.get("mAP")
        if "precision" in row:
            payload[f"retrieval/precision_at_{k}"] = row.get("precision")
            payload[f"retrieval/precision_at_{k}/dim_{dim}"] = row.get("precision")
        if "recall" in row:
            payload[f"retrieval/recall_at_{k}"] = row.get("recall")
            payload[f"retrieval/recall_at_{k}/dim_{dim}"] = row.get("recall")
        if "topk" in row:
            payload[f"retrieval/top{k}"] = row.get("topk")
            payload[f"retrieval/top{k}/dim_{dim}"] = row.get("topk")
        for rank in (1, 5):
            key = f"cmc@{rank}"
            if key in row:
                payload[f"retrieval/cmc_at_{rank}"] = row.get(key)
                payload[f"retrieval/cmc_at_{rank}/dim_{dim}"] = row.get(key)
        append_metric(run, metrics_file, "retrieval", payload)


def write_retrieval_summary(metrics_json: Path, summary_csv: Path, args: argparse.Namespace) -> None:
    with metrics_json.open() as handle:
        data = json.load(handle)
    rows = []
    for metric in data.get("metrics", []):
        rows.append({
            "method": "mrl",
            "model": data.get("model", ""),
            "training_mrl_loss_mode": args.training_mrl_loss_mode,
            "training_sampled_prefix_distribution": args.training_sampled_prefix_distribution,
            "training_sampled_prefix_log_interval": args.training_sampled_prefix_log_interval,
            "training_prefix_mask_prob": args.model_prefix_mask_prob,
            "training_prefix_mask_scale": args.model_prefix_mask_scale,
            "training_prefix_mask_scope": args.model_prefix_mask_scope,
            "training_prefix_mask_skip_prob": args.model_prefix_mask_skip_prob,
            "feature_config": data.get("feature_config", ""),
            "eval_config": data.get("eval_config", ""),
            "index_type": data.get("index_type", ""),
            "dim": metric.get("dim", ""),
            "k": metric.get("k", ""),
            "top1": metric.get("top1", ""),
            "cmc@1": metric.get("cmc@1", ""),
            "cmc@5": metric.get("cmc@5", ""),
            "mAP": metric.get("mAP", ""),
            "precision": metric.get("precision", ""),
            "recall": metric.get("recall", ""),
            "topk": metric.get("topk", ""),
            "neighbors_path": metric.get("neighbors_path", ""),
        })

    fieldnames = [
        "method", "model", "training_mrl_loss_mode",
        "training_sampled_prefix_distribution",
        "training_sampled_prefix_log_interval",
        "training_prefix_mask_prob",
        "training_prefix_mask_scale",
        "training_prefix_mask_scope",
        "training_prefix_mask_skip_prob",
        "feature_config", "eval_config", "index_type",
        "dim", "k", "top1", "cmc@1", "cmc@5", "mAP", "precision", "recall", "topk",
        "neighbors_path",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(args: argparse.Namespace, paths: dict[str, Path], metrics_file: Path) -> None:
    manifest = {
        **vars(args),
        "deterministic": 1,
        "experiment_dir": str(paths["root"]),
        "metrics_file": str(metrics_file),
        "checkpoint": str(paths["train_run"] / "final_weights.pt"),
    }
    with (paths["root"] / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    args = parse_args()
    enforce_prefix_mask_scale(args)
    enforce_wandb_project(args)
    force_deterministic(args)
    resolve_data_root(args)
    paths = make_paths(args)
    wandb_dir = args.wandb_dir or str(paths["root"] / "wandb")
    Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path(args, paths["root"])
    write_manifest(args, paths, metrics_file)

    run = init_wandb_run(
        bool(args.wandb_enabled),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name or paths["root"].name,
        job_type="cifar100_resnet18_full_sweep",
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=wandb_dir,
        config={
            **vars(args),
            "deterministic": 1,
            "experiment_dir": str(paths["root"]),
            "wandb_metrics_file": str(metrics_file),
        },
    )

    exit_code = 0
    install_interrupt_handlers()
    try:
        append_metric(run, metrics_file, "sweep", {"stage/sweep/started": 1})
        run_training(args, paths, run, metrics_file)
        run_classification(args, paths, run, metrics_file)
        run_neural_collapse(args, paths, run, metrics_file)
        run_retrieval(args, paths, run, metrics_file)
        append_metric(run, metrics_file, "sweep", {"stage/sweep/completed": 1, "stage/sweep/exit_code": 0})
    except RunInterrupted as exc:
        exit_code = exc.returncode
        append_metric(run, metrics_file, "sweep", {
            "stage/sweep/interrupted": 1,
            "stage/sweep/signal": int(exc.signum),
            "stage/sweep/exit_code": int(exit_code),
        })
        print("Full CIFAR-100 sweep run interrupted; stopping cleanly.", file=sys.stderr)
    except Exception as exc:
        exit_code = normalized_exit_code(getattr(exc, "returncode", 1) or 1)
        append_metric(run, metrics_file, "sweep", {"stage/sweep/failed": 1, "stage/sweep/exit_code": int(exit_code)})
        print(f"Full CIFAR-100 sweep run failed: {exc}", file=sys.stderr)
    finally:
        wandb_finish(run)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
