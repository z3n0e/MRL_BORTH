#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiment_logging import MetricLogger, configure_logging


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PREFIX_DIMS = "8,16,32,64,128,256,512"
LOGGER_NAME = "cifar100"
CHILD_SHUTDOWN_TIMEOUT = 30.0
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


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_data_root() -> str:
    return os.environ.get("CIFAR100_DIR", str(Path.home() / ".cache" / "torchvision"))


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
    if not hasattr(args, "prefix_mask_scale"):
        return
    if str(args.prefix_mask_scale).lower() == "inverted" and env_flag(ALLOW_INVERTED_PREFIX_MASK_ENV, 0) != 1:
        print(
            "prefix-mask-scale=inverted was supplied; forcing prefix-mask-scale=none. "
            f"Set {ALLOW_INVERTED_PREFIX_MASK_ENV}=1 to run an explicit inverted-scaling ablation.",
            file=sys.stderr,
        )
        args.prefix_mask_scale = DEFAULT_PREFIX_MASK_SCALE


def enforce_wandb_project(args: argparse.Namespace) -> None:
    if not hasattr(args, "wandb_project"):
        return
    if str(args.wandb_project) != DEFAULT_WANDB_PROJECT:
        print(
            f"wandb-project={args.wandb_project!r} was supplied; forcing "
            f"wandb-project={DEFAULT_WANDB_PROJECT!r}.",
            file=sys.stderr,
        )
        args.wandb_project = DEFAULT_WANDB_PROJECT


def command_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


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
    import logging

    logger = logging.getLogger(LOGGER_NAME)
    logger.warning("[%s] interrupt received; forwarding signal %s to child process", stage, signum)
    signal_child(process, signum)
    try:
        process.wait(timeout=CHILD_SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        logger.warning("[%s] child did not stop after %.0fs; killing it", stage, CHILD_SHUTDOWN_TIMEOUT)
        kill_child(process)
        process.wait()


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], stage: str, metrics: MetricLogger) -> None:
    import logging

    logger = logging.getLogger(LOGGER_NAME)
    logger.info("[%s] %s", stage, command_text(command))
    start = time.time()
    metrics.log(stage, {f"stage/{stage}/started": 1})
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
    metrics.log(stage, payload)
    if interrupted is not None:
        raise interrupted
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def child_env(args: argparse.Namespace, *, wandb_enabled: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    wandb_env = {
        "WANDB_ENABLED": getattr(args, "wandb_enabled", None),
        "WANDB_PROJECT": DEFAULT_WANDB_PROJECT,
        "WANDB_GROUP": getattr(args, "wandb_group", None),
        "WANDB_NAME": getattr(args, "wandb_name", None),
        "WANDB_TAGS": getattr(args, "wandb_tags", None),
        "WANDB_MODE": getattr(args, "wandb_mode", None),
        "WANDB_DIR": getattr(args, "wandb_dir", None),
    }
    if getattr(args, "wandb_entity", None) not in {None, ""}:
        wandb_env["WANDB_ENTITY"] = getattr(args, "wandb_entity")
    for key, value in wandb_env.items():
        if value not in {None, ""}:
            env[key] = str(value)
    if wandb_enabled is not None:
        env["WANDB_ENABLED"] = str(wandb_enabled)
    return env


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=default_data_root(), help="CIFAR-100 torchvision root")
    parser.add_argument("--output-root", default=str(ROOT_DIR / "cifar100_runs"))
    parser.add_argument("--output-dir", default="", help="explicit output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--prefix-dims", default=DEFAULT_PREFIX_DIMS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--wandb-enabled", type=int, default=env_flag("WANDB_ENABLED", 1))
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", ""))
    parser.add_argument("--wandb-tags", default=os.environ.get("WANDB_TAGS", ""))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", ""))
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", ""))


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mrl-loss-mode", default="all")
    parser.add_argument("--sampled-prefix-distribution", default="uniform")
    parser.add_argument("--sampled-prefix-log-interval", type=int, default=100)
    parser.add_argument("--prefix-mask-prob", type=float, default=0.0)
    parser.add_argument("--prefix-mask-scale", default=DEFAULT_PREFIX_MASK_SCALE, choices=("none", "inverted"))
    parser.add_argument("--prefix-mask-scope", default=DEFAULT_PREFIX_MASK_SCOPE, choices=("sample", "batch"))


def add_nc_args(parser: argparse.ArgumentParser, *, default_enabled: int = 1) -> None:
    parser.add_argument("--nc-enabled", type=int, default=default_enabled)
    parser.add_argument("--nc-workers", type=int, default=4)
    parser.add_argument("--nc-batch-size", type=int, default=128)
    parser.add_argument("--nc-interval", type=int, default=10)
    parser.add_argument("--nc-splits", default="train,test")


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-type", default="exactl2")
    parser.add_argument("--neighbor-k", type=int, default=2048)
    parser.add_argument("--shortlist", default="1,5,10,25,50,100")
    parser.add_argument("--use-gpu", type=int, default=1)
    parser.add_argument("--rebuild-index", type=int, default=0)
    parser.add_argument("--force-arrays", type=int, default=0)


def output_dir(args: argparse.Namespace, suffix: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (Path(args.output_root).expanduser() / f"cifar100_{suffix}_seed_{args.seed}_{timestamp()}").resolve()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    checkpoint = getattr(args, "checkpoint", "")
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    candidates = [
        run_dir / "trainlogs" / "mrl" / "final_weights.pt",
        run_dir / "checkpoints" / "mrl_final_weights.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find checkpoint. Tried:\n  {tried}")


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in str(value).replace(",", " ").split() if part.strip()]


def train_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    return [
        args.python,
        "train_imagenet.py",
        "--config-file",
        "rn50_configs/rn18_cifar100.yaml",
        "--model.mrl=1",
        f"--model.fixed_feature={args.feature_dim}",
        "--model.nesting_start=3",
        f"--model.prefix_mask_prob={args.prefix_mask_prob}",
        f"--model.prefix_mask_scale={args.prefix_mask_scale}",
        f"--model.prefix_mask_scope={args.prefix_mask_scope}",
        f"--data.root={args.data_root}",
        f"--data.num_workers={args.workers}",
        f"--training.batch_size={args.train_batch_size}",
        f"--training.epochs={args.epochs}",
        f"--training.seed={args.seed}",
        "--training.deterministic=1",
        f"--training.weight_decay={args.weight_decay}",
        f"--training.label_smoothing={args.label_smoothing}",
        f"--training.mrl_loss_mode={args.mrl_loss_mode}",
        f"--training.sampled_prefix_distribution={args.sampled_prefix_distribution}",
        f"--training.sampled_prefix_log_interval={args.sampled_prefix_log_interval}",
        f"--lr.lr={args.lr}",
        f"--lr.warmup_epochs={args.warmup_epochs}",
        f"--lr.min_lr={args.min_lr}",
        f"--validation.batch_size={args.val_batch_size}",
        f"--nc.enabled={args.nc_enabled}",
        f"--nc.interval={args.nc_interval}",
        f"--nc.splits={args.nc_splits}",
        f"--nc.batch_size={args.nc_batch_size}",
        f"--nc.workers={args.nc_workers}",
        f"--logging.folder={run_dir / 'trainlogs'}",
        "--logging.run_name=mrl",
    ]


def run_train(args: argparse.Namespace) -> int:
    run_dir = output_dir(args, "train")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricLogger(run_dir / "metrics.jsonl")
    run_command(
        train_command(args, run_dir),
        cwd=ROOT_DIR / "train",
        env=child_env(args),
        stage="train",
        metrics=metrics,
    )
    checkpoint = run_dir / "trainlogs" / "mrl" / "final_weights.pt"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        shutil.copy2(checkpoint, checkpoint_dir / "mrl_final_weights.pt")
    print(run_dir)
    return 0


def run_eval(args: argparse.Namespace) -> int:
    checkpoint = resolve_checkpoint(args)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else output_dir(args, "eval")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else eval_dir / "mrl.json"
    metrics = MetricLogger(run_dir / "metrics.jsonl")
    command = [
        args.python,
        "pytorch_inference.py",
        "--path",
        str(checkpoint),
        "--dataset",
        "CIFAR100",
        "--data_root",
        str(args.data_root),
        "--arch",
        "resnet18",
        "--rep_size",
        str(args.feature_dim),
        "--prefix-dims",
        str(args.prefix_dims),
        "--use_blurpool",
        "0",
        "--workers",
        str(args.eval_workers),
        "--seed",
        str(args.seed),
        "--deterministic",
        "--metrics_output",
        str(output_json),
        "--mrl",
    ]
    run_command(command, cwd=ROOT_DIR / "inference", env=child_env(args), stage="eval", metrics=metrics)
    return 0


def run_nc(args: argparse.Namespace) -> int:
    checkpoint = resolve_checkpoint(args)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else output_dir(args, "nc")
    nc_dir = run_dir / "neural_collapse"
    nc_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else nc_dir / "cifar100_nc_metrics.csv"
    output_json = Path(args.output_json) if args.output_json else nc_dir / "cifar100_nc_metrics.json"
    metrics = MetricLogger(run_dir / "metrics.jsonl")
    command = [
        args.python,
        "cifar100_neural_collapse.py",
        "--path",
        str(checkpoint),
        "--data-root",
        str(args.data_root),
        "--arch",
        "resnet18",
        "--rep-size",
        str(args.feature_dim),
        "--prefix-dims",
        str(args.prefix_dims),
        "--batch-size",
        str(args.nc_batch_size),
        "--workers",
        str(args.nc_workers),
        "--seed",
        str(args.seed),
        "--deterministic",
        "--output-csv",
        str(output_csv),
        "--output-json",
        str(output_json),
        "--mrl",
    ]
    run_command(command, cwd=ROOT_DIR, env=child_env(args), stage="nc", metrics=metrics)
    return 0


def run_retrieval(args: argparse.Namespace) -> int:
    checkpoint = resolve_checkpoint(args)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else output_dir(args, "retrieval")
    retrieval_method = run_dir / "retrieval" / "mrl"
    metrics_dir = run_dir / "retrieval_metrics"
    retrieval_method.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricLogger(run_dir / "metrics.jsonl")
    feature_config = f"mrl1_e0_ff{args.feature_dim}"
    train_x = retrieval_method / f"CIFAR100_train_{feature_config}-X.npy"
    val_x = retrieval_method / f"CIFAR100_val_{feature_config}-X.npy"

    if int(args.force_arrays) == 1 or not train_x.exists() or not val_x.exists():
        dump_command = [
            args.python,
            "pytorch_inference.py",
            "--retrieval",
            "--path",
            str(checkpoint),
            "--dataset",
            "CIFAR100",
            "--data_root",
            str(args.data_root),
            "--retrieval_array_path",
            str(retrieval_method),
            "--arch",
            "resnet18",
            "--rep_size",
            str(args.feature_dim),
            "--prefix-dims",
            str(args.prefix_dims),
            "--use_blurpool",
            "0",
            "--workers",
            str(args.eval_workers),
            "--seed",
            str(args.seed),
            "--deterministic",
            "--mrl",
        ]
        run_command(dump_command, cwd=ROOT_DIR / "inference", env=child_env(args), stage="retrieval_arrays", metrics=metrics)

    dims = parse_int_list(args.prefix_dims)
    faiss_command = [
        args.python,
        "faiss_nn.py",
        "--root",
        str(retrieval_method),
        "--dataset",
        "CIFAR100",
        "--model",
        "mrl",
        "--feature-config",
        feature_config,
        "--rep-size",
        str(args.feature_dim),
        "--index-type",
        str(args.index_type),
        "--k",
        str(args.neighbor_k),
        "--dims",
        *[str(dim) for dim in dims],
    ]
    if int(args.use_gpu) == 1:
        faiss_command.append("--gpu")
    if int(args.rebuild_index) == 1:
        faiss_command.append("--rebuild-index")
    run_command(faiss_command, cwd=ROOT_DIR / "retrieval", env=child_env(args), stage="retrieval_faiss", metrics=metrics)

    metric_command = [
        args.python,
        "compute_metrics.py",
        "--root",
        str(retrieval_method),
        "--dataset",
        "CIFAR100",
        "--model",
        "mrl",
        "--feature-config",
        feature_config,
        "--rep-size",
        str(args.feature_dim),
        "--eval-config",
        "vanilla",
        "--index-type",
        str(args.index_type),
        "--neighbor-k",
        str(args.neighbor_k),
        "--output-json",
        str(metrics_dir / "mrl.json"),
        "--dims",
        *[str(dim) for dim in dims],
        "--shortlist",
        *[str(k) for k in parse_int_list(args.shortlist)],
    ]
    run_command(metric_command, cwd=ROOT_DIR / "retrieval", env=child_env(args), stage="retrieval_metrics", metrics=metrics)
    return 0


def run_full(args: argparse.Namespace) -> int:
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    elif os.environ.get("WANDB_RUN_ID"):
        run_dir = (Path(args.output_root).expanduser() / os.environ["WANDB_RUN_ID"]).resolve()
    else:
        run_dir = output_dir(args, "full")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricLogger(run_dir / "metrics.jsonl")
    env = child_env(args)
    env.setdefault("WANDB_RUN_ID", run_dir.name)
    command = [
        args.python,
        "scripts/cifar100_resnet18_full_sweep.py",
        "--output_root",
        str(run_dir.parent),
        f"--data.root={args.data_root}",
        f"--data.num_workers={args.workers}",
        f"--model.feature_dim={args.feature_dim}",
        f"--model.prefix_dims={args.prefix_dims}",
        f"--model.prefix_mask_prob={args.prefix_mask_prob}",
        f"--model.prefix_mask_scale={args.prefix_mask_scale}",
        f"--model.prefix_mask_scope={args.prefix_mask_scope}",
        f"--training.epochs={args.epochs}",
        f"--training.batch_size={args.train_batch_size}",
        f"--training.seed={args.seed}",
        f"--training.weight_decay={args.weight_decay}",
        f"--training.label_smoothing={args.label_smoothing}",
        f"--training.mrl_loss_mode={args.mrl_loss_mode}",
        f"--training.sampled_prefix_distribution={args.sampled_prefix_distribution}",
        f"--training.sampled_prefix_log_interval={args.sampled_prefix_log_interval}",
        f"--validation.batch_size={args.val_batch_size}",
        f"--eval.workers={args.eval_workers}",
        f"--nc.enabled={args.nc_enabled}",
        f"--nc.interval={args.nc_interval}",
        f"--nc.splits={args.nc_splits}",
        f"--nc.workers={args.nc_workers}",
        f"--nc.batch_size={args.nc_batch_size}",
        f"--retrieval.index_type={args.index_type}",
        f"--retrieval.k={args.neighbor_k}",
        f"--retrieval.shortlist={args.shortlist}",
        f"--retrieval.use_gpu={args.use_gpu}",
        f"--retrieval.rebuild_index={args.rebuild_index}",
        f"--retrieval.force_arrays={args.force_arrays}",
        f"--lr.lr={args.lr}",
        f"--lr.warmup_epochs={args.warmup_epochs}",
        f"--lr.min_lr={args.min_lr}",
    ]
    run_command(command, cwd=ROOT_DIR, env=env, stage="full", metrics=metrics)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CIFAR-100 MRL workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train CIFAR-100 ResNet-18 MRL")
    add_common_args(train_parser)
    add_training_args(train_parser)
    add_nc_args(train_parser, default_enabled=0)
    train_parser.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser("eval", help="evaluate classification accuracy")
    add_common_args(eval_parser)
    eval_parser.add_argument("--run-dir", default="")
    eval_parser.add_argument("--checkpoint", default="")
    eval_parser.add_argument("--output-json", default="")
    eval_parser.set_defaults(func=run_eval)

    nc_parser = subparsers.add_parser("nc", help="compute NC/GNC metrics")
    add_common_args(nc_parser)
    add_nc_args(nc_parser)
    nc_parser.add_argument("--run-dir", default="")
    nc_parser.add_argument("--checkpoint", default="")
    nc_parser.add_argument("--output-csv", default="")
    nc_parser.add_argument("--output-json", default="")
    nc_parser.set_defaults(func=run_nc)

    retrieval_parser = subparsers.add_parser("retrieval", help="compute retrieval and CMC metrics")
    add_common_args(retrieval_parser)
    add_retrieval_args(retrieval_parser)
    retrieval_parser.add_argument("--run-dir", default="")
    retrieval_parser.add_argument("--checkpoint", default="")
    retrieval_parser.set_defaults(func=run_retrieval)

    full_parser = subparsers.add_parser("full", help="run train, eval, NC/GNC, and retrieval")
    add_common_args(full_parser)
    add_training_args(full_parser)
    add_nc_args(full_parser)
    add_retrieval_args(full_parser)
    full_parser.set_defaults(func=run_full)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    install_interrupt_handlers()
    enforce_prefix_mask_scale(args)
    enforce_wandb_project(args)
    if getattr(args, "command", "") in {"eval", "nc", "retrieval"} and not args.run_dir and not args.checkpoint:
        parser.error(f"{args.command} requires --run-dir or --checkpoint")
    try:
        return int(args.func(args))
    except RunInterrupted as exc:
        return int(exc.returncode)
    except subprocess.CalledProcessError as exc:
        return normalized_exit_code(int(exc.returncode))


if __name__ == "__main__":
    raise SystemExit(main())
