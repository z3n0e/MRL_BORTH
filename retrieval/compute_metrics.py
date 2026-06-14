import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from wandb_utils import env_default, env_flag, init_wandb_run, wandb_finish, wandb_log

try:
    from retrieval.common import (
        DEFAULT_NESTING_LIST,
        default_feature_config_candidates,
        model_subdir,
        normalize_model_name,
        resolve_feature_pair,
    )
    from retrieval.metrics import compute_retrieval_metrics, top1_accuracy
except ImportError:
    from common import (
        DEFAULT_NESTING_LIST,
        default_feature_config_candidates,
        model_subdir,
        normalize_model_name,
        resolve_feature_pair,
    )
    from metrics import compute_retrieval_metrics, top1_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Compute retrieval metrics from k-NN neighbor CSV files.")
    parser.add_argument("--root", type=Path, default=Path("../inference"), help="Directory containing retrieval arrays and neighbors/")
    parser.add_argument("--dataset", default="1K", help="Dataset name used in query filenames, e.g. 1K, V2, 4K, CIFAR100")
    parser.add_argument("--db-dataset", default="", help="Optional database dataset name. Defaults to 1K for V2, otherwise --dataset")
    parser.add_argument("--query-dataset", default="", help="Optional query dataset name. Defaults to --dataset")
    parser.add_argument("--model", default="mrl", help="Model family: mrl, mrl_e, or ff")
    parser.add_argument("--feature-config", default="", help="Override feature filename config, e.g. mrl1_e0_ff2048")
    parser.add_argument("--rep-size", type=int, default=2048, help="Feature filename representation size")
    parser.add_argument("--eval-config", choices=["vanilla"], default="vanilla")
    parser.add_argument("--index-type", default="exactl2", help="Index name used in neighbor filenames")
    parser.add_argument("--dims", type=int, nargs="+", default=None, help="Representation dimensions to evaluate")
    parser.add_argument("--shortlist", type=int, nargs="+", default=[10, 25, 50, 100], help="k values for mAP/precision/recall/top-k")
    parser.add_argument("--neighbor-k", dest="nn_k", type=int, default=2048, help="Shortlist length used in vanilla neighbor filenames")
    parser.add_argument("--neighbors-path", type=Path, default=None, help="Direct neighbor CSV path; use with one dimension/config")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON file for metrics")
    parser.add_argument("--wandb-enabled", type=int, default=env_flag("WANDB_ENABLED", 1), help="enable W&B logging? (1/0)")
    parser.add_argument("--wandb-project", default=env_default("WANDB_PROJECT", "mrl-borth"))
    parser.add_argument("--wandb-entity", default=env_default("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default=env_default("WANDB_GROUP", ""))
    parser.add_argument("--wandb-name", default=env_default("WANDB_NAME", ""))
    parser.add_argument("--wandb-tags", default=env_default("WANDB_TAGS", ""))
    parser.add_argument("--wandb-mode", default=env_default("WANDB_MODE", ""))
    parser.add_argument("--wandb-dir", default=env_default("WANDB_DIR", ""))
    return parser.parse_args()


def default_datasets(args):
    query_dataset = args.query_dataset or args.dataset
    db_dataset = args.db_dataset
    if not db_dataset:
        db_dataset = "1K" if query_dataset.upper() == "V2" else args.dataset
    return db_dataset, query_dataset


def config_candidates(args):
    if args.feature_config:
        return [args.feature_config]
    return default_feature_config_candidates(args.model, args.rep_size)


def load_csv_int_matrix(path):
    matrix = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def neighbors_path(args, dim):
    if args.neighbors_path is not None:
        return args.neighbors_path

    model_dir = model_subdir(args.model)
    filename = f"{args.index_type}_{dim}dim_{args.nn_k}shortlist_{args.dataset}.csv"
    return args.root / "neighbors" / model_dir / filename


def print_metric_row(dim, k, metrics):
    print(f"dim={dim} k={k}")
    print(f"  mAP@{k}: {metrics['mAP']:.6f}")
    print(f"  precision@{k}: {metrics['precision']:.6f}")
    print(f"  recall@{k}: {metrics['recall']:.6f}")
    print(f"  top{k}: {metrics['topk']:.6f}")


def main():
    args = parse_args()
    args.model = normalize_model_name(args.model)
    args.root = args.root.expanduser()
    dims = args.dims
    if dims is None:
        if args.neighbors_path is not None:
            dims = [args.rep_size]
        else:
            dims = [args.rep_size] if args.model == "ff" else DEFAULT_NESTING_LIST
    db_dataset, query_dataset = default_datasets(args)
    resolved_config, db_label_path, query_label_path = resolve_feature_pair(
        args.root,
        db_dataset,
        query_dataset,
        config_candidates(args),
        "y",
    )
    print(f"Feature config: {resolved_config}")
    print(f"Database labels: {db_label_path}")
    print(f"Query labels: {query_label_path}")

    db_labels = np.load(db_label_path)
    query_labels = np.load(query_label_path)

    output = {
        "dataset": args.dataset,
        "db_dataset": db_dataset,
        "query_dataset": query_dataset,
        "model": args.model,
        "feature_config": resolved_config,
        "eval_config": args.eval_config,
        "index_type": args.index_type,
        "metrics": [],
    }
    wandb_run = init_wandb_run(
        bool(args.wandb_enabled),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group or f"{args.dataset}_{args.model}_retrieval",
        name=args.wandb_name or f"{args.model}_{args.dataset}_retrieval",
        job_type="retrieval",
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={**vars(args), "resolved_feature_config": resolved_config},
    )

    for dim in dims:
        path = neighbors_path(args, dim)
        if not path.exists():
            print(f"{path} not found")
            continue

        neighbors = load_csv_int_matrix(path)
        top1 = top1_accuracy(query_labels, db_labels, neighbors)
        print(f"\nNeighbors: {path}")
        print(f"Top1: {top1:.6f}")
        wandb_log(wandb_run, {
            "dim": int(dim),
            "retrieval/top1": top1,
            f"retrieval/top1/dim_{dim}": top1,
        })

        valid_shortlist = [k for k in args.shortlist if k <= neighbors.shape[1]]
        if len(valid_shortlist) != len(args.shortlist):
            skipped = [k for k in args.shortlist if k > neighbors.shape[1]]
            print(f"Skipping k values longer than the neighbor shortlist: {skipped}")

        metric_rows = compute_retrieval_metrics(query_labels, db_labels, neighbors, valid_shortlist)
        for row in metric_rows:
            print_metric_row(dim, row["k"], row)
            output["metrics"].append({
                "dim": int(dim),
                "k": int(row["k"]),
                "top1": top1,
                "neighbors_path": str(path),
                **{key: value for key, value in row.items() if key != "k"},
            })
            k = int(row["k"])
            wandb_log(wandb_run, {
                "dim": int(dim),
                "k": k,
                f"retrieval/mAP_at_{k}": row["mAP"],
                f"retrieval/precision_at_{k}": row["precision"],
                f"retrieval/recall_at_{k}": row["recall"],
                f"retrieval/top{k}": row["topk"],
                f"retrieval/mAP_at_{k}/dim_{dim}": row["mAP"],
                f"retrieval/precision_at_{k}/dim_{dim}": row["precision"],
                f"retrieval/recall_at_{k}/dim_{dim}": row["recall"],
                f"retrieval/top{k}/dim_{dim}": row["topk"],
            })

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(output, handle, indent=2)
        print(f"\nWrote metrics: {args.output_json}")

    wandb_finish(wandb_run)
    return 0 if output["metrics"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
