import argparse
import json
from pathlib import Path

import numpy as np

try:
    from retrieval.common import (
        DEFAULT_NESTING_LIST,
        default_feature_config_candidates,
        format_list_for_filename,
        model_subdir,
        normalize_model_name,
        resolve_feature_pair,
    )
    from retrieval.metrics import compute_retrieval_metrics, top1_accuracy
except ImportError:
    from common import (
        DEFAULT_NESTING_LIST,
        default_feature_config_candidates,
        format_list_for_filename,
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
    parser.add_argument("--model", default="mrl", help="Model family: mrl, residual_aligned_mrl, mrl_e, or ff")
    parser.add_argument("--feature-config", default="", help="Override feature filename config, e.g. mrl1_e0_ff2048")
    parser.add_argument("--rep-size", type=int, default=2048, help="Feature filename representation size")
    parser.add_argument("--eval-config", choices=["vanilla", "reranking", "funnel"], default="vanilla")
    parser.add_argument("--index-type", default="exactl2", help="Index name used in neighbor filenames")
    parser.add_argument("--dims", type=int, nargs="+", default=None, help="Representation dimensions to evaluate")
    parser.add_argument("--residual-interpolate-alpha", type=float, default=0.0, help="Filename tag for residual-interpolated neighbor files")
    parser.add_argument("--shortlist", type=int, nargs="+", default=[10, 25, 50, 100], help="k values for mAP/precision/recall/top-k")
    parser.add_argument("--neighbor-k", type=int, default=2048, help="Shortlist length used in vanilla neighbor filenames")
    parser.add_argument("--ret-dim", type=int, default=8, help="Retrieval dimension for reranking/funnel filenames")
    parser.add_argument("--rerank-shortlist", type=int, default=200, help="Shortlist length used in reranked filenames")
    parser.add_argument("--funnel-rerank-dims", type=int, nargs="+", default=[16, 32, 64, 128, 2048])
    parser.add_argument("--funnel-shortlist", type=int, nargs="+", default=[800, 400, 200, 50, 10])
    parser.add_argument("--neighbors-path", type=Path, default=None, help="Direct neighbor CSV path; use with one dimension/config")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON file for metrics")
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


def residual_interpolation_tag(alpha):
    if abs(float(alpha)) < 1e-12:
        return ""
    value = f"{float(alpha):g}".replace("-", "m").replace(".", "p")
    return f"_resinterp{value}"


def neighbors_path(args, dim):
    if args.neighbors_path is not None:
        return args.neighbors_path

    model_dir = model_subdir(args.model)
    tag = residual_interpolation_tag(args.residual_interpolate_alpha)
    if args.eval_config == "vanilla":
        filename = f"{args.index_type}_{dim}dim_{args.neighbor_k}shortlist_{args.dataset}{tag}.csv"
        return args.root / "neighbors" / model_dir / filename

    if args.eval_config == "reranking":
        filename = (
            f"{args.ret_dim}dim-reranked{dim}_{args.rerank_shortlist}"
            f"shortlist_{args.dataset}_{args.index_type}.csv"
        )
        return args.root / "neighbors" / "reranked" / model_dir / filename

    filename = (
        f"{args.ret_dim}dim-cascade{format_list_for_filename(args.funnel_rerank_dims)}_"
        f"shortlist{format_list_for_filename(args.funnel_shortlist)}_{args.dataset}_{args.index_type}.csv"
    )
    return args.root / "neighbors" / "funnel_retrieval" / model_dir / filename


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
    if args.eval_config == "funnel":
        dims = [args.ret_dim]

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
        "residual_interpolate_alpha": float(args.residual_interpolate_alpha),
        "metrics": [],
    }

    for dim in dims:
        path = neighbors_path(args, dim)
        if not path.exists():
            print(f"{path} not found")
            continue

        neighbors = load_csv_int_matrix(path)
        top1 = top1_accuracy(query_labels, db_labels, neighbors)
        print(f"\nNeighbors: {path}")
        print(f"Top1: {top1:.6f}")

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

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(output, handle, indent=2)
        print(f"\nWrote metrics: {args.output_json}")

    return 0 if output["metrics"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
