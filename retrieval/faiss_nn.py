import argparse
from pathlib import Path

import numpy as np

try:
    from retrieval.common import (
        DEFAULT_NESTING_LIST,
        default_feature_config_candidates,
        model_subdir,
        normalize_model_name,
        resolve_feature_pair,
    )
except ImportError:
    from common import (
        DEFAULT_NESTING_LIST,
        default_feature_config_candidates,
        model_subdir,
        normalize_model_name,
        resolve_feature_pair,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS indexes and write k-NN neighbor CSV files.")
    parser.add_argument("--root", type=Path, default=Path("../inference"), help="Directory containing retrieval feature arrays")
    parser.add_argument("--dataset", default="1K", help="Dataset name used in query filenames, e.g. 1K, V2, 4K, CIFAR100")
    parser.add_argument("--db-dataset", default="", help="Optional database dataset name. Defaults to 1K for V2, otherwise --dataset")
    parser.add_argument("--query-dataset", default="", help="Optional query dataset name. Defaults to --dataset")
    parser.add_argument("--model", default="mrl", help="Model family: mrl, mrl_e, or ff")
    parser.add_argument("--feature-config", default="", help="Override feature filename config, e.g. mrl1_e0_ff2048")
    parser.add_argument("--rep-size", type=int, default=2048, help="Feature filename representation size")
    parser.add_argument("--index-type", default="exactl2", help="exactl2, hnsw8, hnsw32, hnsw_8, or hnsw_32")
    parser.add_argument("--hnsw-max-neighbors", type=int, default=32)
    parser.add_argument("--k", type=int, default=2048, help="Number of nearest neighbors to save")
    parser.add_argument("--dims", type=int, nargs="+", default=None, help="Representation dimensions to search")
    parser.add_argument("--gpu", action="store_true", help="Use all GPUs for exact L2 search when FAISS GPU is available")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild indexes even if index files already exist")
    return parser.parse_args()


def import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("faiss is required for retrieval/faiss_nn.py. Install faiss-cpu or faiss-gpu.") from exc
    return faiss


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


def hnsw_neighbors(index_type, default):
    digits = "".join(ch for ch in index_type if ch.isdigit())
    return int(digits) if digits else int(default)


def build_index(faiss, xb, index_type, hnsw_max_neighbors):
    d = xb.shape[1]
    if index_type == "exactl2":
        print("Building Exact L2 index")
        return faiss.IndexFlatL2(d)

    neighbors = hnsw_neighbors(index_type, hnsw_max_neighbors)
    print(f"Building HNSW{neighbors} index")
    return faiss.IndexHNSWFlat(d, neighbors)


def main():
    args = parse_args()
    args.model = normalize_model_name(args.model)
    args.root = args.root.expanduser()
    dims = args.dims
    if dims is None:
        dims = [args.rep_size] if args.model == "ff" else DEFAULT_NESTING_LIST

    faiss = import_faiss()
    db_dataset, query_dataset = default_datasets(args)
    resolved_config, db_feature_path, query_feature_path = resolve_feature_pair(
        args.root,
        db_dataset,
        query_dataset,
        config_candidates(args),
        "X",
    )

    print(f"Feature config: {resolved_config}")
    print(f"Database vectors: {db_feature_path}")
    print(f"Query vectors: {query_feature_path}")

    database = np.load(db_feature_path)
    queryset = np.load(query_feature_path)
    max_dim = database.shape[1]
    if queryset.shape[1] != max_dim:
        raise ValueError(f"Database/query dims differ: {database.shape[1]} vs {queryset.shape[1]}")

    model_dir = model_subdir(args.model)
    index_dir = args.root / "index_files" / model_dir
    neighbors_dir = args.root / "neighbors" / model_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    neighbors_dir.mkdir(parents=True, exist_ok=True)

    can_use_gpu = args.gpu and args.index_type == "exactl2" and getattr(faiss, "get_num_gpus", lambda: 0)() > 0

    for dim in dims:
        if dim > max_dim:
            print(f"Skipping dim={dim}; feature arrays only have {max_dim} columns")
            continue

        index_file = index_dir / f"{args.dataset}_{dim}dim_{args.index_type}.index"
        if index_file.exists() and not args.rebuild_index:
            print(f"Loading index file: {index_file}")
            cpu_index = faiss.read_index(str(index_file))
        else:
            xb = database[:, :dim]
            xb = np.ascontiguousarray(xb, dtype=np.float32)
            faiss.normalize_L2(xb)
            print(f"Database @ dim={dim}: {xb.shape}")
            cpu_index = build_index(faiss, xb, args.index_type, args.hnsw_max_neighbors)
            cpu_index.add(xb)
            faiss.write_index(cpu_index, str(index_file))
            print(f"Wrote index file: {index_file}")

        index = faiss.index_cpu_to_all_gpus(cpu_index) if can_use_gpu else cpu_index
        xq = queryset[:, :dim]
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        faiss.normalize_L2(xq)
        print(f"Queries @ dim={dim}: {xq.shape}")

        _, neighbors = index.search(xq, args.k)
        neighbors_file = neighbors_dir / f"{args.index_type}_{dim}dim_{args.k}shortlist_{args.dataset}.csv"
        np.savetxt(neighbors_file, neighbors, fmt="%d", delimiter=",")
        print(f"Wrote neighbors: {neighbors_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
