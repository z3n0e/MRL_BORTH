import numpy as np


def labels_to_1d(labels):
    labels = np.asarray(labels)
    if labels.ndim == 0:
        labels = labels.reshape(1)
    return labels.reshape(-1).astype(np.int64)


def neighbors_to_2d(neighbors):
    neighbors = np.asarray(neighbors)
    if neighbors.ndim == 1:
        neighbors = neighbors.reshape(1, -1)
    if neighbors.ndim != 2:
        raise ValueError(f"neighbors must be a 2D array, got shape {neighbors.shape}")
    return neighbors.astype(np.int64, copy=False)


def relevant_counts_by_label(db_labels):
    db_labels = labels_to_1d(db_labels)
    labels, counts = np.unique(db_labels, return_counts=True)
    return dict(zip(labels.tolist(), counts.astype(np.int64).tolist()))


def top1_accuracy(query_labels, db_labels, neighbors):
    query_labels = labels_to_1d(query_labels)
    db_labels = labels_to_1d(db_labels)
    neighbors = neighbors_to_2d(neighbors)
    _validate_shapes(query_labels, db_labels, neighbors, 1)
    top1_labels = db_labels[neighbors[:, 0]]
    return float(np.mean(top1_labels == query_labels))


def compute_retrieval_metrics_at_k(query_labels, db_labels, neighbors, k):
    """
    Compute the repository's retrieval metrics for a k-NN shortlist.

    AP follows the original notebook definition: precision is summed at matching
    ranks and divided by k. Recall is dataset-agnostic and divides by the number
    of database images with the same label as the query.
    """
    query_labels = labels_to_1d(query_labels)
    db_labels = labels_to_1d(db_labels)
    neighbors = neighbors_to_2d(neighbors)
    k = int(k)
    _validate_shapes(query_labels, db_labels, neighbors, k)

    relevant_counts = relevant_counts_by_label(db_labels)
    ranks = np.arange(1, k + 1, dtype=np.float64)
    aps, precisions, recalls, topk_hits = [], [], [], []

    for query_index, target in enumerate(query_labels):
        indices = neighbors[query_index, :k]
        labels = db_labels[indices]
        matches = labels == target
        hits = int(matches.sum())
        relevant_count = relevant_counts.get(int(target), 0)
        if relevant_count == 0:
            raise ValueError(f"Query label {target} has no matching database labels")

        true_positives = np.cumsum(matches)
        precision_at_ranks = true_positives.astype(np.float64) / ranks

        aps.append(float(precision_at_ranks[matches].sum() / k))
        precisions.append(float(hits / k))
        recalls.append(float(hits / relevant_count))
        topk_hits.append(float(hits > 0))

    return {
        "mAP": float(np.mean(aps)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "topk": float(np.mean(topk_hits)),
    }


def compute_retrieval_metrics(query_labels, db_labels, neighbors, ks):
    return [
        {"k": int(k), **compute_retrieval_metrics_at_k(query_labels, db_labels, neighbors, k)}
        for k in ks
    ]


def _validate_shapes(query_labels, db_labels, neighbors, k):
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if neighbors.shape[0] != query_labels.shape[0]:
        raise ValueError(
            f"neighbors rows ({neighbors.shape[0]}) must match query labels ({query_labels.shape[0]})"
        )
    if k > neighbors.shape[1]:
        raise ValueError(f"k={k} exceeds neighbor shortlist length {neighbors.shape[1]}")
    if neighbors.size == 0:
        raise ValueError("neighbors must not be empty")
    if neighbors.min() < 0 or neighbors.max() >= db_labels.shape[0]:
        raise ValueError("neighbors contain indices outside the database label array")
