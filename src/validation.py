"""
Recommendation evaluation metrics.

Expects:
- y_true: list of arrays (or sets/sequences) of relevant item ids per user
- y_pred: list of arrays (or sequences) of recommended item ids per user (ordered by score, best first)

Uses numpy for internal computation and array aggregation.
"""
from typing import Iterable, List, Optional, Sequence, Set, Union

import numpy as np


def _to_arrays(y_true: List[Union[Set, Sequence]]) -> List[np.ndarray]:
    """Convert each user's relevants to a 1D numpy array (for fast np.isin)."""
    out = []
    for u in y_true:
        if isinstance(u, np.ndarray):
            out.append(u.ravel())
        else:
            out.append(np.array(list(u)))
    return out


def _rec_k(rec: Union[Sequence, np.ndarray], k: Optional[int]) -> np.ndarray:
    """Top-k recommendation as 1D array. k=None means use full list."""
    rec = np.asarray(rec).ravel()
    if k is None:
        return rec
    return rec[:k] if rec.size >= k else rec


def precision_at_k(
    y_true: List[Union[Set, Sequence]],
    y_pred: List[Sequence],
    k: int = 10,
) -> float:
    """
    Precision@k: proportion of top-k recommendations that are relevant.
    Averaged over users (no relevant items => 0 contribution).
    """
    y_true = _to_arrays(y_true)
    precisions = np.empty(len(y_true))
    n = 0
    for i, (rel, rec) in enumerate(zip(y_true, y_pred)):
        if rel.size == 0:
            continue
        rec_k = _rec_k(rec, k)
        if rec_k.size == 0:
            precisions[n] = 0.0
        else:
            hits = np.isin(rec_k, rel).sum()
            precisions[n] = hits / min(k, rec_k.size)
        n += 1
    return float(np.mean(precisions[:n])) if n > 0 else 0.0


def recall_at_k(
    y_true: List[Union[Set, Sequence]],
    y_pred: List[Sequence],
    k: int = 10,
) -> float:
    """
    Recall@k: proportion of relevant items that appear in top-k.
    Averaged over users.
    """
    y_true = _to_arrays(y_true)
    recalls = np.empty(len(y_true))
    n = 0
    for rel, rec in zip(y_true, y_pred):
        if rel.size == 0:
            continue
        rec_k = _rec_k(rec, k)
        hits = np.isin(rec_k, rel).sum()
        recalls[n] = hits / rel.size
        n += 1
    return float(np.mean(recalls[:n])) if n > 0 else 0.0


def f_score_at_k(
    y_true: List[Union[Set, Sequence]],
    y_pred: List[Sequence],
    k: int = 10,
    beta: float = 1.0,
) -> float:
    """
    F-score@k (F_beta). beta=1 gives F1. Uses precision@k and recall@k.
    """
    p = precision_at_k(y_true, y_pred, k)
    r = recall_at_k(y_true, y_pred, k)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def _average_precision_one(rel: np.ndarray, rec: np.ndarray) -> float:
    """Average Precision for one user: AP = (1/|rel|) * sum_k P(k) * rel(k)."""
    if rel.size == 0:
        return 0.0
    rec = np.asarray(rec).ravel()
    hit_ranks = np.where(np.isin(rec, rel))[0]
    if hit_ranks.size == 0:
        return 0.0
    # P@position = (hits up to position) / position; rel(k)=1 at hit positions
    precisions_at_hits = (np.arange(1, hit_ranks.size + 1, dtype=np.float64) / (hit_ranks + 1))
    return float(np.sum(precisions_at_hits) / rel.size)


def map_at_k(
    y_true: List[Union[Set, Sequence]],
    y_pred: List[Sequence],
    k: int = 10,
) -> float:
    """
    Mean Average Precision@k: average of per-user AP over users with at least one relevant.
    """
    y_true = _to_arrays(y_true)
    aps = np.empty(len(y_true))
    n = 0
    for rel, rec in zip(y_true, y_pred):
        if rel.size == 0:
            continue
        rec_k = _rec_k(rec, k)
        aps[n] = _average_precision_one(rel, rec_k)
        n += 1
    return float(np.mean(aps[:n])) if n > 0 else 0.0


def _reciprocal_rank_one(rel: np.ndarray, rec: np.ndarray) -> float:
    """Reciprocal rank for one user: 1/rank of first relevant item, else 0."""
    rec = np.asarray(rec).ravel()
    if rec.size == 0 or rel.size == 0:
        return 0.0
    in_rel = np.isin(rec, rel)
    hit = np.flatnonzero(in_rel)
    if hit.size == 0:
        return 0.0
    return 1.0 / (hit[0] + 1)


def mrr(
    y_true: List[Union[Set, Sequence]],
    y_pred: List[Sequence],
    k: Optional[int] = None,
) -> float:
    """
    Mean Reciprocal Rank: average of 1/rank of first hit. Optional cap at k.
    """
    y_true = _to_arrays(y_true)
    rrs = np.empty(len(y_true))
    n = 0
    for rel, rec in zip(y_true, y_pred):
        if rel.size == 0:
            continue
        rec_list = _rec_k(rec, k)
        rrs[n] = _reciprocal_rank_one(rel, rec_list)
        n += 1
    return float(np.mean(rrs[:n])) if n > 0 else 0.0


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """DCG@k with relevances in rank order (position 0 = rank 1)."""
    rel = np.asarray(relevances, dtype=np.float64).ravel()[:k]
    if rel.size == 0:
        return 0.0
    ranks = np.arange(1, rel.size + 1, dtype=np.float64)
    return float(np.sum(rel / np.log2(ranks + 1)))


def _ndcg_at_k_one(rel: np.ndarray, rec: np.ndarray, k: int) -> float:
    """NDCG@k for one user (binary relevance)."""
    if rel.size == 0:
        return 0.0
    rec_k = _rec_k(rec, k)
    if rec_k.size == 0:
        return 0.0
    ideal_relevances = np.ones(min(rel.size, k), dtype=np.float64)
    idcg = _dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    relevances = np.isin(rec_k, rel).astype(np.float64)
    dcg = _dcg_at_k(relevances, k)
    return dcg / idcg


def ndcg_at_k(
    y_true: List[Union[Set, Sequence]],
    y_pred: List[Sequence],
    k: int = 10,
) -> float:
    """
    Normalized Discounted Cumulative Gain@k (binary relevance).
    Averaged over users that have at least one relevant item.
    """
    y_true = _to_arrays(y_true)
    ndcgs = np.empty(len(y_true))
    n = 0
    for rel, rec in zip(y_true, y_pred):
        if rel.size == 0:
            continue
        ndcgs[n] = _ndcg_at_k_one(rel, rec, k)
        n += 1
    return float(np.mean(ndcgs[:n])) if n > 0 else 0.0


def evaluate(
    y_true: List[Union[Set, Sequence]],
    y_pred: List[Sequence],
    k_values: Iterable[int] = (5, 10, 20),
) -> dict:
    """
    Compute common recommendation metrics at given k values.

    Returns
    -------
    dict
        Keys like "precision@5", "recall@10", "ndcg@10", "f1@10", "mrr", "map@10".
    """
    k_values = list(k_values)
    results = {}

    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(y_true, y_pred, k)
        results[f"recall@{k}"] = recall_at_k(y_true, y_pred, k)
        results[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k)
        results[f"f1@{k}"] = f_score_at_k(y_true, y_pred, k)
        results[f"map@{k}"] = map_at_k(y_true, y_pred, k)

    k_max = max(k_values)
    results["mrr"] = mrr(y_true, y_pred, k=k_max)

    return results
