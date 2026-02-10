"""
ALS evaluation: recalculate (user or item) factors from viewable data, predict, and evaluate.

Takes a viewable user×item matrix and held-out y_true; returns metrics dict.
"""
from typing import Iterable, List, Literal, Set

import numpy as np
from scipy.sparse import csr_matrix

from .validation import evaluate


def _to_numpy(x):
    """Convert to numpy if needed (e.g. implicit GPU matrices)."""
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def evaluate_als_dataset(
    model,
    user_items_viewable: csr_matrix,
    y_true: List[Set[int]],
    mode: Literal["user", "item"],
    N_REC: int = 50,
    k_values: Iterable[int] = (5, 10, 20),
) -> dict:
    """
    Run recalc, predictions, and evaluation for one dataset (train-holdout, cold-users, or cold-artists).

    Parameters
    ----------
    model : implicit.als.AlternatingLeastSquares (or compatible)
        Fitted ALS model (trained on train user×item matrix).
    user_items_viewable : scipy.sparse.csr_matrix
        Shape (n_users, n_items). Viewable interactions only (held-out are in y_true).
    y_true : list of set of int
        y_true[i] = set of item indices held out for user i (ground truth for evaluation).
    mode : "user" or "item"
        - "user": items are the same as model was trained on (train artists). Recalculate
          user factors from user_items_viewable and recommend.
        - "item": rows are train users, columns are new items (e.g. val artists). Recalculate
          item factors from viewable data, then score users × items and return top-k per user.
    N_REC : int, default 50
        Number of recommendations per user (top-N).
    k_values : iterable of int, default (5, 10, 20)
        k values for metrics (e.g. precision@k, ndcg@k).

    Returns
    -------
    dict
        Metric names -> values (e.g. "precision@5", "ndcg@10", "mrr").
    """
    k_values = list(k_values)
    n_users, n_items = user_items_viewable.shape

    if mode == "user":
        user_ids = np.arange(n_users)
        rec_item_ids, _ = model.recommend(
            user_ids,
            user_items_viewable,
            N=N_REC,
            filter_already_liked_items=False,
            recalculate_user=True,
        )
        y_pred = [rec_item_ids[i].tolist() for i in range(rec_item_ids.shape[0])]
    else:
        # mode == "item": recalculate item factors (items × users)
        item_users = user_items_viewable.T.tocsr()
        item_factors = model.recalculate_item(
            np.arange(n_items),
            item_users,
        )
        item_factors = _to_numpy(item_factors)
        user_factors = _to_numpy(model.user_factors)
        scores = user_factors @ item_factors.T
        top_k = np.argsort(-scores, axis=1)[:, :N_REC]
        y_pred = [top_k[i].tolist() for i in range(top_k.shape[0])]

    return evaluate(y_true, y_pred, k_values=k_values)
