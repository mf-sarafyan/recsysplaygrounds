"""
Train/validation split by sampling users and artists.

Level 1 (split_matrix): seen vs unseen users/artists → train, val_unseen_users,
  val_unseen_artists, val_cold_start.

Level 2 (holdout_per_user / holdout_per_item): within any user×item matrix, hold
  out a fraction of each user's (or item's) interactions for evaluation.

Sparse layout (CSR/CSC):
  indptr has length n_rows+1 (CSR) or n_cols+1 (CSC). Row i's nonzeros (CSR) are
  in data[indptr[i]:indptr[i+1]] with column indices indices[indptr[i]:indptr[i+1]].
  So indptr[i+1] - indptr[i] = nnz in row i. Same for CSC with columns.
"""
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class MatrixSplit:
    """Result of splitting a user×artist CSR matrix by row/column indices."""

    # Submatrices (all CSR)
    train: csr_matrix  # (n_train_users, n_train_artists)
    val_unseen_users: csr_matrix  # (n_val_users, n_train_artists)
    val_unseen_artists: csr_matrix  # (n_train_users, n_val_artists)
    val_cold_start: csr_matrix  # (n_val_users, n_val_artists)

    # Original matrix row/column indices (for mapping back to IDs if needed)
    train_user_idx: np.ndarray  # (n_train_users,) int
    val_user_idx: np.ndarray  # (n_val_users,) int
    train_artist_idx: np.ndarray  # (n_train_artists,) int
    val_artist_idx: np.ndarray  # (n_val_artists,) int


def split_matrix(
    user_artist: csr_matrix,
    user_ratio: float = 0.8,
    artist_ratio: float = 0.8,
    random_state: Optional[int] = None,
) -> MatrixSplit:
    """
    Split a user×artist CSR matrix by sampling rows (users) and columns (artists).

    No pandas involved: slices the matrix into four blocks. Use this when you
    already have the CSR built (e.g. from 360K plays).

    Parameters
    ----------
    user_artist : scipy.sparse.csr_matrix
        Shape (n_users, n_artists); rows = users, columns = artists.
    user_ratio : float, default 0.8
        Fraction of user rows for training.
    artist_ratio : float, default 0.8
        Fraction of artist columns for training.
    random_state : int or None, default None
        Random seed for reproducible splits.

    Returns
    -------
    MatrixSplit
        train, val_unseen_users, val_unseen_artists, val_cold_start matrices
        plus train_user_idx, val_user_idx, train_artist_idx, val_artist_idx.
    """
    rng = np.random.default_rng(random_state)
    n_users, n_artists = user_artist.shape

    perm_u = rng.permutation(n_users)
    perm_a = rng.permutation(n_artists)

    n_train_u = max(1, int(n_users * user_ratio))
    n_train_a = max(1, int(n_artists * artist_ratio))

    train_user_idx = perm_u[:n_train_u]
    val_user_idx = perm_u[n_train_u:]
    train_artist_idx = perm_a[:n_train_a]
    val_artist_idx = perm_a[n_train_a:]

    # Row slice then column slice (CSR row slice is efficient)
    train = user_artist[train_user_idx, :][:, train_artist_idx].tocsr()
    val_unseen_users = user_artist[val_user_idx, :][:, train_artist_idx].tocsr()
    val_unseen_artists = user_artist[train_user_idx, :][:, val_artist_idx].tocsr()
    val_cold_start = user_artist[val_user_idx, :][:, val_artist_idx].tocsr()

    return MatrixSplit(
        train=train,
        val_unseen_users=val_unseen_users,
        val_unseen_artists=val_unseen_artists,
        val_cold_start=val_cold_start,
        train_user_idx=train_user_idx,
        val_user_idx=val_user_idx,
        train_artist_idx=train_artist_idx,
        val_artist_idx=val_artist_idx,
    )


def _holdout_counts(
    nnz_per_entity: np.ndarray,
    holdout_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each entity (row or column), compute how many entries to hold out and to keep viewable.
    Ensures at least 1 viewable and, when nnz > 1, at least 1 held-out.
    """
    n_held = np.zeros_like(nnz_per_entity)
    multi = nnz_per_entity > 1
    if np.any(multi):
        n_held[multi] = np.minimum(
            np.maximum(1, (holdout_ratio * nnz_per_entity[multi]).astype(np.intp)),
            nnz_per_entity[multi] - 1,
        )
    n_viewable = nnz_per_entity - n_held
    return n_held, n_viewable


def _shuffle_within_entities(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    nnz_per_entity: np.ndarray,
    n_entities: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle nonzero entries independently within each entity (row or column).
    Returns indices and data in shuffled order (by entity, then random within entity).
    """
    total_nnz = indices.size
    entity_id = np.repeat(np.arange(n_entities, dtype=np.intp), nnz_per_entity)
    random_key = rng.random(total_nnz)
    sort_idx = np.lexsort((random_key, entity_id))
    return indices[sort_idx], data[sort_idx]


def _viewable_mask(
    total_nnz: int,
    indptr: np.ndarray,
    nnz_per_entity: np.ndarray,
    n_viewable: np.ndarray,
) -> np.ndarray:
    """Boolean mask: True for entries that remain viewable (first n_viewable per entity after shuffle)."""
    position_in_entity = np.arange(total_nnz, dtype=np.intp) - np.repeat(
        indptr[:-1], nnz_per_entity
    )
    return position_in_entity < np.repeat(n_viewable, nnz_per_entity)


def _held_out_splits(
    indices_sorted: np.ndarray,
    viewable_mask: np.ndarray,
    n_held: np.ndarray,
    n_entities: int,
    dtype: np.dtype,
) -> List[np.ndarray]:
    """Split held-out indices into one array per entity (for y_true)."""
    total_held = int(n_held.sum())
    if total_held == 0:
        return [np.array([], dtype=dtype) for _ in range(n_entities)]
    held_out_mask = ~viewable_mask
    return list(
        np.split(indices_sorted[held_out_mask], np.cumsum(n_held)[:-1])
    )


def holdout_per_user(
    user_items: csr_matrix,
    holdout_ratio: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[csr_matrix, List[Union[Set[int], np.ndarray]]]:
    """
    Split each user's interactions into viewable (for model input) and held-out (for y_true).

    For each row, a random fraction of nonzero entries are masked to 0 in the viewable
    matrix and recorded as y_true. Ensures at least 1 viewable (so user factor can be
    computed) and at least 1 held-out when the user has 2+ interactions (so we can evaluate).

    Parameters
    ----------
    user_items : scipy.sparse.csr_matrix
        Shape (n_users, n_items). Rows = users, columns = items.
    holdout_ratio : float, default 0.2
        Target fraction of each user's interactions to hold out (0 < holdout_ratio < 1).
    random_state : int or None, default None
        Random seed for reproducible splits.

    Returns
    -------
    viewable : csr_matrix
        Same shape as user_items; a subset of nonzeros per row set to 0 (viewable = rest).
    y_true_held_out : list of set or ndarray of int
        y_true_held_out[i] = column indices held out for user i (for evaluation).
    """
    rng = np.random.default_rng(random_state)
    n_users, n_items = user_items.shape
    user_items = user_items.tocsr()
    indptr = user_items.indptr
    indices = user_items.indices
    data = user_items.data
    total_nnz = indices.size

    nnz_per_row = np.diff(indptr)
    n_held, n_viewable = _holdout_counts(nnz_per_row, holdout_ratio)

    indices_sorted, data_sorted = _shuffle_within_entities(
        indptr, indices, data, nnz_per_row, n_users, rng
    )
    viewable_mask = _viewable_mask(total_nnz, indptr, nnz_per_row, n_viewable)

    viewable_indptr = np.empty(n_users + 1, dtype=np.intp)
    viewable_indptr[0] = 0
    np.cumsum(n_viewable, out=viewable_indptr[1:])
    viewable = csr_matrix(
        (data_sorted[viewable_mask], indices_sorted[viewable_mask], viewable_indptr),
        shape=(n_users, n_items),
        dtype=user_items.dtype,
    )
    y_true_held_out = _held_out_splits(
        indices_sorted, viewable_mask, n_held, n_users, indices.dtype
    )
    return viewable, y_true_held_out


def holdout_per_item(
    user_items: csr_matrix,
    holdout_ratio: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[csr_matrix, List[Union[Set[int], np.ndarray]]]:
    """
    Split each item's interactions into viewable (for model input) and held-out (for y_true).

    For each column (item), a random fraction of nonzero entries are masked to 0 in the
    viewable matrix and recorded as y_true. Use for cold-artist inverse evaluation:
    rank users per artist and compare to held-out listeners.

    Parameters
    ----------
    user_items : scipy.sparse.csr_matrix
        Shape (n_users, n_items). Rows = users, columns = items.
    holdout_ratio : float, default 0.2
        Target fraction of each item's interactions to hold out (0 < holdout_ratio < 1).
    random_state : int or None, default None
        Random seed for reproducible splits.

    Returns
    -------
    viewable : csr_matrix
        Same shape as user_items; a subset of nonzeros per column set to 0 (viewable = rest).
    y_true_held_out : list of set or ndarray of int
        y_true_held_out[j] = row (user) indices held out for item j (for evaluation).
    """
    rng = np.random.default_rng(random_state)
    n_users, n_items = user_items.shape
    csc = user_items.tocsc()
    indptr = csc.indptr
    indices = csc.indices
    data = csc.data
    total_nnz = indices.size

    nnz_per_col = np.diff(indptr)
    n_held, n_viewable = _holdout_counts(nnz_per_col, holdout_ratio)

    indices_sorted, data_sorted = _shuffle_within_entities(
        indptr, indices, data, nnz_per_col, n_items, rng
    )
    viewable_mask = _viewable_mask(total_nnz, indptr, nnz_per_col, n_viewable)

    viewable_rows = indices_sorted[viewable_mask]
    viewable_cols = np.repeat(np.arange(n_items, dtype=indices.dtype), n_viewable)
    viewable = csr_matrix(
        (data_sorted[viewable_mask], (viewable_rows, viewable_cols)),
        shape=(n_users, n_items),
        dtype=user_items.dtype,
    )
    y_true_held_out = _held_out_splits(
        indices_sorted, viewable_mask, n_held, n_items, indices.dtype
    )
    return viewable, y_true_held_out


@dataclass
class UserArtistSplit:
    """Result of splitting by users and artists."""

    # Interaction DataFrames (same schema as input: user_id, artist_id, artist_name, plays)
    train: pd.DataFrame
    val_unseen_users: pd.DataFrame  # unseen users × seen artists
    val_unseen_artists: pd.DataFrame  # seen users × unseen artists
    val_cold_start: pd.DataFrame  # unseen users × unseen artists

    # Index sets (for building matrices / filtering)
    train_user_ids: np.ndarray
    val_user_ids: np.ndarray
    train_artist_ids: np.ndarray
    val_artist_ids: np.ndarray


def split_user_artist(
    plays: pd.DataFrame,
    user_ratio: float = 0.8,
    artist_ratio: float = 0.8,
    random_state: Optional[int] = None,
) -> UserArtistSplit:
    """
    Split interactions by sampling users and artists into train/validation.

    Users are split into train/val; artists are split into train/val.
    Then interactions are assigned to:
    - train: user in train_users AND artist in train_artists
    - val_unseen_users: user in val_users AND artist in train_artists
    - val_unseen_artists: user in train_users AND artist in val_artists
    - val_cold_start: user in val_users AND artist in val_artists

    Parameters
    ----------
    plays : pd.DataFrame
        Columns: user_id, artist_id, artist_name, plays (or at least user_id, artist_id).
    user_ratio : float, default 0.8
        Fraction of users to use for training (rest for validation).
    artist_ratio : float, default 0.8
        Fraction of artists to use for training (rest for validation).
    random_state : int or None, default None
        Random seed for reproducible splits.

    Returns
    -------
    UserArtistSplit
        Dataclass with train/val DataFrames and train/val user and artist id arrays.
    """
    rng = np.random.default_rng(random_state)

    users_unique = plays["user_id"].unique()
    artists_unique = plays["artist_id"].unique()

    n_users = len(users_unique)
    n_artists = len(artists_unique)

    perm_u = rng.permutation(n_users)
    perm_a = rng.permutation(n_artists)

    n_train_u = max(1, int(n_users * user_ratio))
    n_train_a = max(1, int(n_artists * artist_ratio))

    train_user_ids = users_unique[perm_u[:n_train_u]]
    val_user_ids = users_unique[perm_u[n_train_u:]]
    train_artist_ids = artists_unique[perm_a[:n_train_a]]
    val_artist_ids = artists_unique[perm_a[n_train_a:]]

    train_users_set = set(train_user_ids)
    val_users_set = set(val_user_ids)
    train_artists_set = set(train_artist_ids)
    val_artists_set = set(val_artist_ids)

    in_train_u = plays["user_id"].isin(train_users_set).values
    in_val_u = plays["user_id"].isin(val_users_set).values
    in_train_a = plays["artist_id"].isin(train_artists_set).values
    in_val_a = plays["artist_id"].isin(val_artists_set).values

    train_mask = in_train_u & in_train_a
    val_unseen_users_mask = in_val_u & in_train_a
    val_unseen_artists_mask = in_train_u & in_val_a
    val_cold_mask = in_val_u & in_val_a

    return UserArtistSplit(
        train=plays.loc[train_mask].reset_index(drop=True),
        val_unseen_users=plays.loc[val_unseen_users_mask].reset_index(drop=True),
        val_unseen_artists=plays.loc[val_unseen_artists_mask].reset_index(drop=True),
        val_cold_start=plays.loc[val_cold_mask].reset_index(drop=True),
        train_user_ids=train_user_ids,
        val_user_ids=val_user_ids,
        train_artist_ids=train_artist_ids,
        val_artist_ids=val_artist_ids,
    )
