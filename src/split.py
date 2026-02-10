"""
Train/validation split by sampling users and artists.

Level 1 (split_matrix): seen vs unseen users/artists → train, val_unseen_users,
  val_unseen_artists, val_cold_start.

Level 2 (holdout_per_user): within any user×item matrix, hold out a fraction of
  each user's interactions for evaluation. Use viewable matrix for recalc/filter,
  held-out sets as y_true.
"""
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

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


def holdout_per_user(
    user_items: csr_matrix,
    holdout_ratio: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[csr_matrix, List[Set[int]]]:
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
    y_true_held_out : list of set of int
        y_true_held_out[i] = set of column indices held out for user i (for evaluation).
    """
    rng = np.random.default_rng(random_state)
    n_users, n_items = user_items.shape
    user_items = user_items.tocsr()

    # Build viewable matrix: same structure but only "viewable" entries kept
    viewable_data = []
    viewable_indices = []
    viewable_indptr = [0]

    y_true_held_out: List[Set[int]] = []

    for i in range(n_users):
        row = user_items.getrow(i)
        cols = row.indices
        data = row.data
        nnz = len(cols)

        if nnz == 0:
            y_true_held_out.append(set())
            viewable_indptr.append(viewable_indptr[-1])
            continue

        if nnz == 1:
            # Keep only viewable (all); hold out nothing so we can still compute user factor
            viewable_data.extend(data.tolist())
            viewable_indices.extend(cols.tolist())
            y_true_held_out.append(set())
        else:
            # Shuffle and split: at least 1 viewable, at least 1 held out
            perm = rng.permutation(nnz)
            cols_perm = cols[perm]
            data_perm = data[perm]
            n_held = min(max(1, int(holdout_ratio * nnz)), nnz - 1)
            n_viewable = nnz - n_held
            viewable_cols = cols_perm[:n_viewable]
            viewable_vals = data_perm[:n_viewable]
            held_cols = cols_perm[n_viewable:]
            viewable_data.extend(viewable_vals.tolist())
            viewable_indices.extend(viewable_cols.tolist())
            y_true_held_out.append(set(held_cols.tolist()))

        viewable_indptr.append(len(viewable_data))

    viewable = csr_matrix(
        (viewable_data, viewable_indices, viewable_indptr),
        shape=(n_users, n_items),
        dtype=user_items.dtype,
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
