"""
Dataset loading for Last.fm 360K.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_360K = _PROJECT_ROOT / "data" / "lastfm-dataset-360K"

# Rows to use when limit_rows=True
LIMIT_ROWS = 500_000


def load_360k(limit_rows: bool = True) -> pd.DataFrame:
    """
    Load the Last.fm 360K user–artist plays dataset.

    Parameters
    ----------
    limit_rows : bool, default True
        If True, load only the first 500_000 rows (faster, for development).
        If False, load the entire dataset (~17.5M rows).

    Returns
    -------
    pd.DataFrame
        Columns: user_id, artist_id, artist_name, plays.
    """
    nrows = LIMIT_ROWS if limit_rows else None
    plays = pd.read_csv(
        _DATA_360K / "usersha1-artmbid-artname-plays.tsv",
        sep="\t",
        header=None,
        names=["user_id", "artist_id", "artist_name", "plays"],
        dtype={"user_id": str, "artist_id": str, "artist_name": str, "plays": np.int32},
        nrows=nrows,
    )
    return plays


def load_360k_profiles() -> pd.DataFrame:
    """
    Load the Last.fm 360K user profiles (no row limit).

    Returns
    -------
    pd.DataFrame
        Columns: user_id, gender, age, country, registered.
    """
    return pd.read_csv(
        _DATA_360K / "usersha1-profile.tsv",
        sep="\t",
        header=None,
        names=["user_id", "gender", "age", "country", "registered"],
        dtype={"user_id": str, "gender": str, "age": "Int64", "country": str, "registered": str},
    )
