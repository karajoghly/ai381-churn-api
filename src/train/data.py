"""Loading, hashing and splitting the training data (shared by every training entry point)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import FEATURES, RANDOM_STATE, TARGET

DEFAULT_DATA = Path("data/churn_v2.csv")
TEST_SIZE = 0.2


def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV and fail loudly if the schema is not what the model expects."""
    df = pd.read_csv(path)
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}. Found: {list(df.columns)}")
    return df


def file_sha256(path: Path) -> str:
    """Hash of the exact bytes trained on: the same hash means the same data."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split(df: pd.DataFrame):
    """Fixed, stratified split, so every candidate is scored on the same test rows."""
    return train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
