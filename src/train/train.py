"""Train the churn model and save it as one versioned artifact.

Usage:
    python -m src.train.train --data data/churn_v2.csv --out models/model.joblib
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)

DEFAULT_DATA = Path("data/churn_v2.csv")
DEFAULT_OUT = Path("models/model.joblib")


def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV and fail loudly if the schema is not what the model expects."""
    df = pd.read_csv(path)
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}. Found: {list(df.columns)}")
    return df


def build_pipeline() -> Pipeline:
    """Preprocessing and model in one object, so serving can never skip a step."""
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train(data_path: Path = DEFAULT_DATA, out_path: Path = DEFAULT_OUT) -> dict:
    df = load_data(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline().fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, proba >= 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "baseline_accuracy": round(float(max(y_test.mean(), 1 - y_test.mean())), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    bundle = {
        "pipeline": pipeline,
        "features": FEATURES,
        "metrics": metrics,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(json.dumps({"saved_to": str(out_path), **metrics}, indent=2))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
