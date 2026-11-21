"""Train the configured model WITHOUT MLflow and save the serving bundle.

Used by the lint-test CI job and the unit tests, which must not need a tracking server.
The tracked paths are src/train/candidates.py (pick a champion) and src/train/gate.py
(retrain, compare with the champion, promote).

Usage:
    python -m src.train.train --data data/churn_v2.csv --out models/model.joblib
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn

from src.features import FEATURES
from src.train.config import DEFAULT_CONFIG, load_config
from src.train.data import DEFAULT_DATA, file_sha256, load_data, split
from src.train.evaluate import evaluate
from src.train.models import build_pipeline

__all__ = ["DEFAULT_DATA", "DEFAULT_OUT", "load_data", "save_bundle", "train"]

DEFAULT_OUT = Path("models/model.joblib")


def save_bundle(
    pipeline,
    *,
    out_path: Path,
    kind: str,
    params: dict,
    metrics: dict,
    data_sha256: str,
    mlflow_run_id: str = "untracked",
    registry_version: str = "none",
) -> dict:
    """The one file the image serves: model + everything needed to trace where it came from."""
    bundle = {
        "pipeline": pipeline,
        "features": FEATURES,
        "model_kind": kind,
        "params": params,
        "metrics": metrics,
        "data_sha256": data_sha256,
        "mlflow_run_id": mlflow_run_id,
        "registry_version": registry_version,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return bundle


def train(
    data_path: Path = DEFAULT_DATA,
    out_path: Path = DEFAULT_OUT,
    config_path: Path = DEFAULT_CONFIG,
) -> dict:
    config = load_config(config_path)
    kind, params = config["model"], config["params"]
    X_train, X_test, y_train, y_test = split(load_data(data_path))
    pipeline = build_pipeline(kind, params).fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    bundle = save_bundle(
        pipeline,
        out_path=out_path,
        kind=kind,
        params=params,
        metrics=metrics,
        data_sha256=file_sha256(data_path),
    )
    print(json.dumps({"saved_to": str(out_path), "model": kind, **metrics}, indent=2))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model (no tracking).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    train(args.data, args.out, args.config)


if __name__ == "__main__":
    main()
