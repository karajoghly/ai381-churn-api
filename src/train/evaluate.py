"""Everything the gate and MODEL_SELECTION.md compare: quality, quality per slice, speed, size."""

from __future__ import annotations

import io
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.features import CATEGORICAL_FEATURES

LATENCY_ROWS = 200
WARMUP_ROWS = 10


def p95_latency_ms(pipeline, X: pd.DataFrame, n: int = LATENCY_ROWS) -> float:
    """p95 time to score ONE row, the way /predict is called (not a batch average)."""
    rows = [X.iloc[[i % len(X)]] for i in range(n + WARMUP_ROWS)]
    for row in rows[:WARMUP_ROWS]:
        pipeline.predict_proba(row)
    timings = []
    for row in rows[WARMUP_ROWS:]:
        start = time.perf_counter()
        pipeline.predict_proba(row)
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(timings, 95))


def model_size_kb(pipeline) -> float:
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return len(buffer.getvalue()) / 1024


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "p95_latency_ms": p95_latency_ms(pipeline, X_test),
        "model_size_kb": model_size_kb(pipeline),
    }
    # F1 per data slice: a model can look fine overall and fail one customer group.
    for column in CATEGORICAL_FEATURES:
        for value in sorted(X_test[column].unique()):
            mask = (X_test[column] == value).to_numpy()
            metrics[f"f1_slice/{column}/{value}"] = f1_score(
                y_test[mask], pred[mask], zero_division=0
            )
    return {k: round(float(v), 4) for k, v in metrics.items()}
