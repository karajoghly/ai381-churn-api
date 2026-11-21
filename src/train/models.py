"""The model families we compare. Every one shares the same preprocessing."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE

MODEL_KINDS = {
    "logreg": (LogisticRegression, {"max_iter": 1000}),
    "random_forest": (RandomForestClassifier, {"n_jobs": 1}),
    "hist_gb": (HistGradientBoostingClassifier, {}),
}


def build_pipeline(kind: str = "logreg", params: dict | None = None) -> Pipeline:
    """Preprocessing and model in one object, so serving can never skip a step."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"Unknown model kind {kind!r}. Choose one of {sorted(MODEL_KINDS)}")
    cls, defaults = MODEL_KINDS[kind]
    model = cls(**{**defaults, "random_state": RANDOM_STATE, **(params or {})})
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])
