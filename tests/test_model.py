"""Tests for the model artifact and the training script."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES
from src.train.train import DEFAULT_DATA, load_data, train


# 7. Model loads: the artifact exists, has the right pieces and can predict.
def test_model_loads(model_path):
    bundle = joblib.load(model_path)
    assert {"pipeline", "features", "metrics", "data_sha256", "trained_at"} <= bundle.keys()
    assert bundle["features"] == FEATURES
    assert hasattr(bundle["pipeline"], "predict_proba")


# 8. Quality floor: the model must beat "always predict the majority class".
def test_model_beats_majority_baseline(model_path):
    m = joblib.load(model_path)["metrics"]
    assert m["accuracy"] > m["baseline_accuracy"]
    assert m["roc_auc"] > 0.6


# 9. Deterministic training: same data and seed give the same model.
def test_training_is_deterministic(tmp_path):
    a = train(DEFAULT_DATA, tmp_path / "a.joblib")["pipeline"]
    b = train(DEFAULT_DATA, tmp_path / "b.joblib")["pipeline"]
    X = pd.read_csv(DEFAULT_DATA)[FEATURES]
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


# 10. Schema drift fails loudly at training time, naming the missing column.
def test_training_fails_clearly_on_renamed_column(tmp_path):
    bad = pd.read_csv(DEFAULT_DATA).rename(columns={"monthly_fee": "monthly_charges"})
    bad_csv = tmp_path / "churn_renamed.csv"
    bad.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="monthly_fee"):
        load_data(bad_csv)
