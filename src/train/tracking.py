"""MLflow logging and model registry helpers (talks to the server from docker-compose.yml)."""

from __future__ import annotations

import os
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.features import RANDOM_STATE, TARGET
from src.train.data import TEST_SIZE

EXPERIMENT = "churn"
REGISTERED_MODEL = "churn-model"
CHAMPION_ALIAS = "champion"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")


def git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_run(
    *,
    run_name: str,
    kind: str,
    params: dict,
    pipeline,
    metrics: dict,
    df: pd.DataFrame,
    X_sample: pd.DataFrame,
    data_path: str,
    data_sha256: str,
    tags: dict | None = None,
) -> tuple[str, str]:
    """Log one training run: params, metrics, data hash, signature and the model itself."""
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model_kind": kind,
                **params,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "data_sha256": data_sha256,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "data_sha256": data_sha256,
                "data_path": data_path,
                "git_sha": git_sha(),
                **(tags or {}),
            }
        )
        # The dataset itself, with MLflow's own digest, under the run's "Datasets" section.
        dataset = mlflow.data.from_pandas(df, source=data_path, targets=TARGET, name="churn_v2")
        mlflow.log_input(dataset, context="training")

        # The signature records the exact input columns and types the model accepts.
        signature = infer_signature(X_sample, pipeline.predict(X_sample))
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_sample.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}"],
        )
        return run.info.run_id, info.model_uri


def register_champion(model_uri: str, run_id: str, tags: dict | None = None) -> str:
    """Create a new registry version from the run's model and move @champion onto it."""
    version = mlflow.register_model(
        model_uri, REGISTERED_MODEL, tags={"run_id": run_id, **(tags or {})}
    ).version
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, version)
    return str(version)


def get_champion() -> dict | None:
    """The current @champion: its registry version, run ID and logged metrics (None if unset)."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    except MlflowException:
        return None
    run_id = mv.tags.get("run_id") or mv.run_id
    run = client.get_run(run_id)
    return {
        "version": str(mv.version),
        "run_id": run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
    }
