"""Churn prediction service: GET /health and POST /predict."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.features import CONTRACT_VALUES, FEATURES, INTERNET_SERVICE_VALUES

logger = logging.getLogger("churn-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
# Set at image build time to the git commit SHA, so the running container can say
# exactly which code and model it serves (the Lab 1 Twist 3 question).
APP_VERSION = os.getenv("APP_VERSION", "dev")

_state: dict = {"bundle": None}


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("features") != FEATURES:
        raise ValueError(f"Model expects {bundle.get('features')}, API sends {FEATURES}")
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["bundle"] = load_bundle(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except (FileNotFoundError, ValueError) as exc:
        _state["bundle"] = None
        logger.error("Model not loaded: %s", exc)
    yield
    _state["bundle"] = None


app = FastAPI(title="Customer Churn Prediction API", version=APP_VERSION, lifespan=lifespan)


class ChurnInput(BaseModel):
    """Request schema. extra='forbid' rejects unknown fields such as the old
    'monthly_charges' name instead of silently dropping them."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "tenure_months": 12,
                "monthly_fee": 70.5,
                "contract": "month-to-month",
                "internet_service": "fiber",
                "support_calls": 3,
            }
        },
    )

    tenure_months: int = Field(ge=0, le=600)
    monthly_fee: float = Field(ge=0, le=10_000)
    contract: Literal[CONTRACT_VALUES]
    internet_service: Literal[INTERNET_SERVICE_VALUES]
    support_calls: int = Field(ge=0, le=100)


class PredictionOut(BaseModel):
    churn_prediction: int
    churn_probability: float
    model_version: str


def _require_model() -> dict:
    bundle = _state["bundle"]
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return bundle


@app.get("/health")
def health() -> dict:
    bundle = _require_model()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": APP_VERSION,
        "trained_at": bundle["trained_at"],
        "data_sha256": bundle["data_sha256"][:12],
    }


@app.post("/predict", response_model=PredictionOut)
def predict(data: ChurnInput) -> PredictionOut:
    bundle = _require_model()
    row = pd.DataFrame([data.model_dump()], columns=FEATURES)
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return PredictionOut(
        churn_prediction=int(probability >= 0.5),
        churn_probability=round(probability, 6),
        model_version=APP_VERSION,
    )
