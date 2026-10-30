from pathlib import Path

import pytest

from src.train.train import DEFAULT_DATA, DEFAULT_OUT, train

VALID_PAYLOAD = {
    "tenure_months": 12,
    "monthly_fee": 70.5,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


@pytest.fixture(scope="session")
def model_path() -> Path:
    """Train once per test session if no model artifact exists yet."""
    if not DEFAULT_OUT.exists():
        train(DEFAULT_DATA, DEFAULT_OUT)
    return DEFAULT_OUT


@pytest.fixture(scope="session")
def client(model_path):
    from fastapi.testclient import TestClient

    from src.app.main import app

    with TestClient(app) as c:  # "with" runs the lifespan, which loads the model
        yield c


@pytest.fixture
def payload() -> dict:
    return dict(VALID_PAYLOAD)
