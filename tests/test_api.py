"""Tests for the HTTP contract of the service."""

import math

import pytest


# 1. API contract: /health tells a load balancer (and Docker HEALTHCHECK) the truth.
def test_health_contract(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert {"model_version", "trained_at", "data_sha256"} <= body.keys()


# 2. API contract: /predict always returns the same, well-formed shape.
def test_predict_contract(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_prediction", "churn_probability", "model_version"}
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] == int(body["churn_probability"] >= 0.5)


# 3. Input schema: the exact Lab 1 Twist 1 bug. The old column name must be
#    rejected with 422, not silently dropped and replaced by 0.
def test_old_column_name_is_rejected(client, payload):
    payload["monthly_charges"] = payload.pop("monthly_fee")
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


# 4. Input schema: bad types, unknown categories, missing and negative values.
@pytest.mark.parametrize(
    "field, value",
    [
        ("contract", "weekly"),  # unknown category
        ("internet_service", "5G"),  # unknown category
        ("tenure_months", -1),  # negative
        ("monthly_fee", "a lot"),  # wrong type
        ("support_calls", None),  # null
    ],
)
def test_invalid_input_is_rejected(client, payload, field, value):
    payload[field] = value
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_field_is_rejected(client, payload):
    del payload["contract"]
    assert client.post("/predict", json=payload).status_code == 422


# 5. Edge cases: boundary values are valid input and must not crash the model.
@pytest.mark.parametrize(
    "overrides",
    [
        # brand-new customer: every number at its minimum
        {"tenure_months": 0, "monthly_fee": 0.0, "support_calls": 0},
        # maximum allowed values, far outside the training range
        {"tenure_months": 600, "monthly_fee": 10_000, "support_calls": 100},
    ],
)
def test_edge_values_return_valid_probability(client, payload, overrides):
    payload.update(overrides)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    p = r.json()["churn_probability"]
    assert math.isfinite(p) and 0.0 <= p <= 1.0


# 6. Determinism: the same request always gets the same answer.
def test_prediction_is_deterministic(client, payload):
    answers = {client.post("/predict", json=payload).json()["churn_probability"] for _ in range(5)}
    assert len(answers) == 1
