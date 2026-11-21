"""Unit tests for the promotion gate's decision logic (no MLflow server needed)."""

from src.train.gate import decide

GATE = {"metric": "f1", "min_delta": 0.0, "p95_latency_ms_budget": 50}
CHAMPION = {"version": "1", "run_id": "abc", "metrics": {"f1": 0.59}}


def test_equal_or_better_and_fast_is_promoted():
    promote, _ = decide({"f1": 0.59, "p95_latency_ms": 5}, CHAMPION, GATE)
    assert promote


def test_worse_model_is_blocked():
    promote, checks = decide({"f1": 0.19, "p95_latency_ms": 5}, CHAMPION, GATE)
    assert not promote
    assert checks[0]["passed"] is False


def test_better_but_too_slow_is_blocked():
    promote, checks = decide({"f1": 0.70, "p95_latency_ms": 80}, CHAMPION, GATE)
    assert not promote
    assert checks[1]["passed"] is False


def test_min_delta_demands_a_real_improvement():
    promote, _ = decide({"f1": 0.595, "p95_latency_ms": 5}, CHAMPION, {**GATE, "min_delta": 0.01})
    assert not promote


def test_first_model_is_promoted_when_no_champion_exists():
    promote, _ = decide({"f1": 0.30, "p95_latency_ms": 5}, None, GATE)
    assert promote
