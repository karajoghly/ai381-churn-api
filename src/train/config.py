"""Reads config/train.yaml: which model the training workflow trains, and the gate's rules."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("config/train.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    config.setdefault("params", {})
    gate = config.setdefault("gate", {})
    gate.setdefault("metric", "f1")
    gate.setdefault("min_delta", 0.0)
    gate.setdefault("p95_latency_ms_budget", 50)
    return config


def write_config(kind: str, params: dict, gate: dict, path: Path = DEFAULT_CONFIG) -> None:
    header = (
        "# The challenger the training workflow retrains on every run.\n"
        "# Change it in a pull request to propose a new model; the gate decides.\n"
    )
    body = yaml.safe_dump({"model": kind, "params": params, "gate": gate}, sort_keys=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(header + body)
