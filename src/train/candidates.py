"""Train the candidate models, log each to MLflow, and crown the best one @champion.

Usage (MLflow must be running: docker compose up -d --wait):
    python -m src.train.candidates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.train.config import DEFAULT_CONFIG, load_config, write_config
from src.train.data import DEFAULT_DATA, file_sha256, load_data, split
from src.train.evaluate import evaluate
from src.train.models import build_pipeline

CANDIDATES = [
    ("logreg", {"C": 1.0, "class_weight": "balanced"}),
    ("random_forest", {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced"}),
    ("hist_gb", {"max_iter": 200, "learning_rate": 0.05, "max_depth": 4}),
]
REPORT = Path("reports/candidates.md")


def pick(results: list[dict], budget_ms: float) -> dict:
    """Highest F1 among models inside the latency budget; smaller model wins a tie."""
    eligible = [r for r in results if r["metrics"]["p95_latency_ms"] <= budget_ms]
    if not eligible:
        raise SystemExit(f"No candidate meets the {budget_ms} ms p95 latency budget.")
    return max(eligible, key=lambda r: (r["metrics"]["f1"], -r["metrics"]["model_size_kb"]))


def report(results: list[dict], winner: dict, budget_ms: float) -> str:
    slices = sorted(k for k in results[0]["metrics"] if k.startswith("f1_slice/"))
    header = ["Model", "F1", "ROC AUC", "p95 ms", "Size KB", *[s[9:] for s in slices], "Run ID"]
    lines = [
        f"p95 latency budget: {budget_ms} ms. Winner: **{winner['kind']}**.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for r in results:
        m = r["metrics"]
        name = f"**{r['kind']}** (champion)" if r is winner else r["kind"]
        row = [name, m["f1"], m["roc_auc"], m["p95_latency_ms"], m["model_size_kb"]]
        row += [m[s] for s in slices] + [f"`{r['run_id']}`"]
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    from src.train.tracking import log_run, register_champion

    parser = argparse.ArgumentParser(description="Train and compare candidate models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    gate = load_config(args.config)["gate"]
    df = load_data(args.data)
    data_sha = file_sha256(args.data)
    X_train, X_test, y_train, y_test = split(df)

    results = []
    for kind, params in CANDIDATES:
        pipeline = build_pipeline(kind, params).fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        run_id, model_uri = log_run(
            run_name=f"candidate-{kind}",
            kind=kind,
            params=params,
            pipeline=pipeline,
            metrics=metrics,
            df=df,
            X_sample=X_test.head(20),
            data_path=str(args.data),
            data_sha256=data_sha,
            tags={"stage": "candidate"},
        )
        results.append(
            {"kind": kind, "params": params, "metrics": metrics, "run_id": run_id, "uri": model_uri}
        )
        print(
            f"{kind:14s} f1={metrics['f1']:.4f} p95={metrics['p95_latency_ms']:.2f}ms run={run_id}"
        )

    winner = pick(results, gate["p95_latency_ms_budget"])
    version = register_champion(winner["uri"], winner["run_id"], tags={"selected_by": "candidates"})
    # The training workflow retrains exactly this configuration from now on.
    write_config(winner["kind"], winner["params"], gate, args.config)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report(results, winner, gate["p95_latency_ms_budget"]))
    Path("reports/candidates.json").write_text(
        json.dumps(
            [{k: r[k] for k in ("kind", "params", "metrics", "run_id")} for r in results], indent=2
        )
    )
    print(f"\nChampion: {winner['kind']} -> churn-model v{version} (@champion)")
    print(f"Champion run ID: {winner['run_id']}")
    print(f"Comparison table written to {REPORT}; config/train.yaml updated.")


if __name__ == "__main__":
    main()
