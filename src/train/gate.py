"""The promotion gate: retrain the configured challenger, compare it with @champion,
and promote it only if it is at least as good AND inside the latency budget.

Exit code 0 = promoted (or would be, with --dry-run); exit code 1 = BLOCKED.

Usage (MLflow must be running):
    python -m src.train.gate --dry-run        # pull requests: decide, change nothing
    python -m src.train.gate --promote        # main: decide, register, write the bundle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.train.config import DEFAULT_CONFIG, load_config
from src.train.data import DEFAULT_DATA, file_sha256, load_data, split
from src.train.evaluate import evaluate
from src.train.models import build_pipeline
from src.train.train import DEFAULT_OUT, save_bundle

RESULT = Path("reports/gate.json")


def decide(challenger: dict, champion: dict | None, gate: dict) -> tuple[bool, list[dict]]:
    """Pure decision logic (unit-tested). Returns (promote?, list of checks)."""
    metric, budget = gate["metric"], gate["p95_latency_ms_budget"]
    if champion is None:
        champ_score, need = None, float("-inf")
    else:
        champ_score = round(champion["metrics"][metric], 4)
        need = round(champ_score + gate["min_delta"], 4)
    checks = [
        {
            "check": f"{metric} >= champion + min_delta",
            "challenger": challenger[metric],
            "required": "no champion yet" if champion is None else f">= {need}",
            "passed": challenger[metric] >= need,
        },
        {
            "check": "p95 latency within budget",
            "challenger": challenger["p95_latency_ms"],
            "required": f"<= {budget} ms",
            "passed": challenger["p95_latency_ms"] <= budget,
        },
    ]
    return all(c["passed"] for c in checks), checks


def summary(
    promote: bool, checks: list[dict], run_id: str, champion: dict | None, mode: str
) -> str:
    verdict = (
        "PROMOTED"
        if promote and mode == "promote"
        else "PASSED (dry run)"
        if promote
        else "BLOCKED"
    )
    icon = "✅" if promote else "❌"
    champ = "none" if champion is None else f"v{champion['version']} (run `{champion['run_id']}`)"
    lines = [
        f"## {icon} Promotion gate: {verdict}",
        "",
        f"Challenger run: `{run_id}`  ",
        f"Current champion: {champ}",
        "",
        "| Check | Challenger | Required | Result |",
        "|---|---|---|---|",
    ]
    for c in checks:
        result = "pass" if c["passed"] else "FAIL"
        lines.append(f"| {c['check']} | {c['challenger']} | {c['required']} | {result} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    from src.train.tracking import get_champion, git_sha, log_run, register_champion

    parser = argparse.ArgumentParser(
        description="Retrain, compare with @champion, promote or block."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--promote", action="store_true", help="register + write bundle if it passes")
    mode.add_argument("--dry-run", action="store_true", help="decide only, change nothing")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    config = load_config(args.config)
    kind, params, gate = config["model"], config["params"], config["gate"]
    df = load_data(args.data)
    data_sha = file_sha256(args.data)
    X_train, X_test, y_train, y_test = split(df)

    pipeline = build_pipeline(kind, params).fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    run_id, model_uri = log_run(
        run_name=f"challenger-{kind}",
        kind=kind,
        params=params,
        pipeline=pipeline,
        metrics=metrics,
        df=df,
        X_sample=X_test.head(20),
        data_path=str(args.data),
        data_sha256=data_sha,
        tags={"stage": "challenger", "trigger": os.getenv("GITHUB_EVENT_NAME", "local")},
    )

    champion = get_champion()
    promote, checks = decide(metrics, champion, gate)
    mode_name = "promote" if args.promote else "dry-run"
    text = summary(promote, checks, run_id, champion, mode_name)
    print(text)

    result = {
        "promote": promote,
        "mode": mode_name,
        "run_id": run_id,
        "checks": checks,
        "champion": champion and {k: champion[k] for k in ("version", "run_id")},
    }
    if promote and args.promote:
        version = register_champion(
            model_uri, run_id, tags={"git_sha": git_sha(), "image_tag": git_sha()}
        )
        save_bundle(
            pipeline,
            out_path=args.out,
            kind=kind,
            params=params,
            metrics=metrics,
            data_sha256=data_sha,
            mlflow_run_id=run_id,
            registry_version=version,
        )
        result["registry_version"] = version
        print(f"Promoted: churn-model v{version} is now @champion. Bundle: {args.out}")

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(text)
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"run_id={run_id}\nregistry_version={result.get('registry_version', '')}\n")

    if not promote:
        print("GATE BLOCKED: the challenger is worse than @champion or too slow. Nothing shipped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
