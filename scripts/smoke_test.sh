#!/usr/bin/env bash
# Smoke test a running deployment of the churn API, and check its lineage.
#
#   bash scripts/smoke_test.sh <base-url> [expected-git-sha] [expected-mlflow-run-id]
#   bash scripts/smoke_test.sh http://localhost:8000
#
# Exits non-zero if the service is down, /predict fails, or the running image is not
# the commit / MLflow run you expected.
set -euo pipefail

URL="${1:?usage: smoke_test.sh <base-url> [git-sha] [mlflow-run-id]}"
EXPECTED_SHA="${2:-}"
EXPECTED_RUN="${3:-}"

echo "==> GET $URL/health"
# Retry while the container starts (or wakes up from scale-to-zero).
HEALTH=$(curl -fsS --retry 15 --retry-delay 4 --retry-all-errors --max-time 30 "$URL/health")
echo "$HEALTH"

HEALTH="$HEALTH" EXPECTED_SHA="$EXPECTED_SHA" EXPECTED_RUN="$EXPECTED_RUN" python3 - <<'PY'
import json, os
h = json.loads(os.environ["HEALTH"])
assert h["status"] == "ok" and h["model_loaded"], "service is not healthy"
sha, run = os.environ["EXPECTED_SHA"], os.environ["EXPECTED_RUN"]
if sha:
    assert h["git_sha"] == sha, f"wrong image: running {h['git_sha']}, expected {sha}"
if run:
    assert h["mlflow_run_id"] == run, f"wrong model: {h['mlflow_run_id']}, expected {run}"
print()
print("Lineage:")
print(f"  image tag / commit SHA : {h['git_sha']}")
print(f"  MLflow run ID          : {h['mlflow_run_id']}")
print(f"  registry version       : churn-model v{h['registry_version']} ({h['model_kind']})")
PY

echo
echo "==> POST $URL/predict"
curl -fsS -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"tenure_months": 12, "monthly_fee": 70.5, "contract": "month-to-month", "internet_service": "fiber", "support_calls": 3}'
echo
echo
echo "SMOKE TEST PASSED"
