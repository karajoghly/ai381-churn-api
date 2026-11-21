# Churn Prediction API

[![CI](https://github.com/karajoghly/ai381-churn-api/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/karajoghly/ai381-churn-api/actions/workflows/ci.yml)
[![Train, gate, deploy](https://github.com/karajoghly/ai381-churn-api/actions/workflows/train-deploy.yml/badge.svg?branch=main)](https://github.com/karajoghly/ai381-churn-api/actions/workflows/train-deploy.yml)

Deployment: every promoted champion is published to GHCR and smoke-tested by CI; run it locally with the command below (Azure Container Apps is optional in Lab 3, see the end of this file).

AI381 Lab 2. A logistic-regression churn model served by FastAPI, tested with pytest,
packaged in a multi-stage non-root Docker image, and built, scanned (Trivy) and pushed
to GHCR by GitHub Actions on every merge to `main`.

## Run it (one command)

```bash
docker run --rm -p 8000:8000 ghcr.io/karajoghly/ai381-churn-api:latest
```

Then open http://localhost:8000/docs, or:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 12, "monthly_fee": 70.5, "contract": "month-to-month", "internet_service": "fiber", "support_calls": 3}'
```

To run an exact, reproducible version, use the commit SHA tag instead of `latest`:
`ghcr.io/karajoghly/ai381-churn-api:<commit-sha>`.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | `200` with model version, training time and data hash; `503` if the model is not loaded |
| `POST /predict` | Returns `churn_prediction` (0/1), `churn_probability` and `model_version` |

Input fields: `tenure_months` (int ≥ 0), `monthly_fee` (float ≥ 0),
`contract` (`month-to-month` \| `one-year` \| `two-year`),
`internet_service` (`fiber` \| `dsl` \| `none`), `support_calls` (int ≥ 0).
Unknown fields (such as the old `monthly_charges`) are rejected with `422`.

## Develop

```bash
pip install -r requirements-dev.txt
python -m src.train.train          # writes models/model.joblib
pytest                             # runs the test suite
ruff check . && ruff format --check .
uvicorn src.app.main:app --reload  # http://localhost:8000/docs
```

Build and run the image locally:

```bash
docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t churn-api .
docker run --rm -p 8000:8000 churn-api
```

## Lab 3: tracked, gated, shipped to Azure

```
config/train.yaml --PR--> gate (dry run): retrain, compare with @champion -> red if worse
                 --main--> gate --promote--> build + Trivy + push (tag = commit SHA)
                                         --> Azure Container Apps (OIDC) --> smoke test
```

### Track experiments (MLflow in Docker Compose)

```bash
pip install -r requirements-train.txt
docker compose up -d --wait            # MLflow UI on http://localhost:5000
python -m src.train.candidates         # 3 candidates -> MLflow, winner -> churn-model@champion
```

Every run logs params, metrics (F1, F1 per slice, p95 latency, size), the data SHA-256,
the dataset, the model signature and the model. MLflow stores everything in `./mlflow`,
which is committed, so CI compares against the same `@champion`.
Why the champion won: [`MODEL_SELECTION.md`](MODEL_SELECTION.md).

### The promotion gate

`python -m src.train.gate` retrains the model in `config/train.yaml` and promotes it only if
**F1 >= champion F1 + min_delta** and **p95 latency <= 50 ms**; otherwise it exits 1 and the
workflow fails, so nothing is built or deployed. Pull requests run it as a dry run.

### Lineage

| Where | What it shows |
|---|---|
| Image tag | `ghcr.io/karajoghly/ai381-churn-api:<commit SHA>` |
| Image labels | `org.opencontainers.image.revision` = commit SHA, `ai381.mlflow.run_id` |
| `GET /health` | `git_sha`, `mlflow_run_id`, `registry_version` |
| MLflow registry | model version tags `run_id`, `git_sha`, `image_tag` |
| Workflow summary | image -> commit -> run ID, plus the `mlflow-export` artifact |

### Smoke test a running deployment

```bash
docker run -d --name churn -p 8000:8000 ghcr.io/karajoghly/ai381-churn-api:latest
bash scripts/smoke_test.sh http://localhost:8000
```

### Azure (optional, one time)

Run [`scripts/azure_setup.sh`](scripts/azure_setup.sh) in Azure Cloud Shell, then add the
three secrets it prints (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`)
and a repository variable `AZURE_DEPLOY=true`.
GitHub logs in with OIDC: no Azure password is stored anywhere.
