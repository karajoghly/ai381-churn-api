# Churn Prediction API

[![CI](https://github.com/karajoghly/ai381-churn-api/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/karajoghly/ai381-churn-api/actions/workflows/ci.yml)

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

## Project layout

```
src/features.py        input contract shared by training and serving
src/train/train.py     trains the sklearn pipeline, saves models/model.joblib
src/app/main.py        FastAPI service (/health, /predict)
tests/                 pytest suite (API contract, schema, edge cases, determinism, model)
Dockerfile             multi-stage, non-root, digest-pinned base, HEALTHCHECK
.github/workflows/     CI: lint+test on PRs; build, Trivy scan, push to GHCR on main
scripts/pin_base_image.sh  refreshes the base image digest in the Dockerfile
```

## CI pipeline

- **Pull request to `main`**: ruff lint and format check, base-image pin check, train, pytest.
- **Push to `main`** (after merge): build image → Trivy report (HIGH, CRITICAL) →
  Trivy gate (fails on fixable CRITICAL) → push `ghcr.io/karajoghly/ai381-churn-api:<sha>` and `:latest`.
  The image digest is printed in the run summary.
- All third-party actions are pinned to full commit SHAs. `main` is protected:
  changes land only through a pull request with a green `lint-test` check.
