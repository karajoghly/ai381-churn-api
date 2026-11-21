# syntax=docker/dockerfile:1

# Base image pinned by tag AND digest: the tag is for humans, the digest makes the
# build byte-for-byte reproducible. Run scripts/pin_base_image.sh to set the digest.
ARG BASE_IMAGE=python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

# ---------- Stage 1: builder (has pip and build tools) ----------
FROM ${BASE_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies first: this layer is cached until requirements.txt changes.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Stage 2: runtime (only what the service needs) ----------
FROM ${BASE_IMAGE} AS runtime

# Lineage, readable with `docker inspect`: image tag == commit SHA -> MLflow run ID.
ARG GIT_SHA=dev
ARG MLFLOW_RUN_ID=untracked
LABEL org.opencontainers.image.revision=${GIT_SHA} \
      org.opencontainers.image.source="https://github.com/karajoghly/ai381-churn-api" \
      ai381.mlflow.run_id=${MLFLOW_RUN_ID}
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    APP_VERSION=${GIT_SHA}

# Unprivileged user with no shell and no home directory.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# The model is NOT trained here. The image ships exactly the bundle the promotion
# gate approved (models/model.joblib, written by `python -m src.train.gate --promote`).
COPY models/model.joblib ./models/model.joblib
COPY src/ ./src/

# Files stay owned by root, so the app user can read them but not change them.
USER app

EXPOSE 8000

# slim images have no curl, so the check uses Python's standard library.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
