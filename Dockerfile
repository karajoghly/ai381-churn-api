# syntax=docker/dockerfile:1

# Base image pinned by tag AND digest: the tag is for humans, the digest makes the
# build byte-for-byte reproducible. Run scripts/pin_base_image.sh to set the digest.
ARG BASE_IMAGE=python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

# ---------- Stage 1: builder (has pip, build tools, training data) ----------
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

# Train inside the build, so the model in the image always matches this commit.
COPY src/ src/
COPY data/ data/
RUN python -m src.train.train --data data/churn_v2.csv --out models/model.joblib

# ---------- Stage 2: runtime (only what the service needs) ----------
FROM ${BASE_IMAGE} AS runtime

ARG GIT_SHA=dev
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
COPY --from=builder /build/models ./models
COPY src/ ./src/

# Files stay owned by root, so the app user can read them but not change them.
USER app

EXPOSE 8000

# slim images have no curl, so the check uses Python's standard library.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
