#!/usr/bin/env bash
# Resolve the base image tag to its current digest and write "tag@sha256:..." into
# the Dockerfile. Re-run it on purpose (e.g. monthly) to pick up security patches.
set -euo pipefail

TAG="${1:-python:3.12-slim-trixie}"
DOCKERFILE="${2:-Dockerfile}"

docker pull --quiet "$TAG" >/dev/null
DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$TAG" | cut -d@ -f2)"

if [[ ! "$DIGEST" =~ ^sha256:[a-f0-9]{64}$ ]]; then
  echo "Could not resolve a digest for $TAG" >&2
  exit 1
fi

sed -i -E "s|^ARG BASE_IMAGE=.*$|ARG BASE_IMAGE=${TAG}@${DIGEST}|" "$DOCKERFILE"
grep '^ARG BASE_IMAGE=' "$DOCKERFILE"
