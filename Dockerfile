# syntax=docker/dockerfile:1.7
#
# One image, one process: FastAPI (uvicorn, 1 worker) serves the API and the static Next.js export at `/`.
# Starts and serves explained recommendations with no credentials at all (docs/DEPLOY.md).
#
#   docker build -t trailsaga-demo .
#   docker run --rm -p 8000:8000 -v trailsaga-data:/data trailsaga-demo
#
# Stages: frontend (pnpm build -> out/), pydeps (runtime wheels only, into /opt/venv), seed (migrate + seed the
# pristine DB from the committed catalogue), runtime (python slim, non-root, no build tools, no secrets).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG NODE_IMAGE=node:22-bookworm-slim

# ---------------------------------------------------------------------------------------------- frontend
FROM ${NODE_IMAGE} AS frontend
WORKDIR /src/frontend
ENV NEXT_TELEMETRY_DISABLED=1 \
    CI=1
RUN corepack enable && corepack prepare pnpm@10.23.0 --activate
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm-store,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile
COPY frontend/ ./
# Same origin as the API: the static bundle calls relative api/... paths (frontend/lib/api-base.ts).
RUN pnpm build && test -f out/index.html

# ----------------------------------------------------------------------------------------------- pydeps
FROM ${PYTHON_IMAGE} AS pydeps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
COPY backend/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN /opt/venv/bin/pip install --only-binary=:all: -r /tmp/requirements-runtime.txt \
    && find /opt/venv -name '__pycache__' -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.12/site-packages/pip /opt/venv/lib/python3.12/site-packages/setuptools*

# ------------------------------------------------------------------------------------------------- app
# Backend source the server needs (the .dockerignore keeps out .env, caches, raw OSM data, the venv, tests,
# evaluation harness and data pipeline outputs).
FROM ${PYTHON_IMAGE} AS app
WORKDIR /app/backend
COPY backend/alembic.ini ./
COPY backend/alembic ./alembic
COPY backend/app ./app
COPY backend/scripts ./scripts
# app.chat.vocab imports pipeline.tags (pure Python); the rest of the package is the offline data pipeline.
COPY backend/pipeline/__init__.py backend/pipeline/tags.py ./pipeline/
COPY backend/data/catalog/routes.jsonl.gz backend/data/catalog/manifest.json backend/data/catalog/ATTRIBUTION.md \
     ./data/catalog/
COPY backend/data/narratives ./data/narratives
COPY backend/data/historical_context ./data/historical_context

# ------------------------------------------------------------------------------------------------- seed
FROM app AS seed
COPY --from=pydeps /opt/venv /opt/venv
COPY backend/data/catalog/geometry.jsonl.gz ./data/catalog/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    LLM_PROVIDER=mock \
    LOG_ENABLE_FILE=false \
    LOG_ENABLE_CONSOLE=false
RUN mkdir -p /seed-db \
    && DATABASE_URL=sqlite:////seed-db/app.db python -m alembic upgrade head \
    && python scripts/seed_from_catalog.py --catalog data/catalog --db /seed-db/app.db \
    && python -c "import hashlib,sys; print(hashlib.sha256(open('data/catalog/routes.jsonl.gz','rb').read()).hexdigest())" \
       > /seed-db/catalogue.sha256 \
    && python scripts/check_runtime_imports.py

# ---------------------------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="trailsaga-demo" \
      org.opencontainers.image.description="Outdoor route recommender demo: FastAPI + static Next.js, OSM data (ODbL)" \
      org.opencontainers.image.licenses="route data: ODbL 1.0, (c) OpenStreetMap contributors"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown app:app /data
COPY --from=pydeps /opt/venv /opt/venv
COPY --from=app /app/backend /app/backend
COPY --from=seed /seed-db /app/backend/data/seed-db
COPY --from=frontend /src/frontend/out /app/frontend/out
# app/logger.py creates backend/logs/ at import time even with LOG_ENABLE_FILE=false; the code is read-only
# for the app user, so this one directory is pre-created and writable (it stays empty: logs go to stdout).
RUN mkdir -p /app/backend/logs && chown app:app /app/backend/logs
# Demo defaults (docs/DEPLOY.md, "Configuration"): no provider credentials in the image; narratives from the
# pre-generated store; protections on. Every value can be overridden with `docker run -e`.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    DATABASE_URL=sqlite+aiosqlite:////data/app.db \
    LLM_CACHE_DIR=/data/llm-cache \
    FRONTEND_DIST_DIR=/app/frontend/out \
    LOG_ENABLE_FILE=false \
    LOG_DETAILED_FORMAT=false \
    LLM_PROVIDER=vertex \
    NARRATIVES_LIVE=false \
    RATE_LIMIT_ENABLED=true \
    DAILY_BUDGET_USD=2 \
    LLM_BUDGET_USD_PER_RUN=1 \
    CHAT_BUDGET_USD=1 \
    RECSYS_BUDGET_USD=0.5
USER app
WORKDIR /app/backend
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'), timeout=4).status == 200 else 1)"
CMD ["python", "scripts/container_start.py"]
