# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Slim toolchain for any wheel that lacks a manylinux build, plus curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

ENV HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/api/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
