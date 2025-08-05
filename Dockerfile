# syntax = docker/dockerfile:1
# =================
#  Builder Stage
# =================
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d AS builder

# Configure apt retries
# CORRECTED: Combined into a single, valid RUN command
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Pipeline-Depth 0;' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::No-Cache true;' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::BrokenProxy true;' >> /etc/apt/apt.conf.d/80-retries

WORKDIR /app

# Install system dependencies with cache mounts
# FIXED: Prefixed cache mount IDs
RUN --mount=type=cache,id=doc-query-api-apt-cache,target=/var/cache/apt \
    --mount=type=cache,id=doc-query-api-apt-lib,target=/var/lib/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg62-turbo \
        zlib1g \
        libfreetype6 \
        lcms2 \
        libopenjp2-7 \
        libtiff6 \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and directories
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/logs /app/data

# Python env
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install Python dependencies with pip cache mount
COPY requirements.txt .
# FIXED: Prefixed cache mount ID
RUN --mount=type=cache,id=doc-query-api-pip-cache,target=/root/.cache/pip \
    pip install --user --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# =================
#  Runtime Stage
# =================
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d

# Configure apt retries
# CORRECTED: Combined into a single, valid RUN command
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

WORKDIR /app

# Install runtime deps (including curl for healthcheck)
# FIXED: Prefixed cache mount IDs
RUN --mount=type=cache,id=doc-query-api-apt-cache-runtime,target=/var/cache/apt \
    --mount=type=cache,id=doc-query-api-apt-lib-runtime,target=/var/lib/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure user and paths
RUN useradd --create-home --shell /bin/bash appuser

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/opt/poetry" \
    VENV_PATH="/opt/pysetup/.venv" \
    PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Copy application code (as appuser)
COPY --chown=appuser:appuser app ./app

# Create directories and make entrypoint executable
# Note: The original Dockerfile references a docker-entrypoint.sh which was not provided.
# This command assumes that file exists in your app directory.
RUN mkdir -p /app/logs /app/cache && \
    chmod +x /app/docker-entrypoint.sh && \
    chown -R appuser:appuser /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--reload"]