# syntax = docker/dockerfile:1

# FIXED: Add a build argument for the cache key.
# The build system can override this value.
ARG CACHE_KEY=doc-query-api-cache

# =================
#  Builder Stage
# =================
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d AS builder

# Configure apt retries
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

WORKDIR /app

# Install system dependencies with cache mounts
# FIXED: Use the CACHE_KEY build argument as a prefix for the mount ID
RUN --mount=type=cache,id=${CACHE_KEY}-apt,target=/var/cache/apt \
    --mount=type=cache,id=${CACHE_KEY}-apt-lib,target=/var/lib/apt \
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
# FIXED: Use the CACHE_KEY build argument as a prefix for the mount ID
RUN --mount=type=cache,id=${CACHE_KEY}-pip,target=/root/.cache/pip \
    pip install --user --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# =================
#  Runtime Stage
# =================
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d

# Configure apt retries
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

WORKDIR /app

# Install runtime deps (including curl for healthcheck)
# FIXED: Use the CACHE_KEY build argument as a prefix for the mount ID
RUN --mount=type=cache,id=${CACHE_KEY}-apt-runtime,target=/var/cache/apt \
    --mount=type=cache,id=${CACHE_KEY}-apt-lib-runtime,target=/var/lib/apt \
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
    PIP_DEFAULT_TIMEOUT=100

# Copy application code (as appuser)
COPY --chown=appuser:appuser app ./app

# Create directories and make entrypoint executable
RUN mkdir -p /app/logs /app/cache && \
    # The original Dockerfile references docker-entrypoint.sh which was not provided.
    # If this file doesn't exist, this line will fail.
    if [ -f /app/docker-entrypoint.sh ]; then chmod +x /app/docker-entrypoint.sh; fi && \
    chown -R appuser:appuser /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--reload"]