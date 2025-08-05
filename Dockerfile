# syntax = docker/dockerfile:1

# Accept a build argument for the cache key. A default is provided for local builds.
ARG CACHE_KEY=local-build-cache

# =================
#  Builder Stage
# =================
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d AS builder

# Configure apt retries
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

WORKDIR /app

# Use the CACHE_KEY argument to prefix the mount ID
RUN --mount=type=cache,id=${CACHE_KEY}-apt,target=/var/cache/apt \
    --mount=type=cache,id=${CACHE_KEY}-apt-lib,target=/var/lib/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates libjpeg62-turbo zlib1g libfreetype6 lcms2 libopenjp2-7 libtiff6 g++ \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && mkdir -p /app/logs /app/data

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
# Use the CACHE_KEY argument to prefix the mount ID
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

# Use the CACHE_KEY argument to prefix the mount ID
RUN --mount=type=cache,id=${CACHE_KEY}-apt-runtime,target=/var/cache/apt \
    --mount=type=cache,id=${CACHE_KEY}-apt-lib-runtime,target=/var/lib/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
RUN useradd --create-home --shell /bin/bash appuser
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --chown=appuser:appuser app ./app
RUN mkdir -p /app/logs /app/cache && chown -R appuser:appuser /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--reload"]