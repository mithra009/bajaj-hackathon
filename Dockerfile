# syntax = docker/dockerfile:1
# =================
#  Builder Stage
# =================
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d AS builder

# Configure apt retries
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

WORKDIR /app

# Install system dependencies.
# REMOVED: --mount=type=cache flags. Railway will handle caching automatically.
RUN apt-get update && \
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

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Python env
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install Python dependencies.
# REMOVED: --mount=type=cache flag. Railway will handle caching automatically.
COPY requirements.txt .
RUN pip install --user --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# =================
#  Runtime Stage
# =================
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d

WORKDIR /app

# Install runtime dependencies.
# REMOVED: --mount=type=cache flags. Railway will handle caching automatically.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONHASHSEED=random

# Copy application code (as appuser)
COPY --chown=appuser:appuser . .

# FIXED: Create all necessary directories including /app/data for the key index file
RUN mkdir -p /app/logs /app/cache /app/data && \
    chown -R appuser:appuser /app

EXPOSE 8000

# The healthcheck is now handled by Railway's deployment settings, but this is good practice
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root
USER appuser

# Use the command from your main.py as the entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]