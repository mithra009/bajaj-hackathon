# syntax = docker/dockerfile:1
# =================
#  Builder Stage
# =================
# FIXED: Removed the specific @sha256 digest to use the flexible tag
FROM python:3.11-slim AS builder

# Configure apt retries
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

WORKDIR /app

# Install system dependencies.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates libjpeg62-turbo zlib1g libfreetype6 lcms2 libopenjp2-7 libtiff6 g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Python env
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --user --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# =================
#  Runtime Stage
# =================
# FIXED: Removed the specific @sha256 digest to use the flexible tag
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 curl \
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

# Create all necessary directories including /app/data for the key index file
RUN mkdir -p /app/logs /app/cache /app/data && \
    chown -R appuser:appuser /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]