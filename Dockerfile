# =================
#  Builder Stage
# =================
# Use a more reliable base image mirror
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d

# Configure apt to retry downloads
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Pipeline-Depth 0;' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::No-Cache true;' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::BrokenProxy true;' >> /etc/apt/apt.conf.d/80-retries

WORKDIR /app

# Install system dependencies with retries
RUN --mount=type=cache,id=cache-key-apt-cache,target=/var/cache/apt \
    --mount=type=cache,id=cache-key-apt-lib,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libjpeg62-turbo \
    zlib1g \
    libfreetype6 \
    lcms2 \
    libopenjp2-7 \
    libtiff6 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and required directories
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install Python dependencies with retries and cache
COPY requirements.txt .
RUN --mount=type=cache,id=cache-key-pip-cache,target=/root/.cache/pip \
    pip install --user --no-cache-dir \
    --retries 5 \
    --timeout 60 \
    --default-timeout 60 \
    -r requirements.txt

# Runtime stage
FROM python:3.11-slim@sha256:2c5f9c323c381d5439d80f8f7d8b5e0c0f1e1f3b1c1d1f0a1b3c1d1f0a1b3c1d

WORKDIR /app

# Configure apt to retry downloads in runtime stage
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

# Install runtime dependencies with retries
RUN --mount=type=cache,id=cache-key-apt-cache-runtime,target=/var/cache/apt \
    --mount=type=cache,id=cache-key-apt-lib-runtime,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser app ./app

# Create necessary directories
RUN mkdir -p /app/logs && \
    mkdir -p /app/cache && \
    chmod +x /app/docker-entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
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

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--reload"]