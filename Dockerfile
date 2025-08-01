# =================
#  Builder Stage
# =================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =================
#  Runtime Stage
# =================
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/app && \
    touch /app/logs/app.log /app/logs/query_logs.json && \
    chown -R appuser:appuser /app && \
    chmod -R 777 /app/logs && \
    chmod 666 /app/logs/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser
WORKDIR /app

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application with auto-reload in development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--reload"]
