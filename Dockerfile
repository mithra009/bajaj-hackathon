# =================
#  Builder Stage
# =================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =================
#  Runtime Stage
# =================
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-create-home -r -g appuser appuser

# Create directories and set base permissions
# The appuser will own the /app directory
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables for the runtime
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code as the non-root user
COPY --chown=appuser:appuser . .

# Ensure the logs directory is writable by the appuser
# 775 permissions (rwxrwxr-x) are secure and effective
RUN chmod -R 775 /app/logs

# Switch to the non-root user for security
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Health check to ensure the API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application in production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]