# =================
#  Builder Stage
# =================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build-time system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libjpeg62-turbo \
    zlib1g \
    libfreetype6 \
    liblcms2-2 \
    libopenjp2-7 \
    libtiff6 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache to reduce image size
    rm -rf /root/.cache/pip/*

# =================
#  Runtime Stage
# =================
FROM python:3.11-slim

WORKDIR /app



# Create a non-root user and group
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Create app and logs directories with correct permissions
RUN mkdir -p /app/app /app/logs && \
    chown -R appuser:appuser /app && \
    chmod 755 /app/logs

# Copy application code
COPY --chown=appuser:appuser app /app/app
COPY --chown=appuser:appuser requirements.txt /app/

# Switch to non-root user
USER appuser
WORKDIR /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]