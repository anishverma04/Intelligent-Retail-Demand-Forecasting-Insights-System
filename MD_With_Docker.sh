# Multi-stage Docker build for ML model deployment

# ==================== Stage 1: Base Image ====================
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==================== Stage 2: Dependencies ====================
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== Stage 3: Application ====================
FROM dependencies as application

# Copy application code
COPY . .

# Copy pre-trained models (in production, mount as volume or download from S3)
COPY models/ /app/models/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "deployment_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
