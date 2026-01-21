# Multi-stage production Dockerfile for Victor AI
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=0.5.1
ARG VCS_REF

# Set labels
LABEL maintainer="Victor AI Team <team@victor.ai>" \
      org.opencontainers.image.title="Victor AI" \
      org.opencontainers.image.description="Open-source AI coding assistant" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/victor-ai/victor"

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DOCKERIZE_VERSION=v0.6.1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY pyproject.toml README.md ./

# Install core dependencies first for better caching
RUN pip install --no-deps -e . && \
    pip install "poetry-core>=1.0.0"

# Install dependencies
RUN pip install -e ".[api]" --no-cache-dir && \
    pip install aiohttp asyncio uvicorn fastapi && \
    pip install -e .

# Copy application code
COPY victor/ /app/victor/
COPY victor/config/ /app/victor/config/

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Set runtime arguments
ARG BUILD_DATE
ARG VERSION=0.5.1
ARG VCS_REF

# Set labels
LABEL maintainer="Victor AI Team <team@victor.ai>" \
      org.opencontainers.image.title="Victor AI - Runtime" \
      org.opencontainers.image.description="Open-source AI coding assistant" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    VICTOR_PROFILE=production \
    VICTOR_LOG_LEVEL=INFO \
    PATH="/opt/venv/bin:$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/bash victor

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/.cache /app/data /app/logs /tmp && \
    chown -R victor:victor /app

# Create non-root user
USER victor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["uvicorn", "victor.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 3: Slim (minimal image for production)
FROM runtime as slim

# Remove unnecessary files to reduce image size
RUN rm -rf /opt/venv/lib/python3.11/site-packages/*.dist-info \
    && find /opt/venv/lib/python3.11/site-packages -type d -name __pycache__ -exec rm -rf {} + \
    && find /opt/venv/lib/python3.11/site-packages -type f -name "*.pyc" -delete \
    && find /opt/venv/lib/python3.11/site-packages -type f -name "*.pyo" -delete

# Stage 4: Development (with dev tools)
FROM runtime as development

# Install development tools
USER root
RUN pip install pytest pytest-cov pytest-asyncio black ruff mypy ipython && \
    apt-get update && apt-get install -y --no-install-recommends vim && \
    rm -rf /var/lib/apt/lists/*
USER victor

# Override command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]
