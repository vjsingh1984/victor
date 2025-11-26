# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Victor - Universal AI Coding Assistant
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.12-slim as builder

LABEL maintainer="Vijaykumar Singh <singhvjd@gmail.com>"
LABEL description="Universal terminal-based AI coding assistant"
LABEL version="0.1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt pyproject.toml README.md ./
COPY victor ./victor

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 victor && \
    mkdir -p /home/victor/.victor && \
    chown -R victor:victor /home/victor

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/victor /usr/local/bin/victor
COPY --from=builder /usr/local/bin/vic /usr/local/bin/vic
COPY --from=builder /app /app

# Copy examples and demos
COPY examples ./examples
COPY docs ./docs

# Copy configuration templates
COPY docker/config/profiles.yaml.template /home/victor/.victor/profiles.yaml.template

# Set ownership
RUN chown -R victor:victor /app /home/victor

# Switch to non-root user
USER victor

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VICTOR_HOME=/home/victor/.victor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD victor --version || exit 1

# Default command
CMD ["bash"]
