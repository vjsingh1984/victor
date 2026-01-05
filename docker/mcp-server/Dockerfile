# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

# Victor MCP Server
# Exposes Victor's 55+ tools via Model Context Protocol
#
# Build:
#   docker build -t victor-mcp-server -f docker/mcp-server/Dockerfile .
#
# Run (stdio mode for Claude Desktop):
#   docker run -it --rm -v $(pwd):/workspace victor-mcp-server
#
# Run with specific tools:
#   docker run -it --rm -e VICTOR_MCP_TOOLS=read,write,ls victor-mcp-server
#
# Run with specific vertical:
#   docker run -it --rm -e VICTOR_MCP_VERTICALS=coding victor-mcp-server

FROM python:3.12-slim

LABEL maintainer="Vijaykumar Singh <singhvjd@gmail.com>"
LABEL description="Victor MCP Server - Exposes Victor's tools via Model Context Protocol"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/victor-ai/victor"
LABEL org.opencontainers.image.documentation="https://github.com/victor-ai/victor/blob/main/docs/guides/VICTOR_AS_MCP_SERVER.md"

# Install system dependencies
# - git: for git tools
# - curl: for HTTP tools
# - build-essential: for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash victor && \
    mkdir -p /home/victor/.victor && \
    mkdir -p /workspace && \
    chown -R victor:victor /home/victor /workspace

WORKDIR /app

# Copy Python package files first for better caching
COPY pyproject.toml setup.py ./
COPY victor/ ./victor/

# Install Victor with all dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Switch to non-root user
USER victor

# Set working directory for file operations
WORKDIR /workspace

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VICTOR_HOME=/home/victor/.victor

# MCP-specific environment variables
# Tools to expose (comma-separated, empty = all)
ENV VICTOR_MCP_TOOLS=""
# Verticals to expose (comma-separated: coding,devops,rag,dataanalysis,research)
ENV VICTOR_MCP_VERTICALS=""
# Log level for MCP server
ENV VICTOR_MCP_LOG_LEVEL=WARNING
# Enable sandbox mode for additional security
ENV VICTOR_SANDBOX_MODE=false
# Sandbox execution timeout (seconds)
ENV VICTOR_SANDBOX_TIMEOUT=30
# Sandbox memory limit
ENV VICTOR_SANDBOX_MEMORY=512M
# Air-gapped mode (disable network tools)
ENV VICTOR_AIRGAPPED_MODE=false

# Health check - verify Victor is properly installed
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from victor.integrations.mcp.server import MCPServer; print('OK')" || exit 1

# Default: run MCP server in stdio mode
# This is the format expected by Claude Desktop and other MCP clients
ENTRYPOINT ["victor", "mcp"]

# Default arguments (can be overridden)
CMD ["--log-level", "WARNING"]
