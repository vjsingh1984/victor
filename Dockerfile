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
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir lancedb

# Pre-download embedding model for air-gapped deployment
# This downloads all-MiniLM-L12-v2 (120MB) during build time
# Model will be cached in Docker image at ~/.cache/huggingface/
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    print('📦 Pre-downloading embedding model: all-MiniLM-L12-v2'); \
    model = SentenceTransformer('all-MiniLM-L12-v2'); \
    print('✅ Embedding model cached in Docker image'); \
    print(f'📊 Model dimension: {model.get_sentence_embedding_dimension()}'); \
    import os; print(f'📂 Cache location: {os.path.expanduser(\"~/.cache\")}')"

# Pre-compute tool embeddings cache for faster startup
# This creates the pickle cache during build so it's ready immediately
RUN mkdir -p /root/.victor/embeddings && \
    python3 -c "import asyncio; \
    from pathlib import Path; \
    from victor.config.settings import Settings; \
    from victor.tools.base import ToolRegistry; \
    from victor.tools.semantic_selector import SemanticToolSelector; \
    from victor.tools.filesystem import read_file, write_file, list_directory; \
    from victor.tools.bash import execute_bash; \
    from victor.tools.file_editor_tool import edit_files; \
    async def init(): \
        print('🧠 Pre-computing tool embeddings cache...'); \
        settings = Settings(); \
        selector = SemanticToolSelector( \
            embedding_model=settings.embedding_model, \
            embedding_provider=settings.embedding_provider, \
            cache_embeddings=True \
        ); \
        tools = ToolRegistry(); \
        tools.register(read_file); \
        tools.register(write_file); \
        tools.register(list_directory); \
        tools.register(execute_bash); \
        tools.register(edit_files); \
        await selector.initialize_tool_embeddings(tools); \
        cache_file = Path.home() / '.victor' / 'embeddings' / f'tool_embeddings_{settings.embedding_model}.pkl'; \
        print(f'✅ Tool embeddings cached: {cache_file}'); \
        print(f'📊 Cache size: {cache_file.stat().st_size / 1024:.2f} KB'); \
    asyncio.run(init())" || echo "⚠️  Tool embedding cache will be created at runtime"

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

# Copy pre-downloaded embedding model cache (air-gapped capability)
# This makes the Docker image 100% offline-capable
COPY --from=builder /root/.cache /tmp/.cache

# Copy pre-computed tool embeddings cache
COPY --from=builder /root/.victor /tmp/.victor

# Copy examples and demos
COPY examples ./examples
COPY docs ./docs

# Copy docker scripts and config
COPY docker ./docker

# Create directories and copy caches to victor's home
RUN mkdir -p /home/victor/.cache /home/victor/.victor/embeddings && \
    cp -r /tmp/.cache/* /home/victor/.cache/ && \
    cp -r /tmp/.victor/* /home/victor/.victor/ 2>/dev/null || true && \
    rm -rf /tmp/.cache /tmp/.victor

# Copy default profiles for Docker deployment
RUN cp /app/docker/profiles.yaml /home/victor/.victor/profiles.yaml

# Set ownership
RUN chown -R victor:victor /app /home/victor

# Switch to non-root user
USER victor

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VICTOR_HOME=/home/victor/.victor
ENV HF_HOME=/home/victor/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/victor/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/home/victor/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD victor --version || exit 1

# Default command
CMD ["bash"]
