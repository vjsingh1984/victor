# Embedding Setup Guide: Qwen3-Embedding:8b

Complete guide to setting up Victor with the most accurate open-source embedding model for production-grade semantic code search.

## Overview

Victor now uses **Qwen3-Embedding:8b** as the default embedding model for semantic code search:

- **MTEB Score**: 70.58 (#1 on multilingual leaderboard, January 2025)
- **Context Window**: 40,000 tokens (can embed entire large files)
- **Embedding Dimension**: 4096 (high-quality representations)
- **Languages Supported**: 100+ (all programming languages)
- **License**: Apache 2.0 (production-ready)
- **Cost**: Free (runs locally)

## Quick Start (5 Minutes)

```bash
# 1. Install Ollama
curl https://ollama.ai/install.sh | sh

# 2. Pull Qwen3-Embedding model (4.7GB download)
ollama pull qwen3-embedding:8b

# 3. Start Ollama server (in background)
ollama serve &

# 4. Verify installation
ollama list | grep qwen3-embedding

# 5. Test embedding generation
curl http://localhost:11434/api/embeddings -d '{
  "model": "qwen3-embedding:8b",
  "prompt": "def hello_world(): print(\"Hello!\")"
}'

# 6. Run Victor demo
python examples/qwen3_embedding_demo.py
```

## Detailed Setup

### Step 1: Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

**Verify Installation:**
```bash
ollama --version
# Should output: ollama version 0.x.x
```

### Step 2: Pull Embedding Model

```bash
# Production (maximum accuracy)
ollama pull qwen3-embedding:8b  # 4.7GB

# Alternative options:
# ollama pull qwen3-embedding:4b   # 2.5GB (still very good)
# ollama pull snowflake-arctic-embed2  # 568MB (fast + accurate)
# ollama pull bge-m3  # 1.2GB (great for RAG)
```

**Monitor Download Progress:**
```bash
ollama list
# Shows: qwen3-embedding:8b  4.7 GB  X minutes ago
```

### Step 3: Start Ollama Server

**Option A: Foreground (for testing)**
```bash
ollama serve
# Listens on http://localhost:11434
```

**Option B: Background (for development)**
```bash
# macOS/Linux
ollama serve > /dev/null 2>&1 &

# Or use systemd (Linux)
systemctl start ollama
```

**Verify Server is Running:**
```bash
curl http://localhost:11434/api/tags
# Should return JSON list of models
```

### Step 4: Configure Victor

**Create Configuration File:**

`~/.victor/config.yaml`:
```yaml
codebase:
  # Vector Store
  vector_store: chromadb
  persist_directory: ~/.victor/embeddings/production
  distance_metric: cosine

  # Embedding Model: Qwen3-Embedding:8b
  embedding_model_type: ollama
  embedding_model_name: qwen3-embedding:8b
  embedding_api_key: http://localhost:11434

  # Configuration
  extra_config:
    collection_name: victor_codebase
    dimension: 4096
    batch_size: 8  # Adjust based on RAM (4-16 recommended)
```

**Or Use Python API:**
```python
from victor.codebase.embeddings.base import EmbeddingConfig

config = EmbeddingConfig(
    vector_store="chromadb",
    persist_directory="~/.victor/embeddings/my_project",
    embedding_model_type="ollama",
    embedding_model_name="qwen3-embedding:8b",
    embedding_api_key="http://localhost:11434",
    extra_config={
        "collection_name": "my_project",
        "dimension": 4096,
        "batch_size": 8
    }
)
```

### Step 5: Test Integration

**Run Demo Script:**
```bash
cd /path/to/victor
python examples/qwen3_embedding_demo.py
```

**Expected Output:**
```
================================================================================
🚀 Qwen3-Embedding:8b Demo - Production-Grade Code Embeddings
================================================================================

📋 Configuration:
   Vector Store: chromadb
   Embedding Model: qwen3-embedding:8b (ollama)
   Dimension: 4096
   Persist Directory: ~/.victor/embeddings/qwen3_demo

🔧 Initializing ChromaDB provider
📦 Vector Store: ChromaDB
🤖 Embedding Model: qwen3-embedding:8b (ollama)
🤖 Initializing Ollama embedding model: qwen3-embedding:8b
🔗 Ollama server: http://localhost:11434
✅ Ollama embedding model ready: qwen3-embedding:8b
📊 Embedding dimension: 4096
📁 Using persistent storage: ~/.victor/embeddings/qwen3_demo
📚 Collection: qwen3_demo
✅ ChromaDB provider initialized!

📝 Indexing documents...
✅ Indexed 5 documents

📊 Index Statistics:
   Total documents: 5
   Embedding model: qwen3-embedding:8b
   Model type: ollama
   Dimension: 4096
   Distance metric: cosine

🔍 Semantic Search Results:
================================================================================

1. Query: "How to authenticate a user with username and password?"
--------------------------------------------------------------------------------
   Rank #1 (Score: 0.8542)
   File: app/auth.py:authenticate_user
   Type: function
   Line: 15
   Code: def authenticate_user(username: str, password: str) -> Optional[User]:...
```

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (model loads ~5GB)
- **Disk**: 6GB free space
- **CPU**: Any modern CPU (2+ cores)
- **OS**: macOS, Linux, Windows

### Recommended Requirements
- **RAM**: 16GB (for smooth operation with large codebases)
- **Disk**: 10GB+ free space
- **GPU**: Optional but recommended (CUDA/Metal/ROCm)
- **CPU**: 4+ cores for parallel embedding generation

### GPU Acceleration (Optional)

**NVIDIA GPU (CUDA):**
```bash
# Ollama automatically uses CUDA if available
# Verify: ollama run qwen3-embedding:8b
# Should show: Using GPU: NVIDIA...
```

**Apple Silicon (Metal):**
```bash
# Metal is automatically used on M1/M2/M3 Macs
# Provides 2-3x faster embedding generation
```

**AMD GPU (ROCm):**
```bash
# Set environment variable
export HSA_OVERRIDE_GFX_VERSION=10.3.0
ollama serve
```

## Performance Tuning

### Batch Size Optimization

```yaml
extra_config:
  # Low RAM (8GB)
  batch_size: 4

  # Medium RAM (16GB) - RECOMMENDED
  batch_size: 8

  # High RAM (32GB+)
  batch_size: 16
```

**Rule of Thumb**: Each document in batch uses ~50MB RAM for Qwen3:8b

### Context Window Usage

Qwen3-Embedding:8b has a 40K token context window:
- Most code files: < 5K tokens (fully embedded)
- Large files: < 40K tokens (fully embedded)
- Massive files: Automatically chunked

**Comparison:**
| Model | Context | Large File Handling |
|-------|---------|---------------------|
| Qwen3:8b | 40K | ✅ Excellent |
| Arctic-Embed2 | 8K | ✅ Good |
| BGE-M3 | 8K | ✅ Good |
| mxbai-embed-large | 512 | ❌ Poor (truncates) |

### Caching Strategy

**Enable Response Caching:**
```yaml
extra_config:
  # Cache embeddings to disk (speeds up re-indexing)
  cache_embeddings: true
  cache_directory: ~/.victor/cache/embeddings
```

## Alternative Models

### When to Use Alternatives

**Choose Snowflake Arctic-Embed 2.0 if:**
- Need faster performance (100+ docs/sec)
- RAM constrained (< 8GB)
- Context < 8K tokens is sufficient

```yaml
embedding_model_name: snowflake-arctic-embed2
extra_config:
  dimension: 1024
  batch_size: 32
```

**Choose BGE-M3 if:**
- Need multi-functional retrieval (dense + sparse + multi-vector)
- Working on RAG applications
- Want excellent balance of speed/accuracy

```yaml
embedding_model_name: bge-m3
extra_config:
  dimension: 1024
  batch_size: 32
```

**Choose Qwen3-Embedding:4b if:**
- Want excellent accuracy with less RAM
- Need faster inference
- 4B params still beats most models

```yaml
embedding_model_name: qwen3-embedding:4b
extra_config:
  dimension: 4096
  batch_size: 16
```

### Model Comparison Table

| Model | Size | MTEB | Dim | Context | Speed | RAM |
|-------|------|------|-----|---------|-------|-----|
| **qwen3:8b** (default) | 4.7GB | 70.58 | 4096 | 40K | Medium | 8GB+ |
| qwen3:4b | 2.5GB | ~68* | 4096 | 40K | Fast | 6GB+ |
| snowflake-arctic-embed2 | 568MB | ~58-60 | 1024 | 8K | Very Fast | 4GB+ |
| bge-m3 | 1.2GB | 59.56 | 1024 | 8K | Fast | 4GB+ |
| mxbai-embed-large | 670MB | SOTA** | 1024 | 512 | Fast | 3GB+ |

*Estimated based on model size scaling
**State-of-the-art for Bert-large size class

## Troubleshooting

### Model Not Found
```bash
# Error: model 'qwen3-embedding:8b' not found
# Solution:
ollama pull qwen3-embedding:8b
ollama list  # Verify it appears
```

### Connection Refused
```bash
# Error: Connection refused to localhost:11434
# Solution:
ollama serve  # Start server in separate terminal

# Or check if port is in use:
lsof -i :11434
```

### Out of Memory
```python
# Error: RuntimeError: CUDA out of memory
# Solution 1: Reduce batch size
extra_config:
  batch_size: 4  # Lower from 8

# Solution 2: Use smaller model
embedding_model_name: qwen3-embedding:4b

# Solution 3: Use CPU only
CUDA_VISIBLE_DEVICES="" ollama serve
```

### Slow Performance
```bash
# Issue: Embeddings taking too long
# Solutions:

# 1. Enable GPU acceleration
# Verify GPU is being used:
ollama run qwen3-embedding:8b

# 2. Reduce batch size (paradoxically can help on CPU)
extra_config:
  batch_size: 4

# 3. Use faster model:
embedding_model_name: snowflake-arctic-embed2
```

### Import Errors
```python
# Error: ModuleNotFoundError: No module named 'httpx'
# Solution:
pip install httpx

# Error: ModuleNotFoundError: No module named 'chromadb'
# Solution:
pip install chromadb
```

## Production Deployment

### Docker Setup

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install Ollama
RUN curl https://ollama.ai/install.sh | sh

# Install Victor
COPY . /app
WORKDIR /app
RUN pip install -e ".[dev]"

# Pull embedding model
RUN ollama pull qwen3-embedding:8b

# Start services
CMD ollama serve & python your_app.py
```

### Kubernetes Deployment

**ollama-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-embedding
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        resources:
          limits:
            memory: "16Gi"
            nvidia.com/gpu: 1  # Optional GPU
        volumeMounts:
        - name: models
          mountPath: /root/.ollama
```

### Performance Monitoring

```python
import time
from victor.codebase.embeddings.chromadb_provider import ChromaDBProvider

# Benchmark embedding generation
start = time.time()
embeddings = await provider.embed_batch(texts)
duration = time.time() - start

print(f"Embedded {len(texts)} documents in {duration:.2f}s")
print(f"Throughput: {len(texts)/duration:.2f} docs/sec")
```

**Expected Throughput:**
- **CPU (8-core)**: 5-10 docs/sec
- **M1/M2 Mac (Metal)**: 15-25 docs/sec
- **NVIDIA A10 GPU**: 50-100 docs/sec

## Next Steps

1. **Index Your Codebase**: Use Victor's CodebaseIndex to index your full repository
2. **Integrate with AI Agent**: Connect semantic search to your coding assistant
3. **Optimize for Scale**: Profile performance and adjust batch sizes
4. **Monitor Quality**: Track search relevance and adjust models as needed

## Support

- **Issues**: https://github.com/vijaysingh/victor/issues
- **Discussions**: https://github.com/vijaysingh/victor/discussions
- **Documentation**: See CLAUDE.md and EMBEDDING_ARCHITECTURE.md

## References

- [Qwen3-Embedding Paper](https://arxiv.org/abs/2506.05176)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Ollama Documentation](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
