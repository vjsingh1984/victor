# Victor Docker Deployment Guide

Complete guide for deploying Victor with air-gapped semantic tool selection in Docker.

> Canonical Docker docs now live in `docker/README.md` (guided) and `docker/QUICKREF.md` (commands-only). This file remains as a deeper reference.

## Quick Start (5 Minutes)

```bash
# One-command setup
./docker-quickstart.sh

# Or manual steps:
docker-compose build victor
docker-compose --profile demo up -d ollama
docker-compose exec ollama ollama pull qwen3-coder:30b
docker-compose run --rm victor victor main "Write hello world"
```

## Features

### ✅ Air-Gapped Capabilities
- **Embedding Model**: all-MiniLM-L12-v2 (120MB) pre-downloaded in Docker image
- **Tool Embeddings**: Pre-computed during build (31 tools, ~50KB cache)
- **Offline Operation**: 100% functional without internet after initial setup
- **No External APIs**: All inference happens locally via Ollama

### ✅ Pre-Configured Models
- **Default**: qwen3-coder:30b (18 GB) - Best code quality
- **Fast**: qwen2.5-coder:7b (4.7 GB) - Resource-constrained
- **General**: llama3.1:8b (4.9 GB) - Non-coding tasks

### ✅ Semantic Tool Selection
- **Selection Method**: Cosine similarity on sentence embeddings
- **Threshold**: 0.15 (configurable)
- **Top-K**: 5 tools per query
- **Cache**: Persistent across container restarts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │              │      │              │                     │
│  │   Victor     │─────▶│   Ollama     │                     │
│  │ (Main App)   │      │  (LLM Server)│                     │
│  │              │      │              │                     │
│  └──────────────┘      └──────────────┘                     │
│         │                                                    │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────┐                   │
│  │  Pre-Bundled Components              │                   │
│  ├──────────────────────────────────────┤                   │
│  │ • Embedding Model (all-MiniLM-L12-v2)│                   │
│  │ • Tool Embeddings Cache (31 tools)   │                   │
│  │ • Default Profiles (qwen3-coder:30b) │                   │
│  │ • Fallback JSON Parser               │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Docker Images

### Victor Image
**Size**: ~1.5 GB (compressed)

**Contents**:
- Python 3.12-slim base
- Victor application + dependencies
- all-MiniLM-L12-v2 embedding model (120MB)
- Pre-computed tool embeddings cache
- Example scripts and documentation

**Build Time**: 3-5 minutes

### Ollama Image
**Size**: Variable (depends on models pulled)

**Models**:
- qwen3-coder:30b: 18 GB
- qwen2.5-coder:7b: 4.7 GB
- llama3.1:8b: 4.9 GB

## Configuration

### Profiles (`docker/profiles.yaml`)

```yaml
profiles:
  default:
    provider: ollama
    model: qwen3-coder:30b
    temperature: 0.2
    max_tokens: 8192

  fast:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.3
    max_tokens: 4096

  general:
    provider: ollama
    model: llama3.1:8b
    temperature: 0.7
    max_tokens: 4096
```

### Environment Variables

```bash
# Ollama connection
OLLAMA_HOST=http://ollama:11434

# API Keys (optional, for cloud providers)
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
XAI_API_KEY=...

# Victor configuration
VICTOR_HOME=/home/victor/.victor
HF_HOME=/home/victor/.cache/huggingface
```

## Usage

### Interactive Mode

```bash
# Start interactive shell
docker-compose run --rm victor

# Inside container:
victor                                    # Interactive REPL
victor "Write a function to sort arrays"  # One-shot
victor --profile fast "Quick task"        # Use fast profile
victor profiles                           # List all profiles
```

### One-Shot Commands

```bash
# Simple function
docker-compose run --rm victor \
  victor main "Write a function to calculate factorial"

# Multiple functions
docker-compose run --rm victor \
  victor main "Write a calculator with add, subtract, multiply, divide"

# With specific profile
docker-compose run --rm victor \
  victor --profile fast main "Write hello world"

# Non-streaming mode
docker-compose run --rm victor \
  victor main "Write prime checker" --no-stream
```

### Demo Script

```bash
# Run comprehensive demo
docker-compose run --rm victor bash /app/docker/demo-semantic-tools.sh

# Expected output:
# ✓ Demonstrates semantic tool selection
# ✓ Creates 4 example files
# ✓ Shows tool execution with different prompts
# ✓ Displays similarity scores and selected tools
```

## Volumes

### Persistent Storage

```yaml
volumes:
  # Ollama models (persistent)
  ollama_data:
    name: victor_ollama_data
    # Location: Models pulled via ollama

  # Victor configuration (persistent)
  victor_home:
    name: victor_home
    # Location: ~/.victor/profiles.yaml, embeddings cache

  # Demo output (temporary)
  demo_output:
    name: victor_demo_output
    # Location: Generated files from demos
```

### Volume Management

```bash
# List volumes
docker volume ls | grep victor

# Inspect volume
docker volume inspect victor_ollama_data

# Backup Ollama models
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama_backup.tar.gz -C /data .

# Restore Ollama models
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/ollama_backup.tar.gz -C /data

# Remove all Victor volumes (caution!)
docker volume rm victor_ollama_data victor_home victor_demo_output
```

## Performance

### Resource Requirements

**Minimum (fast profile)**:
- RAM: 8 GB
- CPU: 4 cores
- Disk: 20 GB
- Model: qwen2.5-coder:7b (4.7 GB)

**Recommended (default profile)**:
- RAM: 32 GB
- CPU: 8 cores
- Disk: 50 GB
- Model: qwen3-coder:30b (18 GB)

**High Performance**:
- RAM: 64 GB
- CPU: 16 cores
- Disk: 100 GB
- GPU: Optional (NVIDIA/AMD)
- Model: llama3.3:70b (40 GB)

### Benchmarks

| Model | RAM Usage | Inference Speed | Quality |
|-------|-----------|-----------------|---------|
| qwen2.5-coder:7b | ~8 GB | ~15 tok/s | Good |
| qwen3-coder:30b | ~20 GB | ~5 tok/s | Excellent |
| llama3.1:8b | ~10 GB | ~12 tok/s | Good |
| llama3.3:70b | ~45 GB | ~2 tok/s | Exceptional |

### GPU Support

**NVIDIA (Linux only)**:
```yaml
# In docker-compose.yml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**Apple Silicon (M1/M2/M3)**:
- GPU acceleration automatic via Metal
- Docker Desktop required
- Set resource limits in Docker Desktop preferences

## Troubleshooting

### Build Issues

**Problem**: Embedding model download fails
```bash
# Solution: Manual download
docker-compose build victor --no-cache --build-arg HTTP_PROXY=http://proxy:port
```

**Problem**: Tool embeddings cache not created
```bash
# Solution: Will be created at runtime automatically
# Or manually initialize:
docker-compose run --rm victor bash /app/docker/init-victor.sh
```

### Runtime Issues

**Problem**: Ollama not responding
```bash
# Check Ollama status
docker-compose ps ollama

# View Ollama logs
docker-compose logs ollama

# Restart Ollama
docker-compose restart ollama
```

**Problem**: Model not found
```bash
# List available models
docker-compose exec ollama ollama list

# Pull missing model
docker-compose exec ollama ollama pull qwen3-coder:30b
```

**Problem**: Semantic selection returns 0 tools
```bash
# Check tool embeddings cache
docker-compose run --rm victor ls -la /home/victor/.victor/embeddings/

# Reinitialize cache
docker-compose run --rm victor bash /app/docker/init-victor.sh
```

**Problem**: Tool calling not working
```bash
# Verify fallback parser is included
docker-compose run --rm victor python3 -c "from victor.providers.ollama import OllamaProvider; print(hasattr(OllamaProvider, '_parse_json_tool_call_from_content'))"

# Test with specific model
docker-compose run --rm victor victor --profile general main "Write hello world"
```

### Performance Issues

**Problem**: Out of memory
```bash
# Solution: Use smaller model
docker-compose run --rm victor victor --profile fast "Your prompt"

# Or increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 32 GB
```

**Problem**: Slow inference
```bash
# Solution 1: Use faster model
docker-compose run --rm victor victor --profile fast "Your prompt"

# Solution 2: Enable GPU (if available)
# See GPU Support section above

# Solution 3: Reduce max_tokens
# Edit docker/profiles.yaml: max_tokens: 4096 → 2048
```

## Air-Gapped Deployment

### Offline Setup Process

1. **Online Machine** (with internet):
```bash
# Build image
./docker-quickstart.sh

# Save image
docker save -o victor-image.tar $(docker images -q codingagent-victor)
docker save -o ollama-image.tar ollama/ollama:latest

# Export volumes
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama-models.tar.gz -C /data .
```

2. **Transfer to Air-Gapped Machine**:
```bash
# Copy files to USB/network share:
# - victor-image.tar
# - ollama-image.tar
# - ollama-models.tar.gz
# - docker-compose.yml
```

3. **Offline Machine** (air-gapped):
```bash
# Load images
docker load -i victor-image.tar
docker load -i ollama-image.tar

# Create volume and restore models
docker volume create victor_ollama_data
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/ollama-models.tar.gz -C /data

# Start services
docker-compose --profile demo up -d

# Verify
docker-compose run --rm victor victor --version
docker-compose run --rm victor victor main "Write hello world"
```

### Verification

```bash
# Confirm air-gapped operation
docker-compose run --rm victor python3 -c "
from victor.config.settings import Settings
s = Settings()
print(f'Airgapped: {s.airgapped_mode}')
print(f'Embedding model: {s.embedding_model}')
print(f'Embedding provider: {s.embedding_provider}')
"

# Expected output:
# Airgapped: True
# Embedding model: all-MiniLM-L12-v2
# Embedding provider: sentence-transformers
```

## Security

### Best Practices

1. **Non-Root User**: Victor runs as user `victor` (UID 1000)
2. **Read-Only Volumes**: Examples and docs mounted as read-only
3. **Network Isolation**: Uses dedicated bridge network
4. **No Exposed Ports**: Victor container has no exposed ports
5. **Environment Variables**: API keys via environment, not in image

### Scanning

```bash
# Scan image for vulnerabilities
docker scan codingagent-victor

# Run as read-only (except volumes)
docker-compose run --rm --read-only victor victor main "Test"
```

## Maintenance

### Updates

```bash
# Update Victor
git pull
docker-compose build victor --no-cache

# Update Ollama
docker-compose pull ollama

# Update models
docker-compose exec ollama ollama pull qwen3-coder:30b
```

### Cleanup

```bash
# Stop all services
docker-compose down

# Remove containers
docker-compose down -v

# Remove images
docker rmi $(docker images -q codingagent-victor)

# Clean build cache
docker builder prune

# Full cleanup (caution: removes volumes!)
docker-compose down -v
docker volume rm victor_ollama_data victor_home victor_demo_output
```

## Integration

### CI/CD Pipelines

```yaml
# .github/workflows/docker-test.yml
name: Docker Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Victor
        run: docker-compose build victor

      - name: Start Ollama
        run: docker-compose up -d ollama

      - name: Pull test model
        run: docker-compose exec ollama ollama pull qwen2.5-coder:1.5b

      - name: Run tests
        run: docker-compose run --rm victor pytest tests/

      - name: Run demo
        run: docker-compose run --rm victor bash /app/docker/demo-semantic-tools.sh
```

### Docker Hub

```bash
# Tag and push
docker tag codingagent-victor username/victor:latest
docker tag codingagent-victor username/victor:v1.0

docker push username/victor:latest
docker push username/victor:v1.0
```

## License

Victor is licensed under the Apache License 2.0. See LICENSE file for details.

## Support

- **Documentation**: See AIR_GAPPED_TOOL_CALLING_SOLUTION.md
- **Issues**: https://github.com/yourusername/victor/issues
- **Discussions**: https://github.com/yourusername/victor/discussions
