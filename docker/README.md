# Victor Docker Setup

Production-ready Docker environment for Victor - Universal AI Coding Assistant supporting Ollama, vLLM, and cloud providers.

## Overview

This Docker setup provides:
- **Ollama**: Local LLM server with GPU support
- **vLLM**: High-performance inference server with tool calling
- **Victor**: Main application with all 25+ enterprise tools
- **Jupyter**: Interactive notebooks for experimentation
- **Automated Demos**: Pre-built demonstrations of all features

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Network                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Ollama  │  │   vLLM   │  │  Victor  │  │ Jupyter │ │
│  │  :11434  │  │  :8000   │  │   App    │  │ :8888   │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│       │             │              │             │       │
│  ┌────▼─────────────▼──────────────▼─────────────▼────┐ │
│  │           Shared Volumes & Environment              │ │
│  │  - ollama_data  - vllm_cache  - demo_output         │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Required
- Docker 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose 2.0+ (included with Docker Desktop)
- 8GB RAM minimum, 16GB recommended
- 20GB disk space for models

### Optional
- NVIDIA GPU + nvidia-docker for GPU acceleration
- Apple Silicon (M1/M2/M3) with Metal support

## Quick Start

### 1. Initial Setup

```bash
# Clone repository
git clone https://github.com/yourusername/victor.git
cd victor

# Run setup script
bash docker/scripts/setup.sh

# Edit .env to add API keys (optional for cloud providers)
nano .env
```

### 2. Start Services

#### Option A: Ollama Only (Recommended for First Run)
```bash
# Start Ollama service
docker-compose --profile ollama up -d

# Pull a model
docker exec victor-ollama ollama pull qwen2.5-coder:7b

# Run Victor interactively
docker-compose run victor bash
```

#### Option B: Full Stack (Ollama + vLLM)
```bash
# Start all services
docker-compose --profile full up -d

# Check service health
docker-compose ps
```

#### Option C: Run Automated Demos
```bash
# Run all demonstrations
docker-compose --profile demo up

# View demo output
ls -lh demo_workspace/
```

## Service Profiles

Docker Compose uses profiles to selectively start services:

| Profile | Services | Use Case |
|---------|----------|----------|
| `ollama` | Ollama + Victor | Local development, quick testing |
| `vllm` | vLLM only | High-performance inference |
| `full` | Ollama + vLLM + Victor + Jupyter | Complete development environment |
| `demo` | Ollama + Demo runner | Automated demonstrations |
| `notebook` | Ollama + Jupyter | Interactive notebooks |

## Service Details

### Ollama (Port 11434)

Local LLM server supporting 100+ models.

**Start:**
```bash
docker-compose --profile ollama up -d
```

**Pull Models:**
```bash
# Recommended models
docker exec victor-ollama ollama pull qwen2.5-coder:7b      # Code generation (4.7GB)
docker exec victor-ollama ollama pull qwen2.5-coder:1.5b    # Fast demos (934MB)
docker exec victor-ollama ollama pull deepseek-coder-v2:16b # Advanced coding (9GB)

# List installed models
docker exec victor-ollama ollama list
```

**Configuration:**
- Base URL: `http://ollama:11434` (internal) or `http://localhost:11434` (external)
- GPU: Automatically uses NVIDIA/Metal if available
- Data: Persisted in `victor_ollama_data` volume

### vLLM (Port 8000)

High-performance inference server with tool calling support.

**Start:**
```bash
docker-compose --profile vllm up -d

# Wait for model download (first run takes 5-10 minutes)
docker-compose logs -f vllm
```

**Features:**
- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (auto-downloaded from HuggingFace)
- Tool calling: Enabled via `--enable-auto-tool-choice`
- Parser: Hermes format (`--tool-call-parser hermes`)
- Cache: HuggingFace models cached in `victor_vllm_cache` volume

**Health Check:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

**Configuration:**
- Base URL: `http://vllm:8000/v1` (internal) or `http://localhost:8000/v1` (external)
- GPU: Requires NVIDIA GPU with CUDA
- CPU Mode: Use `--enforce-eager` flag for Apple Silicon

### Victor Application

Main coding assistant with 25+ enterprise tools.

**Interactive Shell:**
```bash
docker-compose run victor bash

# Inside container:
victor --profile ollama
victor "Write a Python function to calculate Fibonacci numbers"
victor --help
```

**Available Tools:**
- Code Review, Security Scanner, Refactoring
- Testing, CI/CD, Documentation Generation
- Git, Docker, HTTP, Database Tools
- Batch Processing, Metrics, Caching

**Working Directory:**
- Host: `./demo_workspace`
- Container: `/workspace`
- All file operations persist to host

### Jupyter Notebooks (Port 8888)

Interactive Python environment with Victor integration.

**Start:**
```bash
docker-compose --profile notebook up -d

# Get access token
docker-compose logs jupyter | grep "http://127.0.0.1:8888/lab?token="
```

**Open:** [http://localhost:8888](http://localhost:8888)

**Available Notebooks:**
- `/home/jovyan/examples` - Read-only Victor examples
- `/home/jovyan/notebooks` - Your notebooks (persisted)

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Cloud Provider API Keys (Optional)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
XAI_API_KEY=...

# Local Services (Auto-configured)
OLLAMA_HOST=http://ollama:11434
VLLM_API_BASE=http://vllm:8000/v1

# Demo Configuration
DEMO_OUTPUT_DIR=/output
```

### Victor Profiles

Victor uses profiles defined in `docker/config/profiles.yaml.template`:

```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

  ollama-fast:
    provider: ollama
    model: qwen2.5-coder:1.5b
    temperature: 0.5
    max_tokens: 2048

  vllm:
    provider: openai
    model: Qwen/Qwen2.5-Coder-1.5B-Instruct
    temperature: 0.7
    max_tokens: 2048

  claude:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0
    max_tokens: 8192
```

**Usage:**
```bash
docker-compose run victor victor --profile ollama
docker-compose run victor victor --profile vllm
docker-compose run victor victor --profile claude  # Requires ANTHROPIC_API_KEY
```

## Usage Examples

### Example 1: Simple Code Generation

```bash
docker-compose --profile ollama up -d
docker exec victor-ollama ollama pull qwen2.5-coder:1.5b

docker-compose run victor victor --profile ollama-fast \
  "Write a Python function to calculate factorial with memoization"
```

### Example 2: Multi-Turn Conversation

```bash
docker-compose run victor bash

# Inside container:
victor --profile ollama
# Interactive REPL starts
> I'm building a REST API with FastAPI
> What's the best way to handle authentication?
> Show me an example with JWT tokens
```

### Example 3: Tool Calling Demo

```bash
docker-compose --profile demo up

# Runs all 5 demonstrations:
# 1. Simple chat completion
# 2. Code generation
# 3. Streaming responses
# 4. Multi-turn conversation
# 5. Tool calling

# View results
cat demo_workspace/demo_report_*.md
```

### Example 4: Code Review

```bash
docker-compose run victor bash

# Inside container:
cd /workspace
# Create a Python file
cat > example.py << 'EOF'
def calculate(x, y):
    return x / y
EOF

# Run code review
victor --profile ollama
> Review the file example.py and suggest improvements
```

### Example 5: Batch Processing

```bash
docker-compose run victor victor --profile ollama \
  "Search all Python files in /app/victor/tools for TODO comments and list them"
```

### Example 6: vLLM High-Performance Inference

```bash
# Start vLLM (first run downloads model)
docker-compose --profile vllm up -d

# Wait for ready
docker-compose logs -f vllm | grep "Uvicorn running"

# Use vLLM provider
docker-compose run victor victor --profile vllm \
  "Generate a Python class for a binary search tree with insert, search, and delete methods"
```

## Automated Demonstrations

The demo runner (`docker/demos/run_all_demos.py`) showcases Victor's capabilities:

### Running Demos

```bash
docker-compose --profile demo up
```

### Demo Contents

1. **Simple Chat Completion**
   - Basic question answering
   - Model: qwen2.5-coder:1.5b
   - Shows response quality

2. **Code Generation**
   - Fibonacci function with docstring
   - Temperature: 0.3 (deterministic)
   - Demonstrates code quality

3. **Streaming Responses**
   - Real-time token generation
   - Shows UI responsiveness
   - Temperature: 0.7

4. **Multi-Turn Conversation**
   - Context-aware responses
   - Framework recommendations
   - Demonstrates conversation memory

5. **Tool Calling**
   - Weather function calling
   - Shows tool integration
   - JSON schema validation

### Demo Output

Results saved to:
- Console: Rich formatted output with colors
- File: `demo_workspace/demo_report_YYYYMMDD_HHMMSS.md`

## Platform Support

### Apple Silicon (M1/M2/M3)

**Ollama:** Full support with Metal GPU acceleration
```bash
docker-compose --profile ollama up -d
# Automatically uses Metal
```

**vLLM:** CPU mode only (no Metal support yet)
```yaml
# In docker-compose.yml, vLLM command includes:
--enforce-eager  # Required for ARM CPU
```

### Linux with NVIDIA GPU

**Ollama & vLLM:** Full GPU acceleration
```bash
# Requires nvidia-docker installed
docker-compose --profile full up -d
# Automatically uses CUDA
```

**Check GPU Usage:**
```bash
docker exec victor-ollama nvidia-smi
docker exec victor-vllm nvidia-smi
```

### Linux CPU Only

**Ollama:** Works on CPU (slower)
```bash
# Remove GPU reservations from docker-compose.yml:
# Comment out the deploy.resources.reservations section
docker-compose --profile ollama up -d
```

**vLLM:** Add `--enforce-eager` flag
```yaml
# Already included in docker-compose.yml
```

## Troubleshooting

### Ollama Issues

**Problem:** Ollama not starting
```bash
# Check logs
docker-compose logs ollama

# Common fix: Reset data
docker-compose down
docker volume rm victor_ollama_data
docker-compose --profile ollama up -d
```

**Problem:** Model not found
```bash
# List models
docker exec victor-ollama ollama list

# Pull model
docker exec victor-ollama ollama pull qwen2.5-coder:7b
```

**Problem:** Out of memory
```bash
# Use smaller model
docker exec victor-ollama ollama pull qwen2.5-coder:1.5b

# Or set OLLAMA_NUM_PARALLEL=1 in .env
```

### vLLM Issues

**Problem:** vLLM crash on startup
```bash
# Check logs
docker-compose logs vllm

# Common fix for ARM: Verify --enforce-eager is set
# Common fix for NVIDIA: Verify nvidia-docker installed
```

**Problem:** Model download slow/failing
```bash
# Check HuggingFace connection
docker exec victor-vllm curl -I https://huggingface.co

# Set HF mirror (China):
# HF_ENDPOINT=https://hf-mirror.com in .env
```

**Problem:** Tool calling not working
```bash
# Verify flags are set:
docker-compose exec vllm ps aux | grep vllm
# Should see: --enable-auto-tool-choice --tool-call-parser hermes

# Restart with flags:
docker-compose --profile vllm down
docker-compose --profile vllm up -d
```

### Victor Application Issues

**Problem:** Provider not found
```bash
# Check .env file
cat .env

# Verify service URLs
docker-compose exec victor curl http://ollama:11434/api/tags
docker-compose exec victor curl http://vllm:8000/v1/models
```

**Problem:** Permission denied in workspace
```bash
# Fix ownership
sudo chown -R $USER:$USER demo_workspace/
```

**Problem:** Tool execution fails
```bash
# Check tool availability
docker-compose run victor victor
> /tools  # List available tools

# Check logs
docker-compose logs victor
```

### Network Issues

**Problem:** Services can't communicate
```bash
# Verify network
docker network inspect victor_victor-network

# Restart all services
docker-compose down
docker-compose --profile full up -d
```

**Problem:** Port already in use
```bash
# Find process using port
lsof -i :11434  # Ollama
lsof -i :8000   # vLLM
lsof -i :8888   # Jupyter

# Kill process or change port in docker-compose.yml
```

## Performance Tuning

### Ollama Optimization

```bash
# Set in .env:
OLLAMA_NUM_PARALLEL=2        # Parallel requests (default: 1)
OLLAMA_MAX_LOADED_MODELS=2   # Keep models in memory
OLLAMA_MODELS=/models        # Custom model directory

# Use quantized models for speed:
docker exec victor-ollama ollama pull qwen2.5-coder:1.5b  # Fastest
docker exec victor-ollama ollama pull qwen2.5-coder:7b    # Balanced
```

### vLLM Optimization

Edit `docker-compose.yml` command:

```yaml
# For more VRAM:
--max-model-len 4096  # Increase context window

# For more throughput:
--max-num-seqs 256    # Increase batch size

# For lower latency:
--max-num-batched-tokens 512
```

### Resource Limits

Add to services in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      memory: 4G
```

## Development Workflow

### 1. Local Development

```bash
# Mount local code
docker-compose run -v $(pwd):/app victor bash

# Inside container:
pip install -e ".[dev]"
pytest
```

### 2. Testing Changes

```bash
# Rebuild image
docker-compose build victor

# Run tests
docker-compose run victor pytest
```

### 3. Debugging

```bash
# Interactive shell in running container
docker-compose exec victor bash

# View logs
docker-compose logs -f victor

# Python debugger
docker-compose run victor python -m pdb /app/victor/ui/cli.py
```

## Production Deployment

### Recommended Setup

```yaml
# docker-compose.prod.yml
services:
  ollama:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G

  victor:
    restart: unless-stopped
    environment:
      - VICTOR_ENV=production
      - LOG_LEVEL=INFO
```

### Security Hardening

1. **API Keys:**
   ```bash
   # Use secrets instead of .env
   docker secret create anthropic_key /path/to/key
   ```

2. **Network Isolation:**
   ```yaml
   # Remove port mappings for internal services
   # Use reverse proxy for external access
   ```

3. **User Permissions:**
   ```yaml
   # Already running as non-root user 'victor'
   user: "1000:1000"
   ```

### Monitoring

```bash
# Resource usage
docker stats

# Health checks
docker-compose ps

# Logs
docker-compose logs --tail=100 -f
```

## Data Persistence

### Volumes

| Volume | Purpose | Size |
|--------|---------|------|
| `victor_ollama_data` | Ollama models | 5-50GB |
| `victor_vllm_cache` | HuggingFace cache | 2-10GB |
| `victor_home` | Victor config | <100MB |
| `victor_demo_output` | Demo results | <10MB |
| `victor_jupyter_data` | Jupyter packages | 1-5GB |

### Backup

```bash
# Backup all volumes
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama_backup.tar.gz /data

# Restore
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/ollama_backup.tar.gz -C /
```

### Clean Up

```bash
# Stop all services
docker-compose --profile full down

# Remove volumes (WARNING: deletes all data)
docker volume rm victor_ollama_data victor_vllm_cache

# Remove images
docker rmi $(docker images | grep victor | awk '{print $3}')

# Full cleanup
docker-compose down -v --rmi all
```

## Advanced Usage

### Custom Models

#### Ollama Custom Model

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM qwen2.5-coder:7b
SYSTEM You are a Python expert specializing in Django.
PARAMETER temperature 0.5
EOF

# Create custom model
docker exec victor-ollama ollama create django-expert -f Modelfile

# Use in profile
# Add to profiles.yaml:
# django:
#   provider: ollama
#   model: django-expert
```

#### vLLM Custom Model

Edit `docker-compose.yml`:

```yaml
vllm:
  environment:
    - VLLM_MODEL=codellama/CodeLlama-13b-Instruct-hf
  command: >
    --model codellama/CodeLlama-13b-Instruct-hf
    --dtype float16
    ...
```

### Multi-Container Scaling

```bash
# Scale Ollama for load balancing
docker-compose --profile full up -d --scale ollama=3

# Use nginx for load balancing
# Add nginx service to docker-compose.yml
```

### Remote Access

```bash
# Expose via SSH tunnel
ssh -L 11434:localhost:11434 user@remote-server

# Or use Tailscale/Cloudflare Tunnel for secure access
```

## Integration Examples

### CI/CD Integration

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Start services
        run: docker-compose --profile ollama up -d
      - name: Run tests
        run: docker-compose run victor pytest
```

### API Integration

```python
# External Python script
import httpx

# Use Ollama API
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5-coder:7b", "prompt": "Write hello world"}
    )
    print(response.json())
```

### VSCode Integration

```json
// .vscode/settings.json
{
  "docker.defaultRegistryPath": "victor",
  "remote.containers.defaultExtensions": [
    "ms-python.python",
    "github.copilot"
  ]
}
```

## Resources

- [Victor Documentation](../README.md)
- [Ollama Documentation](https://ollama.com/docs)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [GitHub Issues](https://github.com/yourusername/victor/issues)

## License

Same as Victor main project license.

## Support

For issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Search [GitHub Issues](https://github.com/yourusername/victor/issues)
3. Create new issue with:
   - Output of `docker-compose logs`
   - Output of `docker-compose ps`
   - Your `docker-compose.yml` (redact secrets)
   - Host OS and Docker version
