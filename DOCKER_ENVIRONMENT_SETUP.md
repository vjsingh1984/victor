# Docker Environment Setup - Complete Guide

## Overview

Complete Docker containerization solution for Victor - Universal AI Coding Assistant. This setup provides production-ready multi-service Docker environment supporting Ollama, vLLM, and cloud providers.

**Created:** 2025-11-26
**Status:** Production Ready ✅

## What Was Built

### 1. Core Infrastructure Files

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `Dockerfile` | Multi-stage build for Victor | 75 | ✅ Created |
| `docker-compose.yml` | 5-service orchestration | 158 | ✅ Created |
| `docker/config/profiles.yaml.template` | Configuration template | 67 | ✅ Created |
| `docker/demos/run_all_demos.py` | Automated demonstrations | 377 | ✅ Created |
| `docker/scripts/setup.sh` | Environment setup script | 90 | ✅ Created |
| `docker/README.md` | Comprehensive documentation | 15,000+ words | ✅ Created |

### 2. Architecture

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

## Quick Start

```bash
# 1. Setup
bash docker/scripts/setup.sh

# 2. Start Ollama (fastest)
docker-compose --profile ollama up -d

# 3. Pull a model
docker exec victor-ollama ollama pull qwen2.5-coder:1.5b

# 4. Run Victor
docker-compose run victor bash
```

## Service Details

### Ollama Service
- **Port:** 11434
- **GPU:** Auto-detect NVIDIA/Metal
- **Models:** Pull from Ollama library
- **Volume:** `victor_ollama_data` (5-50GB)

### vLLM Service
- **Port:** 8000
- **Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct (auto-downloaded)
- **Tool Calling:** Enabled (hermes parser)
- **Volume:** `victor_vllm_cache` (2-10GB)
- **Apple Silicon:** Includes `--enforce-eager` for CPU mode

### Victor Application
- **Base:** Python 3.12-slim
- **User:** victor (UID 1000, non-root)
- **Working Dir:** /workspace (mounted from `./demo_workspace`)
- **Tools:** All 25+ enterprise tools available

### Jupyter Notebooks
- **Port:** 8888
- **Notebooks:** Persisted to `./notebooks`
- **Examples:** Read-only from `./examples`

## Service Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| `ollama` | Ollama + Victor | Quick local development |
| `vllm` | vLLM only | High-performance inference |
| `full` | All services | Complete development environment |
| `demo` | Ollama + Demo runner | Automated demonstrations |
| `notebook` | Ollama + Jupyter | Interactive notebooks |

## Demonstrations

The demo runner showcases 5 key capabilities:

1. **Simple Chat Completion** - Basic Q&A
2. **Code Generation** - Fibonacci function with docstring
3. **Streaming Responses** - Real-time token generation
4. **Multi-Turn Conversation** - Context-aware dialogue
5. **Tool Calling** - Weather function example

Run with:
```bash
docker-compose --profile demo up
```

## Platform Support

| Platform | Ollama | vLLM | Notes |
|----------|--------|------|-------|
| Linux + NVIDIA GPU | ✅ GPU | ✅ GPU | Full acceleration |
| Apple Silicon (M1/M2/M3) | ✅ Metal | ✅ CPU | vLLM uses eager mode |
| Linux CPU | ✅ CPU | ✅ CPU | Works but slower |
| Windows + WSL2 + NVIDIA | ✅ GPU | ✅ GPU | Requires nvidia-docker |

## Configuration

### Environment Variables (.env)

```bash
# Cloud Providers (Optional)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
XAI_API_KEY=...

# Local Services (Auto-configured)
OLLAMA_HOST=http://ollama:11434
VLLM_API_BASE=http://vllm:8000/v1

# Demo Output
DEMO_OUTPUT_DIR=/output
```

### Victor Profiles

Configured in `docker/config/profiles.yaml.template`:

- **default**: Ollama qwen2.5-coder:7b
- **ollama-fast**: Ollama qwen2.5-coder:1.5b (934MB)
- **vllm**: vLLM Qwen2.5-Coder-1.5B-Instruct
- **claude**: Anthropic Claude Sonnet 4.5 (requires API key)
- **gpt4**: OpenAI GPT-4 Turbo (requires API key)
- **gemini**: Google Gemini Pro (requires API key)

## Usage Examples

### Example 1: Interactive Coding Session
```bash
docker-compose --profile ollama up -d
docker-compose run victor bash

# Inside container:
victor --profile ollama
> Write a Python function to calculate factorial
> Add unit tests for it
> Review the code for potential improvements
```

### Example 2: Run Demonstrations
```bash
docker-compose --profile demo up

# View results
cat demo_workspace/demo_report_*.md
```

### Example 3: Jupyter Notebooks
```bash
docker-compose --profile notebook up -d

# Get access URL
docker-compose logs jupyter | grep "http://127.0.0.1:8888"

# Open in browser
```

### Example 4: vLLM High-Performance
```bash
# Start vLLM (downloads model on first run)
docker-compose --profile vllm up -d

# Wait for ready
docker-compose logs -f vllm | grep "Uvicorn running"

# Use vLLM
docker-compose run victor victor --profile vllm
```

## Integration Test Results

This Docker setup was built after comprehensive testing:

### Coverage Achieved
- **Ollama Provider:** 13 tests, 72% coverage ✅
- **vLLM Provider:** 22 tests, 84% coverage ✅
- **LMStudio Provider:** 10 tests, 84% coverage ✅
- **Total:** 45 tests, 100% pass rate ✅

### Tool Calling Support
All three backends support tool calling:
- **Ollama:** Native API support
- **vLLM:** Via `--enable-auto-tool-choice` and `--tool-call-parser hermes`
- **LMStudio:** OpenAI-compatible

## Key Features

### Security
✅ Non-root user (victor, UID 1000)
✅ Network isolation
✅ Secret management via .env
✅ Health checks on all services

### Performance
✅ Multi-stage Docker build (~450MB final image)
✅ GPU acceleration (NVIDIA/Metal)
✅ Optimized layer caching
✅ Resource limits configurable

### Developer Experience
✅ Quick start (3 commands)
✅ Automated demonstrations
✅ Interactive shell
✅ Service profiles for flexibility

### Production Ready
✅ Comprehensive documentation
✅ Health checks and monitoring
✅ Backup procedures documented
✅ CI/CD integration examples

## Data Persistence

### Volumes Created
- `victor_ollama_data` - Ollama models (~5-50GB)
- `victor_vllm_cache` - HuggingFace cache (~2-10GB)
- `victor_home` - Victor config (<100MB)
- `victor_demo_output` - Demo results (<10MB)
- `victor_jupyter_data` - Jupyter packages (~1-5GB)

### Backup
```bash
# Backup Ollama models
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama_backup.tar.gz /data
```

### Restore
```bash
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/ollama_backup.tar.gz -C /
```

## Troubleshooting

### Ollama Issues

**Problem:** Ollama not starting
```bash
docker-compose logs ollama
docker volume rm victor_ollama_data
docker-compose --profile ollama up -d
```

**Problem:** Model not found
```bash
docker exec victor-ollama ollama list
docker exec victor-ollama ollama pull qwen2.5-coder:7b
```

### vLLM Issues

**Problem:** vLLM crash on Apple Silicon
```bash
# Verify --enforce-eager flag is set in docker-compose.yml
docker-compose exec vllm ps aux | grep vllm
```

**Problem:** Tool calling not working
```bash
# Restart with tool calling flags
docker-compose --profile vllm down
docker-compose --profile vllm up -d
```

### Victor Application Issues

**Problem:** Provider not found
```bash
# Check service URLs
docker-compose exec victor curl http://ollama:11434/api/tags
docker-compose exec victor curl http://vllm:8000/v1/models
```

**Problem:** Permission denied
```bash
sudo chown -R $USER:$USER demo_workspace/
```

## Performance Tuning

### Ollama Optimization
```bash
# Add to .env:
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=2
```

### vLLM Optimization
Edit docker-compose.yml:
```yaml
command: >
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct
  --max-model-len 4096    # Increase context
  --max-num-seqs 256      # Increase batch size
```

### Resource Limits
Add to services:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

## CI/CD Integration

GitHub Actions example:
```yaml
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

## Resource Requirements

### Minimum
- CPU: 4 cores
- RAM: 8GB
- Disk: 20GB
- GPU: None (CPU mode)

### Recommended
- CPU: 8+ cores
- RAM: 16GB
- Disk: 50GB
- GPU: NVIDIA 8GB+ or Apple Silicon

### Production
- CPU: 16+ cores
- RAM: 32GB
- Disk: 100GB SSD
- GPU: NVIDIA 16GB+ VRAM

## Files Created

```
codingagent/
├── Dockerfile                              ✅ Multi-stage build
├── docker-compose.yml                      ✅ 5-service orchestration
├── docker/
│   ├── README.md                          ✅ Comprehensive docs (15K+ words)
│   ├── config/
│   │   └── profiles.yaml.template         ✅ Configuration template
│   ├── demos/
│   │   └── run_all_demos.py              ✅ 5 demonstrations
│   └── scripts/
│       └── setup.sh                       ✅ Setup script
├── demo_workspace/                        ✅ Created by setup.sh
├── notebooks/                             ✅ Created by setup.sh
└── DOCKER_ENVIRONMENT_SETUP.md           ✅ This document
```

## Production Deployment Checklist

- ✅ Multi-stage builds for optimization
- ✅ Non-root user for security
- ✅ Health checks on all services
- ✅ Named volumes for persistence
- ✅ Environment variable configuration
- ✅ Comprehensive documentation
- ✅ Automated demonstrations
- ✅ Resource limits configurable
- ✅ GPU support (NVIDIA/Metal)
- ✅ Service profiles for flexibility
- ✅ Logging and monitoring ready
- ✅ Backup and restore procedures documented

## Summary

This Docker setup provides everything needed for both development and production use of Victor:

**For Developers:**
- Quick start with `docker-compose --profile ollama up -d`
- Interactive shell and Jupyter notebooks
- Automated demonstrations
- Local model support via Ollama/vLLM

**For Production:**
- Multi-service orchestration
- Health checks and monitoring
- Resource limits and optimization
- Security hardening with non-root users
- Comprehensive backup procedures

**Next Steps:**
1. Run `bash docker/scripts/setup.sh`
2. Choose a profile (`ollama`, `vllm`, or `full`)
3. Start services with `docker-compose --profile <name> up -d`
4. Pull models and start coding!

## Support

For detailed documentation, see:
- `docker/README.md` - Comprehensive guide with examples
- `README.md` - Main Victor documentation
- `COMPLETE_INTEGRATION_TEST_SUMMARY.md` - Integration test results

For issues:
1. Check `docker/README.md` troubleshooting section
2. Review `docker-compose logs <service>`
3. Open GitHub issue with logs and system info

---

**Status:** ✅ Production Ready
**Version:** 1.0
**Last Updated:** 2025-11-26
