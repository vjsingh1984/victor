# Getting Started with Victor

Welcome to Victor! Get up and running in minutes.

## What is Victor?

**Victor** is a provider-agnostic AI coding assistant supporting 21 LLM providers through a unified interface.

**Key Features:**
- **21 LLM Providers**: Cloud (Anthropic, OpenAI, Google) and local (Ollama, LM Studio, vLLM)
- **Provider Switching**: Switch providers mid-conversation without losing context
- **55+ Tools**: File operations, code editing, git, testing, search, and more
- **5 Domain Verticals**: Coding, DevOps, RAG, Data Analysis, Research
- **YAML Workflows**: Multi-step automation with scheduling

## Fastest Path to Running Victor

### Local Model (No API Key)

```bash
# Install Victor
pipx install victor-ai

# Install and start Ollama
brew install ollama  # macOS
ollama serve

# Pull a model
ollama pull qwen2.5-coder:7b

# Start Victor
victor chat "Hello, Victor!"
```

### Cloud Provider (API Key Required)

```bash
# Install Victor
pipx install victor-ai

# Set API key
export ANTHROPIC_API_KEY=sk-ant-your-key

# Start Victor
victor chat --provider anthropic "Hello, Victor!"
```

### Docker

```bash
docker pull ghcr.io/vjsingh1984/victor:latest
docker run -it -v ~/.victor:/root/.victor ghcr.io/vjsingh1984/victor:latest
```

## Getting Started Guides

| Guide | Description | Time |
|-------|-------------|------|
| [Installation](installation.md) | Install Victor (pipx, pip, Docker) | 5 min |
| [First Run](first-run.md) | First steps and basic usage | 5 min |
| [Local Models](local-models.md) | Ollama, LM Studio, vLLM setup | 10 min |
| [Cloud Models](cloud-models.md) | Cloud provider setup (17 providers) | 10 min |
| [Docker](docker.md) | Docker deployment | 5 min |
| [Configuration](configuration.md) | Advanced configuration | 10 min |
| [Troubleshooting](troubleshooting.md) | Solve common issues | As needed |

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 500 MB | 1 GB (local models: 8-32 GB) |
| **OS** | Linux, macOS, Windows (WSL2) | Linux or macOS |

## Quick Verification

```bash
# Check version
victor --version

# List providers
victor providers

# Test chat
victor chat "Hello, Victor!"

# Open TUI mode
victor
```

## Common Tasks

```bash
# Explore codebase
victor "Summarize this repository"

# Get help with code
victor "Explain what main() does"

# Make changes
victor "Add error handling to db.py"

# Run tests
victor "Run tests and summarize failures"
```

## What's Next?

1. **[User Guide](../user-guide/)** - Daily usage patterns and workflows
2. **[Provider Reference](../reference/providers/)** - All 21 providers detailed
3. **[Tool Catalog](../reference/tools/catalog.md)** - 55+ available tools
4. **[Workflow DSL](../guides/workflow-development/dsl.md)** - YAML workflow automation

## Need Help?

- **Troubleshooting**: [Troubleshooting Guide](troubleshooting.md)
- **Documentation**: [Full docs](../README.md)
- **Community**: [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)
- **Issues**: [Report bugs](https://github.com/vjsingh1984/victor/issues)

---

**Next**: [Installation](installation.md) | [First Run](first-run.md) | [Local Models](local-models.md) | [Cloud Models](cloud-models.md)
