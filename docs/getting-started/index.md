# Getting Started with Victor

Welcome! This guide will help you install, configure, and run Victor in under 5 minutes.

## What is Victor?

**Victor** is a provider-agnostic AI coding assistant that supports 21 LLM providers (both local and cloud) through a unified CLI/TUI interface.

**Key Features:**
- **Provider Switching**: Switch between any LLM provider without losing conversation context
- **No API Key Required**: Use local models (Ollama, LM Studio, vLLM) by default
- **55+ Tools**: File operations, code editing, git, testing, search, and more
- **5 Verticals**: Specialized assistants for Coding, DevOps, RAG, Data Analysis, Research
- **Workflows**: YAML-based automation with scheduling and versioning
- **Multi-Agent Teams**: Coordinate specialized AI agents for complex tasks

## Choose Your Path

Select the installation path that matches your use case:

### Path 1: Local Model (Recommended for Beginners)

**Best for**: Privacy, offline use, free tier

- No API key required
- Privacy-focused (data stays local)
- Free and open source
- [Get Started →](installation.md#local-models)

### Path 2: Cloud Provider

**Best for**: Maximum capability, faster execution

- More powerful models
- Faster execution
- Pay per use
- [Get Started →](installation.md#cloud-providers)

### Path 3: Docker

**Best for**: Isolated environment, easy deployment

- Containerized deployment
- Easy to scale
- Cross-platform
- [Get Started →](installation.md#docker)

## System Requirements

- **Python**: 3.10 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 500MB free space
- **OS**: Linux, macOS, Windows (WSL2)

## Quick Verification

After installation, verify Victor is working:

```bash
# Check version
victor --version

# Test with local model
victor chat "Hello, Victor!" --provider ollama

# Test TUI mode
victor
```

## What's Next?

After installation, continue your journey:

1. **[First Run](first-run.md)** - Initial configuration and setup wizard
2. **[Basic Usage](basic-usage.md)** - Your first conversation
3. **[User Guide](../user-guide/)** - Daily usage patterns
4. **[Reference](../reference/)** - Provider and tool documentation

## Common Installation Options

| Method | Command | Best For |
|--------|---------|----------|
| **pipx** (Recommended) | `pipx install victor-ai` | Isolated installation |
| **pip** | `pip install victor-ai` | Virtual environments |
| **Docker** | `docker pull ghcr.io/vjsingh1984/victor:latest` | Containers |
| **Source** | `pip install -e .[dev]` | Development |

[Full Installation Guide →](installation.md)

## Need Help?

- **Troubleshooting**: See [Troubleshooting Guide](../user-guide/troubleshooting.md)
- **Documentation**: Browse [full docs](../README.md)
- **Community**: Join [Discord](https://discord.gg/...) or [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)
- **Issues**: [Report bugs](https://github.com/vjsingh1984/victor/issues)

---

**Next**: [Installation Guide →](installation.md)
