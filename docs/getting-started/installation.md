# Installation

Install Victor AI in minutes.

## Quick Install

```bash
pipx install victor-ai
```

## Alternative Methods

### pip (Development)
```bash
git clone https://github.com/vjsingh1984/victor
cd victor
pip install -e ".[dev]"
```

### pip (Direct)
```bash
pip install victor-ai
```

### Docker
```bash
docker pull ghcr.io/vjsingh1984/victor:latest
docker run -it -v ~/.victor:/root/.victor ghcr.io/vjsingh1984/victor:latest
```

## Optional Dependencies

| Dependency | Command | Purpose |
|------------|---------|---------|
| **API Server** | `pip install victor-ai[api]` | IDE integration |
| **All Languages** | `pip install victor-ai[lang-all]` | Full language support |
| **Development** | `pip install victor-ai[dev]` | Testing, linting |

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 500 MB | 1 GB (8-32 GB for local models) |
| **OS** | Linux, macOS, Windows (WSL2) | Linux or macOS |

## Verify Installation

```bash
victor --version
victor --help
```

## Next Steps

- [First Run](first-run.md) - Get started with your first query
- [Local Models](local-models.md) - Use Ollama for free, private AI
- [Cloud Models](cloud-models.md) - Use Anthropic, OpenAI, etc.
- [Configuration](../user-guide/configuration.md) - Advanced configuration
