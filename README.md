<div align="center">

# Victor

**Provider-agnostic coding assistant. Local or cloud, one CLI.**

[![PyPI version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/vjsingh1984/victor/actions/workflows/test.yml/badge.svg)](https://github.com/vjsingh1984/victor/actions/workflows/test.yml)

</div>

---

## At a glance

```
[You] -> [Victor CLI/TUI] -> [Context + Orchestration] -> [Providers] -> [Tools + Workflows] -> [Results]
                                      |                     |
                                      |                     +--> Local: Ollama, LM Studio, vLLM
                                      +--> Files, tests, git       Cloud: Anthropic, OpenAI, etc
```

## Choose your path

| Persona | Start here | Typical goals |
|---------|------------|---------------|
| **New user** | [Getting Started](docs/getting-started/) | Install, first run, local vs cloud |
| **Daily user** | [User Guide](docs/user-guide/) | Commands, modes, profiles, workflows |
| **Operator** | [Operations](docs/operations/) | Deployment, monitoring, security |
| **Contributor** | [Development](docs/development/) | Setup, testing, architecture, extending |
| **Architect** | [Architecture](ARCHITECTURE.md) | System overview, core components |

## Quick start

| Path | Commands | Best for |
|------|----------|----------|
| **Local model** | `pipx install victor-ai`<br>`ollama pull qwen2.5-coder:7b`<br>`victor chat "Hello"` | Privacy, offline, free tier |
| **Cloud model** | `pipx install victor-ai`<br>`export ANTHROPIC_API_KEY=...`<br>`victor chat --provider anthropic` | Max capability |
| **Docker** | `docker pull ghcr.io/vjsingh1984/victor:latest`<br>`docker run -it -v ~/.victor:/root/.victor ghcr.io/vjsingh1984/victor:latest` | Isolated env |

## Core capabilities

| Capability | What it means | Docs |
|------------|---------------|------|
| **Provider switching** | Change models mid-thread without losing context | [Provider switching](docs/user-guide/index.md#1-provider-switching) |
| **Workflows** | YAML DSL for multi-step automation | [Workflow development](docs/guides/workflow-development/) |
| **Multi-agent teams** | Coordinate specialized agents | [Multi-agent](docs/guides/multi-agent/) |
| **Tooling** | File ops, git, tests, search, web | [Tool catalog](docs/reference/tools/) |
| **Verticals** | Domain-focused assistants | [Verticals](docs/reference/verticals/) |

## Command quick reference

| Command | Purpose | Example |
|---------|---------|---------|
| `victor` | TUI mode | `victor` |
| `victor chat` | CLI mode | `victor chat "refactor this"` |
| `victor chat --mode plan` | Plan-only analysis | `victor chat --mode plan` |
| `victor serve` | HTTP API | `victor serve --port 8080` |
| `victor mcp` | MCP server | `victor mcp --stdio` |
| `/provider` | Switch provider in chat | `/provider openai --model gpt-4` |

## Documentation

- [Getting Started](docs/getting-started/)
- [User Guide](docs/user-guide/)
- [Guides](docs/guides/)
- [Reference](docs/reference/)
- [Operations](docs/operations/)
- [Development](docs/development/)
- [Architecture](ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)

## Contributing

We welcome contributions. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Community

- [GitHub](https://github.com/vjsingh1984/victor)
- [Discussions](https://github.com/vjsingh1984/victor/discussions)
- [Issues](https://github.com/vjsingh1984/victor/issues)
- [Discord](https://discord.gg/...)

## License

Apache License 2.0 - see [LICENSE](LICENSE).
