<div align="center">

# Victor

**Provider-agnostic AI coding assistant. Local or cloud, one CLI.**

[![PyPI version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://badge.fury.io/py/victor-ai.svg)](LICENSE)
[![Tests](https://github.com/vjsingh1984/victor/actions/workflows/test.yml/badge.svg)](https://github.com/vjsingh1984/victor/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](https://github.com/vjsingh1984/victor/actions)

**21 LLM Providers** • **55+ Tools** • **5 Domain Verticals** • **92% Test Coverage**

</div>

---

## Quick Start

```bash
# Local model (free, offline)
pipx install victor-ai
ollama pull qwen2.5-coder:7b
victor chat "Hello"

# Cloud model (full capability)
pipx install victor-ai
export ANTHROPIC_API_KEY=sk-...
victor chat --provider anthropic
```

## Why Victor?

- **Multi-Provider**: Switch between 21 providers without losing context
- **Air-Gapped**: Full functionality with local models (Ollama, LM Studio, vLLM)
- **Verticals**: Specialized assistants for Coding, DevOps, RAG, Data Analysis, Research
- **Workflows**: YAML DSL for automation with StateGraph
- **Extensible**: Plugin system for custom tools and verticals

## Architecture

Victor uses a **two-layer coordinator architecture** with protocol-based design:

```
Clients (CLI/TUI, HTTP, MCP)
         ↓
ServiceContainer (55+ services, DI)
         ↓
AgentOrchestrator (facade)
         ↓
Coordinators (20: Chat, Tool, Context, ...)
         ↓
Providers (21) + Tools (55) + Workflows + Verticals (5)
         ↓
EventBus (5 backends: In-Memory, Kafka, SQS, RabbitMQ, Redis)
```

**Key Patterns:**
- **Protocol-Based**: 98 protocols for loose coupling
- **Dependency Injection**: ServiceContainer with lifecycle control
- **Event-Driven**: Async communication across components
- **YAML-First**: Configuration via YAML files

## Usage

### CLI Mode

```bash
victor chat "Explain this codebase"
victor chat --mode plan "Design a REST API"
victor chat --provider openai --model gpt-4
```

### Python API

```python
from victor.framework import Agent

# Simple
agent = await Agent.create(provider="anthropic")
result = await agent.run("Refactor this function")

# Streaming
async for event in agent.stream("Analyze code"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")

# Multi-provider (cost optimization)
brainstormer = await Agent.create(provider="ollama")      # FREE
implementer = await Agent.create(provider="openai")       # CHEAP
reviewer = await Agent.create(provider="anthropic")       # QUALITY
```

### Workflows

```yaml
# workflows/code-review.yaml
name: code_review
nodes:
  - id: analyze
    type: agent
    provider: anthropic
    tools: [read, grep, run_tests]

  - id: review
    type: agent
    provider: anthropic
    depends_on: [analyze]
```

```bash
victor workflow run code-review.yaml
```

## Providers

| Type | Providers |
|------|-----------|
| **Cloud** | Anthropic, OpenAI, Google, Azure, AWS, xAI, DeepSeek, Mistral, Cohere, Groq, Together, Fireworks, OpenRouter, Replicate, Hugging Face, Moonshot, Cerebras |
| **Local** | Ollama, LM Studio, vLLM, llama.cpp |

[Full provider reference →](docs/reference/providers/)

## Verticals

- **Coding**: Code analysis, refactoring, testing
- **DevOps**: Docker, Kubernetes, CI/CD
- **RAG**: Document search, retrieval, Q&A
- **Data Analysis**: Pandas, visualization, statistics
- **Research**: Literature search, synthesis

[Vertical documentation →](docs/reference/verticals/)

## Development

```bash
# Setup
git clone https://github.com/vjsingh1984/victor
cd victor
pip install -e ".[dev]"

# Development
make test           # Unit tests
make test-all       # All tests
make lint           # Linters
make format         # Format code

# Single test
pytest tests/unit/path/test_file.py::test_name -v
```

## Documentation

- **[Getting Started](docs/getting-started/)** - Installation and first steps
- **[User Guide](docs/user-guide/)** - Daily usage
- **[Developer Guide](docs/development/)** - Contributing
- **[Architecture](docs/architecture/)** - System design with 30 diagrams
- **[API Reference](docs/reference/api.md)** - Complete API docs
- **[Documentation Index](docs/)** - Full documentation hub

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

**[Homepage](https://github.com/vjsingh1984/victor)** • **[Documentation](docs/)** • **[Examples](docs/examples/)** • **[Changelog](CHANGELOG.md)**
