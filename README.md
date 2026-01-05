<div align="center">

# Victor

**Provider-agnostic coding assistant. Local or cloud, one CLI.**

[![PyPI version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---

## Why Victor?

**Switch models without losing context.** Start with Claude, continue with GPT-4, finish with a local model—all in one conversation. Victor manages context independently of the LLM provider.

```bash
# Same conversation, different models
victor chat --provider anthropic    # Start with Claude
# ... switch mid-conversation ...
victor chat --provider ollama       # Continue locally
```

**No API key required.** Run local models (Ollama, LM Studio, vLLM) by default. Add cloud providers when you need them.

**One CLI for everything.** Code review, refactoring, testing, multi-file edits, workflows—all from the terminal.

---

## Quick Start

```bash
# Install
pipx install victor-ai

# Local model (no API key)
ollama pull qwen2.5-coder:7b

# Start
victor init && victor chat
```

Or with a cloud provider:
```bash
victor keys --set anthropic --keyring
victor chat --provider anthropic
```

---

## What It Does

| Capability | Details |
|------------|---------|
| **21 LLM Providers** | Anthropic, OpenAI, Google, DeepSeek, Ollama, vLLM, and 15 more |
| **55+ Tools** | File ops, code editing, git, testing, semantic search |
| **5 Verticals** | Coding, DevOps, RAG, Data Analysis, Research |
| **Workflows** | YAML DSL with scheduling, versioning, parallel execution |
| **Multi-Agent** | Team coordination with 5 formations |

---

## Core Commands

```bash
victor                              # TUI mode
victor chat                         # CLI mode
victor chat --mode plan             # Analysis without edits
victor "refactor for clarity"       # One-shot task
victor serve                        # HTTP API
victor mcp                          # MCP server
```

---

## Installation

```bash
pipx install victor-ai          # Recommended
pip install victor-ai           # Alternative
docker pull vjsingh1984/victor  # Container
```

---

## Documentation

| Topic | Link |
|-------|------|
| Getting Started | [docs/getting-started.md](docs/getting-started.md) |
| User Guide | [docs/user-guide.md](docs/user-guide.md) |
| Provider Reference | [docs/reference/PROVIDERS.md](docs/reference/PROVIDERS.md) |
| Tool Catalog | [docs/TOOL_CATALOG.md](docs/TOOL_CATALOG.md) |
| Workflow DSL | [docs/guides/WORKFLOW_DSL.md](docs/guides/WORKFLOW_DSL.md) |
| Developer Guide | [docs/development/README.md](docs/development/README.md) |

Full documentation: [docs/README.md](docs/README.md)

---

## Contributing

```bash
git clone https://github.com/vijayksingh/victor.git
cd victor && pip install -e ".[dev]" && pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

<div align="center">

**Open source. Provider agnostic. Local first.**

[GitHub](https://github.com/vijayksingh/victor) · [PyPI](https://pypi.org/project/victor-ai/) · [Issues](https://github.com/vijayksingh/victor/issues)

</div>
