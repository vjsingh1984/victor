<div align="center">

# Victor

**Open-source agentic AI framework. Build, orchestrate, and evaluate AI agents across 22 providers.**

[![PyPI version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/vjsingh1984/victor/actions/workflows/test.yml/badge.svg)](https://github.com/vjsingh1984/victor/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/vjsingh1984/victor/actions)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue.svg)](https://ghcr.io/vjsingh1984/victor)

</div>

---

## Features

```
┌─────────────────────────────────────────────────────────────┐
│                     VICTOR FRAMEWORK                        │
│                                                             │
│  Agents ─── Teams ─── Workflows ─── Evaluation              │
│    │          │          │              │                    │
│  run()    Sequential   StateGraph    SWE-bench              │
│  stream()  Parallel    YAML DSL      Harnesses              │
│  chat()   Hierarchical Checkpoints   Code Quality           │
│           Pipeline                                          │
│                                                             │
│  22 Providers │ 33 Tool Modules │ 9 Verticals │ 4 Scopes   │
└─────────────────────────────────────────────────────────────┘
```

- **22 LLM Providers** — Cloud (Anthropic, OpenAI, Google, Azure, Bedrock, Vertex) + local (Ollama, LM Studio, vLLM)
- **33 Tool Modules** — File ops, git, shell, web, search, docker, testing, refactoring, analysis
- **9 Domain Verticals** — Coding, DevOps, RAG, Data Analysis, Research, Security, IaC, Classification, Benchmark
- **Multi-Agent Teams** — 4 formations: sequential, parallel, hierarchical, pipeline
- **Stateful Workflows** — YAML DSL compiled to StateGraph with typed state and checkpointing
- **Air-Gapped Mode** — Full functionality with local models for secure, offline environments

## At a glance

```
                              ┌─────────────────────────────────┐
                              │       Agent Orchestrator        │
                              │                                 │
[You] ──▶ [CLI/TUI/API] ──▶  │  ProviderManager ──▶ 22 LLMs   │ ──▶ [Response]
                              │  ToolPipeline    ──▶ 33 Tools   │
                              │  TeamCoordinator ──▶ Agents     │
                              │  StateManager    ──▶ 4 Scopes   │
                              └─────────────────────────────────┘
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

## Supported Providers

Victor supports **22 LLM providers** — switch mid-conversation without losing context.

| Category | Providers |
|----------|-----------|
| **Frontier Cloud** | Anthropic, OpenAI, Google Gemini, Azure OpenAI |
| **Cloud Platforms** | AWS Bedrock, Google Vertex |
| **Specialized** | xAI, DeepSeek, Mistral, Groq, Cerebras, Moonshot, ZAI |
| **Aggregators** | OpenRouter, Together AI, Fireworks AI, Replicate, Hugging Face |
| **Local (air-gapped)** | Ollama, LM Studio, vLLM, llama.cpp |

[Full Provider Reference](docs/reference/providers/)

## Python API

Victor provides a clean Python API for programmatic use:

```python
from victor.framework import Agent, EventType

# Simple use case
agent = await Agent.create(provider="anthropic")
result = await agent.run("Explain this codebase structure")
print(result.content)

# Streaming responses
async for event in agent.stream("Refactor this function"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")
    elif event.type == EventType.TOOL_CALL:
        print(f"\nUsing tool: {event.tool_name}")

# With tool configuration
from victor.framework import ToolSet

agent = await Agent.create(
    provider="openai",
    model="gpt-4o",
    tools=ToolSet.default()  # or ToolSet.minimal(), ToolSet.full()
)

# Multi-turn conversation
session = agent.chat()
await session.send("What files are in this project?")
await session.send("Now explain the main entry point")
```

### StateGraph Workflows

```python
from victor.framework import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    query: str
    result: str

graph = StateGraph(MyState)

graph.add_node("research", research_fn)
graph.add_node("synthesize", synthesize_fn)

graph.add_edge("research", "synthesize")
graph.add_edge("synthesize", END)

compiled = graph.compile()
result = await compiled.invoke({"query": "AI trends 2025"})
```

## Core capabilities

| Capability | What it means | Docs |
|------------|---------------|------|
| **Agent abstractions** | `run()`, `stream()`, `chat()`, `run_workflow()`, `run_team()` | [Framework](docs/development/architecture/) |
| **22 Providers** | Cloud + local LLMs; switch mid-thread without losing context | [Providers](docs/reference/providers/) |
| **33 Tool modules** | File ops, git, shell, web, search, docker, testing, analysis | [Tool catalog](docs/reference/tools/) |
| **Workflows** | YAML DSL compiled to StateGraph with typed state + checkpointing | [Workflows](docs/guides/workflow-development/) |
| **Multi-agent teams** | 4 formations: sequential, parallel, hierarchical, pipeline | [Multi-agent](docs/guides/multi-agent/) |
| **State management** | 4 scopes: workflow, conversation, team, global | [State](docs/development/architecture/) |
| **9 Verticals** | Domain-focused agents with tools, prompts, and workflows | [Verticals](docs/reference/verticals/) |
| **Evaluation** | Agent harnesses, code quality analysis, SWE-bench integration | [Evaluation](docs/development/) |

## Command quick reference

| Command | Purpose | Example |
|---------|---------|---------|
| `victor` | TUI mode | `victor` |
| `victor chat` | CLI mode | `victor chat "refactor this"` |
| `victor chat --mode plan` | Plan-only analysis | `victor chat --mode plan` |
| `victor serve` | HTTP API | `victor serve --port 8080` |
| `victor mcp` | MCP server | `victor mcp --stdio` |
| `/provider` | Switch provider in chat | `/provider openai --model gpt-4` |

## Screenshots

<!-- TUI Screenshot -->
![Victor TUI](docs/assets/tui-screenshot.png)
*The Victor TUI provides an interactive terminal interface with syntax highlighting and tool status.*

<!-- CLI Screenshot -->
![Victor CLI](docs/assets/cli-screenshot.png)
*CLI mode for quick queries and script integration.*

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

## Acknowledgments

Victor is built on the shoulders of excellent open-source projects:

- **[Pydantic](https://docs.pydantic.dev/)** - Data validation and settings management
- **[Tree-sitter](https://tree-sitter.github.io/)** - Incremental parsing for code analysis
- **[Textual](https://textual.textualize.io/)** - Modern TUI framework
- **[Typer](https://typer.tiangolo.com/)** - CLI interface with type hints
- **[Rich](https://rich.readthedocs.io/)** - Beautiful terminal formatting
- **[httpx](https://www.python-httpx.org/)** - Async HTTP client
- **[Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)** - Claude API client
- **[OpenAI SDK](https://github.com/openai/openai-python)** - OpenAI API client

## License

Apache License 2.0 - see [LICENSE](LICENSE).
