<div align="center">

# Victor

**Provider-agnostic coding assistant. Local or cloud, one CLI.**

[![PyPI version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/vjsingh1984/victor/actions/workflows/test.yml/badge.svg)](https://github.com/vjsingh1984/victor/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/vjsingh1984/victor/actions)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue.svg)](https://ghcr.io/vjsingh1984/victor)

</div>

---

## Features

- **21 LLM Providers** - Cloud (Anthropic, OpenAI, Google, Azure, AWS Bedrock) and local (Ollama, LM Studio, vLLM)
- **55 Specialized Tools** - Across 5 domain verticals (Coding, DevOps, RAG, Data Analysis, Research)
- **Air-Gapped Mode** - Full functionality with local models for secure, offline environments
- **Semantic Codebase Search** - Context-aware code understanding with Tree-sitter and embeddings
- **YAML Workflow DSL** - Define multi-step automation with StateGraph and checkpointing
- **Multi-Agent Teams** - Coordinate specialized agents for complex tasks
- **MCP Protocol Support** - Model Context Protocol for IDE integration
- **Provider Switching** - Change models mid-conversation without losing context
- **Multi-Provider Workflows** - Use different providers for optimal cost/quality balance
- **Plugin System** - Extend workflow compilation with custom sources (JSON, S3, databases)
- **Coordinator Architecture** - Modular, testable orchestrator design (experimental)

## At a glance

```
[You] -> [Victor CLI/TUI] -> [Context + Orchestration] -> [Providers] -> [Tools + Workflows] -> [Results]
                                      |                     |
                                      |                     +--> Local: Ollama, LM Studio, vLLM
                                      +--> Files, tests, git       Cloud: Anthropic, OpenAI, etc
```

## Architecture

Victor uses a **two-layer coordinator architecture** that separates application-specific orchestration from framework-agnostic workflow infrastructure:

### Application Layer (`victor/agent/coordinators/`)

Manages the AI agent conversation lifecycle with Victor-specific business logic:

- **ChatCoordinator**: LLM chat operations and streaming
- **ToolCoordinator**: Tool validation, execution, and budget enforcement
- **ContextCoordinator**: Context management and compaction strategies
- **AnalyticsCoordinator**: Session metrics and analytics collection
- **PromptCoordinator**: System prompt building from contributors
- **SessionCoordinator**: Conversation session lifecycle
- **ProviderCoordinator**: Provider switching and management
- **ModeCoordinator**: Agent modes (build, plan, explore)
- **ToolSelectionCoordinator**: Semantic tool selection
- **CheckpointCoordinator**: Workflow checkpoint management
- **EvaluationCoordinator**: LLM evaluation and benchmarking
- **MetricsCoordinator**: System metrics collection

### Framework Layer (`victor/framework/coordinators/`)

Provides domain-agnostic workflow infrastructure reusable across all verticals:

- **YAMLWorkflowCoordinator**: YAML workflow loading and execution
- **GraphExecutionCoordinator**: StateGraph/CompiledGraph execution
- **HITLCoordinator**: Human-in-the-loop workflow integration
- **CacheCoordinator**: Workflow caching system

### Key Benefits

- **Single Responsibility**: Each coordinator has one clear purpose
- **Layered Design**: Application layer builds on framework foundation
- **Reusability**: Framework coordinators work across all verticals (Coding, DevOps, RAG, DataAnalysis, Research)
- **Testability**: Coordinators can be tested independently
- **Maintainability**: Clear boundaries reduce coupling

**For detailed architecture documentation**, see:
- [Coordinator Architecture](docs/architecture/coordinator_separation.md) - Two-layer design principles
- [Architecture Diagrams](docs/architecture/diagrams/coordinators.mmd) - Visual representations
- [Architecture Overview](ARCHITECTURE.md) - System overview and components

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

## Optional Dependencies

Victor supports several optional dependency groups for extended functionality:

### Development Tools
```bash
pip install victor-ai[dev]  # Testing, linting, type checking
```

### API Server
```bash
pip install victor-ai[api]  # FastAPI server for IDE integrations
```

### Checkpoint Persistence
```bash
pip install victor-ai[checkpoints]  # SQLite backend for workflow checkpoints
```

### Language Grammars
```bash
pip install victor-ai[lang-all]  # All tree-sitter language grammars
pip install victor-ai[lang-c]  # C language support
pip install victor-ai[lang-web]  # HTML, CSS, JavaScript
```

### Experimental Features
```bash
pip install victor-ai[vector-experimental]  # Experimental vector stores (ProximaDB)
```

**Note**: Victor includes LanceDB as the default vector store for semantic code search and conversation embeddings. The `vector-experimental` group provides additional vector stores like ProximaDB, which are optional and not required for core functionality.

## Supported Providers

Victor supports **21 LLM providers** for maximum flexibility:

### Cloud Providers

| Provider | Models | Tool Calling | Vision | Setup |
|----------|--------|--------------|--------|-------|
| **Anthropic** | Claude Sonnet 4, Claude 3.5, Opus, Haiku | Yes | Yes | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-4o, GPT-4, o1-preview, o1-mini | Yes | Yes | `OPENAI_API_KEY` |
| **Google** | Gemini 2.0 Flash, Gemini 1.5 Pro (2M context) | Yes | Yes | `GOOGLE_API_KEY` |
| **Azure OpenAI** | GPT-4, Claude (via Azure) | Yes | Yes | `AZURE_OPENAI_API_KEY` |
| **AWS Bedrock** | Claude 3, Llama 3, Titan | Yes | Yes | AWS credentials |
| **xAI** | Grok | Yes | No | `XAI_API_KEY` |
| **DeepSeek** | DeepSeek-V3, DeepSeek Coder | Yes | No | `DEEPSEEK_API_KEY` |
| **Mistral** | Mistral Large, Mistral Small | Yes | Partial | `MISTRAL_API_KEY` |
| **Cohere** | Command R, Command R+ | Yes | No | `COHERE_API_KEY` |
| **Groq** | Llama, Mistral (ultra-fast, 300+ tok/s) | Yes | No | `GROQ_API_KEY` |
| **Together AI** | 100+ open models | Yes | Partial | `TOGETHER_API_KEY` |
| **Fireworks AI** | 50+ open models | Yes | Partial | `FIREWORKS_API_KEY` |
| **OpenRouter** | 200+ models (unified API) | Yes | Partial | `OPENROUTER_API_KEY` |
| **Replicate** | 20,000+ models | Yes | Partial | `REPLICATE_API_TOKEN` |
| **Hugging Face** | 100,000+ models | Partial | Partial | `HF_API_KEY` (optional) |
| **Moonshot** | Moonshot v1 (8K/32K/128K) | Yes | No | `MOONSHOT_API_KEY` |
| **Cerebras** | Llama, Mistral (500+ tok/s) | Yes | No | `CEREBRAS_API_KEY` |

### Local Providers (No API Key Required)

| Provider | Best For | Setup |
|----------|----------|-------|
| **Ollama** | Beginners, ease of use | `ollama pull qwen2.5-coder:7b` |
| **LM Studio** | GUI model management, Windows | Download from lmstudio.ai |
| **vLLM** | Production, high throughput | `pip install vllm` |
| **llama.cpp** | CPU inference, minimal footprint | Build from source |

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

# Multi-provider workflow (cost optimization)
from victor import Agent

# Use different providers for different tasks
brainstormer = await Agent.create(provider="ollama", model="qwen2.5-coder:7b")  # FREE
implementer = await Agent.create(provider="openai", model="gpt-4o-mini")  # CHEAP
reviewer = await Agent.create(provider="anthropic", model="claude-sonnet-4-5")  # QUALITY

ideas = await brainstormer.run("Brainstorm 3 feature ideas")
code = await implementer.run(f"Implement: {ideas.content[0]}")
review = await reviewer.run(f"Review this code: {code.content}")
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
| **Provider switching** | Change models mid-thread without losing context | [Provider switching](docs/user-guide/index.md#1-provider-switching) |
| **Multi-provider workflows** | Use different providers for cost/quality optimization | [Multi-provider guide](docs/features/multi_provider_workflows.md) |
| **Workflows** | YAML DSL for multi-step automation | [Workflow development](docs/guides/workflow-development/) |
| **Plugins** | Custom workflow compilers (JSON, S3, databases) | [Plugin development](docs/features/plugin_development.md) |
| **Multi-agent teams** | Coordinate specialized agents | [Multi-agent](docs/guides/multi-agent/) |
| **Tooling** | File ops, git, tests, search, web | [Tool catalog](docs/reference/tools/) |
| **Verticals** | Domain-focused assistants | [Verticals](docs/reference/verticals/) |
| **Coordinator mode** | Modular orchestrator architecture | [Coordinator guide](docs/features/orchestrator_modes.md) |

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
- [Quick Start Guides](docs/quickstart/)
  - [Multi-Provider Workflows](docs/quickstart/multi_provider_quickstart.md)
  - [Plugin Development](docs/quickstart/plugin_development_quickstart.md)
  - [Coordinator Mode](docs/quickstart/coordinator_mode_quickstart.md)
- [User Guide](docs/user-guide/)
- [Features](docs/features/)
  - [Multi-Provider Workflows](docs/features/multi_provider_workflows.md)
  - [Plugin Development](docs/features/plugin_development.md)
  - [Orchestrator Modes](docs/features/orchestrator_modes.md)
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
