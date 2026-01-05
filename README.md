<div align="center">

# Victor

**Open-source, local-first coding assistant**

*Run local or cloud models with one CLI.*

[![PyPI version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Quick Start](#quick-start) | [Why Victor](#why-victor) | [Providers](#providers) | [Docs](#documentation)

</div>

---

## Quick Start

**60-second quickstart (no API key):**
```bash
# 1. Install (recommended)
pipx install victor-ai

# Or:
# pip install victor-ai

# 2. Local model
ollama pull qwen2.5-coder:7b

# 3. Initialize and chat
victor init
victor chat
```

**One-shot:**
```bash
victor "refactor this file for clarity"
```

See `docs/QUICKSTART_60S.md` for a concise walkthrough.

```mermaid
flowchart LR
    A[Install] --> B[Init] --> C[Chat]
    C --> D{Provider?}
    D -->|Local| E[Ollama]
    D -->|Cloud| F[API Key]
```

<details>
<summary><b>Local Model (No API Key)</b></summary>

```bash
ollama pull qwen2.5-coder:7b
victor chat
```
</details>

<details>
<summary><b>Cloud Provider</b></summary>

```bash
victor keys --set anthropic --keyring
victor chat --provider anthropic --model claude-sonnet-4-5
```
</details>

---

## Why Victor?

- Local-first by default: run Ollama/LM Studio/vLLM without an API key.
- Provider-agnostic: switch models without rewriting workflows.
- Practical workflows: review, refactor, and test from the CLI.
- Extensible: add tools, verticals, and MCP integrations as needed.

---

## Capability Matrix

| Category | Capability | Count | Status |
|----------|------------|:-----:|--------|
| **LLM Providers** | Anthropic, OpenAI, Google, Groq, DeepSeek, Mistral, xAI, Ollama, LM Studio, vLLM, llama.cpp, OpenRouter, Together, Cohere, Replicate, Bedrock, Azure, Vertex AI, Fireworks, Perplexity, Cerebras | 21 | Stable |
| **Tools** | File operations, code editing, shell execution, search, git, test runners, semantic search | 55+ | Stable |
| **Verticals** | Coding, DevOps, RAG, Data Analysis, Research | 5 | Stable |
| **Workflows** | YAML DSL, StateGraph, scheduling, versioning, parallel execution | - | Stable |
| **Multi-Agent** | Team coordination, personas, formations (hierarchical, flat, sequential, collaborative) | 4 | Stable |
| **Observability** | EventBus, OpenTelemetry, live streaming, TUI dashboard | - | Stable |
| **Resilience** | Circuit breaker, retry policies, provider fallback, budget controls | - | Stable |
| **Integrations** | MCP protocol, VS Code extension, HTTP API, WebSocket events | 4 | Beta |

### Tools by Category

| Category | Tools |
|----------|-------|
| **File System** | read_file, write_file, list_directory, search_files, glob_files |
| **Code Editing** | edit_file, multi_edit, apply_patch, undo_edit |
| **Shell** | run_command, run_background, kill_process |
| **Git** | git_status, git_diff, git_commit, git_log, git_branch |
| **Search** | semantic_search, grep_search, symbol_search, find_references |
| **Testing** | run_tests, coverage_report, test_generation |
| **Analysis** | ast_parse, dependency_graph, code_review |
| **Web** | web_search, fetch_url, scrape_page |

### Domain Verticals

| Vertical | Description | Key Tools |
|----------|-------------|-----------|
| **Coding** | Code editing, AST analysis, LSP integration | 20+ code-specific tools |
| **DevOps** | Docker, Terraform, CI/CD, infrastructure | Container, IaC tools |
| **RAG** | Document ingestion, vector search, QA | Embeddings, retrieval |
| **Data Analysis** | Pandas, statistics, visualization | Data processing tools |
| **Research** | Web search, citations, synthesis | Research workflows |

### Provider-Agnostic Context Management

Victor maintains conversation context **independent of the LLM provider**, enabling:

| Feature | Description |
|---------|-------------|
| **Unified Conversation History** | Same context works across Anthropic, OpenAI, DeepSeek, Ollama, or any provider |
| **Mid-Conversation Provider Switching** | Switch from Claude to GPT-4 to local Qwen without losing context |
| **Stateless Model Support** | Works with models that don't maintain session state (most open-source models) |
| **Context Windowing** | Smart truncation to fit each model's context limits |
| **Summarization** | Automatic summarization for long conversations |
| **State Checkpoints** | Save/restore conversation state across sessions |

**This is the core USP**: Start a conversation with Claude, continue with DeepSeek for cost savings, then switch to a local model for sensitive code - all in one session. Victor injects the full conversation context to each model transparently.

### Discover Capabilities Programmatically

```bash
# List all capabilities (CLI)
victor capabilities              # Full manifest
victor capabilities --tools      # Tools only
victor capabilities --teams      # Multi-agent teams
victor capabilities --json       # JSON for tooling

# HTTP API
curl http://localhost:8765/capabilities
```

---

## Providers

Use the same CLI with local or cloud models. Pick based on privacy, cost, and speed.

- Local: Ollama, LM Studio, vLLM, llama.cpp
- Cloud: Anthropic, OpenAI, Google, Groq, DeepSeek, Mistral, xAI, and more

See `docs/reference/PROVIDERS.md` for setup and examples.

---

## What You Can Do

- Run local or cloud models with the same CLI.
- Use built-in workflows for review, refactor, tests, and docs.
- Make multi-file edits with previews and rollback.
- Add MCP tools and custom verticals when needed.

See `docs/ARCHITECTURE_DEEP_DIVE.md` for system internals.

---

## Workflow Capabilities

Victor includes a YAML-based workflow system with scheduling and versioning.

### What's Included

| Feature | Status | Notes |
|---------|--------|-------|
| YAML Workflow DSL | Stable | Agent, compute, condition, parallel nodes |
| Cron Scheduling | Stable | Standard cron syntax, aliases (@daily, @hourly) |
| Execution Limits | Stable | Timeouts, retries, iteration limits |
| Workflow Versioning | Stable | Semver, migrations, deprecation |
| CLI Management | Stable | `victor scheduler add/list/remove` |

### Quick Example

```yaml
# my_workflow.yaml
workflows:
  daily_analysis:
    schedule:
      cron: "0 9 * * *"
    execution:
      max_timeout_seconds: 3600
    nodes:
      - id: analyze
        type: agent
        role: analyst
        goal: "Analyze daily metrics"
        timeout: 300
        next: []
```

```bash
victor scheduler add daily_analysis --cron "0 9 * * *"
victor scheduler start
```

### Honest Limitations

The built-in scheduler is designed for **single-instance deployments**:

- No distributed execution (one scheduler process only)
- In-memory state (no persistence across restarts without YAML reload)
- No web dashboard (CLI only)
- No DAG-style dependencies between workflows

**For production at scale**, use:
- **Airflow**: Complex DAGs, web UI, distributed workers
- **Temporal.io**: Microservices, durable execution
- **AWS Step Functions**: Serverless, AWS-native

Victor's scheduler is ideal for: dev environments, single-server deployments, and simple recurring tasks where an external orchestrator is overkill.

See `docs/guides/WORKFLOW_SCHEDULER.md` for complete documentation.

---

## Installation

| Method | Command |
|--------|---------|
| pip | `pip install victor-ai` |
| pipx | `pipx install victor-ai` |
| Docker | `docker pull vjsingh1984/victor-ai` |
| Source | `pip install -e ".[dev]"` |

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `victor` | Start TUI |
| `victor chat` | CLI mode |
| `victor chat --provider X` | Use specific provider |
| `victor chat --mode explore` | Exploration mode |
| `victor serve` | API server |
| `victor mcp` | MCP server |
| `victor keys --set X` | Configure API key |
| `victor workflow validate <path>` | Validate YAML workflow file |
| `victor workflow render <path>` | Render workflow as diagram (SVG/PNG/ASCII) |
| `victor scheduler start` | Start workflow scheduler daemon |
| `victor scheduler add <name>` | Add scheduled workflow |
| `victor scheduler list` | List scheduled workflows |
| `victor vertical create <name>` | Scaffold new vertical structure |
| `victor vertical list` | List available verticals |
| `victor dashboard` | Launch observability dashboard |

### Agent Modes

| Mode | Edits | Exploration | Use Case |
|------|:-----:|:-----------:|----------|
| BUILD | Full | 1.0x | Implementation |
| PLAN | Sandbox | 2.5x | Analysis |
| EXPLORE | Notes | 3.0x | Understanding |

---

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Complete usage |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Contributing |
| [Tool Catalog](docs/TOOL_CATALOG.md) | Tool list |
| [Provider Setup](docs/guides/PROVIDER_SETUP.md) | Provider config |
| [Local Models](docs/guides/LOCAL_MODELS.md) | Ollama, LM Studio, vLLM, llama.cpp |
| [Air-Gapped Mode](docs/embeddings/AIRGAPPED.md) | Offline operation |
| [Workflow DSL](docs/guides/WORKFLOW_DSL.md) | StateGraph guide |
| [Workflow Scheduling](docs/guides/WORKFLOW_SCHEDULER.md) | Cron, versioning, limits |

---

## Project Status

- CLI and core workflows are stable for daily use.
- VS Code extension is beta.
- See `docs/ARCHITECTURE_DEEP_DIVE.md` for system details.

---

## Contributing

```bash
git clone https://github.com/vijayksingh/victor.git
cd victor
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)

---

<div align="center">

**Open source. Provider agnostic. Privacy first.**

[GitHub](https://github.com/vijayksingh/victor)

</div>
