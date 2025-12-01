<div align="center">

![Victor Banner](./assets/victor-banner.png)

<h3>Community-Built AI Coding Assistant for Secure, Hybrid Software & Data Delivery</h3>
<p><strong>Free & Open Source • Apache 2.0 Licensed</strong></p>

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Features](#key-features) • [Quick Start](#quick-start) • [Use Cases](#use-cases) • [Documentation](#documentation) • [Contribute](#contributing)

</div>

---

## Project Status: Alpha

**Victor is currently in an alpha state.** The project has a comprehensive and ambitious roadmap, but the current implementation is a foundational subset of the full vision.

Much of the documentation, especially `docs/ARCHITECTURE_DEEP_DIVE.md`, describes the target architecture and features that are under active development. For a detailed breakdown of the current state and action items, please see our **[Codebase Analysis Report](docs/CODEBASE_ANALYSIS_REPORT.md)**.

We share Victor freely for anyone to use, learn from, and improve—there are no paid tiers or monetization plans. We welcome contributors who are excited to help us build the future of this project.

**Docs + Onboarding:** use `docs/guides/QUICKSTART.md` as the canonical setup guide and treat other longform docs as aspirational unless explicitly labeled “Current State.” The active code lives in `victor/`; `archive/victor-legacy/` is frozen for historical reference.

---

## What is Victor?

**Victor** is a **community-driven**, terminal-based AI coding assistant where **multiple LLMs can work together**—generating, critiquing, and refining code collaboratively. Instead of being locked into one provider's ecosystem, Victor lets you orchestrate Claude Opus 4.5, GPT-5, Gemini 3, and local models like Qwen3 or Llama as a unified team.

### Why This Matters Now

The AI frontier is shifting *weekly*, not yearly:
- **August 2025**: GPT-5 launched with unified reasoning (74.9% SWE-bench)
- **November 18**: Gemini 3 topped 19/20 benchmarks with 37.4% on Humanity's Last Exam
- **November 24**: Claude Opus 4.5 reclaimed coding crown (80.9% SWE-bench)

Three frontier releases in one quarter. Each claiming "best for coding." If your tooling locks you into one provider, you're permanently one release behind.

Victor is built for this reality: **add a new provider, keep your workflows**.

### The Problem with Current AI Coding Tools

| Tool | Lock-in | Privacy Concern | Cost Trend |
|------|---------|-----------------|------------|
| **GitHub Copilot** | OpenAI only | Code sent to Microsoft servers | Increasing ($19→$39/mo for enterprise) |
| **Cursor** | Proprietary models | Caches/indexes your codebase on their servers | Rising (free tier shrinking) |
| **Claude Code** | Anthropic only | Cloud-dependent | Pay per token |
| **Continue.dev** | Multi-provider ✓ | Local option ✓ | Open source ✓ |
| **Victor** | **7 providers** | **100% air-gapped option** | **Apache 2.0 open source** |

**The hidden cost of Cursor and similar tools**: They index and cache your codebase on remote servers to provide "context-aware" suggestions. For enterprises with proprietary code, trade secrets, or compliance requirements (HIPAA, SOC2, ITAR)—this is a non-starter.

Victor offers the same intelligent codebase understanding using **local embeddings that never leave your machine**.

### Victor's Approach: LLMs as Collaborative Team Members

Instead of one model doing everything, Victor enables workflows like:

```
1. DRAFT    → Fast local model (qwen3-coder:30b) generates initial code
2. REVIEW   → Claude Opus 4.5 analyzes for edge cases and security issues
3. REFINE   → GPT-5 optimizes for performance and readability
4. VALIDATE → Local model runs tests (code never leaves your network)
```

Each model contributes its strength. The result is better than any single model alone.

### Why Consider Victor

| Feature | Benefit |
|---------|---------|
| **7 LLM Providers** | Claude Opus 4.5, GPT-5, Gemini 3, Grok, Ollama, LMStudio, vLLM |
| **Future-Proof** | Add GPT-6, Claude 5, or any new model without changing workflows |
| **100% Air-Gapped** | Complete offline operation—no code ever transmitted |
| **Local Embeddings** | Codebase indexing that stays on your machine |
| **54 Enterprise Tools** | Unified toolset works across all providers |
| **Apache 2.0** | Truly open source, safe for commercial use |

---

## Key Features

### Universal Provider Support

Switch between AI providers as easily as changing a config file:

| Provider | Models | Tool Calling | Streaming | Cost |
|----------|--------|--------------|-----------|------|
| **Anthropic Claude** | Opus 4.5, Sonnet 4.5, Haiku | Yes | Yes | Pay per use |
| **OpenAI GPT** | GPT-5, GPT-5 Pro, GPT-5 Mini | Yes | Yes | Pay per use |
| **Google Gemini** | Gemini 3 Pro, 3 Deep Think | Yes | Yes | Pay per use |
| **xAI Grok** | Grok, Grok Vision | Yes | Yes | Pay per use |
| **Ollama** | Qwen3, Llama 3.3, CodeLlama, +100 | Yes | Yes | **FREE** |
| **vLLM** | Any HuggingFace model | Yes | Yes | **FREE** |
| **LMStudio** | Any GGUF model | Yes | Yes | **FREE** |

### Enterprise-Grade Tool Suite

<details>
<summary><b>Code Management Tools</b></summary>

- **Multi-File Editor** - Atomic edits across multiple files with rollback
- **Batch Processor** - Parallel operations on hundreds of files
- **Refactoring Engine** - AST-based safe transformations (rename, extract, inline)
- **Git Integration** - AI-powered commits, smart staging, conflict analysis, PR creation

</details>

<details>
<summary><b>Code Quality & Analysis</b></summary>

- **Code Review** - Automated quality analysis with complexity metrics
- **Security Scanner** - Detect secrets (12+ patterns) and insecure configurations
- **Code Metrics** - Complexity analysis, maintainability index, technical debt
- **Semantic Search** - AI-powered codebase indexing with context-aware search

</details>

<details>
<summary><b>Testing & CI/CD</b></summary>

- **Test Runner** - Execute pytest with configurable options
- **CI/CD Automation** - Generate GitHub Actions pipelines (GitLab/CircleCI planned)
- *Test Generation* - Coming soon
- *Coverage Analysis* - Coming soon

</details>

<details>
<summary><b>Documentation</b></summary>

- **Docstring Generator** - Auto-generate function/class documentation with AST analysis
- *API Documentation* - OpenAPI/Swagger generation planned
- *README Automation* - Coming soon

</details>

<details>
<summary><b>Development Tools</b></summary>

- **Database Tools** - Query SQLite, PostgreSQL, MySQL, SQL Server safely
- **Docker Management** - Container and image operations
- **HTTP/API Testing** - Test endpoints and validate responses
- **Web Search** - Fetch documentation and online resources
- **Project Scaffolding** - 5+ production-ready templates (FastAPI, Flask, React, CLI)

</details>

### Security & Privacy First

```
┌─────────────────────────────────────────────────────────┐
│                  Security Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Air-Gapped Mode                                        │
│     - Complete offline operation                        │
│     - Zero network calls                                │
│     - Enterprise compliance ready                       │
│                                                         │
│  Sandboxed Execution                                    │
│     - Docker containerized code execution               │
│     - Isolated from host system                         │
│     - Automatic cleanup                                 │
│                                                         │
│  Secret Detection                                       │
│     - 12+ pattern types (API keys, tokens, passwords)   │
│     - Pre-commit scanning                               │
│     - Dependency vulnerability checks                   │
│                                                         │
│  Type-Safe Architecture                                 │
│     - Pydantic validation throughout                    │
│     - Runtime type checking                             │
│     - Zero tolerance for type errors                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Reality Check (Current Implementation)

- **Tooling surface**: 54 tools dynamically registered (editor, git, test runner, Docker, docs, refactors, cache/database/http, basic CI/CD stub). Test generation, coverage analysis, and rich pipeline generators are not implemented yet.
- **Tool selection & budgets**: Semantic tool selection is on by default; tool call budget/loop-guarding is enforced by the orchestrator. If stage pruning removes everything, Victor falls back to a small core set capped by `fallback_max_tools` (default 8) to avoid broadcasting all tools. Disable semantic selection via `profiles.yaml` → `tools` config if you want keyword-only.
- **Tool cache**: Allowlisted tools (defaults: code_search, semantic_code_search, list_directory, plan_files) are cached for `tool_cache_ttl` seconds to avoid rerunning pure/idempotent operations. Configure via `tool_cache_*` settings.
- **Planning scaffold**: A minimal dependency graph is registered for search→read→analyze flows; deeper auto-planning is a future enhancement.
- **Code search**: Use `semantic_code_search` for embedding-backed search (auto-reindexes on file changes); `code_search` remains keyword-only. Requires embedding deps (sentence-transformers + lancedb/chromadb).
- **Security scan**: Current `security_scan` detects secrets and obvious config flags via regex only; no dependency/IaC/CVE scanning is wired in yet.
- **Docs**: Prefer `docs/guides/QUICKSTART.md` for setup and `docker/README.md` for containers. Embedding/air-gapped material is being consolidated under `docs/embeddings/` (see consolidation plan).
- **Package layout**: The active package lives at `victor/`.
- **Tool catalog**: See `docs/TOOL_CATALOG.md` (generated via `python scripts/generate_tool_catalog.py`) for the exact tool surface registered today.

| Area | Implemented now | Planned/Gap |
| --- | --- | --- |
| Code search | Keyword `code_search` + new `semantic_code_search` (embeddings required) | Improve CLI UX (progress, cache stats), add selective reindex |
| Security | Regex secret/config checks | Dependency/IaC/CVE scanning |
| CI/CD & coverage | Basic `cicd` stub, no coverage tool | Rich pipeline/coverage analyzers |
| Package layout | Active: `victor/` | ✅ Completed |

---

## Use Cases

- Share workflows for code review, refactoring, and test authoring across multiple models.
- Run everything locally for privacy or mix in cloud models when you choose.
- Automate docs, scaffolding, CI helpers, and batch edits for personal or team projects.
- Experiment with new providers quickly and contribute improvements back to the community.

---

## Why Choose Victor?

### Potential Cost Optimization

Victor enables hybrid deployment where you can:

```
Example Cost Comparison (Theoretical):

Traditional Cloud-Only Approach:
    • Premium API usage: $200-500/developer/month
    • No control over costs
    • Vendor lock-in

Victor Hybrid Approach:
    • Local models (Ollama): FREE
    • Cloud APIs (optional): Pay only for what you use
    • Full cost control and flexibility
    • Zero vendor lock-in

Potential Savings: Use free local models for development/testing,
reserve paid APIs for production/critical tasks
```

### Capabilities

Victor provides enterprise-grade tools for:

| Capability | What Victor Offers |
|------------|-------------------|
| Code Review | Automated analysis with complexity metrics |
| Test Generation | Pytest-compatible test suites with fixtures |
| Documentation | Auto-generate docstrings and API docs |
| Refactoring | AST-based safe transformations |
| Security | Secret detection and vulnerability scanning |
| Batch Operations | Process multiple files in parallel |

---

## Quick Start

### Installation (2 minutes)

```bash
# Clone Victor
 git clone https://github.com/vjsingh1984/victor.git
 cd victor

# Install
 pip install -e ".[dev]"

# Initialize
 victor init
```

### Setup Provider (1 minute)

<details open>
<summary><b>Option 1: Free Local Model (LMStudio default)</b></summary>

```bash
# Install LMStudio (download from https://lmstudio.ai)
# Launch LMStudio → Local Server tab → start qwen2.5-coder:7b

# Configure Victor (tiered LMStudio endpoints)
cat > ~/.victor/profiles.yaml <<EOF
profiles:
  default:
    provider: lmstudio
    model: qwen2.5-coder:7b
    temperature: 0.7

providers:
  lmstudio:
    base_url:
      - http://127.0.0.1:1234      # Primary (localhost)
      # Add LAN servers if needed:
      # - http://your-lan-server:1234
    timeout: 300
EOF

# Start coding (no API key required)
victor main
```

**Cost**: FREE | **Speed**: Fast | **Privacy**: 100% local

Tip: If `~/.victor/profiles.yaml` is missing, Victor will probe the LMStudio tiered endpoints above, list their available models, and pick a preferred coder model it finds (prefers `qwen2.5-coder:7b/14b/32b`, then first available).

Connectivity check:
```bash
python scripts/check_lmstudio.py
```
Shows which LMStudio endpoint is reachable, which models it exposes, and (if VRAM is detected) recommends the most capable model that fits your GPU.
You can also cap selection via `lmstudio_max_vram_gb` in settings (default 48 GB) if you want to stay under a target budget.

Interactive command:
```
/lmstudio
```
From inside Victor’s REPL, probes LMStudio endpoints and shows a VRAM-aware recommendation.

</details>

<details>
<summary><b>Option 2: Cloud Provider (Claude/GPT/Gemini)</b></summary>

```bash
# Set API key
 export ANTHROPIC_API_KEY="your-key"

# Configure Victor
 cat > ~/.victor/profiles.yaml <<EOF
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
EOF

# Start coding
 victor main
```

**Cost**: Pay per use | **Speed**: Very fast | **Privacy**: Cloud-based

</details>

### Your First Session

```bash
$ victor main

╦  ╦╦╔═╗╔╦╗╔═╗╦═╗
╚╗╔╝║║   ║ ║ ║╠╦╝
 ╚╝ ╩╚═╝ ╩ ╚═╝╩╚═

Welcome to Victor v0.1.0
Using: Ollama (qwen2.5-coder:7b)

> Create a FastAPI app with user authentication

I'll help you create a production-ready FastAPI application...

[✓] Created project structure
[✓] Generated authentication endpoints
[✓] Added JWT token handling
[✓] Created database models
[✓] Generated tests
[✓] Set up Docker configuration

> Run the tests

[✓] Running pytest...
All 12 tests passed!

> Deploy this with docker-compose

[✓] Generated docker-compose.yml
[✓] Created Dockerfile
Ready to deploy with: docker-compose up
```

---

## Docker Deployment

Victor includes production-ready Docker configuration:

```bash
# Quick setup
 bash docker/scripts/setup.sh

# Start Ollama (local LLM)
 docker-compose --profile ollama up -d

# Or full stack (Ollama + vLLM + Jupyter)
 docker-compose --profile full up -d

# Run automated demos
 docker-compose --profile demo up
```

**Service Profiles**:
- `ollama` - Local LLM server (fast, free)
- `vllm` - High-performance inference (GPU optimized)
- `full` - Complete dev environment
- `demo` - Automated demonstrations
- `notebook` - Jupyter integration

See [docker/README.md](docker/README.md) for complete guide.

---

## Advanced Features

### Model Sharing (Save 300+ GB)

Share GGUF models between Ollama and LMStudio:

```bash
# Install Gollama
 go install github.com/sammcj/gollama@HEAD

# Link all models
 ~/go/bin/gollama -L

# Result: 27+ models, 300GB saved
```

**Before**: 681 GB (Ollama: 350 GB + LMStudio: 331 GB)
**After**: 369 GB (Ollama: 350 GB + LMStudio: 19 GB + symlinks: 0 GB)
**Saved**: 312 GB (45.8% reduction)

See [docs/guides/MODEL_SHARING.md](docs/guides/MODEL_SHARING.md) for details.

### Semantic Search & Codebase Intelligence

```python
# Configure AI-powered search
 victor --profile default

> Index this codebase with semantic search

[✓] Analyzing codebase structure...
[✓] Generating embeddings with Qwen3-Embedding:8b
[✓] Indexed 1,247 files in 3.2 minutes

> Find all authentication-related code

[✓] Found 47 matches across 12 files:
    - auth/jwt_handler.py (relevance: 98%)
    - auth/middleware.py (relevance: 95%)
    - api/endpoints/auth.py (relevance: 92%)
    ...
```

**Powered by**: Qwen3-Embedding:8b (#1 MTEB multilingual, 70.58 score)

### Batch Operations

```python
> Replace all print statements with logger.info across the project

[✓] Scanning 432 Python files...
[✓] Found 156 print statements
[✓] Preview mode - showing changes:

    file1.py:42    print("Starting")  →  logger.info("Starting")
    file2.py:88    print(f"User {u}") →  logger.info(f"User {u}")
    ... (154 more)

> Apply changes

[✓] Modified 89 files
[✓] All changes committed atomically
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Clients                               │
│            (Claude Desktop · VS Code · Others)               │
└────────────────────┬────────────────────────────────────────┘
                     │ Model Context Protocol
                     │
┌────────────────────▼────────────────────────────────────────┐
│                 Victor Terminal UI                           │
│  • Interactive REPL    • Rich Formatting                    │
│  • Streaming Responses • Command History                    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Agent Orchestrator                              │
│  • Conversation Management  • Context Tracking              │
│  • Tool Execution          • Semantic Search                │
│  • Transaction Management  • MCP Server                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           Unified Provider Interface                         │
│  • Format Normalization    • Streaming Support              │
│  • Tool Call Translation   • Error Handling                 │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬──────────┬────────────┐
        ▼            ▼            ▼          ▼            ▼
    ┌───────┐  ┌─────────┐  ┌────────┐  ┌───────┐  ┌────────┐
    │Claude │  │  GPT-4  │  │ Gemini │  │Ollama │  │ vLLM   │
    └───────┘  └─────────┘  └────────┘  └───────┘  └────────┘
                     │
        ┌────────────┼────────────┬──────────┬────────────┐
        ▼            ▼            ▼          ▼            ▼
    ┌───────┐  ┌─────────┐  ┌────────┐  ┌───────┐  ┌────────┐
    │  Git  │  │Database │  │ Docker │  │  HTTP │  │Security│
    │ Tools │  │  Tools  │  │  Tools │  │ Tools │  │  Scan  │
    └───────┘  └─────────┘  └────────┘  └───────┘  └────────┘
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Installation and first steps |
| [User Guide](docs/USER_GUIDE.md) | Complete usage guide |
| [Tool Catalog](docs/TOOL_CATALOG.md) | All 54 tools with examples |
| [Model Comparison](docs/MODEL_COMPARISON.md) | Ollama model benchmarks |
| [Docker Deployment](docker/README.md) | Container deployment guide |
| [Air-Gapped Mode](docs/embeddings/AIRGAPPED.md) | Offline operation |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Contributing and development |
| [Full Documentation Index](docs/README.md) | All documentation |

---

## Roadmap

This roadmap provides a high-level overview of the project's future direction. For a detailed breakdown of the current implementation and action items, please see our **[Codebase Analysis Report](docs/CODEBASE_ANALYSIS_REPORT.md)**.

### Completed

- [x] Universal provider abstraction with 6+ LLMs
- [x] 54 enterprise-grade tools (consolidated from 86)
- [x] MCP protocol (server + client)
- [x] Multi-file editing with transactions
- [x] Semantic search & codebase indexing
- [x] Docker deployment (production-ready)
- [x] Air-gapped mode for enterprise
- [x] Batch processing engine
- [x] Code review & secret scanning (regex-based)
- [x] Test runner & GitHub Actions CI/CD
- [x] Tiered caching system
- [x] Workspace snapshots & auto-commit
- [x] Browser automation (Playwright)

### In Progress

- [ ] Comprehensive test coverage (90%+ target)
- [ ] Performance profiling & optimization
- [ ] Additional providers (Azure OpenAI, Bedrock)
- [ ] GitLab CI & CircleCI pipeline generation

### Planned

- [ ] Test generation (pytest scaffolding)
- [ ] Coverage analysis integration
- [ ] Dependency vulnerability scanning
- [ ] VS Code extension
- [ ] JetBrains IDE plugin
- [ ] Web UI (optional)
- [ ] Multi-agent collaboration
- [ ] Plugin marketplace

---

## Contributing

We welcome contributions! Here's how to get started:

See `CONTRIBUTING.md` for detailed guidelines and `SUPPORT.md` for how to raise questions or issues.

```bash
# Fork and clone
git clone https://github.com/yourusername/victor.git
cd victor

# Create virtual environment
 python -m venv venv
 source venv/bin/activate

# Install dev dependencies
 pip install -e ".[dev]"

# Run tests
 pytest

# Submit PR
 git checkout -b feature/amazing-feature
 git commit -m "Add amazing feature"
 git push origin feature/amazing-feature
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Project Stats

<div align="center">

![Lines of Code](https://img.shields.io/tokei/lines/github/vjsingh1984/victor?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/vjsingh1984/victor?style=for-the-badge)
![Issues](https://img.shields.io/github/issues/vjsingh1984/victor?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/vjsingh1984/victor?style=for-the-badge)

</div>

---

## Community & Support

<div align="center">

[![GitHub Issues](https://img.shields.io/badge/Issues-GitHub-red?style=for-the-badge&logo=github)](https://github.com/vjsingh1984/victor/issues)
[![Discussions](https://img.shields.io/badge/Discussions-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/vjsingh1984/victor/discussions)
[![Documentation](https://img.shields.io/badge/Docs-Read-green?style=for-the-badge&logo=readthedocs)](https://github.com/vjsingh1984/victor/docs)

</div>

- **Bug Reports**: [GitHub Issues](https://github.com/vjsingh1984/victor/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)
- **Questions**: [GitHub Discussions Q&A](https://github.com/vjsingh1984/victor/discussions/categories/q-a)

---

## License

Victor is open source software licensed under the **Apache License 2.0**.

See [LICENSE](LICENSE) for full text.

---

<div align="center">

### Star Us on GitHub

If Victor helps you code faster, please consider starring the repository.
It helps others discover this project and motivates us to keep improving.

[![GitHub stars](https://img.shields.io/github/stars/vjsingh1984/victor?style=social)](https://github.com/vjsingh1984/victor/stargazers)

---

**Made with care by developers, for developers**

[Get Started](#quick-start) • [Documentation](#documentation) • [Contribute](#contributing)

</div>
