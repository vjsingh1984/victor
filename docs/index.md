<div align="center">

# Victor Documentation

**Provider-agnostic AI coding assistant with multi-provider support, workflow orchestration, and extensible tool system**

[![PyPI version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/vjsingh1984/victor/actions/workflows/test.yml/badge.svg)](https://github.com/vjsingh1984/victor/actions/workflows/test.yml)

</div>

---

## Welcome! 👋

Victor is an open-source AI coding assistant supporting **21 LLM providers** with **55 specialized tools** across **5 domain verticals**. Whether you need local development with Ollama, cloud capabilities with Claude/GPT, or enterprise deployments with Azure/AWS, Victor provides a unified interface with powerful features like semantic codebase search, YAML-first workflows, multi-agent coordination, and air-gapped mode support.

## What is Victor?

Victor is a **provider-agnostic AI coding assistant** that gives you the freedom to choose between local and cloud LLM providers through a single, consistent CLI/TUI interface. Built with extensibility in mind, Victor's vertical architecture lets you specialize workflows for coding, DevOps, RAG, data analysis, and research tasks.

### Key Highlights

- 🌐 **21 LLM Providers** - Switch between Anthropic, OpenAI, Google, Azure, AWS Bedrock, Ollama, LM Studio, vLLM, and more
- 🛠️ **55 Specialized Tools** - Across 5 domain verticals (Coding, DevOps, RAG, Data Analysis, Research)
- 🔄 **YAML-First Workflows** - Define multi-step automation with StateGraph DSL and checkpointing
- 🔒 **Air-Gapped Mode** - Full functionality with local models for secure, offline environments
- 🔍 **Semantic Codebase Search** - Context-aware code understanding with Tree-sitter and embeddings
- 🤖 **Multi-Agent Teams** - Coordinate specialized agents for complex tasks
- 🔌 **MCP Protocol Support** - Model Context Protocol for IDE integration
- ⚡ **Provider Switching** - Change models mid-conversation without losing context

## Quick Navigation

### 🚀 Getting Started
New to Victor? Start here!

- **[Installation](getting-started/installation.md)** - Install Victor via pip, pipx, or Docker
- **[Quick Start](getting-started/quickstart.md)** - Get up and running in 5 minutes
- **[Configuration](getting-started/configuration.md)** - Configure providers, profiles, and settings
- **[First Run](getting-started/first-run.md)** - Initialize your Victor environment

### 📖 User Guide
Master Victor's features and capabilities.

- **[CLI Commands](reference/cli-commands.md)** - Complete command reference
- **[User Guide Index](user-guide/index.md)** - Comprehensive user documentation
- **[Providers](user-guide/providers.md)** - LLM provider configuration and comparison
- **[Tools](user-guide/tools.md)** - Available tools and usage examples
- **[Workflows](user-guide/workflows.md)** - Create and manage YAML workflows
- **[TUI Mode](user-guide/tui-mode.md)** - Terminal user interface guide
- **[Session Management](user-guide/session-management.md)** - Manage conversation sessions
- **[Troubleshooting](user-guide/troubleshooting.md)** - Common issues and solutions

### 📚 Tutorials
Learn by doing with hands-on tutorials.

- **[Build a Custom Tool](tutorials/build-custom-tool.md)** - Create your own tools
- **[Create a Workflow](tutorials/create-workflow.md)** - Design multi-step automation
- **[Integrate a Provider](tutorials/integrate-provider.md)** - Add a new LLM provider

### 🔧 Reference
In-depth technical documentation.

- **[Configuration Options](reference/configuration-options.md)** - Complete settings reference
- **[Environment Variables](reference/environment-variables.md)** - All environment variables
- **[CLI Commands](reference/cli-commands.md)** - Command-line interface reference
- **[API Reference](api-reference/)** - Python API documentation
  - [Protocols](api-reference/protocols.md) - Core protocol definitions
  - [Providers](api-reference/providers.md) - Provider API
  - [Tools](api-reference/tools.md) - Tool API
  - [Workflows](api-reference/workflows.md) - Workflow API

### 🏗️ Development
Contributor and architect documentation.

- **[Development Setup](development/setup.md)** - Development environment setup
- **[Code Style](development/code-style.md)** - Coding standards and conventions
- **[Testing](development/testing.md)** - Testing strategy and guidelines
- **[Architecture Overview](architecture/overview.md)** - System architecture and design
- **[Extending Victor](development/index.md)** - Plugin development and extensions
  - [Plugins](development/extending/plugins.md) - Build plugins
  - [Verticals](development/extending/verticals.md) - Create domain verticals
- **[Releasing](development/releasing/publishing.md)** - Release and publishing process

### 🎯 Verticals
Domain-specific documentation.

- **[Coding](verticals/coding.md)** - AST analysis, code review, test generation
- **[DevOps](verticals/devops.md)** - CI/CD, Docker, Terraform, infrastructure
- **[RAG](verticals/rag.md)** - Document ingestion, vector search, retrieval
- **[Data Analysis](verticals/data-analysis.md)** - Pandas, visualization, statistics
- **[Research](verticals/research.md)** - Web search, citations, synthesis

## Project Information

### Version
**Current Version:** v0.5.0 (see [CHANGELOG](../CHANGELOG.md))

### Links
- **[GitHub Repository](https://github.com/vjsingh1984/victor)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/victor-ai/)** - Install via pip
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute
- **[Code of Conduct](../CODE_OF_CONDUCT.md)** - Community guidelines
- **[Security Policy](../SECURITY.md)** - Security reporting and policies

### License
Apache License 2.0 - see [LICENSE](../LICENSE) for details.

---

## Quick Start

### Choose Your Path

```bash
# Local model (privacy, offline, free tier)
pipx install victor-ai
ollama pull qwen2.5-coder:7b
victor chat "Hello"

# Cloud model (max capability)
pipx install victor-ai
export ANTHROPIC_API_KEY=sk-...
victor chat --provider anthropic

# Docker (isolated environment)
docker pull ghcr.io/vjsingh1984/victor:latest
docker run -it -v ~/.victor:/root/.victor ghcr.io/vjsingh1984/victor:latest
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENTS                                       │
│  CLI/TUI  │  VS Code (HTTP)  │  MCP Server  │  API Server       │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AGENT ORCHESTRATOR (Facade)                      │
│  Delegates to: ConversationController, ToolPipeline,            │
│  StreamingController, ProviderManager, ToolRegistrar            │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌───────────┬─────────────┬───────────────┬───────────────────────┐
│ PROVIDERS │   TOOLS     │  WORKFLOWS    │  VERTICALS            │
│  21       │   55        │  StateGraph   │  Coding/DevOps/RAG/   │
│           │             │  + YAML       │  DataAnalysis/Research│
└───────────┴─────────────┴───────────────┴───────────────────────┘
```

### Key Concepts

- **Provider Agnosticism** - Switch between 21 LLM providers seamlessly
- **Vertical Architecture** - Self-contained domains with specialized tools
- **YAML Workflows** - Declarative workflow definitions with Python escape hatches
- **StateGraph DSL** - LangGraph-compatible workflow API for complex automation
- **Multi-Agent Coordination** - Team formations with specialized roles
- **SOLID Compliance** - Clean architecture following SOLID principles

## Support & Community

- **Documentation** - You are here! Explore the sections above
- **GitHub Issues** - [Report bugs or request features](https://github.com/vjsingh1984/victor/issues)
- **Discussions** - [Join community discussions](https://github.com/vjsingh1984/victor/discussions)
- **Contributing** - See [Contributing Guide](../CONTRIBUTING.md) for details

---

**Built with ❤️ by the Victor community**
