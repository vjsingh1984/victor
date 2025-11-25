# Victor

> A universal terminal-based AI coding assistant that works with any LLM provider

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Victor is a powerful terminal-based AI coding assistant that provides a **unified interface** for working with multiple LLM providers. Whether you're using frontier models like Claude, GPT-4, and Gemini, or running open-source models locally via Ollama, LMStudio, or vLLM, Victor has you covered.

### Key Features

- **Universal Provider Support**: Seamlessly switch between Anthropic Claude, OpenAI GPT, Google Gemini, and local models
- **Cost-Effective Development**: Use free local models (Ollama) for development and testing
- **Enterprise-Grade Tools**: 20+ production-ready tools for professional development workflows
- **Batch Processing**: Parallel multi-file operations with search, replace, and analysis
- **Code Refactoring**: AST-based safe transformations (rename, extract, inline, organize)
- **Test Generation**: Automated pytest-compatible test suite creation with fixtures
- **CI/CD Automation**: Generate and validate GitHub Actions, GitLab CI, CircleCI pipelines
- **Code Quality**: Automated reviews with complexity analysis, best practices, security checks
- **Security First**: Secret detection (12+ patterns), vulnerability scanning, dependency auditing
- **Caching System**: Tiered caching (memory + disk) for cost savings and performance
- **MCP Protocol Support**: Full Model Context Protocol server and client implementation
- **Semantic Search**: AI-powered codebase indexing with context-aware search
- **Multi-File Editing**: Transaction-based atomic edits across multiple files with rollback
- **Advanced Tool Suite**: Database, Docker, HTTP/API, Git, web search, and more
- **Rich Terminal UI**: Beautiful, interactive terminal experience with streaming responses
- **Extensible Architecture**: Easy to add new providers, tools, and capabilities
- **Type-Safe**: Built with Pydantic for robust type checking and validation

## Supported Providers

### Frontier Models
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **xAI**: Grok, Grok with Vision

### Local/Self-Hosted Models
- **Ollama**: Any model from the Ollama library (Llama 3, Qwen, CodeLlama, etc.)
- **LMStudio**: Any model compatible with LMStudio
- **vLLM**: High-performance inference server

## Installation

### From PyPI (Coming Soon)

```bash
pip install victor
```

### From Source

```bash
git clone https://github.com/vijaysingh/victor.git
cd victor
pip install -e ".[dev]"
```

## Quick Start

### 1. Set Up API Keys

Create a `.env` file or export environment variables:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export XAI_API_KEY="your-key-here"
```

### 2. Configure Provider Profile

Create `~/.victor/profiles.yaml`:

```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7

  claude:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

  ollama:
    base_url: http://localhost:11434
```

### 3. Start Coding

```bash
# Use default profile (Ollama)
victor

# Use specific profile
victor --profile claude

# One-shot command
victor "Write a Python function to calculate Fibonacci numbers"

# List available providers
victor providers

# List configured profiles
victor profiles

# List Ollama models
victor models

# Test a provider
victor test-provider ollama
```

## Usage Examples

### Interactive REPL Mode

```bash
$ victor

Welcome to Victor v0.1.0
Using: Ollama (qwen2.5-coder:7b)

> Create a FastAPI endpoint for user authentication

I'll help you create a FastAPI authentication endpoint...

[Agent creates files, writes code, explains changes]

> Add tests for the endpoint

[Agent writes tests using pytest]
```

### Tool Calling

Victor has access to a comprehensive suite of tools:

#### Core Tools
- **File Operations**: Read, write, edit files with transaction support
- **Multi-File Editor**: Atomic edits across multiple files with rollback
- **Bash Commands**: Execute shell commands safely
- **Git Operations**: AI-powered git with smart commits, staging, branching

#### Advanced Tools
- **Database**: Query SQLite, PostgreSQL, MySQL, SQL Server with safety controls
- **Docker**: Container and image management (list, run, stop, logs, stats)
- **HTTP/API Testing**: Make requests, test endpoints, validate responses
- **Web Search**: Fetch documentation and online resources
- **Semantic Search**: AI-powered codebase indexing and context-aware search

#### Enterprise Tools
- **Code Review**: Automated quality analysis, complexity metrics, security checks, code smells
- **Security Scanner**: Secret detection (12+ patterns), vulnerability scanning, dependency auditing
- **Project Scaffolding**: Generate production-ready project templates (FastAPI, Flask, React, CLI, Microservice)
- **Batch Processing**: Multi-file parallel operations (search, replace, analyze) with dry-run mode
- **Refactoring**: Safe code transformations (rename, extract function, inline variable, organize imports)
- **Testing**: Automated test generation (pytest-compatible, fixtures, coverage analysis)
- **CI/CD Integration**: Generate and validate pipelines (GitHub Actions, GitLab CI, CircleCI)
- **Caching System**: Tiered caching (memory + disk) for LLM responses and embeddings

#### MCP Integration
- **MCP Server**: Expose Victor's tools to Claude Desktop and other MCP clients
- **MCP Client**: Connect to external MCP servers and use their tools

### Switching Models On-The-Fly

```python
> use claude-sonnet-4-5
Switched to Anthropic Claude Sonnet 4.5

> use ollama:llama3:8b
Switched to Ollama (llama3:8b)
```

### Enterprise Workflow Examples

#### Batch Processing
```python
> Search for all TODO comments across the codebase
[Agent uses batch_process tool to search across all Python files]
Found in 15 files (47 matches)

> Replace all print statements with logger.info in preview mode
[Agent shows preview of changes across files]

> Apply the changes
[Agent applies changes to all files]
```

#### Code Refactoring
```python
> Rename the function 'process' to 'process_user_data' across the project
[Agent uses refactor tool with AST analysis]
Renamed in 8 locations across 3 files

> Organize imports in all Python files
[Agent organizes imports following PEP 8]
Organized imports in 12 files
```

#### Test Generation
```python
> Generate pytest tests for the ShoppingCart class
[Agent analyzes code and generates comprehensive test suite]
Generated test file: tests/test_cart.py
  • test_add_item()
  • test_remove_item()
  • test_get_total()
  • Edge cases and error handling
```

#### CI/CD Setup
```python
> Create GitHub Actions workflow for testing and deployment
[Agent generates workflow files]
Created workflows:
  • .github/workflows/test.yml (Python 3.10, 3.11, 3.12 matrix)
  • .github/workflows/release.yml (PyPI publishing)
All configurations validated ✅
```

## Architecture

```
┌──────────────────────────────────────────────┐
│   MCP Clients (Claude Desktop, VS Code)      │
└───────────────────┬──────────────────────────┘
                    │ MCP Protocol
                    │
┌───────────────────▼──────────────────────────┐
│          Terminal Interface (Rich)           │
│        • Interactive REPL                    │
│        • Streaming Responses                 │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│           Agent Orchestrator                 │
│   • Conversation Management                  │
│   • Context & Semantic Search                │
│   • Tool Execution & MCP Server              │
│   • Multi-File Transaction Manager           │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│      Unified Provider Interface              │
│   • Request/Response Normalization           │
│   • Streaming Support                        │
│   • Tool Call Translation                    │
└───────────────────┬──────────────────────────┘
                    │
        ┌───────────┼──────────┬───────────┐
        │           │          │           │
    ┌───▼───┐  ┌───▼───┐  ┌───▼────┐  ┌──▼─────┐
    │Claude │  │  GPT  │  │ Gemini │  │ Ollama │
    └───────┘  └───────┘  └────────┘  └────────┘
                    │
        ┌───────────┴──────────┬────────────────┐
        │                      │                │
    ┌───▼────┐          ┌─────▼─────┐    ┌────▼────┐
    │Database│          │  Docker   │    │  HTTP   │
    │ Tools  │          │   Tools   │    │  Tools  │
    └────────┘          └───────────┘    └─────────┘
```

## Configuration

### Provider Configuration

See [docs/configuration.md](docs/configuration.md) for detailed configuration options.

### Custom Tools

Create custom tools by extending the `BaseTool` class:

```python
from victor.tools.base import BaseTool

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "What my tool does"

    async def execute(self, **kwargs):
        # Your implementation
        return result
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/vijaysingh/victor.git
cd victor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/unit/test_providers.py

# Integration tests (requires Ollama running)
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black victor tests

# Lint
ruff check victor tests

# Type check
mypy victor
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- New provider integrations
- Additional tools and capabilities
- Documentation improvements
- Bug fixes and performance enhancements
- Test coverage

## Roadmap

### Completed ✅
- [x] Core provider abstraction with unified interface
- [x] Ollama, Anthropic, OpenAI, Google, xAI integration
- [x] Comprehensive tool system (20+ tools)
- [x] MCP protocol server and client
- [x] Multi-file editing with transactions
- [x] Advanced git integration with AI commits
- [x] Semantic search and codebase indexing
- [x] Database tool (SQLite, PostgreSQL, MySQL, SQL Server)
- [x] Docker management tool
- [x] HTTP/API testing tool
- [x] Web search integration
- [x] Enterprise code review and security scanning
- [x] Project scaffolding with 5+ templates
- [x] Batch processing for multi-file operations
- [x] Code refactoring tool (AST-based transformations)
- [x] Automated test generation (pytest-compatible)
- [x] CI/CD pipeline generation and validation
- [x] Tiered caching system (memory + disk)

### In Progress 🚧
- [ ] Comprehensive test coverage (target: 90%+)
- [ ] Performance profiling and optimization
- [ ] Additional provider integrations (Anthropic Bedrock, Azure OpenAI)

### Planned 📋
- [ ] IDE Extensions (VS Code, JetBrains)
- [ ] Multi-agent collaboration system
- [ ] Web UI (optional)
- [ ] Plugin marketplace
- [ ] Community templates and workflows

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- Inspired by [Claude Code](https://github.com/anthropics/claude-code)
- Built with [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
- Uses [Model Context Protocol](https://modelcontextprotocol.io/)

## Support

- **Issues**: [GitHub Issues](https://github.com/vijaysingh/victor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vijaysingh/victor/discussions)
- **Documentation**: [Full Documentation](https://github.com/vijaysingh/victor/docs)

---

Made with ❤️ by the open source community
