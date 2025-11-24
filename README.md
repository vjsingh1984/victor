# CodingAgent

> A universal terminal-based coding agent that works with any LLM provider

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

CodingAgent is a powerful terminal-based AI coding assistant that provides a **unified interface** for working with multiple LLM providers. Whether you're using frontier models like Claude, GPT-4, and Gemini, or running open-source models locally via Ollama, LMStudio, or vLLM, CodingAgent has you covered.

### Key Features

- **Universal Provider Support**: Seamlessly switch between Anthropic Claude, OpenAI GPT, Google Gemini, and local models
- **Cost-Effective Development**: Use free local models (Ollama) for development and testing
- **Unified Tool Calling**: Standardized tool interface across all providers with MCP support
- **Rich Terminal UI**: Beautiful, interactive terminal experience with streaming responses
- **Extensible Architecture**: Easy to add new providers, tools, and capabilities
- **Type-Safe**: Built with Pydantic for robust type checking and validation

## Supported Providers

### Frontier Models
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash

### Local/Self-Hosted Models
- **Ollama**: Any model from the Ollama library (Llama 3, Qwen, CodeLlama, etc.)
- **LMStudio**: Any model compatible with LMStudio
- **vLLM**: High-performance inference server

## Installation

### From PyPI (Coming Soon)

```bash
pip install codingagent
```

### From Source

```bash
git clone https://github.com/vijaysingh/codingagent.git
cd codingagent
pip install -e ".[dev]"
```

## Quick Start

### 1. Set Up API Keys

Create a `.env` file or export environment variables:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

### 2. Configure Provider Profile

Create `~/.codingagent/profiles.yaml`:

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
codingagent

# Use specific profile
codingagent --profile claude

# One-shot command
codingagent "Write a Python function to calculate Fibonacci numbers"
```

## Usage Examples

### Interactive REPL Mode

```bash
$ codingagent

Welcome to CodingAgent v0.1.0
Using: Ollama (qwen2.5-coder:7b)

> Create a FastAPI endpoint for user authentication

I'll help you create a FastAPI authentication endpoint...

[Agent creates files, writes code, explains changes]

> Add tests for the endpoint

[Agent writes tests using pytest]
```

### Tool Calling

The agent has access to various tools:

- **File Operations**: Read, write, edit files
- **Bash Commands**: Execute shell commands
- **Code Analysis**: Parse and analyze code structure
- **Web Search**: Fetch documentation and resources
- **Git Operations**: Commit, branch, diff

### Switching Models On-The-Fly

```python
> use claude-sonnet-4-5
Switched to Anthropic Claude Sonnet 4.5

> use ollama:llama3:8b
Switched to Ollama (llama3:8b)
```

## Architecture

```
┌─────────────────────────────────────┐
│     Terminal Interface (Rich)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Agent Orchestrator             │
│  • Conversation Management          │
│  • Context Handling                 │
│  • Tool Execution                   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Unified Provider Interface        │
│  • Request/Response Normalization   │
│  • Streaming Support                │
│  • Tool Call Translation            │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┬───────────┐
    │          │          │           │
┌───▼───┐  ┌──▼───┐  ┌──▼────┐  ┌───▼────┐
│Claude │  │ GPT  │  │Gemini │  │ Ollama │
└───────┘  └──────┘  └───────┘  └────────┘
```

## Configuration

### Provider Configuration

See [docs/configuration.md](docs/configuration.md) for detailed configuration options.

### Custom Tools

Create custom tools by extending the `BaseTool` class:

```python
from codingagent.tools.base import BaseTool

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
git clone https://github.com/vijaysingh/codingagent.git
cd codingagent

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
black codingagent tests

# Lint
ruff check codingagent tests

# Type check
mypy codingagent
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

- [x] Core provider abstraction
- [x] Ollama integration
- [x] Basic tool system
- [ ] Full Anthropic/OpenAI/Google support
- [ ] MCP server integration
- [ ] Multi-agent collaboration
- [ ] Context caching optimization
- [ ] Docker support
- [ ] Web UI (optional)

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- Inspired by [Claude Code](https://github.com/anthropics/claude-code)
- Built with [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
- Uses [Model Context Protocol](https://modelcontextprotocol.io/)

## Support

- **Issues**: [GitHub Issues](https://github.com/vijaysingh/codingagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vijaysingh/codingagent/discussions)
- **Documentation**: [Full Documentation](https://github.com/vijaysingh/codingagent/docs)

---

Made with ❤️ by the open source community
