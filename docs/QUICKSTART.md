<div align="center">

# Victor AI Quick Start Guide

**Get up and running with Victor AI in 5 minutes**

[![Version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Welcome to Victor AI!

Victor AI is an open-source AI coding assistant supporting **21 LLM providers** with **55 specialized tools** across **5 domain verticals**. This guide will get you started in just 5 minutes.

### What You'll Learn

- How to install Victor AI
- How to configure your first provider
- How to run your first AI coding session
- Where to find more resources

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10 or higher** installed
- **pip or pipx** for package installation
- **An LLM provider API key** (or local model like Ollama)

### Check Python Version

```bash
python --version
# Should show Python 3.10 or higher
```

---

## Installation (1 minute)

Choose your installation method:

### Option 1: Using pipx (Recommended)

```bash
pipx install victor-ai
```

### Option 2: Using pip

```bash
pip install victor-ai
```

### Option 3: Using Docker

```bash
docker pull ghcr.io/vjsingh1984/victor:latest
```

### Verify Installation

```bash
victor --version
# Should show version 0.5.1 or higher
```

---

## Choose Your Provider (1 minute)

Victor supports 21 LLM providers. Choose the one that fits your needs:

### Local Providers (Free, Privacy-Focused)

- **Ollama** - Run models locally (recommended for beginners)
- **LM Studio** - Local model management
- **vLLM** - High-performance local inference

### Cloud Providers (Maximum Capability)

- **Anthropic** - Claude models (Sonnet 4.5, Opus 4.5)
- **OpenAI** - GPT-4, GPT-4 Turbo
- **Google** - Gemini models
- **Azure OpenAI** - Enterprise Azure deployment
- **AWS Bedrock** - AWS managed models

### See All Providers

[View all 21 supported providers](./user-guide/providers.md)

---

## Configuration (2 minutes)

### For Local Providers (Ollama)

1. **Install Ollama** (if not already installed):

   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull a model**:

   ```bash
   ollama pull qwen2.5-coder:7b
   # Or: ollama pull codellama:7b
   ```

3. **Run Victor**:

   ```bash
   victor chat --provider ollama --model qwen2.5-coder:7b
   ```

### For Cloud Providers (Anthropic)

1. **Set your API key**:

   ```bash
   export ANTHROPIC_API_KEY=sk-your-api-key-here
   ```

2. **Run Victor**:

   ```bash
   victor chat --provider anthropic --model claude-sonnet-4-5
   ```

### For Other Providers

[View provider-specific setup instructions](./getting-started/configuration.md)

---

## Your First Session (1 minute)

### Start Chatting

```bash
# With default settings
victor chat

# With specific provider and model
victor chat --provider anthropic --model claude-sonnet-4-5

# In TUI mode (interactive terminal UI)
victor chat --tui

# In CLI mode (command-line interface)
victor chat --no-tui
```

### Try These Commands

```bash
# Ask a question
victor chat "Explain how async/await works in Python"

# Generate code
victor chat "Create a FastAPI endpoint for user authentication"

# Analyze code
victor chat "Review this file for security issues" --include-file app.py

# Get help
victor chat --help
```

---

## What's Next?

### Learn the Basics

- [Basic Usage Guide](./getting-started/basic-usage.md) - Learn fundamental commands
- [Configuration Guide](./getting-started/configuration.md) - Advanced configuration
- [User Guide](./user-guide/index.md) - Comprehensive user documentation

### Explore Features

- [Tools Reference](./user-guide/tools.md) - 55 specialized tools
- [Workflows Guide](./user-guide/workflows.md) - YAML workflow automation
- [Multi-Agent Teams](./AGENT_SWARMING_GUIDE.md) - Coordinate multiple AI agents
- [Advanced Features](./ADVANCED_FEATURES.md) - Advanced capabilities

### Get Help

- [FAQ](./user-guide/faq.md) - Frequently asked questions
- [Troubleshooting](./TROUBLESHOOTING.md) - Common issues and solutions
- [GitHub Issues](https://github.com/vjsingh1984/victor/issues) - Report bugs

### Join the Community

- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions) - Community discussions
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute

---

## Quick Reference

### Essential Commands

| Command | Description |
|---------|-------------|
| `victor chat` | Start interactive chat session |
| `victor chat --no-tui` | Use CLI mode instead of TUI |
| `victor chat --provider <name>` | Specify LLM provider |
| `victor chat --model <name>` | Specify model |
| `victor init` | Initialize Victor configuration |
| `victor --version` | Show version information |
| `victor --help` | Show help message |

### Common Options

| Option | Description |
|---------|-------------|
| `--provider` | LLM provider (anthropic, openai, ollama, etc.) |
| `--model` | Model name (claude-sonnet-4-5, gpt-4, etc.) |
| `--tui` | Enable terminal UI (default) |
| `--no-tui` | Disable terminal UI, use CLI |
| `--session` | Session name for persistence |
| `--profile` | Configuration profile |

### Popular Models

| Provider | Model | Description |
|----------|-------|-------------|
| anthropic | claude-sonnet-4-5 | Balanced performance/cost |
| anthropic | claude-opus-4-5 | Maximum capability |
| openai | gpt-4-turbo | High performance |
| openai | gpt-4o | Latest GPT-4 |
| ollama | qwen2.5-coder:7b | Local coding model |
| ollama | codellama:7b | Local code model |

---

## Tips for Success

### 1. Start with a Good Model

For beginners:
- **Local**: Use `qwen2.5-coder:7b` via Ollama (free, capable)
- **Cloud**: Use `claude-sonnet-4-5` via Anthropic (balanced)

### 2. Use TUI Mode for Interactive Sessions

```bash
victor chat --tui
```

TUI mode provides a rich terminal interface with syntax highlighting and multi-line input.

### 3. Use CLI Mode for Quick Questions

```bash
victor chat --no-tui "Your question here"
```

CLI mode is perfect for quick, one-off questions.

### 4. Configure Your Default Provider

Create `~/.victor/config.yml`:

```yaml
providers:
  default:
    name: anthropic
    model: claude-sonnet-4-5
```

Now you can simply run `victor chat` without specifying provider/model.

### 5. Enable Session Persistence

```bash
victor chat --session my-project
```

Sessions persist conversation history across restarts.

---

## Architecture Overview

Victor uses a vertical architecture with specialized tools:

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
- **YAML Workflows** - Declarative workflow definitions
- **StateGraph DSL** - Complex automation workflows
- **Multi-Agent Teams** - Coordinate specialized agents

---

## Common Use Cases

### 1. Code Generation

```bash
victor chat "Create a Python function to validate email addresses"
```

### 2. Code Review

```bash
victor chat "Review this code for bugs and security issues" \
  --include-file app.py
```

### 3. Debugging

```bash
victor chat "Help me debug this error: TypeError in line 42" \
  --include-file app.py \
  --include-file error.log
```

### 4. Documentation

```bash
victor chat "Generate API documentation for this module" \
  --include-file mymodule.py
```

### 5. Refactoring

```bash
victor chat "Refactor this code to follow SOLID principles" \
  --include-file legacy.py
```

### 6. Test Generation

```bash
victor chat "Generate unit tests for this class" \
  --include-file user.py
```

---

## Next Steps

### For Users

- [User Guide](./user-guide/index.md) - Complete user documentation
- [Tools Reference](./user-guide/tools.md) - All 55 tools explained
- [Workflows Guide](./user-guide/workflows.md) - Automate with workflows

### For Developers

- [Contributor Quick Start](./CONTRIBUTOR_QUICKSTART.md) - Start contributing
- [Extension Development](./extensions/README.md) - Build extensions
- [API Reference](./api/README.md) - Python API documentation

### For Architects

- [Architect Quick Start](./ARCHITECT_QUICKSTART.md) - Architecture overview
- [Architecture Documentation](./architecture/README.md) - System design
- [Design Patterns](./architecture/DESIGN_PATTERNS.md) - SOLID patterns

---

## Troubleshooting

### Installation Issues

**Problem**: `pip install victor-ai` fails

**Solution**:
```bash
# Try pipx instead
pipx install victor-ai

# Or upgrade pip first
pip install --upgrade pip
pip install victor-ai
```

### Provider Connection Issues

**Problem**: Cannot connect to provider

**Solution**:
```bash
# Check API key is set
echo $ANTHROPIC_API_KEY

# Test provider explicitly
victor chat --provider anthropic --model claude-sonnet-4-5 "test"
```

### Model Not Found

**Problem**: Model name not recognized

**Solution**:
```bash
# List available models
victor chat --provider anthropic --list-models

# Use exact model name from provider
victor chat --provider anthropic --model claude-sonnet-4-5-20250114
```

### More Help

[Visit the Troubleshooting Guide](./TROUBLESHOOTING.md)

---

## Resources

### Documentation

- [Documentation Index](./INDEX.md) - Complete documentation hub
- [Getting Started](./getting-started/) - Getting started guides
- [User Guide](./user-guide/) - User documentation
- [API Reference](./api/) - API documentation

### Community

- [GitHub Repository](https://github.com/vjsingh1984/victor) - Source code
- [GitHub Issues](https://github.com/vjsingh1984/victor/issues) - Bug reports
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions) - Discussions

### Related Projects

- [Claude Code](https://claude.com/claude-code) - Anthropic's CLI tool
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow framework
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) - AST parsing

---

## Version Information

- **Current Version**: 0.5.1
- **Python Required**: 3.10+
- **License**: Apache 2.0
- **Last Updated**: January 18, 2026

---

<div align="center">

**Ready to dive deeper?**

[Read the User Guide](./user-guide/index.md) •
[Explore Tools](./user-guide/tools.md) •
[Join the Community](https://github.com/vjsingh1984/victor/discussions)

**[Back to Documentation Index](./INDEX.md)**

</div>
