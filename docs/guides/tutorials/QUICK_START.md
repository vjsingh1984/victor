# Victor AI - Quick Start Guide

Welcome to Victor AI! This guide will get you up and running in minutes.

## Table of Contents

1. [Installation](#installation)
2. [First Steps](#first-steps)
3. [Your First Project](#your-first-project)
4. [Common Workflows](#common-workflows)
5. [Next Steps](#next-steps)

## Installation

### Option 1: Basic Installation

```bash
# Install Victor AI
pip install victor-ai

# Verify installation
victor --version
```

### Option 2: Development Installation

```bash
# Clone repository
git clone https://github.com/yourusername/victor.git
cd victor

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Verify installation
victor --version
```

### Option 3: With Optional Dependencies

```bash
# Install with specific feature sets
pip install victor-ai[google]      # Google Gemini support
pip install victor-ai[api]         # FastAPI server
pip install victor-ai[checkpoints] # SQLite checkpoint persistence
pip install victor-ai[lang-all]    # All tree-sitter language grammars
pip install victor-ai[native]      # Rust-accelerated extensions
```

## First Steps

### 1. Choose Your Provider

Victor supports multiple LLM providers. Start with one that fits your needs:

#### Free & Local (Good for Getting Started)

**Ollama** (Free to use, runs offline; hardware/compute costs apply):
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull a coding model
ollama pull qwen2.5-coder:7b

# Test
victor chat --provider ollama --model qwen2.5-coder:7b
```

#### Cloud Providers (API Key Required)

**Anthropic Claude** (Best Reasoning):
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
victor chat --provider anthropic --model claude-sonnet-4-5
```

**OpenAI GPT** (Fast & Popular):
```bash
export OPENAI_API_KEY="sk-..."
victor chat --provider openai --model gpt-4
```

**Google Gemini** (Longest Context - 1M tokens):
```bash
export GOOGLE_API_KEY="..."
victor chat --provider google --model gemini-1.5-pro
```

See the [Provider Reference](../reference/providers/index.md) for the current list.

### 2. Basic Chat

Start a simple conversation:

```bash
# Interactive chat (TUI mode)
victor chat

# Or specify provider and model
victor chat --provider ollama --model qwen2.5-coder:7b

# CLI mode (no TUI)
victor chat --no-tui "Explain what this code does"
```

### 3. Your First Question

Try these example prompts:

```bash
# Code explanation
victor chat "Explain how list comprehensions work in Python"

# Code generation
victor chat "Write a function to calculate fibonacci numbers"

# Code review
victor chat "Review this file for bugs: src/main.py"

# Documentation
victor chat "Generate API documentation for my codebase"
```

## Your First Project

### Step 1: Initialize Victor

```bash
# Navigate to your project
cd /path/to/your/project

# Initialize Victor in your project
victor init

# This creates .victor/ directory with configuration
```

### Step 2: Configure Your Project

Create `.victor/config.yaml`:

```yaml
# Project-specific configuration
provider: ollama
model: qwen2.5-coder:7b
temperature: 0.7

# Tool preferences
tools:
  enabled:
    - read_file
    - write_file
    - search_code
    - run_tests
  disabled:
    - web_search  # Disable if offline

# Mode selection
mode: build  # Options: build, plan, explore

# Vertical (domain) selection
vertical: coding  # Options: coding, devops, rag, dataanalysis, research
```

### Step 3: Create Project Instructions

Create `.victor/project_context.md`:

```markdown
# My Project

## Project Overview
This is a web application for task management.

## Tech Stack
- Backend: Python, FastAPI
- Frontend: React, TypeScript
- Database: PostgreSQL

## Coding Standards
- Follow PEP 8 for Python
- Use TypeScript strict mode
- Write tests for all new features
- Document public APIs

## Development Workflow
1. Create feature branch
2. Implement changes
3. Write/update tests
4. Run tests
5. Create pull request
```

### Step 4: Start Working

```bash
# Start Victor in your project
victor chat

# Victor will load your project context automatically
# Try: "Analyze my codebase and suggest improvements"
```

## Common Workflows

### Workflow 1: Code Analysis

```bash
# Analyze entire codebase
victor chat "Analyze my codebase for:
1. Security vulnerabilities
2. Performance bottlenecks
3. Code quality issues
4. Technical debt"

# Analyze specific file
victor chat "Review src/auth.py for security issues"

# Find code patterns
victor chat "Find all database queries that aren't using parameterized queries"
```

### Workflow 2: Code Generation

```bash
# Generate new feature
victor chat "Create a REST API endpoint for user authentication with:
- POST /auth/login
- POST /auth/logout
- POST /auth/refresh
Use JWT tokens and include proper error handling"

# Generate tests
victor chat "Generate unit tests for the authentication module"

# Generate documentation
victor chat "Generate API documentation for all endpoints in src/api/"
```

### Workflow 3: Refactoring

```bash
# Refactor code
victor chat "Refactor this code to be more maintainable:
- Extract magic numbers to constants
- Improve function names
- Add type hints
- Reduce complexity"

# Apply design patterns
victor chat "Apply the Repository pattern to the database access layer"
```

### Workflow 4: Debugging

```bash
# Debug error
victor chat "I'm getting this error:
TypeError: 'NoneType' object is not subscriptable
at line 45 in src/users.py
Help me fix it"

# Find bug
victor chat "Find why the user session is expiring prematurely"

# Performance issue
victor chat "Identify why this API endpoint is slow:
GET /api/users/list"
```

### Workflow 5: Testing

```bash
# Generate tests
victor chat "Generate comprehensive tests for src/utils.py"

# Improve test coverage
victor chat "Add tests to achieve 90% coverage for src/models/"

# Debug test failures
victor chat "Fix these failing tests:
pytest tests/test_auth.py::test_login_failed"
```

### Workflow 6: Documentation

```bash
# Generate README
victor chat "Generate a comprehensive README.md for this project including:
- Project description
- Installation instructions
- Usage examples
- API documentation
- Contributing guidelines"

# Generate API docs
victor chat "Generate OpenAPI specification for all REST endpoints"

# Generate code comments
victor chat "Add docstrings to all functions in src/utils.py"
```

## Advanced Features

### Multi-Provider Strategy

Save 90% on costs by using multiple providers strategically:

```bash
# Use free local model for brainstorming
victor chat --provider ollama "Brainstorm 5 approaches to implement feature X"

# Use cheap model for implementation
victor chat --provider openai --model gpt-3.5-turbo "Implement approach 3"

# Use premium model for code review
victor chat --provider anthropic "Review the implementation for quality"
```

### Agent Modes

```bash
# BUILD mode (default) - Full edits allowed
victor chat --mode build "Implement the feature"

# PLAN mode - 2.5x exploration, sandbox only
victor chat --mode plan "Plan the architecture for this feature"

# EXPLORE mode - 3.0x exploration, no edits
victor chat --mode explore "Explore different implementation options"
```

### Profiles

Create `~/.victor/profiles.yaml`:

```yaml
profiles:
  development:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7

  production:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.3

  fast:
    provider: openai
    model: gpt-3.5-turbo
    temperature: 0.7
```

Use profiles:

```bash
victor chat --profile development
victor chat --profile production
victor chat --profile fast
```

### Workflow Automation

Create ready-to-use workflows in `examples/workflows/`:

```bash
# Code refactoring
bash examples/workflows/refactor.sh src/app.py

# Code review
bash examples/workflows/review.sh src/app.py

# Test generation
bash examples/workflows/tests.sh src/app.py
```

## Air-Gapped Mode

For offline or secure environments:

```bash
# Enable air-gapped mode
export VICTOR_AIRGAPPED=true

# Only local providers available (Ollama, LMStudio, vLLM)
# No web tools
# Local embeddings only

victor chat --provider ollama
```

## Tips & Best Practices

### 1. Start Simple

- Begin with Ollama (free, local)
- Move to cloud providers when needed
- Use profiles for different scenarios

### 2. Use Project Context

- Always run `victor init` in your project
- Create `.victor/project_context.md` with project details
- Victor will provide better, contextual responses

### 3. Be Specific

- Good: "Add input validation to the user registration endpoint"
- Bad: "Fix the code"

### 4. Iterate

- Start with high-level requests
- Gradually add more detail
- Review and refine iteratively

### 5. Use Tools Wisely

- Enable only the tools you need
- Disable expensive tools in production
- Use tool budgets to control costs

## Troubleshooting

### "Connection refused" with Ollama

```bash
# Make sure Ollama is running
ollama serve

# Check models
ollama list

# Pull a model
ollama pull qwen2.5-coder:7b
```

### API Key Issues

```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Set key if missing
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e ".[dev]"
```

### Rate Limiting

- Wait and retry
- Use Ollama instead (no rate limits)
- Upgrade your API plan

## Next Steps

1. **Explore Examples**: Check out `examples/` directory for more examples
2. **Read Documentation**: See `docs/` for comprehensive guides
3. **Create Custom Tools**: Learn to extend Victor (see `docs/tutorials/CREATING_TOOLS.md`)
4. **Build Workflows**: Automate repetitive tasks (see `docs/tutorials/CREATING_WORKFLOWS.md`)
5. **Join Community**: Get help and share ideas

## Getting Help

- **Documentation**: `docs/`
- **Examples**: `examples/`
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions and share experiences

## Cost-Saving Tips

1. **Use Ollama for development** - free to use (local compute required)
2. **Mix providers strategically** - 90% cost savings
3. **Use appropriate models** - Don't overpay for simple tasks
4. **Set tool budgets** - Control tool usage costs
5. **Use caching** - Avoid redundant API calls

Happy coding with Victor! 🚀
