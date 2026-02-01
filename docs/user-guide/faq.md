# Victor AI FAQ

**Frequently Asked Questions about Victor AI**

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation--setup)
3. [Providers & Models](#providers--models)
4. [Usage & Features](#usage--features)
5. [Performance & Cost](#performance--cost)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)
8. [Community & Support](#community--support)

---

## General Questions

### What is Victor AI?

Victor AI is an open-source AI coding assistant that supports multiple LLM providers, tools, and domain verticals (e.g., Coding, DevOps, RAG, Data Analysis, Research). It provides both local and cloud LLM support through a unified CLI and Python API.

**Key differentiators:**
- Provider agnostic - switch models without losing context
- Multi-agent teams - coordinate specialized agents
- Air-gapped mode - local/offline operation with supported providers
- Semantic codebase search - context-aware understanding
- YAML workflows - automate repetitive tasks

### Is Victor AI free?

Yes. Victor AI is:
- **Open source** - No licensing fees
- **Free to use** - With local models (compute costs apply)
- **Bring your own key** - Use your own API keys for cloud providers
- **No subscription** - Pay only for provider usage where applicable

### How is Victor different from GitHub Copilot?

| Feature | Victor AI | GitHub Copilot |
|---------|-----------|----------------|
| **Providers** | Multiple providers | OpenAI only |
| **Local Support** | Local/offline options | None |
| **Multi-Agent** | Team workflows | Single agent |
| **Provider Switching** | Yes, mid-conversation | No |
| **Extensibility** | Custom verticals | Limited |
| **Cost** | Open-source + provider costs | Subscription only |
| **Privacy** | Local-only possible with supported providers | Cloud-only |

### Can I use Victor offline?

Yes. Victor supports **air-gapped mode** with local providers:
- Use local models (Ollama, LM Studio, vLLM)
- No internet connection required
- Full tool functionality
- Data stays on your machine with local-only configurations

```bash
export VICTOR_AIRGAPPED_MODE=true
victor chat --provider ollama
```

---

## Installation & Setup

### How do I install Victor AI?

**Recommended (pipx):**
```bash
pipx install victor-ai
```

**Alternative (pip):**
```bash
pip install victor-ai
```

**Verify:**
```bash
victor --version
```

### What are the requirements?

**Minimum:**
- Python 3.10 or higher
- 4GB RAM
- 1GB disk space

**Recommended for local models:**
- Python 3.10+
- 16GB RAM (for 7B models)
- 20GB disk space
- GPU (optional, but faster)

### How do I update Victor?

```bash
pipx upgrade victor-ai
# or
pip install --upgrade victor-ai
```

### Installation fails with "Permission denied"

**Solution:**
```bash
# Use user directory
pip install --user victor-ai

# Or use virtual environment
python -m venv .venv
source .venv/bin/activate
pip install victor-ai
```

---

## Providers & Models

### Which provider should I use?

**For beginners:**
- **Ollama** - Free, easy to set up, good models
- **Claude Sonnet** - Balanced capability/cost
- **GPT-4o** - Fast, capable

**For specific tasks:**
- **Brainstorming**: Ollama (free)
- **Implementation**: GPT-4o (fast)
- **Review**: Claude Sonnet (thorough)
- **Complex analysis**: Claude Opus (best)

### How do I set up Anthropic Claude?

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-your-key-here

# Use Victor
victor chat --provider anthropic --model claude-sonnet-4-5
```

Get API key: https://console.anthropic.com/

### How do I set up OpenAI GPT?

```bash
# Set API key
export OPENAI_API_KEY=sk-your-key-here

# Use Victor
victor chat --provider openai --model gpt-4o
```

Get API key: https://platform.openai.com/api-keys

### How do I set up Ollama (local)?

```bash
# Install Ollama
brew install ollama  # macOS
# or
curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Pull a model
ollama pull qwen2.5-coder:7b

# Use Victor
victor chat --provider ollama
```

### Can I switch providers mid-conversation?

Yes! Victor's unique feature:
```bash
# Start with Ollama
victor chat --provider ollama "Brainstorm ideas"

# Switch to Claude (in TUI mode)
/provider anthropic --model claude-sonnet-4-5
"Implement the first idea"

# Context is preserved!
```

### Which models are supported?

Models depend on your configured providers. Examples include:
- **Anthropic**: Claude Sonnet 4, Opus, Haiku
- **OpenAI**: GPT-4o, GPT-4, o1-preview
- **Google**: Gemini 2.0, Gemini 1.5 Pro
- **Local (Ollama)**: Qwen2.5, DeepSeek, CodeLlama, etc.
- And more...

[Full provider list](reference/providers/index.md)

---

## Usage & Features

### How do I start using Victor?

```bash
# Interactive TUI mode
victor

# CLI mode
victor chat "Your question here"

# With specific provider
victor chat --provider anthropic "Explain async/await"
```

### What's the difference between TUI and CLI mode?

**TUI Mode** (`victor chat`):
- Interactive terminal UI
- Syntax highlighting
- Multi-line input
- Session history
- Rich formatting

**CLI Mode** (`victor chat --no-tui`):
- Quick, one-off queries
- Script-friendly
- Pipe-able output
- Minimal overhead

### How do I save conversations?

```bash
# In TUI mode
/save "My Session Title"

# From CLI
victor chat --session my-session-name

# List sessions
victor sessions

# Resume session
victor resume my-session-name
```

### What are agent modes?

Three modes for different use cases:

| Mode | Description | Use When |
|------|-------------|----------|
| **build** | Full capabilities, can edit files | Implementing features |
| **plan** | Analysis only, no edits | Architecture review |
| **explore** | Deep exploration, no edits | Learning codebase |

```bash
victor chat --mode plan "Analyze this architecture"
```

### What tools does Victor have?

Tools across multiple domains (see the catalog for the current list):

**Coding:**
- File operations (read, write, search)
- AST parsing and analysis
- Code review
- Test generation

**DevOps:**
- Git operations
- Docker management
- CI/CD configuration

**Research:**
- Web search
- Document analysis
- Citation generation

**Data Analysis:**
- Pandas operations
- Visualization
- Statistics

**RAG:**
- Vector search
- Document indexing
- Semantic retrieval

[List all tools](user-guide/tools.md)

### How do I use workflows?

```bash
# List available workflows
victor workflow list

# Run a workflow
victor workflow run code-review --file myfile.py

# Create custom workflow
victor workflow create my-workflow.yaml
```

[Learn more about workflows](user-guide/workflows.md)

### What are multi-agent teams?

Coordinate multiple specialized agents:
```bash
# Create team
victor team create review-team.yaml

# Run team
victor team run review-team "Review this PR"
```

[Learn more about teams](guides/MULTI_AGENT_TEAMS.md)

---

## Performance & Cost

### How much does Victor cost?

**Victor itself**: Free (open source)

**Usage costs**:
- **Local models**: $0 (Ollama, LM Studio)
- **Cloud models**: Pay for API usage only
  - Claude Sonnet: ~$3/M input, ~$15/M output
  - GPT-4o: ~$2.50/M input, ~$10/M output
  - GPT-4o-mini: ~$0.15/M input, ~$0.60/M output

### How can I reduce costs?

**1. Use local models for brainstorming:**
```bash
victor chat --provider ollama "Brainstorm ideas"
```

**2. Use cheaper models for drafts:**
```bash
victor chat --provider openai --model gpt-4o-mini "Draft this"
```

**3. Use best models only for final polish:**
```bash
/provider anthropic --model claude-sonnet-4-5
"Refine this implementation"
```

**4. Enable caching:**
```yaml
# In ~/.victor/config.yaml
cache_enabled: true
cache_ttl: 3600  # 1 hour
```

### Victor is slow, how can I speed it up?

**1. Use faster models:**
```bash
victor chat --provider groq "Quick question"  # 300+ tok/s
```

**2. Use caching:**
```bash
export VICTOR_CACHE_ENABLED=true
```

**3. Reduce context:**
```bash
victor chat --max-tokens 1000 "Brief answer"
```

**4. Use local model for offline:**
```bash
victor chat --provider ollama
```

### How many tokens will this cost?

Estimate: 1 token ≈ 4 characters for English text

**Example costs (GPT-4o):**
- 1000 words (prompt): ~$0.006
- 1000 words (response): ~$0.024
- Typical coding session: ~$0.05-0.15

**Use local models (Ollama) for $0**

---

## Troubleshooting

### "Command not found: victor"

**Solutions:**
```bash
# If using pipx
pipx ensurepath

# If using pip
export PATH="$HOME/.local/bin:$PATH"

# Or use full path
~/.local/bin/victor --version
```

### "Provider not found"

**Solutions:**
```bash
# Check provider name
victor providers list

# Install missing dependencies
pip install victor-ai[google]  # For Google
pip install victor-ai[all]     # For all providers

# Verify provider is supported
victor providers list | grep anthropic
```

### "Model not supported"

**Solutions:**
```bash
# List supported models for provider
victor providers models anthropic

# Use exact model name
victor chat --provider anthropic --model claude-sonnet-4-5

# Check model capabilities
cat victor/config/model_capabilities.yaml
```

### "Ollama connection failed"

**Solutions:**
```bash
# Ensure Ollama is running
ollama serve

# Test Ollama directly
ollama run qwen2.5-coder:7b "Hello"

# Check if port is accessible
curl http://localhost:11434/api/tags

# Try explicit model
victor chat --provider ollama --model qwen2.5-coder:7b
```

### "API key not found"

**Solutions:**
```bash
# Check if key is set
echo $ANTHROPIC_API_KEY

# Set key temporarily
export ANTHROPIC_API_KEY=sk-your-key

# Add to ~/.bashrc or ~/.zshrc
echo 'export ANTHROPIC_API_KEY=sk-your-key' >> ~/.bashrc
source ~/.bashrc

# Or use .env file
echo 'ANTHROPIC_API_KEY=sk-your-key' > ~/.victor/.env
```

### "Import error: No module named victor"

**Solutions:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall in development mode
pip install -e .

# Check installation
pip show victor-ai

# If missing, reinstall
pip install victor-ai
```

### Low disk space

**Solutions:**
```bash
# Remove cached models
rm -rf ~/.victor/cache/models/

# Clear conversation history
rm -rf ~/.victor/sessions/

# Clean package cache
pip cache purge

# Check disk usage
du -sh ~/.victor
```

### Victor hangs or freezes

**Solutions:**
```bash
# Check provider status
curl https://api.anthropic.com/v1/messages  # For Anthropic

# Try different provider
victor chat --provider ollama

# Reduce max tokens
victor chat --max-tokens 1000 "Test"

# Check logs
export VICTOR_LOG_LEVEL=DEBUG
victor chat "Test"
```

---

## Advanced Usage

### How do I use Victor programmatically?

```python
import asyncio
from victor import Agent

async def main():
    agent = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-5"
    )

    result = await agent.run("Explain async/await")
    print(result.content)

    await agent.close()

asyncio.run(main())
```

[Full API documentation](api/README.md)

### How do I configure Victor?

**Configuration file** (`~/.victor/config.yaml`):
```yaml
# Default provider
provider: anthropic
model: claude-sonnet-4-5

# Default mode (build, plan, explore)
mode: build

# Tool selection strategy (keyword, semantic, hybrid)
tool_selection_strategy: hybrid

# Air-gapped mode (local providers only)
airgapped_mode: false

# Maximum tool executions
max_tool_executions: 50

# Caching
cache_enabled: true
cache_ttl: 3600
```

**Environment variables:**
```bash
export VICTOR_PROVIDER=anthropic
export VICTOR_MODEL=claude-sonnet-4-5
export VICTOR_MODE=build
export VICTOR_LOG_LEVEL=INFO
```

### How do I create custom workflows?

**Create `my_workflow.yaml`:**
```yaml
name: my_workflow
description: "My custom workflow"

steps:
  - name: analyze
    agent: true
    prompt: "Analyze this code"

  - name: improve
    agent: true
    prompt: "Suggest improvements"

  - name: implement
    agent: true
    prompt: "Implement the improvements"
```

**Run workflow:**
```bash
victor workflow run my_workflow --file myfile.py
```

[Learn more about workflows](user-guide/workflows.md)

### How do I create custom tools?

**Create `my_tool.py`:**
```python
from victor.tools.base import BaseTool, ToolCostTier

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    cost_tier = ToolCostTier.FREE

    async def execute(self, **kwargs):
        return ToolResult(output="Result")
```

**Register tool:**
```python
from victor.tools.registry import register_tool

register_tool(MyTool())
```

### How do I use Victor in CI/CD?

**GitHub Actions example:**
```yaml
name: Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Victor
        run: pipx install victor-ai
      - name: Run review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          victor chat --provider anthropic "
          Review the changes in this PR
          "
```

### How do I monitor Victor usage?

**Enable logging:**
```bash
export VICTOR_LOG_LEVEL=DEBUG
victor chat 2>&1 | tee victor.log
```

**Check session history:**
```bash
victor sessions
```

**Monitor costs:**
```bash
# Track token usage
export VICTOR_TRACK_TOKENS=true
```

---

## Community & Support

### Where can I get help?

- **GitHub Discussions**: https://github.com/vjsingh1984/victor/discussions
- **GitHub Issues**: https://github.com/vjsingh1984/victor/issues
- **Documentation**: https://github.com/vjsingh1984/victor/tree/main/docs
- **Email**: singhvjd@gmail.com

### How do I report bugs?

1. Search existing issues first
2. Create new issue with:
   - Clear title
   - Victor version (`victor --version`)
   - Python version (`python --version`)
   - OS version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs

### How do I request features?

1. Check if feature already exists
2. Search for existing feature requests
3. Create new issue with:
   - Feature description
   - Use case
   - Proposed solution
   - Alternatives considered

### How do I contribute?

[Contributing guide](contributing/index.md)

**Quick start:**
```bash
# Fork repository
git clone https://github.com/YOUR_USERNAME/victor.git
cd victor

# Create branch
git checkout -b feature/my-feature

# Make changes
# Add tests
# Run tests
pytest

# Submit PR
```

### Where can I learn more?

**Documentation:**
- [Quick Start](tutorials/QUICK_START.md)
- [Tutorials](tutorials/README.md)
- [Recipe Book](tutorials/COMMON_WORKFLOWS.md)
- [User Guide](user-guide/index.md)
- [API Reference](api/README.md)
- [Architecture](architecture/ARCHITECTURE.md)

**Examples:**
- [Examples](examples/coordinator_examples.md)
- [GitHub Issues](https://github.com/vjsingh1984/victor/issues)
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)

---

## Quick Tips

### Beginner Tips

1. **Start with local model** - Free and private
2. **Use plan mode** for learning - No accidental edits
3. **Save sessions** - Build knowledge over time
4. **Ask specific questions** - Better responses
5. **Review generated code** - Always verify

### Intermediate Tips

1. **Switch providers** - Use right tool for job
2. **Create workflows** - Automate repetitive tasks
3. **Use multi-agent teams** - For complex reviews
4. **Enable caching** - Speed up repeated queries
5. **Configure profiles** - Quick environment switching

### Advanced Tips

1. **Custom tools** - Extend Victor's capabilities
2. **Python API** - Integrate into applications
3. **Air-gapped mode** - Secure, offline operation
4. **Multi-provider workflows** - Optimize cost/quality
5. **CI/CD integration** - Automate code reviews

---

## Common Pitfalls

### Don't

- Don't share API keys
- Don't use production API keys for testing
- Don't trust generated code blindly
- Don't forget to review changes
- Don't skip testing generated code

### Do

- Do use version control
- Do review generated code
- Do test thoroughly
- Do save important sessions
- Do keep API keys secure
- Do use appropriate modes

---

## Version Information

- **Current Version**: 0.5.0
- **Python Required**: 3.10+
- **License**: Apache 2.0
- **Last Updated**: January 20, 2026

---

## Additional Resources

### Official Documentation
- [Documentation Home](index.md) - Project overview
- `CLAUDE.md` - Developer quick reference (repo root)
- [Architecture](architecture/ARCHITECTURE.md) - System architecture

### Community
- [GitHub Repository](https://github.com/vjsingh1984/victor)
- [Contributing Guide](contributing/index.md)
- Code of Conduct: `CODE_OF_CONDUCT.md` (repo root)

### Related Projects
- [Ollama](https://ollama.ai/) - Local model runner
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Typer](https://typer.tiangolo.com/) - CLI framework

---

**Still have questions?**

[Ask on GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)

[Report an Issue](https://github.com/vjsingh1984/victor/issues)

---

**Version**: 0.5.0
**Last Updated**: January 20, 2026

---

**Last Updated:** February 01, 2026
**Reading Time:** 7 minutes
