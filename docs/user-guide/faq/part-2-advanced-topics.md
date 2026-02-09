# Victor AI FAQ - Part 2

**Part 2 of 2:** Troubleshooting, Advanced Usage, Community & Support, Quick Tips, Common Pitfalls, Version Information,
  and Additional Resources

---

## Navigation

- [Part 1: Basic Questions](part-1-basic-questions.md)
- **[Part 2: Advanced Topics](#)** (Current)
- [**Complete Guide](../faq.md)**

---

## Table of Contents

1. [General Questions](#general-questions) *(in Part 1)*
2. [Installation & Setup](#installation--setup) *(in Part 1)*
3. [Providers & Models](#providers--models) *(in Part 1)*
4. [Usage & Features](#usage--features) *(in Part 1)*
5. [Performance & Cost](#performance--cost) *(in Part 1)*
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)
8. [Community & Support](#community--support)
9. [Quick Tips](#quick-tips)
10. [Common Pitfalls](#common-pitfalls)
11. [Version Information](#version-information)
12. [Additional Resources](#additional-resources)

---

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

**Check session history:**
```bash
victor sessions
```

**Monitor costs:**
```bash
# Track token usage
export VICTOR_TRACK_TOKENS=true
```text

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
