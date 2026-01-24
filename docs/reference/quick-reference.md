# Quick Reference

Essential commands and patterns for Victor AI.

## Essential Commands

```bash
# Install
pipx install victor-ai

# Initialize
victor init

# TUI mode
victor

# CLI mode
victor chat "Your message"

# Specific provider
victor chat --provider anthropic --model claude-sonnet-4-5
```

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--provider, -p` | LLM provider | `--provider openai` |
| `--model, -m` | Model name | `--model gpt-4o` |
| `--mode` | Agent mode | `--mode plan` |
| `--airgapped` | Local providers only | `--airgapped` |
| `--tool-budget` | Max tool calls | `--tool-budget 20` |
| `--tools` | Specific tools | `--tools read,write,edit` |

## Agent Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `build` | Full edits (default) | Active development |
| `plan` | 2.5x exploration, sandbox | Research & analysis |
| `explore` | 3.0x exploration, no edits | Codebase understanding |

## Provider Switching

```bash
# Start with Claude
victor chat --provider anthropic

# Switch to GPT-4 (context preserved)
/provider openai --model gpt-4o

# Switch to local (context preserved)
/provider ollama --model qwen2.5-coder:7b
```

## Python API

```python
from victor.framework import Agent, ToolSet

# Create agent
agent = await Agent.create(provider="anthropic")

# Run task
result = await agent.run("Write tests for auth.py")

# Stream events
async for event in agent.stream("Analyze code"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")

# Multi-provider (cost optimization)
brainstormer = await Agent.create(provider="ollama")      # FREE
implementer = await Agent.create(provider="openai")       # CHEAP
reviewer = await Agent.create(provider="anthropic")       # QUALITY
```

## ToolSet Presets

```python
from victor.framework import ToolSet

# Minimal (read, write, edit)
ToolSet.minimal()

# Default (balanced)
ToolSet.default()

# Full (all 55+ tools)
ToolSet.full()

# Custom selection
ToolSet.custom(["read", "grep", "run_tests"])
```

## API Keys

```bash
# Set environment variable
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...

# Or use secure keyring
victor keys --set anthropic --keyring
victor keys --set openai --keyring
```

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `profiles.yaml` | `~/.victor/` | Provider/model profiles |
| `config.yaml` | `~/.victor/` | Global settings |
| `.victor.md` | Project root | Project context |
| `CLAUDE.md` | Project root | AI instructions |

## Profile Configuration

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    tools: [read, write, edit]
    mode: build

  fast:
    provider: groq
    model: llama3.1-70b

  free:
    provider: ollama
    model: qwen2.5-coder:7b
```

## Slash Commands (In-Chat)

```
/provider <name>     # Switch provider
/mode <mode>         # Change mode
/profile <name>      # Load profile
/clear               # Clear conversation
/help                # Show help
/exit                # Exit session
```

## Workflows

```bash
# List workflows
victor workflow list

# Run workflow
victor workflow run code-review

# Validate workflow
victor workflow validate my-workflow.yaml

# Schedule workflow
victor workflow schedule code-review --cron "0 9 * * 1"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `command not found` | Run `pipx ensurepath` |
| Ollama connection error | Run `ollama serve` |
| API key not found | Set env var or use `victor keys --set` |
| Permission denied | Never use `sudo pip` |

## Cost Optimization

| Strategy | Provider | Best For |
|----------|----------|---------|
| **Free** | Ollama | Brainstorming |
| **Fast** | Groq (300+ tok/s) | Rapid iteration |
| **Cheap** | OpenAI (gpt-4o-mini) | Implementation |
| **Quality** | Anthropic (claude-sonnet-4) | Review, analysis |

## Version Info

```bash
victor --version              # Show version
victor list providers         # List providers
victor list tools             # List tools
victor config show            # Show configuration
```

## See Also

- [Installation](../getting-started/installation.md)
- [First Run](../getting-started/first-run.md)
- [Local Models](../getting-started/local-models.md)
- [Cloud Models](../getting-started/cloud-models.md)
- [Configuration](configuration/)
- [API Reference](api.md)

---

<div align="center">

**[← Back to Reference](index.md)**

**Quick Reference**

*Essential commands and patterns*

</div>
