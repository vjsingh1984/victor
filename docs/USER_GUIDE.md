# Victor User Guide

> Complete guide for using Victor - Your Universal AI Coding Assistant

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Working with Providers](#working-with-providers)
- [Agent Modes](#agent-modes)
- [Using Tools](#using-tools)
- [Workflows](#workflows)
- [Tips and Tricks](#tips-and-tricks)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```mermaid
flowchart LR
    A[Install] --> B[Pull Model] --> C[Run Victor]
    C --> D[Start Chatting]
```

```bash
# 1. Install Victor
pip install victor-ai

# 2. Quick start with Ollama (FREE)
ollama pull qwen2.5-coder:7b

# 3. Run Victor
victor
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Methods

| Method | Command | Notes |
|--------|---------|-------|
| pip | `pip install victor-ai` | Recommended |
| pipx | `pipx install victor-ai` | Isolated |
| Source | `pip install -e ".[dev]"` | Development |

### Provider Setup

| Provider | Setup |
|----------|-------|
| **Ollama** | `ollama pull qwen2.5-coder:7b` |
| **Cloud** | `victor keys --set anthropic --keyring` |

---

## CLI Reference

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `victor` | Start TUI | `victor` |
| `victor chat` | CLI mode | `victor chat "Hello"` |
| `victor chat --no-tui` | Non-interactive | `victor chat --no-tui "Explain async"` |
| `victor serve` | API server | `victor serve --port 8765` |
| `victor mcp` | MCP server | `victor mcp` |
| `victor keys` | Key management | `victor keys --set anthropic` |

### Common Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--provider` | Select provider | `--provider ollama` |
| `--model` | Select model | `--model gpt-4` |
| `--mode` | Agent mode | `--mode explore` |
| `--vertical` | Domain vertical | `--vertical devops` |
| `--tool-budget` | Max tool calls | `--tool-budget 20` |
| `--renderer` | Output format | `--renderer text` |
| `--log-level` | Logging | `--log-level DEBUG` |

### Output Modes

| Flag | Use Case |
|------|----------|
| `--code-only` | Extract code blocks only |
| `--json` | JSON output for parsing |
| `--plain` | No formatting |
| `--quiet` | Minimal output |
| `--stdin` | Pipe input |

---

## Working with Providers

### Provider Selection

| Task Type | Recommended | Why |
|-----------|-------------|-----|
| Brainstorming | Ollama | Free, fast |
| Quick code | GPT-3.5 / Groq | Cheap, fast |
| Complex logic | Claude | Best quality |
| Long context | Gemini | 1M tokens |

### Switching Providers

**In REPL**:
```
/provider anthropic
/model gpt-4
```

**Via Profiles** (`~/.victor/profiles.yaml`):
```yaml
profiles:
  dev:
    provider: ollama
    model: qwen2.5-coder:7b
  production:
    provider: anthropic
    model: claude-sonnet-4-5
```

```bash
victor --profile dev
victor --profile production
```

---

## Agent Modes

| Mode | Purpose | Edits | Exploration |
|------|---------|:-----:|:-----------:|
| **BUILD** | Implementation | Full | 1.0x |
| **PLAN** | Analysis | Sandbox only | 2.5x |
| **EXPLORE** | Understanding | Notes only | 3.0x |

```bash
victor chat --mode plan "Plan the auth refactor"
victor chat --mode explore "Explain the caching architecture"
victor chat --mode build "Implement the feature"
```

> **Note**: PLAN and EXPLORE modes restrict file edits but allow more iterations for thorough analysis.

---

## Using Tools

### Tool Categories

| Category | Tools | Cost |
|----------|-------|:----:|
| **File Ops** | read, write, edit, ls, grep | FREE |
| **Git** | status, diff, commit, branch | FREE |
| **Code** | review, refactor, metrics | LOW |
| **Web** | search, fetch | MEDIUM |
| **Database** | connect, query, schema | LOW |
| **Docker** | ps, logs, stats | FREE |
| **Batch** | multi-file operations | HIGH |

### Examples

**File Operations**:
```
> Read the README.md file
> Edit main.py and add error handling
> Create a new file called utils.py
```

**Git Integration**:
```
> Show me what files have changed
> Create a commit with an appropriate message
> Push to main branch
```

**Database**:
```
> Connect to SQLite at ./data/app.db
> Show all tables
> Query users created last week
```

**Semantic Search**:
```
> Index my codebase
> Find code related to "user authentication"
> Where is the database connection logic?
```

---

## Workflows

### StateGraph DSL

LangGraph-compatible workflow engine for stateful, cyclic agent processes.

```python
from victor.framework import StateGraph, END
from typing import TypedDict

class TaskState(TypedDict):
    messages: list[str]
    result: str | None

async def analyze(state: TaskState) -> TaskState:
    state["messages"].append("Analyzed")
    return state

graph = StateGraph(TaskState)
graph.add_node("analyze", analyze)
graph.add_edge("analyze", END)
graph.set_entry_point("analyze")

app = graph.compile()
result = await app.invoke({"messages": [], "result": None})
```

| Feature | Description |
|---------|-------------|
| Typed State | TypedDict schemas |
| Conditional Edges | Branch on state |
| Cycles | Retry loops |
| Checkpointing | Resume workflows |

### Multi-Agent Teams

| Formation | Description | Use Case |
|-----------|-------------|----------|
| SEQUENTIAL | One after another | Simple handoffs |
| PARALLEL | Simultaneous | Independent tasks |
| PIPELINE | Output flows through | Feature implementation |
| HIERARCHICAL | Manager coordinates | Complex projects |

```python
from victor.framework.teams import TeamMemberSpec, TeamFormation

team = await Agent.create_team(
    name="Feature Team",
    goal="Implement authentication",
    members=[
        TeamMemberSpec(role="researcher", goal="Find patterns"),
        TeamMemberSpec(role="executor", goal="Write code", tool_budget=30),
    ],
    formation=TeamFormation.PIPELINE,
)
result = await team.run()
```

---

## Tips and Tricks

### Cost Optimization

| Task | Provider | Cost |
|------|----------|:----:|
| Brainstorm | Ollama | $0 |
| Implementation | GPT-3.5 | $0.002/1K |
| Review | Claude | $0.015/1K |
| Tests | Ollama | $0 |

> **Tip**: Use Ollama for 90%+ tasks, escalate to Claude only for complex logic.

### Best Practices

| Do | Don't |
|----|-------|
| Be specific: "Fix null pointer in user_service.py:45" | "Fix the bug" |
| Provide context: "FastAPI app with SQLAlchemy" | "Add auth" |
| Iterate: model -> validate -> test | Request everything at once |
| Use modes: EXPLORE first, then BUILD | Jump straight to implementation |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel operation |
| `Ctrl+D` | Exit Victor |
| `/help` | Show help |
| `/clear` | Clear screen |
| `/history` | Show history |
| `/provider X` | Switch provider |

---

## Troubleshooting

### Quick Fixes

| Issue | Solution |
|-------|----------|
| "Connection refused" | Check Ollama: `ollama list` |
| Slow responses | Use smaller model or `/clear` context |
| Poor quality | Be more specific, use better model |
| API errors | Verify key: `victor keys --list` |

### Ollama Issues

```bash
# Check if running
ollama list

# Start service
ollama serve

# Pull model if missing
ollama pull qwen2.5-coder:7b
```

### Cloud Provider Issues

```bash
# Verify API key
victor keys --list

# Test connection
victor chat --provider anthropic "Hello"
```

### Performance Tips

| Issue | Solution |
|-------|----------|
| Slow | Use smaller model (GPT-3.5, Gemini Flash) |
| High latency | Use local model (Ollama) |
| Context too large | `/clear` to reset |
| Streaming delays | Check network, use `--renderer text` |

---

## Resources

| Resource | Link |
|----------|------|
| Documentation | [GitHub Docs](https://github.com/vjsingh1984/victor/docs) |
| Issues | [GitHub Issues](https://github.com/vjsingh1984/victor/issues) |
| Examples | `examples/` directory |
| Developer Guide | [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) |

---

*Last Updated: 2025-12-30*
