# User Guide

Daily usage patterns for the Victor CLI.

## Basic Usage

```bash
victor                          # TUI mode
victor chat                     # CLI mode
victor "task description"       # One-shot
victor chat --provider X        # Specific provider
victor --profile local          # Use saved profile
```

## Modes

| Mode | Behavior |
|------|----------|
| **BUILD** | Full edits, default for implementation |
| **PLAN** | Sandbox edits, 2.5x exploration budget |
| **EXPLORE** | Notes only, 3x exploration budget |

```bash
victor chat --mode plan
victor chat --mode explore
```

## Tool Usage

Victor automatically selects tools based on your task. Request capabilities naturally:

```
Read README.md and summarize it
Create unit tests for src/api.py
Show git status and recent commits
Search for authentication-related code
Refactor this function for clarity
```

See [TOOL_CATALOG.md](TOOL_CATALOG.md) for the full list of 55+ tools.

## Profiles

Save provider/model combinations in `~/.victor/profiles.yaml`:

```yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
  cheap:
    provider: deepseek
    model: deepseek-coder
```

```bash
victor --profile local
victor --profile claude
```

## Switching Providers Mid-Conversation

Victor maintains context across provider switches:

```bash
# Start with Claude
victor chat --provider anthropic
# ... work on task ...
# Exit and continue with local model
victor chat --provider ollama
# Context preserved, conversation continues
```

## Workflows

### Quick Commands

```bash
victor workflow validate my_workflow.yaml
victor workflow render my_workflow.yaml --format ascii
```

### YAML Workflows

```yaml
# my_workflow.yaml
workflows:
  code_review:
    nodes:
      - id: review
        type: agent
        role: reviewer
        goal: "Review code for bugs and style"
        next: []
```

See [guides/WORKFLOW_DSL.md](guides/WORKFLOW_DSL.md) for workflow syntax.

### Scheduling

```bash
victor scheduler add daily_check --cron "0 9 * * *"
victor scheduler start
victor scheduler list
```

## Project Context

Victor loads project-specific instructions from:
- `.victor.md` (preferred)
- `CLAUDE.md`
- `.victor/init.md`

Add project-specific context, conventions, or restrictions here.

## HTTP API

```bash
victor serve --port 8765

# Endpoints
curl http://localhost:8765/chat -d '{"message": "hello"}'
curl http://localhost:8765/capabilities
```

## MCP Integration

Victor as MCP server:
```bash
victor mcp
```

Connect to MCP servers:
```yaml
# ~/.victor/mcp.yaml
servers:
  - name: filesystem
    command: npx
    args: ["-y", "@anthropic/mcp-server-filesystem"]
```

See [guides/MCP_INTEGRATION.md](guides/MCP_INTEGRATION.md) for details.

## Troubleshooting

### Provider Issues

```bash
victor providers --list          # List available providers
victor test-provider anthropic   # Test connectivity
victor chat --debug              # Verbose output
```

### Common Fixes

| Issue | Solution |
|-------|----------|
| "Connection refused" (Ollama) | Run `ollama serve` |
| "Model not found" | Pull model: `ollama pull model-name` |
| "API key invalid" | Re-set: `victor keys --set provider --keyring` |
| Config issues | Reset: `victor init --force` |

## More Resources

- [Getting Started](getting-started.md) - Installation, first run
- [Workflow DSL](guides/WORKFLOW_DSL.md) - YAML workflow syntax
- [Observability](guides/OBSERVABILITY.md) - Events, metrics, tracing
- [Provider Reference](reference/PROVIDERS.md) - All provider details
