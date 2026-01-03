# Victor User Guide

A short, practical guide for day-to-day CLI use.

## Start
```bash
pipx install victor-ai
victor init
victor chat
```

One-shot:
```bash
victor "review src/main.py for bugs"
```

## Choose a Model
- Local: Ollama / LM Studio / vLLM (no API key)
- Cloud: Anthropic / OpenAI / Google / Groq / DeepSeek / others

Profiles live in `~/.victor/profiles.yaml`:
```yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
```

Use a profile:
```bash
victor --profile local
```

## Modes
- **BUILD**: make edits
- **PLAN**: analyze without edits
- **EXPLORE**: understand and take notes

```bash
victor chat --mode plan "Plan the auth refactor"
```

## Tools (What the Agent Can Do)
Tools are enabled by your profile, mode, and permissions. See the full list in `docs/TOOL_CATALOG.md`.

Examples:
```
Read README.md and summarize it
Create tests for src/api.py
Show git status and recent commits
```

## Workflows
- CLI workflows: `examples/workflows/README.md`
- YAML workflows: `docs/guides/WORKFLOW_DSL.md`

## Troubleshooting
- Ollama not reachable: `ollama serve`
- Model missing: `ollama pull <model>`
- API key issues: `victor keys --list`

## More Help
- Quick Start: `docs/guides/QUICKSTART.md`
- Provider setup: `docs/guides/PROVIDER_SETUP.md`
- Examples: `examples/README.md`
