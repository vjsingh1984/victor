# First Run

Get productive with Victor in minutes.

## 1. Initialize

```bash
victor init
```

Creates `~/.victor/profiles.yaml` and project context file.

## 2. Start Chatting

```bash
# Interactive chat
victor chat

# One-shot query
victor "explain this codebase"
```

## 3. Try These Prompts

- "Summarize this repo and list risky areas"
- "Generate unit tests for src/utils.py"
- "Refactor src/api.py to improve readability"
- "Explain the architecture of this project"

## 4. Switch Providers

### In Chat
```
/provider openai --model gpt-4
/provider anthropic --model claude-sonnet-4-5
```

### Or Edit Config
```yaml
# ~/.victor/profiles.yaml
default_profile:
  provider: anthropic
  model: claude-sonnet-4-5
```

## 5. Choose Your Mode

```bash
# Build mode (default, full edits)
victor chat --mode build

# Plan mode (2.5x exploration, sandbox)
victor chat --mode plan

# Explore mode (3.0x exploration, no edits)
victor chat --mode explore
```

## Next Steps

- [Local Models](local-models.md) - Free, private AI with Ollama
- [Cloud Models](cloud-models.md) - Full capability with cloud providers
- [Configuration](./configuration.md) - Advanced setup

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
