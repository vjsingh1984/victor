# Configuration Guide - Part 3

**Part 3 of 3:** Example Configurations, Validation and Testing, Troubleshooting, and Best Practices

---

## Navigation

- [Part 1: Configuration Basics](part-1-configuration-basics.md)
- [Part 2: Project Configuration](part-2-project-configuration.md)
- **[Part 3: Validation & Best Practices](#)** (Current)
- [**Complete Guide](../configuration.md)**

---

## Table of Contents

1. [Configuration Overview](#configuration-overview) *(in Part 1)*
2. [Quick Setup](#quick-setup) *(in Part 1)*
3. [Configuration Locations](#configuration-locations) *(in Part 1)*
4. [Environment Variables](#environment-variables) *(in Part 1)*
5. [Profiles Configuration](#profiles-configuration) *(in Part 1)*
6. [Global Settings](#global-settings) *(in Part 2)*
7. [Project Context Files](#project-context-files) *(in Part 2)*
8. [Modes Configuration](#modes-configuration) *(in Part 2)*
9. [MCP Configuration](#mcp-configuration) *(in Part 2)*
10. [Example Configurations](#example-configurations)
11. [Validation and Testing](#validation-and-testing)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)

---

## Example Configurations

### Developer Setup

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b

  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7

  fast:
    provider: groq
    model: llama-3.1-70b-versatile

default_profile: default
```text

```yaml
# ~/.victor/config.yaml
logging:
  level: INFO

cache:
  enabled: true
  backend: memory

tools:
  lazy_load: true
  parallel_execution: true
```

### Team Setup

```yaml
# ~/.victor/profiles.yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7

  code-review:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.3  # More consistent

  testing:
    provider: openai
    model: gpt-4o
```text

```yaml
# ~/.victor/config.yaml
logging:
  level: INFO
  format: json
  file: /var/log/victor/team.log

cache:
  enabled: true
  backend: redis
  redis:
    host: redis.internal.example.com
    port: 6379
```

### Air-Gapped Setup

For secure, offline environments:

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:14b
    base_url: http://localhost:11434
```text

```yaml
# ~/.victor/config.yaml
airgapped_mode: true

# Only local providers allowed
providers:
  allowed:
    - ollama
    - lmstudio
    - vllm
    - llamacpp

# Disable web tools
tools:
  disabled:
    - web_search
    - fetch_url
```

---

## Validation and Testing

### Validate Configuration

```bash
# Validate all configuration files
victor config validate

# Check current configuration
victor config show

# Test a specific profile
victor --profile development chat "Test"
```text

### Debug Configuration

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG

# Run with debug output
victor --debug chat "Test"

# View configuration
victor config show

# Check logs
victor logs --tail 100
```

---

## Troubleshooting

### Configuration Not Loading

```bash
# Check file exists and has correct permissions
ls -la ~/.victor/profiles.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('$HOME/.victor/profiles.yaml'))"

# Check for tab characters (should use spaces)
cat -A ~/.victor/profiles.yaml | grep '\t'
```text

### Profile Not Found

```bash
# List available profiles
victor config profiles

# Check profile name in file
grep -A 5 "profiles:" ~/.victor/profiles.yaml
```

### API Key Not Found

```bash
# Check environment variable
echo $ANTHROPIC_API_KEY

# Check keyring
victor keys --list

# Set key
victor keys --set anthropic --keyring
```text

---

## Best Practices

1. **Use profiles** for different contexts (development, testing, review)
2. **Store API keys in keyring**, not in config files
3. **Create project context** (`.victor.md`) for each project
4. **Use version control** for `.victor.md` (but not API keys)
5. **Start with PLAN mode** for unfamiliar codebases
6. **Use environment variables** for CI/CD

---

## Next Steps

- [User Guide](../user-guide/index.md) - Daily usage patterns
- [Provider Reference](../reference/providers/index.md) - Current provider list
- [Tool Catalog](../reference/tools/catalog.md) - Current tool list
- [Workflow Guide](../guides/workflow-development/dsl.md) - Automation

---

**Need help?** See [Troubleshooting](../user-guide/troubleshooting.md) or [open an
  issue](https://github.com/vjsingh1984/victor/issues).

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
