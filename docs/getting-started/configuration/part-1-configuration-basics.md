# Configuration Guide - Part 1

**Part 1 of 3:** Configuration Overview, Quick Setup, Configuration Locations, Environment Variables, Profiles, and Global Settings

---

## Navigation

- **[Part 1: Configuration Basics](#)** (Current)
- [Part 2: Project Configuration](part-2-project-configuration.md)
- [Part 3: Validation & Best Practices](part-3-validation-best-practices.md)
- [**Complete Guide](../configuration.md)**

---

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Quick Setup](#quick-setup)
3. [Configuration Locations](#configuration-locations)
4. [Environment Variables](#environment-variables)
5. [Profiles Configuration](#profiles-configuration)
6. [Global Settings](#global-settings)
7. [Project Context Files](#project-context-files) *(in Part 2)*
8. [Modes Configuration](#modes-configuration) *(in Part 2)*
9. [MCP Configuration](#mcp-configuration) *(in Part 2)*
10. [Example Configurations](#example-configurations) *(in Part 2)*
11. [Validation and Testing](#validation-and-testing) *(in Part 3)*
12. [Troubleshooting](#troubleshooting) *(in Part 3)*
13. [Best Practices](#best-practices) *(in Part 3)*

---

# Configuration Guide

This guide covers all configuration options for Victor, from basic setup to advanced customization.

## Configuration Overview

Victor uses a layered configuration system:

```
Priority (highest to lowest):
1. CLI flags              --provider anthropic --model claude-4
2. Environment variables  ANTHROPIC_API_KEY, VICTOR_LOG_LEVEL
3. Project context        .victor.md, CLAUDE.md
4. Profile config         ~/.victor/profiles.yaml
5. Global config          ~/.victor/config.yaml
6. Built-in defaults      Sensible defaults for all settings
```

---

## Quick Setup

### Minimal Configuration

**For local models** (no configuration needed):
```bash
# Victor auto-detects Ollama
victor chat
```

**For cloud providers**:
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Use Victor
victor chat --provider anthropic
```

**For persistent configuration** (recommended):
```bash
# Initialize Victor
victor init

# Edit profiles
nano ~/.victor/profiles.yaml
```

---

## Configuration Locations

### Directory Structure

```
~/.victor/                      # User configuration (global)
+-- profiles.yaml               # Provider/model profiles
+-- config.yaml                 # Global settings
+-- mcp.yaml                    # MCP server config
+-- api_keys.yaml               # API keys (use keyring instead)
+-- cache/                      # Cache storage
+-- logs/                       # Log files
+-- conversations/              # Saved conversation history

<your-project>/                 # Project-specific (local)
+-- .victor.md                  # Project context for Victor
+-- CLAUDE.md                   # AI instructions (also read by Victor)
+-- .victor/
    +-- init.md                 # Alternative project context location
```

### Finding Your Config

```bash
# Show current configuration
victor config show

# Show config file locations
ls -la ~/.victor/

# Show active profile
victor config profiles
```

---

## Environment Variables

### API Keys

Set API keys for cloud providers:

```bash
# Cloud providers
export ANTHROPIC_API_KEY=sk-ant-...      # Anthropic (Claude)
export OPENAI_API_KEY=sk-proj-...        # OpenAI (GPT-4)
export GOOGLE_API_KEY=...                # Google (Gemini)
export XAI_API_KEY=...                   # xAI (Grok)
export DEEPSEEK_API_KEY=...              # DeepSeek
export MISTRAL_API_KEY=...               # Mistral
export GROQ_API_KEY=...                  # Groq
export TOGETHER_API_KEY=...              # Together AI
export FIREWORKS_API_KEY=...             # Fireworks AI
export OPENROUTER_API_KEY=...            # OpenRouter
export CEREBRAS_API_KEY=...              # Cerebras
export MOONSHOT_API_KEY=...              # Moonshot

# Enterprise
export AZURE_OPENAI_API_KEY=...          # Azure OpenAI
export AZURE_OPENAI_ENDPOINT=https://... # Azure endpoint
export AWS_PROFILE=default               # AWS Bedrock
export GCP_PROJECT=...                   # Google Vertex AI
```

**Secure storage (recommended):**
```bash
# Store in system keyring (macOS Keychain, Windows Credential Manager, Linux Secret Service)
victor keys --set anthropic --keyring
victor keys --set openai --keyring

# List stored keys
victor keys --list
```

### Local Provider Hosts

Override default hosts for local providers:

```bash
export OLLAMA_HOST=127.0.0.1:11434      # Ollama (default)
export VICTOR_LM_STUDIO_HOST=127.0.0.1:1234   # LM Studio
export VICTOR_VLLM_HOST=127.0.0.1:8000        # vLLM
export VICTOR_LLAMACPP_HOST=127.0.0.1:8080    # llama.cpp
```

### Victor Settings

Configure Victor behavior:

```bash
export VICTOR_CONFIG_DIR=~/.victor       # Config directory
export VICTOR_PROFILE=development        # Default profile
export VICTOR_LOG_LEVEL=INFO             # Log level (DEBUG, INFO, WARNING, ERROR)
export VICTOR_LOG_FILE=/var/log/victor.log  # Log file path
export VICTOR_CACHE_ENABLED=true         # Enable caching
export VICTOR_TIMEOUT=30                 # Provider timeout (seconds)
```

### Shell Configuration

Add to your shell profile (`~/.bashrc`, `~/.zshrc`):

```bash
# Victor API Keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-proj-..."

# Victor defaults
export VICTOR_PROFILE="development"
export VICTOR_LOG_LEVEL="INFO"

# Optional: aliases
alias v="victor"
alias vc="victor chat"
```

---

## Profiles Configuration

Profiles let you save provider/model combinations for easy reuse.

### Basic Profile

**File**: `~/.victor/profiles.yaml`

```yaml
profiles:
  # Default profile (used when no --profile specified)
  default:
    provider: ollama
    model: qwen2.5-coder:7b

  # Cloud provider profile
  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096

  # Fast cloud profile
  fast:
    provider: groq
    model: llama-3.1-70b-versatile

  # Cost-effective profile
  cheap:
    provider: deepseek
    model: deepseek-coder

# Optional: set default profile
default_profile: default
```

### Using Profiles

```bash
# Use specific profile
victor --profile claude chat "Hello"

# Or set default profile
export VICTOR_PROFILE=claude
victor chat "Hello"

# List profiles
victor config profiles
```

### Profile Options

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `provider` | string | Provider name | `anthropic`, `openai`, `ollama` |
| `model` | string | Model name | `claude-sonnet-4-20250514` |
| `temperature` | float | Sampling temperature (0.0-2.0) | `0.7` |
| `max_tokens` | int | Maximum tokens to generate | `4096` |
| `top_p` | float | Nucleus sampling (0.0-1.0) | `0.9` |
| `top_k` | int | Top-k sampling | `40` |
| `api_key_env` | string | Environment variable for API key | `ANTHROPIC_API_KEY` |
| `base_url` | string | Custom API endpoint | `https://api.anthropic.com` |
| `timeout` | int | Request timeout (seconds) | `30` |

### Advanced Profile Examples

**Development profile with debugging:**
```yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096
```

**Production profile (more conservative):**
```yaml
profiles:
  production:
    provider: openai
    model: gpt-4o
    temperature: 0.3  # Lower for consistency
    max_tokens: 8192
```

**Local private profile:**
```yaml
profiles:
  private:
    provider: ollama
    model: qwen2.5-coder:14b
    base_url: http://127.0.0.1:11434
    # No API key needed
```

---

