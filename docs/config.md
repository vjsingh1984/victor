# Victor Configuration Guide

This guide explains Victor's unified configuration system, which provides a streamlined way to manage provider accounts, authentication, and settings.

## Overview

Victor uses a single configuration file (`~/.victor/config.yaml`) that unifies:

- **Provider accounts** - Which LLM providers you use
- **Model selection** - Which models to use for each account
- **Authentication** - API keys, OAuth, or no authentication
- **Settings** - Temperature, max tokens, and other parameters

## Quick Start

### First-Time Setup

The easiest way to get started is with the interactive setup wizard:

```bash
victor auth setup
```

The wizard will:
1. Detect local providers (Ollama, LM Studio)
2. Help you choose a cloud provider
3. Guide you through authentication
4. Test your connection
5. Save your configuration

### Quick Add an Account

To quickly add a provider account:

```bash
# Add Anthropic Claude
victor auth add --provider anthropic --model claude-sonnet-4-5

# Add OpenAI GPT
victor auth add --provider openai --model gpt-4o

# Add ZhipuAI GLM with coding plan endpoint
victor auth add --provider zai --model glm-4.6:coding --name glm-coding

# Add OAuth-enabled provider
victor auth add --provider openai --model gpt-4o --auth-method oauth
```

## Configuration File

### Location

- **Unix/Linux/macOS**: `~/.victor/config.yaml`
- **Windows**: `C:\Users\<username>\.victor\config.yaml`

### File Format

```yaml
accounts:
  # Default account
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    auth:
      method: api_key
      source: keyring
    tags: [chat, coding]
    temperature: 0.7
    max_tokens: 4096

  # Coding-focused account
  coding:
    provider: openai
    model: gpt-4o
    auth:
      method: api_key
      source: keyring
    tags: [coding, premium]
    temperature: 0.5
    max_tokens: 8192

  # GLM with coding plan endpoint
  glm-coding:
    provider: zai
    model: glm-4.6:coding  # Model suffix selects coding endpoint
    auth:
      method: api_key
      source: keyring
    tags: [coding, glm]

  # Local provider (no API key)
  local:
    provider: ollama
    model: llama3
    auth:
      method: none
    tags: [local, free]

  # OAuth-enabled account
  openai-oauth:
    provider: openai
    model: gpt-4o
    auth:
      method: oauth
      source: keyring
    tags: [oauth, chat]

# Default account to use
defaults:
  account: default
```

## Account Properties

### Required Fields

- **`provider`**: Provider name (e.g., `anthropic`, `openai`, `zai`)
- **`model`**: Model identifier
- **`auth`**: Authentication configuration

### Optional Fields

- **`name`**: Account name (defaults to entry key in YAML)
- **`tags`**: List of tags for categorization
- **`temperature`**: Default temperature (0.0-1.0)
- **`max_tokens`**: Default max tokens for generation
- **`endpoint`**: Custom endpoint URL (overrides default)

## Authentication Methods

### API Key Authentication

```yaml
auth:
  method: api_key
  source: keyring  # or "env" or "file"
```

**Storage Options:**
- **`keyring`**: System keyring (recommended, most secure)
  - macOS: Keychain
  - Windows: Credential Manager
  - Linux: Secret Service (GNOME Keyring, KWallet)
- **`env`**: Environment variable (for CI/CD)
- **`file`**: Stored in config file (not recommended for production)

### OAuth Authentication

Supported providers: `openai`, `qwen`

```yaml
auth:
  method: oauth
  source: keyring
```

To use OAuth:

```bash
# Authenticate with OAuth (opens browser)
victor providers auth login openai

# Check OAuth status
victor providers auth status openai

# Logout
victor providers auth logout openai
```

### No Authentication (Local Providers)

For local providers like Ollama, LM Studio, vLLM:

```yaml
auth:
  method: none
```

## Model Suffixes

Some providers support multiple endpoints. Use model suffixes to select:

### ZhipuAI (ZAI)

```yaml
# Standard endpoint
model: glm-4.6

# Coding plan endpoint (optimized for code)
model: glm-4.6:coding

# China endpoint
model: glm-4.6:china

# Anthropic-compatible endpoint
model: glm-4.6:anthropic
```

## Provider Reference

### Cloud Providers

| Provider | Models | Auth Methods | Notes |
|----------|--------|--------------|-------|
| `anthropic` | Claude 3.5/4 Sonnet, Opus | API key | `claude-sonnet-4-5`, `claude-opus-4-6` |
| `openai` | GPT-4.1, 4o, o1, o3 | API key, OAuth | Enable OAuth in config |
| `google` | Gemini 2.5, Flash | API key | `gemini-2.5-pro`, `gemini-2.5-flash` |
| `zai` | GLM-4.6, 4.7, 5.0 | API key | Use `:coding` suffix for coding endpoint |
| `xai` | Grok-2 | API key | Aliases: `grok` |
| `moonshot` | Kimi K2 | API key | Aliases: `kimi` |
| `deepseek` | DeepSeek-V3, Coder | API key | `deepseek-chat`, `deepseek-coder` |
| `qwen` | Qwen 2.5, Max | API key, OAuth | Alibaba Cloud |
| `groqcloud` | Llama, Mixtral | API key | Ultra-fast inference |
| `cerebras` | Qwen, Llama | API key | Fast inference |
| `mistral` | Mistral Large, Codestral | API key | 500K tokens/min free |

### Local Providers

| Provider | Models | Auth | Setup |
|----------|--------|------|-------|
| `ollama` | Llama2, Mistral, etc. | None | `brew install ollama` |
| `lmstudio` | Various local models | None | Download LM Studio |
| `vllm` | Various | None | `vllm serve <model>` |

## Commands

### Setup & Configuration

```bash
# Interactive setup wizard
victor auth setup

# Quick add account
victor auth add --provider <provider> --model <model>

# List accounts
victor auth list

# Remove account
victor auth remove <account-name>

# Test connection
victor auth test [--name <account>] [--provider <provider>]

# Migrate from old configuration
victor auth migrate
```

### OAuth Management

```bash
# Login with OAuth
victor providers auth login <provider>

# Check OAuth status
victor providers auth status [<provider>]

# Logout from OAuth
victor providers auth logout <provider>
```

## Resolution Order

Victor resolves configuration in this priority order:

1. **CLI flags** - Highest priority
   ```bash
   victor chat --provider anthropic --model claude-sonnet-4-5
   ```

2. **`~/.victor/config.yaml`** - Accounts section

3. **Environment variables** - For CI/CD
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   export OPENAI_API_KEY=sk-...
   ```

4. **System keyring** - Secure credential storage

## Migration from Old Configuration

If you have old configuration files (`profiles.yaml`, `api_keys.yaml`), Victor will automatically detect them and offer to migrate:

```bash
victor auth migrate
```

### Manual Migration

The migration will:
- Read `~/.victor/profiles.yaml` and `~/.victor/api_keys.yaml`
- Create `~/.victor/config.yaml` with unified format
- Back up old files to `~/.victor/backups/migration_<timestamp>/`
- Preserve all your existing accounts and API keys

### Rollback

If you need to rollback:

```bash
# Automatic rollback
victor auth migrate --dry-run  # Preview first
```

Or manually:

```bash
# Restore from backup
cp ~/.victor/backups/migration_<timestamp>/profiles.yaml ~/.victor/
cp ~/.victor/backups/migration_<timestamp>/api_keys.yaml ~/.victor/
rm ~/.victor/config.yaml
```

## Environment Variables

For CI/CD or automated environments, use environment variables:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...

# Google Gemini
export GOOGLE_API_KEY=...

# ZhipuAI GLM
export ZAI_API_KEY=...

# And more...
```

Victor automatically checks environment variables for API keys.

## Troubleshooting

### Connection Issues

Test your provider connection:

```bash
victor auth test --name <account-name>
```

### API Key Issues

If Victor can't find your API key:

1. Check keyring:
   ```bash
   victor auth list
   ```

2. Try environment variable:
   ```bash
   export ANTHROPIC_API_KEY=your-key
   victor auth test
   ```

3. Re-add the account:
   ```bash
   victor auth remove <account>
   victor auth add --provider <provider> --model <model>
   ```

### OAuth Issues

If OAuth isn't working:

```bash
# Check status
victor providers auth status openai

# Re-authenticate
victor providers auth login openai --force
```

### Local Provider Issues

For Ollama/LM Studio:

```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve

# Check LM Studio (should be accessible at http://localhost:1234)
curl http://localhost:1234/v1/models
```

## Best Practices

1. **Use keyring for API keys** - Most secure storage method
2. **Use OAuth when available** - For OpenAI and Qwen
3. **Use model suffixes** - For ZAI coding plan, etc.
4. **Organize with tags** - Group accounts by purpose (chat, coding, local)
5. **Test connections** - Use `victor auth test` after adding accounts
6. **Use environment variables in CI/CD** - Never commit API keys to git

## Examples

### Chat Configuration

```yaml
accounts:
  chat:
    provider: anthropic
    model: claude-sonnet-4-5
    auth:
      method: api_key
      source: keyring
    tags: [chat]
    temperature: 0.8
    max_tokens: 4096
```

### Coding Configuration

```yaml
accounts:
  coding:
    provider: zai
    model: glm-4.6:coding
    auth:
      method: api_key
      source: keyring
    tags: [coding, glm]
    temperature: 0.3
    max_tokens: 8192
```

### Local Development

```yaml
accounts:
  local:
    provider: ollama
    model: llama3
    auth:
      method: none
    tags: [local, free]
    temperature: 0.7
```

### Multi-Provider Setup

```yaml
accounts:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    auth:
      method: api_key
      source: keyring

  backup:
    provider: openai
    model: gpt-4o
    auth:
      method: api_key
      source: keyring

  local:
    provider: ollama
    model: llama3
    auth:
      method: none

defaults:
  account: default
```

## Advanced Usage

### Custom Endpoints

```yaml
accounts:
  custom-endpoint:
    provider: anthropic
    model: claude-sonnet-4-5
    endpoint: https://custom-api.example.com
    auth:
      method: api_key
      source: keyring
```

### Per-Account Settings

```yaml
accounts:
  creative:
    provider: anthropic
    model: claude-sonnet-4-5
    auth:
      method: api_key
      source: keyring
    temperature: 0.9  # More creative
    max_tokens: 4096

  precise:
    provider: anthropic
    model: claude-sonnet-4-5
    auth:
      method: api_key
      source: keyring
    temperature: 0.1  # More focused
    max_tokens: 2048
```

## Configuration Reference

### Complete Schema

```yaml
accounts:
  <account-name>:
    # Required
    provider: <string>           # Provider name
    model: <string>              # Model identifier
    auth:
      method: <api_key|oauth|none>
      source: <keyring|env|file>
      value: <string>            # Optional: explicit value

    # Optional
    name: <string>               # Account name (default: YAML key)
    tags: [<string>]             # Tags for categorization
    temperature: <float>         # 0.0 - 1.0
    max_tokens: <int>            # Max output tokens
    endpoint: <string>           # Custom endpoint URL
    extra_params:                # Provider-specific parameters
      <key>: <value>

defaults:
  account: <string>              # Default account name
  temperature: <float>           # Default temperature
  max_tokens: <int>              # Default max tokens
```

## See Also

- [Provider Reference](providers.md) - Full list of supported providers
- [OAuth Guide](oauth.md) - OAuth authentication details
- [Local Providers](local-providers.md) - Setting up local inference
- [API Documentation](api.md) - Programmatic configuration
