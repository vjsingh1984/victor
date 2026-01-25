# All-Provider Integration Tests - Complete Guide

This directory contains **comprehensive integration tests** that support **ALL 21 Victor providers** with automatic API key validation and graceful skipping.

## Overview

Tests support:
- **4 Local Providers**: Ollama, LMStudio, vLLM, LlamaCpp (free)
- **7 Premium Cloud Providers**: Anthropic, OpenAI, Google, xAI, Zhipu AI, Moonshot, DeepSeek
- **6 Free-Tier Cloud Providers**: Groq, Mistral, Together, OpenRouter, Fireworks, Cerebras, HuggingFace
- **4 Enterprise Providers**: Vertex AI, Azure OpenAI, AWS Bedrock, Replicate

## Key Features

✅ **Automatic Provider Detection** - Tests run with ALL available providers  
✅ **Robust API Key Validation** - Tests keys with actual API calls  
✅ **Billing Error Detection** - Skips on credit limits, auth failures  
✅ **Cost Optimization** - Uses cheapest/fastest model for each provider  
✅ **GitHub Actions Ready** - Tests skip gracefully when keys not configured  
✅ **Local Development Friendly** - Works with Ollama (no API key needed)  

## Quick Start

### 1. Run Tests with Available Providers

```bash
# Automatically detects and tests with ALL available providers
pytest tests/integration/real_execution/test_all_providers.py -v -m real_execution

# See provider availability summary
pytest tests/integration/real_execution/test_all_providers.py::test_provider_summary -v
```

### 2. Run with Specific Provider

```bash
# Only test with Ollama
pytest tests/integration/real_execution/test_all_providers.py -v -k "ollama"

# Only test with DeepSeek
pytest tests/integration/real_execution/test_all_providers.py -v -k "deepseek"

# Only test with OpenAI
pytest tests/integration/real_execution/test_all_providers.py -v -k "openai"
```

## All Supported Providers

### Local Providers (No API Key Required)

| Provider | Cost | Cheapest Model | Setup |
|----------|------|---------------|-------|
| **Ollama** | Free | qwen2.5-coder:7b | `brew install ollama && ollama serve` |
| **LlamaCpp** | Free | local-model | `llama-cli` |
| LMStudio | Free | local-model | (GUI app) |
| vLLM | Free | local-model | `vllm serve` |

### Premium Cloud Providers

| Provider | API Key Env Var | Cheapest Model | Cost (per 1M tokens) |
|----------|----------------|----------------|---------------------|
| **Anthropic** | `ANTHROPIC_API_KEY` | claude-haiku-3-5 | $0.25 / $1.25 |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o-mini | $0.15 / $0.60 |
| **Google** | `GOOGLE_API_KEY` | gemini-1.5-flash | $0.075 / $0.30 |
| **xAI** | `XAI_API_KEY` | grok-beta | ~$0.50 |
| **Zhipu AI** | `ZAI_API_KEY` | glm-4-flash | ~$0.30 |
| **Moonshot** | `MOONSHOT_API_KEY` | moonshot-v1-8k | ~$1.20 |
| **DeepSeek** | `DEEPSEEK_API_KEY` | deepseek-chat | $0.14 / $0.28 |

### Free-Tier Cloud Providers

| Provider | API Key Env Var | Free Tier | Cheapest Model |
|----------|----------------|-----------|---------------|
| **Groq** | `GROQCLOUD_API_KEY` | 15K RPM | llama-3.3-70b |
| **Mistral** | `MISTRAL_API_KEY` | 500K tokens/min | mistral-small-latest |
| **Together** | `TOGETHER_API_KEY` | $25 credits | Llama-3-8b |
| **OpenRouter** | `OPENROUTER_API_KEY | Daily limits | llama-3-8b:free |
| **Fireworks** | `FIREWORKS_API_KEY | Yes | llama-v3-70b |
| **Cerebras** | `CEREBRAS_API_KEY` | Yes | llama-3.1-8b |
| **Hugging Face** | `HF_TOKEN` | Yes | Meta-Llama-3.1-8B |

## Setting Up Providers

### Local Providers (Recommended for Development)

```bash
# Ollama (easiest option)
brew install ollama
ollama serve
ollama pull qwen2.5-coder:7b

# Run tests
pytest tests/integration/real_execution/test_all_providers.py -v -k "ollama"
```

### Cloud Providers

Set one or more API keys:

```bash
# DeepSeek (cheapest premium option)
export DEEPSEEK_API_KEY="your-api-key"

# Mistral (has free tier)
export MISTRAL_API_KEY="your-api-key"

# Groq (free tier)
export GROQCLOUD_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# OpenAI
export OPENAI_API_KEY="your-api-key"

# Google
export GOOGLE_API_KEY="your-api-key"

# xAI (Grok)
export XAI_API_KEY="your-api-key"

# ... (see table above for all providers)

# Run tests with all available providers
pytest tests/integration/real_execution/test_all_providers.py -v -m real_execution
```

## How Tests Work

### 1. Provider Detection

Tests automatically detect available providers:

```
1. Check if local providers are running (Ollama, LlamaCpp)
2. Check environment variables for cloud provider API keys
3. Test API keys with simple calls (detects billing/auth errors)
4. Skip tests for unavailable providers automatically
```

### 2. API Key Validation

When a cloud provider API key is found:

1. **Test Call**: Makes a minimal API call (e.g., "Hi" → "Hello")
2. **Error Classification**:
   - **Auth Error**: Invalid/expired key → Skip
   - **Billing Error**: Credit limits exceeded → Skip
   - **Rate Limit**: Too many requests → Skip
   - **Network Error**: Connection issues → Skip
3. **Graceful Skip**: Test is marked as SKIPPED with helpful reason

### 3. Test Execution

For each available provider:

```python
test_provider_read_tool[ollama]      ✓ 45.2s
test_provider_read_tool[deepseek]    ✓ 12.3s
test_provider_read_tool[mistral]     ✓ 15.7s
test_provider_read_tool[openai]      ✗ SKIP (OPENAI_API_KEY not set)
test_provider_read_tool[xai]          ✗ SKIP (XAI_API_KEY not set)
...
```

## GitHub Actions Integration

### Setting Up Secrets

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add repository secrets for providers you want to test:
   ```
   ANTHROPIC_API_KEY
   OPENAI_API_KEY
   GOOGLE_API_KEY
   DEEPSEEK_API_KEY
   MISTRAL_API_KEY
   GROQCLOUD_API_KEY
   XAI_API_KEY
   ZAI_API_KEY
   MOONSHOT_API_KEY
   TOGETHER_API_KEY
   OPENROUTER_API_KEY
   FIREWORKS_API_KEY
   CEREBRAS_API_KEY
   HF_TOKEN
   ```

3. Push changes or manually trigger workflow

### Conditional Test Execution

Jobs only run if API keys are set:

```yaml
test-deepseek:
  if: secrets.DEEPSEEK_API_KEY != ''
  steps:
    - name: Run DeepSeek tests
      env:
        DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
      run: pytest ... -k "deepseek"
```

## Test Files

### `test_all_providers.py`
Parametrized tests across ALL 21 providers:

- `test_provider_read_tool` - Read tool (parametrized across all providers)
- `test_provider_shell_tool` - Shell tool (parametrized across all providers)
- `test_provider_multi_tool` - Multi-turn conversation (parametrized across all providers)
- `test_provider_simple_query` - Basic connectivity (parametrized across all providers)
- `test_provider_summary` - Display provider availability

### `conftest_all_providers.py`
Fixtures for all 21 providers with:
- Automatic API key detection
- API key validation with test calls
- Billing/auth error detection
- Graceful skipping

## Running Specific Tests

### Test All Providers

```bash
pytest tests/integration/real_execution/test_all_providers.py -v -m real_execution
```

### Test Specific Provider

```bash
# Ollama only
pytest tests/integration/real_execution/test_all_providers.py -v -k "ollama"

# DeepSeek only
pytest tests/integration/real_execution/test_all_providers.py -v -k "deepseek"

# Multiple providers
pytest tests/integration/real_execution/test_all_providers.py -v -k "ollama or deepseek or mistral"
```

### Test Specific Function

```bash
# Only test simple query
pytest tests/integration/real_execution/test_all_providers.py::test_provider_simple_query -v

# Only test read tool
pytest tests/integration/real_execution/test_all_providers.py::test_provider_read_tool -v
```

## Troubleshooting

### Tests are Skipped

**Problem**: All tests show SKIPPED

**Solution**:
```bash
# Check provider availability
pytest tests/integration/real_execution/test_all_providers.py::test_provider_summary -v

# Ensure at least one provider is available:
# - Ollama: brew install ollama && ollama serve
# - Cloud: export DEEPSEEK_API_KEY="..."
```

### API Key Errors

**Problem**: Test shows "API key validation failed"

**Solutions**:
1. Verify API key is set:
   ```bash
   echo $DEEPSEEK_API_KEY
   ```

2. Check key is valid (not expired):
   - Visit provider dashboard
   - Generate new API key if needed

3. Check billing/credits:
   - Ensure account has credits
   - Check rate limits

### Ollama Connection Refused

**Problem**: `Ollama not available at localhost:11434`

**Solutions**:
1. Start Ollama:
   ```bash
   ollama serve
   ```

2. Check it's running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. Pull model:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

## Cost Estimates

Per full test run (all providers available):

| Provider | Estimated Cost |
|----------|---------------|
| Ollama | **Free** (local) |
| Groq | **Free** (15K RPM free tier) |
| Mistral | **Free** (500K tokens/min) |
| Together | **Free** ($25 credits) |
| DeepSeek | ~$0.01 |
| Google | ~$0.01 |
| Anthropic | ~$0.02 |
| OpenAI | ~$0.02 |
| xAI | ~$0.03 |
| **Total (all providers)** | **~$0.15 per full run** |

**For CI/CD**: Run tests only with cheapest providers (Ollama, Groq, Mistral) for free testing.

## Adding New Providers

To add a new provider:

1. **Add to PROVIDER_CONFIG** in `conftest_all_providers.py`:
```python
"newprovider": {
    "class": NewProvider,
    "model": "cheapest-model",
    "models": ["cheapest-model", "alternative"],
    "api_key_env": "NEWPROVIDER_API_KEY",
    "type": "cloud",
    "cost_tier": "free",
    "description": "Provider description",
},
```

2. **Add to ALL_PROVIDERS list**:
```python
ALL_PROVIDERS = [..., "newprovider"]
```

3. **Add to get_provider_env_vars()**:
```python
"newprovider": ["NEWPROVIDER_API_KEY"],
```

## Best Practices

### For Local Development

Use **Ollama** for free, fast local testing:
```bash
# Install and run Ollama once
brew install ollama && ollama serve

# Pull a fast model
ollama pull qwen2.5-coder:7b

# Run tests anytime
pytest tests/integration/real_execution/test_all_providers.py -v -k "ollama"
```

### For CI/CD

Use **free-tier providers** to minimize costs:
- Groq Cloud (free tier: 15K RPM)
- Mistral (free tier: 500K tokens/min)

```yaml
- name: Run integration tests (free providers only)
  env:
    GROQCLOUD_API_KEY: ${{ secrets.GROQCLOUD_API_KEY }}
    MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
  run: |
    pytest tests/integration/real_execution/test_all_providers.py \
      -v \
      -k "groqcloud or mistral"
```

### For Pre-Release Testing

Test with **multiple providers** to ensure compatibility:
- 1 local provider (Ollama)
- 1-2 cloud providers (DeepSeek, Mistral, etc.)

## API Key Security

⚠️ **IMPORTANT**: Never commit API keys to repository!

### Best Practices

1. **Use Environment Variables** (recommended):
   ```bash
   export DEEPSEEK_API_KEY="sk-..."
   ```

2. **Use GitHub Secrets** for CI/CD

3. **Use Victor's Keyring** for local development:
   ```bash
   victor keys --set deepseek --keyring
   ```

4. **Check for exposed keys**:
   ```bash
   # Scan for accidentally committed keys
   git log --all --full-history -S --source --all -- "**" | grep -i "key\|token\|secret"
   ```

## See Also

- [Victor Key Management](../../docs/KEY_MANAGEMENT.md)
- [Provider Documentation](../../docs/PROVIDERS.md)
- [API Keys Registry](../../victor/config/api_keys_registry.yaml)
- [Original Real Execution Tests](./README.md)

## Contributing

When adding new integration tests:

1. Use parametrization with `ALL_PROVIDERS`
2. Make tests robust (handle model variations)
3. Add multiple success indicators
4. Test locally first with Ollama
5. Document provider-specific behavior

## Summary

This all-provider test infrastructure provides:
- ✅ **Complete coverage** - All 21 Victor providers supported
- ✅ **Automatic detection** - Finds available providers automatically
- ✅ **Robust validation** - Tests API keys with actual calls
- ✅ **Graceful degradation** - Skips on billing/auth errors
- ✅ **Cost-optimized** - Uses cheapest models for each provider
- ✅ **CI/CD ready** - Works with GitHub Actions Secrets
- ✅ **Developer friendly** - Works locally with Ollama

Run tests with confidence knowing they'll automatically skip for providers you don't have configured!
