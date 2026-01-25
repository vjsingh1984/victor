# Multi-Provider Integration Tests

This directory contains real execution integration tests that support multiple LLM providers (Ollama, DeepSeek, xAI, Mistral, OpenAI, ZAI).

## Overview

Tests are designed to:
- Run with **any available provider** (local or cloud)
- **Automatically skip** when providers are not available
- Work seamlessly in **CI/CD environments** like GitHub Actions
- Use **cheapest/fastest models** for cost-effective testing

## Supported Providers

| Provider | Type | API Key Env Var | Cheapest Model | Free Tier |
|----------|------|-----------------|----------------|-----------|
| **Ollama** | Local | None | qwen2.5-coder:7b | ✅ Yes (local) |
| **DeepSeek** | Cloud | `DEEPSEEK_API_KEY` | deepseek-chat | ❌ No |
| **xAI** | Cloud | `XAI_API_KEY` | grok-beta | ❌ No |
| **Mistral** | Cloud | `MISTRAL_API_KEY` | mistral-small-latest | ✅ Yes (500K tokens/min) |
| **OpenAI** | Cloud | `OPENAI_API_KEY` | gpt-4o-mini | ❌ No |
| **ZAI** | Cloud | `ZAI_API_KEY` | glm-4-flash | ❌ No |

## Running Tests Locally

### Option 1: Run with Ollama (Free, Local)

```bash
# Install and start Ollama
brew install ollama
ollama serve

# Pull a model (in another terminal)
ollama pull qwen2.5-coder:14b

# Run tests
pytest tests/integration/real_execution/test_real_tool_execution.py -v -m real_execution
```

### Option 2: Run with Cloud Providers

Set one or more API keys:

```bash
# DeepSeek (cheapest cloud option)
export DEEPSEEK_API_KEY="your-api-key"

# Mistral (has free tier)
export MISTRAL_API_KEY="your-api-key"

# xAI (Grok)
export XAI_API_KEY="your-api-key"

# OpenAI
export OPENAI_API_KEY="your-api-key"

# ZAI
export ZAI_API_KEY="your-api-key"

# Run tests (will use first available provider)
pytest tests/integration/real_execution/test_multi_provider_execution.py -v -m real_execution
```

### Option 3: Run Tests for Specific Provider

```bash
# Only test with DeepSeek
pytest tests/integration/real_execution/test_multi_provider_execution.py \
  -v \
  -m "real_execution" \
  -k "deepseek"

# Only test with OpenAI
pytest tests/integration/real_execution/test_multi_provider_execution.py \
  -v \
  -m "real_execution" \
  -k "openai"
```

## Running All Tests

Run all real execution tests (will skip unavailable providers):

```bash
# Run all real execution tests
pytest tests/integration/real_execution/ -v -m real_execution --timeout=300
```

## GitHub Actions Integration

The `.github/workflows/integration-tests.yml` file demonstrates how to run these tests in CI/CD:

### Setting Up GitHub Secrets

To enable cloud provider tests in GitHub Actions:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add new repository secrets:
   - `DEEPSEEK_API_KEY` - DeepSeek API key
   - `XAI_API_KEY` - xAI/Grok API key
   - `MISTRAL_API_KEY` - Mistral API key
   - `OPENAI_API_KEY` - OpenAI API key
   - `ZAI_API_KEY` - ZAI API key

3. Push changes or manually trigger the workflow

### Workflow Behavior

```yaml
# Only runs if DEEPSEEK_API_KEY is set in GitHub Secrets
deepseek-tests:
  if: secrets.DEEPSEEK_API_KEY != ''
  steps:
    - name: Run DeepSeek tests
      env:
        DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
      run: pytest ... -k "deepseek"
```

## Cost Optimization

Tests use the **cheapest models** for each provider:

| Provider | Model | Approx Cost (per 1M tokens) |
|----------|-------|---------------------------|
| Ollama | qwen2.5-coder:7b | **Free** (local) |
| DeepSeek | deepseek-chat | ~$0.14 (input) / ~$0.28 (output) |
| Mistral | mistral-small-latest | **Free tier** (500K tokens/min) |
| xAI | grok-beta | ~$0.50 (estimated) |
| OpenAI | gpt-4o-mini | ~$0.15 (input) / ~$0.60 (output) |

**Estimated cost per full test run**: $0.01 - $0.05 (depending on provider)
