# Provider Support Guide

## Overview

CodingAgent supports all major LLM providers through a unified interface. This guide covers setup and usage for each provider.

## Supported Providers

### ✅ Fully Implemented

| Provider | Status | Models | Tool Calling | Streaming |
|----------|--------|--------|--------------|-----------|
| **Ollama** | ✅ Ready | All Ollama models | ✅ Yes | ✅ Yes |
| **Anthropic** | ✅ Ready | Claude 3.5, 3 Opus/Sonnet/Haiku | ✅ Yes | ✅ Yes |
| **OpenAI** | ✅ Ready | GPT-4, GPT-3.5, etc. | ✅ Yes | ✅ Yes |
| **Google** | ✅ Ready | Gemini 1.5 Pro/Flash | ✅ Yes | ✅ Yes |
| **xAI (Grok)** | ✅ Ready | Grok Beta, Grok Vision | ✅ Yes | ✅ Yes |

### 🚧 Coming Soon

| Provider | Status | ETA |
|----------|--------|-----|
| **Cohere** | 📋 Planned | v0.3.0 |
| **Mistral** | 📋 Planned | v0.3.0 |
| **LMStudio** | 📋 Planned | v0.2.0 |
| **vLLM** | 📋 Planned | v0.2.0 |

---

## Provider Setup

### 1. Ollama (Local Models)

**No API key needed!** Just install and run Ollama.

#### Installation

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve
```

#### Pull Models

```bash
# Recommended coding models
ollama pull qwen2.5-coder:7b      # Best for coding
ollama pull deepseek-coder:6.7b   # Great code completion
ollama pull codellama:13b          # Meta's coding model
ollama pull llama3:8b              # General purpose

# List installed models
ollama list
```

#### Configuration

```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

providers:
  ollama:
    base_url: http://localhost:11434
    timeout: 300
```

#### Usage

```bash
victor --profile default
```

---

### 2. Anthropic Claude

**Sign up**: https://console.anthropic.com/

#### Get API Key

1. Go to https://console.anthropic.com/
2. Navigate to API Keys
3. Create new key

#### Configuration

```bash
# Set environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Or in .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

```yaml
profiles:
  claude-sonnet:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0
    max_tokens: 8192

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    timeout: 60
```

#### Available Models

- `claude-sonnet-4-5` - Latest Sonnet (best balance)
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Balanced
- `claude-3-haiku-20240307` - Fast & efficient

#### Usage

```bash
victor --profile claude-sonnet "Write a REST API"
```

---

### 3. OpenAI GPT

**Sign up**: https://platform.openai.com/

#### Get API Key

1. Go to https://platform.openai.com/api-keys
2. Create new secret key

#### Configuration

```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or in .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

```yaml
profiles:
  gpt4:
    provider: openai
    model: gpt-4-turbo-preview
    temperature: 0.8
    max_tokens: 4096

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    organization: ${OPENAI_ORG_ID}  # Optional
    timeout: 60
```

#### Available Models

- `gpt-4-turbo-preview` - Latest GPT-4 Turbo
- `gpt-4` - Standard GPT-4
- `gpt-3.5-turbo` - Fast and affordable

#### Usage

```bash
victor --profile gpt4 "Debug this code"
```

---

### 4. Google Gemini

**Sign up**: https://makersuite.google.com/

#### Get API Key

1. Go to https://makersuite.google.com/app/apikey
2. Create API key

#### Configuration

```bash
# Set environment variable
export GOOGLE_API_KEY="..."

# Or in .env file
echo "GOOGLE_API_KEY=..." >> .env
```

```yaml
profiles:
  gemini:
    provider: google
    model: gemini-1.5-pro
    temperature: 0.9
    max_tokens: 8192

providers:
  google:
    api_key: ${GOOGLE_API_KEY}
    timeout: 60
```

#### Available Models

- `gemini-1.5-pro` - Most capable, multimodal
- `gemini-1.5-flash` - Fast and efficient

#### Usage

```bash
victor --profile gemini "Analyze this architecture"
```

---

### 5. xAI Grok

**Sign up**: https://console.x.ai/

#### Get API Key

1. Go to https://console.x.ai/
2. Create API key

#### Configuration

```bash
# Set environment variable
export XAI_API_KEY="xai-..."

# Or in .env file
echo "XAI_API_KEY=xai-..." >> .env
```

```yaml
profiles:
  grok:
    provider: xai
    model: grok-beta
    temperature: 0.8
    max_tokens: 4096

providers:
  xai:
    api_key: ${XAI_API_KEY}
    base_url: https://api.x.ai/v1
    timeout: 60
```

#### Available Models

- `grok-beta` - Latest Grok model
- `grok-vision-beta` - Grok with vision capabilities

#### Usage

```bash
victor --profile grok "Explain quantum computing"
```

---

## Provider Comparison

### Cost (Approximate)

| Provider | Input (per 1M tokens) | Output (per 1M tokens) |
|----------|----------------------|------------------------|
| **Ollama** | FREE | FREE |
| **Claude Sonnet** | $3 | $15 |
| **GPT-4 Turbo** | $10 | $30 |
| **GPT-3.5 Turbo** | $0.50 | $1.50 |
| **Gemini Pro** | $0.50 | $1.50 |
| **Grok** | TBD | TBD |

### Speed

| Provider | Relative Speed | Best For |
|----------|---------------|----------|
| **Ollama** | Depends on hardware | Development, privacy |
| **GPT-3.5** | ⚡⚡⚡ Very Fast | Quick tasks |
| **Gemini Flash** | ⚡⚡⚡ Very Fast | Fast responses |
| **Claude Haiku** | ⚡⚡ Fast | Balanced speed/quality |
| **Claude Sonnet** | ⚡ Moderate | Best quality |
| **GPT-4** | ⚡ Moderate | Complex reasoning |

### Capabilities

| Feature | Ollama | Claude | GPT-4 | Gemini | Grok |
|---------|--------|--------|-------|--------|------|
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Vision** | 🔄 Some | ❌ | ✅ | ✅ | ✅ |
| **Long Context** | 🔄 Varies | ✅ 200K | ✅ 128K | ✅ 1M | ✅ |
| **Code Focus** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Multi-Provider Workflows

### Strategy 1: Local Dev + Cloud Production

```yaml
profiles:
  dev:
    provider: ollama
    model: qwen2.5-coder:7b

  prod:
    provider: anthropic
    model: claude-sonnet-4-5
```

```bash
# Development (free)
victor --profile dev

# Production (when quality matters)
victor --profile prod
```

### Strategy 2: Task-Specific Providers

```yaml
profiles:
  quick:
    provider: openai
    model: gpt-3.5-turbo

  complex:
    provider: anthropic
    model: claude-sonnet-4-5

  vision:
    provider: google
    model: gemini-1.5-pro
```

### Strategy 3: Cost Optimization

```bash
# Prototype with free local models
victor --profile ollama "Draft initial implementation"

# Refine with cheaper cloud model
victor --profile gpt35 "Improve the code"

# Final review with best model
victor --profile claude-sonnet "Review and polish"
```

---

## Switching Providers

### During Runtime

Not yet implemented - coming in v0.3.0

Future syntax:
```
> use claude-sonnet
> use grok
```

### For Each Command

```bash
victor --profile ollama "Task 1"
victor --profile claude "Task 2"
victor --profile gpt4 "Task 3"
```

---

## Provider-Specific Tips

### Ollama
- ✅ Use for development and testing
- ✅ Great for privacy-sensitive work
- ⚠️ Performance depends on hardware
- ⚠️ Model quality varies

### Anthropic Claude
- ✅ Excellent at complex reasoning
- ✅ Very good at following instructions
- ✅ Strong coding capabilities
- 💰 Mid-range pricing

### OpenAI GPT
- ✅ Wide range of models
- ✅ GPT-4 is very capable
- ✅ GPT-3.5 is fast and cheap
- 💰 GPT-4 is expensive

### Google Gemini
- ✅ Huge context window (1M tokens)
- ✅ Good multimodal support
- ✅ Affordable pricing
- ⚠️ Newer platform

### xAI Grok
- ✅ Access to real-time information
- ✅ Strong reasoning
- ⚠️ Newer model, less tested
- ⚠️ Pricing TBD

---

## Troubleshooting

### Provider Not Found

```bash
Error: Provider 'xyz' not found
```

**Solution:**
```bash
# List available providers
python -c "from victor.providers.registry import ProviderRegistry; print(ProviderRegistry.list_providers())"

# Output: ['ollama', 'anthropic', 'openai', 'google', 'xai', 'grok']
```

### Authentication Failed

```bash
Error: Authentication failed
```

**Solution:**
```bash
# Check API key is set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Verify .env file
cat .env

# Test API key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json"
```

### Rate Limit Exceeded

```bash
Error: Rate limit exceeded
```

**Solution:**
- Wait and retry
- Switch to different provider temporarily
- Upgrade your plan with provider
- Use local Ollama for development

---

## Contributing New Providers

Want to add a provider? See [CONTRIBUTING.md](CONTRIBUTING.md)

Template:
1. Create `victor/providers/your_provider.py`
2. Extend `BaseProvider`
3. Implement `chat()`, `stream()`, `supports_tools()`
4. Register in `registry.py`
5. Add tests
6. Update documentation

---

## Provider Feature Matrix

| Feature | Status | Providers |
|---------|--------|-----------|
| Chat Completions | ✅ | All |
| Streaming | ✅ | All |
| Tool Calling | ✅ | All |
| Vision/Multimodal | 🔄 | Gemini, GPT-4V, Grok (planned) |
| Embeddings | 📋 | Planned v0.4 |
| Fine-tuning | 📋 | Planned v0.5 |

---

## FAQ

**Q: Which provider should I use?**
A: Start with Ollama for free development, then use Claude Sonnet or GPT-4 for production.

**Q: Can I use multiple providers in one session?**
A: Not yet - coming in v0.3.0. Currently restart with different profile.

**Q: Do all providers support tools?**
A: Yes! All implemented providers support tool/function calling.

**Q: Which is the cheapest?**
A: Ollama (free), then GPT-3.5 Turbo or Gemini Flash.

**Q: Which is the best for coding?**
A: Claude Sonnet, GPT-4, or local Qwen2.5-Coder are all excellent.

**Q: How do I add a new provider?**
A: See [CONTRIBUTING.md](CONTRIBUTING.md) for the provider template.

---

Ready to start? Pick a provider and run:

```bash
victor --profile <profile-name>
```

Happy coding! 🚀
