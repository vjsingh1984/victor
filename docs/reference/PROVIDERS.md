# Provider Support Guide

## Overview

Victor supports all major LLM providers through a unified interface. This guide covers setup and usage for each provider.

## Supported Providers (25+)

Victor supports 25+ LLM providers organized by category. For detailed setup instructions, see [Provider Setup Guide](../guides/PROVIDER_SETUP.md).

### Local Providers (Free, Private)

| Provider | Models | Tool Calling | Streaming |
|----------|--------|--------------|-----------|
| **Ollama** | 100+ models (Llama, Qwen, DeepSeek, etc.) | ✅ | ✅ |
| **LMStudio** | Any GGUF model | ✅ | ✅ |
| **vLLM** | Any HuggingFace model | ✅ | ✅ |

### Premium Cloud Providers

| Provider | Models | Tool Calling | Streaming |
|----------|--------|--------------|-----------|
| **Anthropic** | Claude Opus 4.5, Sonnet, Haiku | ✅ | ✅ |
| **OpenAI** | GPT-4o, GPT-4, GPT-4o-mini | ✅ | ✅ |
| **Google** | Gemini 2.5 Pro/Flash | ✅ | ✅ |
| **xAI** | Grok 2, Grok 3 | ✅ | ✅ |

### Free-Tier Cloud Providers

| Provider | Free Tier | Models | Tool Calling | Streaming |
|----------|-----------|--------|--------------|-----------|
| **Groq** | 14K TPM | Llama, Mixtral | ✅ | ✅ |
| **DeepSeek** | 1M tokens | DeepSeek-V3, R1 | ✅ | ✅ |
| **Mistral** | 500K TPM | Mistral Large, Codestral | ✅ | ✅ |
| **Together** | $25 credits | 100+ models | ✅ | ✅ |
| **Moonshot** | 1M tokens | Kimi K2 | ✅ | ✅ |
| **Fireworks** | $1 credits | Llama, Mixtral | ✅ | ✅ |
| **OpenRouter** | Daily limits | Gateway to 100+ | ✅ | ✅ |
| **Cerebras** | Unlimited | Llama 3.1, 3.3 | ✅ | ✅ |

### Enterprise Cloud Providers

| Provider | Models | Tool Calling | Streaming |
|----------|--------|--------------|-----------|
| **Google Vertex AI** | Gemini (enterprise) | ✅ | ✅ |
| **Azure OpenAI** | GPT-4, Phi models | ✅ | ✅ |
| **AWS Bedrock** | Claude, Llama, Mistral, Titan | ✅ | ✅ |
| **HuggingFace** | 1000s of open models | ✅ | ✅ |
| **Replicate** | Open models (pay-per-second) | ✅ | ✅ |

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
    model: gpt-4o
    temperature: 0.8
    max_tokens: 4096

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    organization: ${OPENAI_ORG_ID}  # Optional
    timeout: 60
```

#### Available Models

- `gpt-4o` - Latest GPT-4 Omni (recommended)
- `gpt-4o-mini` - Fast and affordable (replaces GPT-3.5 Turbo)
- `gpt-4-turbo` - GPT-4 Turbo

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
    model: gemini-2.5-pro
    temperature: 0.9
    max_tokens: 8192

providers:
  google:
    api_key: ${GOOGLE_API_KEY}
    timeout: 60
```

#### Available Models

- `gemini-2.5-pro` - Latest and most capable
- `gemini-1.5-pro` - Multimodal, 1M context window
- `gemini-1.5-flash` - Fast and efficient

#### Safety Settings

The Google provider supports configurable safety filters for code generation:

```python
from victor.providers.google_provider import GoogleProvider

# Default: No blocking (best for code generation)
provider = GoogleProvider(api_key=key, safety_level="block_none")

# Available levels (least to most restrictive):
# - "block_none"  - No content filtering (default, recommended for coding)
# - "block_few"   - Block only high probability harmful content
# - "block_some"  - Block medium and above
# - "block_most"  - Most restrictive
```

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

### 6. LMStudio (Local Models)

**No API key needed!** LMStudio uses OpenAI-compatible API.

#### Installation

1. Download from https://lmstudio.ai/
2. Install the application
3. Launch LMStudio

#### Download Models

1. Open LMStudio
2. Go to "Discover" tab
3. Search for coding models:
   - **Qwen2.5-Coder-7B-Instruct** (recommended)
   - **CodeLlama-7B-Instruct**
   - **DeepSeek-Coder-6.7B-Instruct**
4. Click "Download"

#### Start Local Server

1. Go to "Local Server" tab
2. Select your downloaded model
3. Click "Start Server"
4. Note the server URL (default: `http://localhost:1234`)

#### Configuration

```bash
# No API key required, just a placeholder
export LMSTUDIO_API_KEY="lm-studio"
```

```yaml
profiles:
  lmstudio:
    provider: lmstudio  # Uses OpenAI-compatible API
    model: local-model
    temperature: 0.3
    max_tokens: 4096

providers:
  lmstudio:
    base_url:
      - http://127.0.0.1:1234      # Primary LMStudio server (localhost)
      # Add LAN servers as needed:
      # - http://your-lan-server:1234
    api_key: lm-studio  # Placeholder; LMStudio ignores it but OpenAI client requires a token
    timeout: 300
```

If `~/.victor/profiles.yaml` does not exist, Victor will probe the LMStudio tiered URLs above, list their models via `/v1/models`, and choose a sensible default (prefers `qwen2.5-coder` family, otherwise first available).

Connectivity check:
```bash
python scripts/check_lmstudio.py
```
Shows which LMStudio endpoint is reachable, which models it exposes, and (if VRAM is detected) recommends the most capable model that fits your GPU.
You can cap selection with `lmstudio_max_vram_gb` in settings (default 48 GB) to keep under a target budget.

Interactive (inside Victor):
```
/lmstudio
```
Probes the tiered endpoints and shows a VRAM-aware recommendation.

#### Usage

```bash
victor --profile lmstudio "Write a Python function"
```

#### Model Sharing with Ollama

**Save disk space** by sharing GGUF models between Ollama and LMStudio!

```bash
# Install Gollama
go install github.com/sammcj/gollama@HEAD

# Link Ollama models to LMStudio
~/go/bin/gollama -L

# Result: 27 models linked, ~300GB disk space saved!
```

See [MODEL_SHARING_GUIDE.md](MODEL_SHARING_GUIDE.md) for details.

---

### 7. vLLM (High-Performance Inference)

**No API key needed!** vLLM provides OpenAI-compatible API.

#### Installation

```bash
# Install vLLM
pip install vllm

# Or with specific CUDA version:
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

#### Start vLLM Server

```bash
# Start with a model
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0

# Or for CPU-only (slower):
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000 \
  --device cpu
```

#### Recommended Models

- `Qwen/Qwen2.5-Coder-7B-Instruct` - Best for coding
- `codellama/CodeLlama-7b-Instruct-hf`
- `deepseek-ai/deepseek-coder-6.7b-instruct`

#### Configuration

```bash
# No real API key needed
export VLLM_API_KEY="EMPTY"
```

```yaml
profiles:
  vllm:
    provider: openai  # Uses OpenAI-compatible API
    model: Qwen/Qwen2.5-Coder-7B-Instruct
    temperature: 0.3
    max_tokens: 4096

providers:
  openai:
    base_url: http://localhost:8000/v1  # vLLM server
    api_key: EMPTY
    timeout: 300
```

#### Usage

```bash
victor --profile vllm "Optimize this code"
```

#### Performance Tuning

```bash
# Optimize for GPU memory
python -m vllm.entrypoints.openai.api_server \
  --model MODEL_NAME \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 256
```

---

## Provider Comparison

### Cost (Approximate)

| Provider | Input (per 1M tokens) | Output (per 1M tokens) |
|----------|----------------------|------------------------|
| **Ollama** | FREE | FREE |
| **LMStudio** | FREE | FREE |
| **vLLM** | FREE | FREE |
| **Claude Sonnet** | $3 | $15 |
| **GPT-4o** | $2.50 | $10 |
| **GPT-4o mini** | $0.15 | $0.60 |
| **Gemini 2.5 Pro** | $1.25 | $5 |
| **Grok** | TBD | TBD |

### Speed

| Provider | Relative Speed | Best For |
|----------|---------------|----------|
| **Ollama** | Depends on hardware | Development, privacy |
| **LMStudio** | Depends on hardware | GUI model management |
| **vLLM** | ⚡⚡⚡ Very Fast | High-throughput inference |
| **GPT-4o mini** | ⚡⚡⚡ Very Fast | Quick tasks |
| **Gemini Flash** | ⚡⚡⚡ Very Fast | Fast responses |
| **Claude Haiku** | ⚡⚡ Fast | Balanced speed/quality |
| **Claude Sonnet** | ⚡ Moderate | Best quality |
| **GPT-4o** | ⚡⚡ Fast | Complex reasoning |

### Capabilities

| Feature | Ollama | LMStudio | vLLM | Claude | GPT-4 | Gemini | Grok |
|---------|--------|----------|------|--------|-------|--------|------|
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Vision** | 🔄 Some | 🔄 Some | 🔄 Some | ❌ | ✅ | ✅ | ✅ |
| **Long Context** | 🔄 Varies | 🔄 Varies | 🔄 Varies | ✅ 200K | ✅ 128K | ✅ 1M | ✅ |
| **Code Focus** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Model Sharing** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

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
    model: gpt-4o-mini

  complex:
    provider: anthropic
    model: claude-sonnet-4-5

  vision:
    provider: google
    model: gemini-2.5-pro
```

### Strategy 3: Cost Optimization

```bash
# Prototype with free local models
victor --profile ollama "Draft initial implementation"

# Refine with cheaper cloud model
victor --profile gpt4o-mini "Improve the code"

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
A: Ollama (free), then GPT-4o mini or Gemini Flash.

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
