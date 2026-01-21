# Victor AI 0.5.1 Provider Reference

Complete reference for all 21 supported LLM providers.

**Table of Contents**
- [Overview](#overview)
- [Local Providers](#local-providers)
  - [Ollama](#ollama)
  - [LMStudio](#lmstudio)
  - [vLLM](#vllm)
  - [Llama.cpp](#llamacpp)
- [Major Cloud Providers](#major-cloud-providers)
  - [Anthropic](#anthropic)
  - [OpenAI](#openai)
  - [Google](#google)
  - [Azure OpenAI](#azure-openai)
  - [AWS Bedrock](#aws-bedrock)
  - [Google Vertex AI](#google-vertex-ai)
- [AI Research Companies](#ai-research-companies)
  - [xAI](#xai)
  - [DeepSeek](#deepseek)
  - [Moonshot](#moonshot)
  - [Zhipu AI (ZAI)](#zhipu-ai-zai)
- [Free-Tier Providers (2025)](#free-tier-providers-2025)
  - [Groq](#groq)
  - [Mistral](#mistral)
  - [Together](#together)
  - [OpenRouter](#openrouter)
  - [Fireworks](#fireworks)
  - [Cerebras](#cerebras)
- [Enterprise/Other](#enterpriseother)
  - [Hugging Face](#hugging-face)
  - [Replicate](#replicate)
- [Provider Switching](#provider-switching)
- [Model Capabilities](#model-capabilities)

---

## Overview

Victor AI supports 21 LLM providers through a unified `BaseProvider` interface. All providers implement:

- `chat()`: Non-streaming chat completion
- `stream_chat()`: Streaming chat completion
- `supports_tools()`: Tool calling capability query
- `name`: Provider identifier

### Provider Categories

1. **Local Providers** (4): Run models locally, air-gapped compatible
2. **Major Cloud Providers** (6): Enterprise-grade cloud APIs
3. **AI Research Companies** (4): Specialized research labs
4. **Free-Tier Providers** (6): Low-cost/free tiers (2025 wave)
5. **Enterprise/Other** (2): Specialized enterprise platforms

### Quick Start

```python
from victor.providers.registry import ProviderRegistry

# List all providers
providers = ProviderRegistry.list_providers()
print(providers)
# ['ollama', 'lmstudio', 'vllm', 'llamacpp', 'anthropic', 'openai', ...]

# Create provider instance
provider = ProviderRegistry.create(
    name="anthropic",
    api_key="sk-ant-...",
    model="claude-sonnet-4-5"
)

# Use provider
response = await provider.chat(messages=[...])
```

---

## Local Providers

### Ollama

**Category:** Local
**Air-gapped:** Yes
**Tool Support:** Yes (via tool calling adapter)

**Description:** Run LLMs locally using Ollama. Supports 100+ models including Llama, Mistral, Qwen, etc.

**Supported Models:**
- `llama3.2`, `llama3.1`, `llama3`
- `qwen2.5:32b`, `qwen2.5:14b`, `qwen2.5:7b`
- `mistral`, `mixtral`
- `codellama`, `deepseek-coder`
- `gemma2`, `phi3`

**Installation:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5:32b
```

**Configuration:**
```python
from victor.providers.registry import ProviderRegistry

provider = ProviderRegistry.create(
    name="ollama",
    model="qwen2.5:32b",
    base_url="http://localhost:11434",  # Default
    timeout=120,  # Seconds
)
```

**Environment Variables:**
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
```

**Performance Characteristics:**
- **Latency:** 2-10s per response (hardware dependent)
- **Quality:** Good for coding (Qwen2.5:32b recommended)
- **Cost:** Free (local compute)
- **Best For:** Privacy-sensitive work, offline development

---

### LMStudio

**Category:** Local
**Air-gapped:** Yes
**Tool Support:** Yes

**Description:** Local inference with GUI model manager. Good for experimentation and model comparison.

**Supported Models:**
- All GGUF format models (Llama, Mistral, Qwen, etc.)
- Auto-downloads from Hugging Face

**Installation:**
```bash
# Download from https://lmstudio.ai/
# Install and start server
lmstudio server
```

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="lmstudio",
    model="lmstudio-community/Qwen2.5-32B-Instruct-GGUF",
    base_url="http://localhost:1234",  # Default
)
```

**Performance:**
- **Latency:** 3-15s per response
- **Quality:** Good (depends on model)
- **Best For:** Visual model selection, GUI workflow

---

### vLLM

**Category:** Local
**Air-gapped:** Yes
**Tool Support:** Experimental

**Description:** High-throughput serving engine with PagedAttention. Best for production local serving.

**Supported Models:**
- All Hugging Face models
- Optimized for batched inference

**Installation:**
```bash
pip install vllm
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000
```

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="vllm",
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="fake",  # vLLM requires this but doesn't validate
)
```

**Performance:**
- **Latency:** 1-5s (fastest local option)
- **Throughput:** 10-100x other local options
- **Best For:** Production local serving, batch inference

---

### Llama.cpp

**Category:** Local
**Air-gapped:** Yes
**Tool Support:** Experimental

**Description:** Lightweight C++ inference. Minimal resource usage.

**Supported Models:**
- GGUF format models

**Installation:**
```bash
# Build from source or download binary
./llama-server --model ./qwen2.5-32b.Q4_K_M.gguf --port 8080
```

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="llamacpp",
    model="qwen2.5-32b",
    base_url="http://localhost:8080",
)
```

**Aliases:** `llama-cpp`, `llama.cpp`

**Performance:**
- **Memory:** Lowest resource usage
- **Latency:** 5-20s per response
- **Best For:** Resource-constrained environments

---

## Major Cloud Providers

### Anthropic (Claude)

**Category:** Cloud
**Tool Support:** Native (all models)
**Streaming:** Yes

**Description:** Enterprise-grade AI with strong focus on safety and helpfulness.

**Supported Models:**
- `claude-sonnet-4-5` (Recommended) - Best for coding
- `claude-sonnet-4-5-20250114` - Specific version
- `claude-3-5-sonnet-20241022` - Previous version
- `claude-3-5-haiku` - Fast, cost-effective
- `claude-3-opus` - Highest quality

**API Key:** https://console.anthropic.com/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="anthropic",
    api_key="sk-ant-...",
    model="claude-sonnet-4-5",
    base_url="https://api.anthropic.com",  # Optional
    timeout=60,  # Seconds
    max_tokens=8192,  # Max tokens per response
)
```

**Environment Variables:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Model Capabilities:**

| Model | Input | Output | Tools | Cost (1M tokens) |
|-------|-------|--------|-------|------------------|
| claude-sonnet-4-5 | 200K | 8K | Native | $3.00 input / $15.00 output |
| claude-3-5-haiku | 200K | 8K | Native | $0.80 input / $4.00 output |
| claude-3-opus | 200K | 4K | Native | $15.00 input / $75.00 output |

**Performance:**
- **Latency:** 1-3s per response
- **Quality:** Excellent for coding
- **Best For:** Production use, complex tasks

---

### OpenAI

**Category:** Cloud
**Tool Support:** Native
**Streaming:** Yes

**Description:** Leading AI research lab with GPT models.

**Supported Models:**
- `gpt-4o` - Multimodal, fast
- `gpt-4o-mini` - Cost-effective
- `gpt-4-turbo` - Previous flagship
- `gpt-3.5-turbo` - Legacy

**API Key:** https://platform.openai.com/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="openai",
    api_key="sk-...",
    model="gpt-4o",
    base_url="https://api.openai.com/v1",
    organization="org-...",  # Optional
)
```

**Environment Variables:**
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_ORGANIZATION="org-..."
```

**Model Capabilities:**

| Model | Input | Output | Tools | Cost (1M tokens) |
|-------|-------|--------|-------|------------------|
| gpt-4o | 128K | 4K | Native | $5.00 input / $15.00 output |
| gpt-4o-mini | 128K | 16K | Native | $0.15 input / $0.60 output |
| gpt-4-turbo | 128K | 4K | Native | $10.00 input / $30.00 output |

**Performance:**
- **Latency:** 1-2s
- **Quality:** Excellent
- **Best For:** General purpose, fast responses

---

### Google

**Category:** Cloud
**Tool Support:** Native
**Streaming:** Yes

**Description:** Google's Gemini models.

**Supported Models:**
- `gemini-2.0-flash-exp` - Experimental, fast
- `gemini-1.5-pro` - Production model
- `gemini-1.5-flash` - Fast, cost-effective

**API Key:** https://aistudio.google.com/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="google",
    api_key="...",
    model="gemini-2.0-flash-exp",
)
```

**Environment Variables:**
```bash
export GOOGLE_API_KEY="..."
```

**Model Capabilities:**

| Model | Input | Output | Tools | Cost (1M tokens) |
|-------|-------|--------|-------|------------------|
| gemini-2.0-flash-exp | 1M | 8K | Native | Free during beta |
| gemini-1.5-pro | 2M | 8K | Native | $3.50 input / $10.50 output |
| gemini-1.5-flash | 1M | 8K | Native | $0.075 input / $0.30 output |

**Aliases:** `gemini`

---

### Azure OpenAI

**Category:** Cloud
**Tool Support:** Native
**Streaming:** Yes

**Description:** OpenAI models hosted on Azure with enterprise features.

**Supported Models:**
- Same as OpenAI (deployed on Azure)

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="azure",
    api_key="...",
    model="gpt-4o",  # Deployment name
    api_base="https://your-resource.openai.azure.com",
    api_version="2024-02-01",
)
```

**Environment Variables:**
```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_VERSION="2024-02-01"
```

**Aliases:** `azure-openai`

**Best For:** Enterprise compliance, data residency

---

### AWS Bedrock

**Category:** Cloud
**Tool Support:** Native (some models)
**Streaming:** Yes

**Description:** AWS managed AI service.

**Supported Models:**
- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `anthropic.claude-3-5-haiku-20241022-v1:0`
- `meta.llama3-1-405b-instruct-v1:0`
- `mistral.mistral-large-2402-v1:0`

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="bedrock",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    # AWS credentials from environment or ~/.aws/credentials
    region_name="us-east-1",
)
```

**Environment Variables:**
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

**Aliases:** `aws`

**Best For:** AWS ecosystem integration

---

### Google Vertex AI

**Category:** Cloud
**Tool Support:** Native
**Streaming:** Yes

**Description:** Enterprise AI platform on Google Cloud.

**Supported Models:**
- `gemini-2.0-flash-exp`
- `gemini-1.5-pro`
- `claude-3-5-sonnet` (via Vertex)

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="vertex",
    model="gemini-2.0-flash-exp",
    project="your-project-id",
    location="us-central1",
    # Credentials from service account JSON
)
```

**Environment Variables:**
```bash
export GOOGLE_CLOUD_PROJECT="your-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

**Aliases:** `vertexai`

**Best For:** GCP integration, enterprise features

---

## AI Research Companies

### xAI (Grok)

**Category:** Cloud
**Tool Support:** None (as of 2025-01)
**Streaming:** Yes

**Description:** Elon Musk's AI company. Grok models.

**Supported Models:**
- `grok-2-1212` - Latest
- `grok-2` - Previous version
- `grok-beta` - Early version

**API Key:** https://x.ai/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="xai",
    api_key="...",
    model="grok-2-1212",
)
```

**Environment Variables:**
```bash
export XAI_API_KEY="..."
```

**Aliases:** `grok`

**Note:** Tool support not available, output deduplication enabled by default due to repetition issues.

---

### DeepSeek

**Category:** Cloud
**Tool Support:** Native
**Streaming:** Yes

**Description:** Chinese AI lab with strong coding models.

**Supported Models:**
- `deepseek-chat` - General purpose
- `deepseek-coder` - Code specialized

**API Key:** https://platform.deepseek.com/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="deepseek",
    api_key="...",
    model="deepseek-chat",
)
```

**Environment Variables:**
```bash
export DEEPSEEK_API_KEY="..."
```

**Cost:** Very low (~$0.14/1M input tokens)

**Best For:** Cost-effective coding

---

### Moonshot (Kimi)

**Category:** Cloud
**Tool Support:** Limited
**Streaming:** Yes

**Description:** Chinese AI lab, strong at long context.

**Supported Models:**
- `moonshot-v1-8k`
- `moonshot-v1-32k`
- `moonshot-v1-128k`

**API Key:** https://platform.moonshot.cn/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="moonshot",
    api_key="...",
    model="moonshot-v1-32k",
)
```

**Environment Variables:**
```bash
export MOONSHOT_API_KEY="..."
```

**Aliases:** `kimi`

**Best For:** Long-context tasks

---

### Zhipu AI (ZAI)

**Category:** Cloud
**Tool Support:** Limited
**Streaming:** Yes

**Description:** Leading Chinese AI research company.

**Supported Models:**
- `glm-4-plus`
- `glm-4-air`
- `glm-4-flash`

**API Key:** https://open.bigmodel.cn/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="zai",
    api_key="...",
    model="glm-4-plus",
)
```

**Environment Variables:**
```bash
export ZHIPUAI_API_KEY="..."
```

**Aliases:** `zhipuai`, `zhipu`

---

## Free-Tier Providers (2025)

### Groq

**Category:** Cloud (Free tier)
**Tool Support:** Native
**Streaming:** Yes

**Description:** Ultra-fast inference with free tier.

**Supported Models:**
- `llama-3.3-70b-versatile`
- `mixtral-8x7b-32768`
- `gemma2-9b-it`

**API Key:** https://console.groq.com/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="groqcloud",
    api_key="gsk_...",
    model="llama-3.3-70b-versatile",
)
```

**Environment Variables:**
```bash
export GROQ_API_KEY="..."
```

**Performance:**
- **Latency:** < 1s (fastest cloud option)
- **Cost:** Free tier available
- **Best For:** Speed-critical applications

---

### Mistral

**Category:** Cloud
**Tool Support:** Native
**Streaming:** Yes

**Description:** European AI company with strong open models.

**Supported Models:**
- `mistral-large-2411` - Flagship
- `mistral-small` - Fast
- `codestral` - Code specialized

**API Key:** https://console.mistral.ai/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="mistral",
    api_key="...",
    model="mistral-large-2411",
)
```

**Environment Variables:**
```bash
export MISTRAL_API_KEY="..."
```

**Cost:** Competitive, free tier available

---

### Together

**Category:** Cloud
**Tool Support:** Native
**Streaming:** Yes

**Description:** Open-source focused platform.

**Supported Models:**
- `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `Qwen/Qwen2.5-72B-Instruct-Turbo`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

**API Key:** https://api.together.xyz/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="together",
    api_key="...",
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
)
```

**Environment Variables:**
```bash
export TOGETHER_API_KEY="..."
```

**Best For:** Open-source model hosting

---

### OpenRouter

**Category:** Cloud (Aggregator)
**Tool Support:** Varies by model
**Streaming:** Yes

**Description:** Unified API for 100+ models from multiple providers.

**Supported Models:**
- `anthropic/claude-sonnet-4`
- `openai/gpt-4o`
- `google/gemini-2.0-flash-exp`
- And 100+ more

**API Key:** https://openrouter.ai/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="openrouter",
    api_key="sk-or-...",
    model="anthropic/claude-sonnet-4",
    base_url="https://openrouter.ai/api/v1",
)
```

**Environment Variables:**
```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Best For:** Model comparison, fallback provider

---

### Fireworks

**Category:** Cloud
**Tool Support:** Experimental
**Streaming:** Yes

**Description:** Fast inference platform.

**Supported Models:**
- `accounts/fireworks/models/llama-3.3-70b-instruct`
- `accounts/fireworks/models/mixtral-8x7b-instruct`

**API Key:** https://fireworks.ai/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="fireworks",
    api_key="...",
    model="accounts/fireworks/models/llama-3.3-70b-instruct",
)
```

**Environment Variables:**
```bash
export FIREWORKS_API_KEY="..."
```

**Best For:** Fast inference, cost efficiency

---

### Cerebras

**Category:** Cloud
**Tool Support:** Native
**Streaming:** Yes

**Description:** Fastest inference (wafer-scale engine).

**Supported Models:**
- `llama3.1-70b` - Ultra-fast
- `mixtral-8x7b`

**API Key:** https://inference.cerebras.ai/

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="cerebras",
    api_key="...",
    model="llama3.1-70b",
)
```

**Environment Variables:**
```bash
export CEREBRAS_API_KEY="..."
```

**Performance:**
- **Latency:** < 1s (fastest cloud)
- **Best For:** Speed-critical applications

---

## Enterprise/Other

### Hugging Face

**Category:** Cloud
**Tool Support:** Varies by model
**Streaming:** Yes

**Description:** Platform for 100k+ open-source models.

**Supported Models:**
- Any model on Hugging Face Hub
- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`
- etc.

**API Key:** https://huggingface.co/settings/tokens

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="huggingface",
    api_key="hf_...",
    model="Qwen/Qwen2.5-72B-Instruct",
)
```

**Environment Variables:**
```bash
export HUGGINGFACE_API_KEY="hf_..."
```

**Aliases:** `hf`

**Best For:** Access to open-source models

---

### Replicate

**Category:** Cloud
**Tool Support:** Experimental
**Streaming:** Limited

**Description:** Platform for running ML models.

**Supported Models:**
- Any model on Replicate
- `meta/meta-llama-3.1-405b-instruct`
- `mistralai/mistral-7b-instruct-v0.3`

**API Key:** https://replicate.com/account/api-tokens

**Configuration:**
```python
provider = ProviderRegistry.create(
    name="replicate",
    api_key="r8_...",
    model="meta/meta-llama-3.1-405b-instruct",
)
```

**Environment Variables:**
```bash
export REPLICATE_API_TOKEN="r8_..."
```

**Best For:** Experimental models, custom deployments

---

## Provider Switching

### Runtime Switching

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(
    provider_name="anthropic",
    model="claude-sonnet-4-5"
)

# Switch to local model
await orchestrator.switch_provider(
    provider_name="ollama",
    model="qwen2.5:32b",
    reason="cost"
)

# Switch back
await orchestrator.switch_provider(
    provider_name="anthropic",
    model="claude-sonnet-4-5",
    reason="quality"
)
```

### Automatic Failover

```python
from victor.providers.pool import ProviderPool

pool = ProviderPool(
    providers=[
        {"name": "anthropic", "model": "claude-sonnet-4-5"},
        {"name": "openai", "model": "gpt-4o"},
        {"name": "ollama", "model": "qwen2.5:32b"},
    ],
    strategy="round_robin"
)

# Automatic failover on errors
response = await pool.execute_with_fallback(
    messages=messages,
    max_retries=2
)
```

### Health Monitoring

```python
from victor.providers.health_monitor import HealthMonitor

monitor = HealthMonitor(check_interval=60)

# Start monitoring
await monitor.start(provider)

# Check health
is_healthy = await monitor.check_health(provider)

# Auto-switch on failure
if not is_healthy:
    await switch_to_backup()
```

---

## Model Capabilities

### Tool Calling Support

Check model capabilities:

```python
from victor.config.model_capabilities import get_model_capabilities

caps = get_model_capabilities("anthropic", "claude-sonnet-4-5")

print(f"Native tools: {caps.native_tool_calls}")
print(f"Parallel tools: {caps.parallel_tool_calls}")
print(f"Streaming: {caps.supports_streaming}")
print(f"Vision: {caps.supports_vision}")
```

### Model Capability Matrix

| Provider | Model | Tools | Parallel | Streaming | Vision | Max Context |
|----------|-------|-------|----------|-----------|--------|-------------|
| Anthropic | claude-sonnet-4-5 | ✓ | ✓ | ✓ | ✓ | 200K |
| OpenAI | gpt-4o | ✓ | ✓ | ✓ | ✓ | 128K |
| Google | gemini-2.0-flash | ✓ | ✓ | ✓ | ✓ | 1M |
| Ollama | qwen2.5:32b | ✓ | ✗ | ✓ | ✗ | 32K |
| vLLM | llama-3.2-3b | Exp | ✗ | ✓ | ✗ | Model-specific |
| xAI | grok-2 | ✗ | ✗ | ✓ | ✗ | Model-specific |

### Cost Optimization

```python
# Automatic cost optimization
from victor.providers.cost_optimizer import CostOptimizer

optimizer = CostOptimizer(
    providers=[
        ("anthropic", "claude-sonnet-4-5", 15.0),  # $15/1M output
        ("openai", "gpt-4o-mini", 0.60),            # $0.60/1M output
        ("ollama", "qwen2.5:32b", 0.0),             # Free
    ]
)

# Select cheapest provider for task
provider = optimizer.select_for_task(
    task_type="coding",
    quality_threshold=0.8
)
```

---

**See Also:**
- [API Reference](API_REFERENCE.md) - Main API documentation
- [Protocol Reference](PROTOCOL_REFERENCE.md) - Protocol interfaces
- [Configuration Reference](CONFIGURATION_REFERENCE.md) - Settings reference
