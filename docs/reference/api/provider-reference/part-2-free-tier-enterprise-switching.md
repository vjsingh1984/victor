# Victor AI 0.5.0 Provider Reference - Part 2

**Part 2 of 2:** Free-Tier Providers, Enterprise/Other, Provider Switching, Model Capabilities

---

## Navigation

- [Part 1: Provider Catalog](part-1-providers-catalog.md)
- **[Part 2: Free-Tier, Enterprise, Switching](#)** (Current)
- [**Complete Reference](../PROVIDER_REFERENCE.md)**

---

## Free-Tier Providers (2025)

### Groq

**Models**: Llama 3.1 70B, Mixtral 8x7B

**Features**:
- Ultra-fast inference
- Free tier available
- Limited to 128K tokens/min

**Setup**:
```bash
export GROQ_API_KEY="your-key"
```

### Mistral

**Models**: Mistral 7B, Mixtral 8x7B, Codestral

**Features**:
- Free tier: 1M tokens/month
- Strong code generation
- Competitive pricing

**Setup**:
```bash
export MISTRAL_API_KEY="your-key"
```

### Together

**Models**: Mixtral, Llama, RedPajama

**Features**:
- Free tier: 1M tokens/month
- Open source models
- Flexible deployment

**Setup**:
```bash
export TOGETHER_API_KEY="your-key"
```

### OpenRouter

**Models**: 100+ models from multiple providers

**Features**:
- Unified API for multiple providers
- Competitive pricing
- Free credits available

**Setup**:
```bash
export OPENROUTER_API_KEY="your-key"
```

### Fireworks

**Models**: Llama 3, Mixtral, FireLLaMA

**Features**:
- Fast inference
- Free tier: 10B tokens/day
- Custom model hosting

**Setup**:
```bash
export FIREWORKS_API_KEY="your-key"
```

### Cerebras

**Models**: Llama 3.1 70B

**Features**:
- Extremely fast inference
- Free tier available
- Edge deployment

**Setup**:
```bash
export CEREBRAS_API_KEY="your-key"
```

---

## Enterprise/Other

### Hugging Face

**Models**: 100K+ models

**Features**:
- Largest model hub
- Inference API
- Free tier: 30K requests/day

**Setup**:
```bash
export HF_API_KEY="your-key"
```

### Replicate

**Models**: 20K+ models

**Features**:
- Easy deployment
- Serverless inference
- Pay-per-use

**Setup**:
```bash
export REPLICATE_API_KEY="your-key"
```

---

## Provider Switching

### Runtime Switching

Switch providers during conversation:

```python
from victor.agent import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Start with Anthropic
response1 = await orchestrator.chat(
    messages=[{"role": "user", "content": "Hello"}],
    provider="anthropic"
)

# Switch to OpenAI
response2 = await orchestrator.chat(
    messages=[{"role": "user", "content": "Continue"}],
    provider="openai"
)
```

### Automatic Fallback

Configure fallback providers:

```python
orchestrator = AgentOrchestrator(
    settings=Settings(
        default_provider="anthropic",
        fallback_providers=["openai", "google"]
    )
)
```

---

## Model Capabilities

### Tool Calling Support

| Provider | Tool Calling | Notes |
|----------|-------------|-------|
| Anthropic | ✅ Native | Excellent |
| OpenAI | ✅ Native | Function calling |
| Google | ✅ Native | Function calling |
| xAI | ✅ Native | Good |
| Ollama | ✅ Native | Via format |
| vLLM | ✅ Native | Via format |
| Groq | ✅ Native | Model-dependent |
| Mistral | ✅ Native | Good |
| Together | ✅ Native | Model-dependent |

### Streaming Support

| Provider | Streaming | Notes |
|----------|----------|-------|
| Anthropic | ✅ | SSE |
| OpenAI | ✅ | SSE |
| Google | ✅ | SSE |
| xAI | ✅ | SSE |
| Ollama | ✅ | Native |
| vLLM | ✅ | Native |
| Groq | ✅ | SSE |
| Mistral | ✅ | SSE |

---

## Related Documentation

- [Provider Implementation Guide](../../../architecture/PROVIDER_IMPLEMENTATION.md)
- [Configuration Reference](./CONFIGURATION_REFERENCE.md)
- [Provider Comparison](../../providers/provider-reference/)

---

**Last Updated:** February 01, 2026
