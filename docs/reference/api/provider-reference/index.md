# Victor AI 0.5.0 Provider Reference

Complete reference for all 21 supported LLM providers.

---

## Quick Summary

Victor AI supports 21 LLM providers through a unified `BaseProvider` interface. All providers implement:

- `chat()`: Non-streaming chat completion
- `stream_chat()`: Streaming chat completion
- `supports_tools()`: Tool calling capability query
- `name`: Provider identifier

**Provider Categories**:
- **Local**: Ollama, LMStudio, vLLM, Llama.cpp (privacy, free)
- **Cloud**: Anthropic, OpenAI, Google, Azure, AWS (easy setup, best models)
- **Research**: xAI, DeepSeek, Moonshot, Zhipu AI (specialized models)
- **Free-Tier**: Groq, Mistral, Together, OpenRouter, Fireworks, Cerebras
- **Enterprise**: Hugging Face, Replicate (platform, scale)

---

## Reference Parts

### [Part 1: Provider Catalog](part-1-providers-catalog.md)
- Overview
- Local Providers (Ollama, LMStudio, vLLM, Llama.cpp)
- Major Cloud Providers (Anthropic, OpenAI, Google, Azure, AWS, Vertex)
- AI Research Companies (xAI, DeepSeek, Moonshot, Zhipu AI)

### [Part 2: Free-Tier, Enterprise, Switching](part-2-free-tier-enterprise-switching.md)
- Free-Tier Providers (Groq, Mistral, Together, OpenRouter, Fireworks, Cerebras)
- Enterprise/Other (Hugging Face, Replicate)
- Provider Switching
- Model Capabilities

---

## Quick Start

```python
from victor.agent import AgentOrchestrator

# Use any provider
orchestrator = AgentOrchestrator()

response = await orchestrator.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    provider="anthropic"  # or "openai", "google", "ollama", etc.
)
```

---

## Related Documentation

- [Provider Implementation Guide](../../architecture/PROVIDER_IMPLEMENTATION.md)
- [Configuration Reference](./CONFIGURATION_REFERENCE.md)
- [Provider Comparison](../../providers/provider-reference/)

---

**Last Updated:** February 01, 2026
**Reading Time:** 15 min (all parts)
