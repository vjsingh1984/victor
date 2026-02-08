# Provider Reference

Complete reference for supported LLM providers in Victor.

---

## Quick Summary

Victor supports 21+ LLM providers across multiple categories:

| Category | Providers | Key Features |
|----------|-----------|--------------|
| **Local** | Ollama, LM Studio, vLLM, llama.cpp | Privacy, free, custom models |
| **Cloud** | Anthropic, OpenAI, Google, xAI | Easy setup, best models |
| **Enterprise** | Azure, AWS Bedrock, Vertex AI | SLA, security, compliance |
| **Platforms** | Hugging Face, Replicate | 100K+ models, serverless |

---

## Reference Parts

### [Part 1: Comparison, Local, Cloud](part-1-comparison-local-cloud.md)
- Provider Comparison Matrix
- Feature Support
- Local Providers (Ollama, vLLM, LM Studio, llama.cpp)
- Cloud Providers (Anthropic, OpenAI, Google, xAI, DeepSeek)

### [Part 2: Enterprise, Platforms, Choosing](part-2-enterprise-platforms-choosing.md)
- Enterprise Providers (Azure, AWS Bedrock, Vertex AI)
- Open Model Platforms (Hugging Face, Replicate)
- Choosing a Provider Guide
- Environment Variables Reference
- Troubleshooting

---

## Quick Start

```bash
# Set provider
export VICTOR_DEFAULT_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use local provider
export VICTOR_DEFAULT_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"

# Test connection
victor providers test anthropic
```

---

## Related Documentation

- [Getting Started](../../../getting-started/README.md)
- [Configuration Reference](../api/CONFIGURATION_REFERENCE.md)
- [Provider Implementation Guide](../../architecture/PROVIDER_IMPLEMENTATION.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 10 min (all parts)
