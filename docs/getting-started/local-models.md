# Local Models

Use free, private AI with local models (no API keys required).

## Quick Start

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b

# Start Victor
victor chat --provider ollama
```

## Supported Local Providers

| Provider | Setup | Best For |
|----------|-------|----------|
| **Ollama** | `brew install ollama` | Easiest, beginner-friendly |
| **LM Studio** | Download from lmstudio.ai | GUI, Windows support |
| **vLLM** | `pip install vllm` | Production, high throughput |
| **llama.cpp** | Build from source | CPU inference, minimal |

## Recommended Models

```bash
# Coding (7B parameters, 4GB RAM)
ollama pull qwen2.5-coder:7b

# General (7B parameters, 4GB RAM)
ollama pull llama3.1:8b

# High quality (14B parameters, 8GB RAM)
ollama pull qwen2.5-coder:14b
```

## Air-Gapped Mode

For completely offline operation:

```python
from victor.framework import Agent

agent = await Agent.create(
    provider="ollama",
    airgapped_mode=True  # Only local providers and tools
)
```

Or set environment variable:
```bash
export VICTOR_AIRGAPPED=true
victor chat --provider ollama
```

## Performance Tips

| RAM | Model Size | Model |
|-----|------------|-------|
| 4 GB | 3B-4B | phi3, qwen2.5-coder:3b |
| 8 GB | 7B-8B | qwen2.5-coder:7b, llama3.1:8b |
| 16 GB | 14B+ | qwen2.5-coder:14b, deepseek-coder:33b |

## Troubleshooting

**Model not found?**
```bash
ollama list  # Check installed models
ollama pull <model>  # Download missing model
```

**Out of memory?**
- Use a smaller model (3B instead of 7B)
- Close other applications
- Check RAM usage: `ollama ps`

## Next Steps

- [Installation](installation.md) - Install Victor
- [First Run](first-run.md) - Get started
- [Cloud Models](cloud-models.md) - Use cloud providers

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
