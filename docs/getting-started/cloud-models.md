# Cloud Models

Use full-capability AI with cloud providers.

## Quick Start

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-...

# Start Victor
victor chat --provider anthropic
```

## Supported Providers

| Provider | API Key | Models | Tool Calling | Vision |
|----------|---------|--------|--------------|--------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Sonnet 4, Opus, Haiku | ✅ | ✅ |
| **OpenAI** | `OPENAI_API_KEY` | GPT-4o, GPT-4, o1 | ✅ | ✅ |
| **Google** | `GOOGLE_API_KEY` | Gemini 2.0 Flash, Pro | ✅ | ✅ |
| **Azure OpenAI** | `AZURE_OPENAI_API_KEY` | GPT-4, Claude | ✅ | ✅ |
| **AWS Bedrock** | AWS credentials | Claude 3, Llama 3 | ✅ | ✅ |
| **xAI** | `XAI_API_KEY` | Grok | ✅ | ❌ |
| **DeepSeek** | `DEEPSEEK_API_KEY` | DeepSeek-V3, Coder | ✅ | ❌ |
| **Mistral** | `MISTRAL_API_KEY` | Mistral Large, Small | ✅ | Partial |
| **Cohere** | `COHERE_API_KEY` | Command R, R+ | ✅ | ❌ |
| **Groq** | `GROQ_API_KEY` | Llama, Mistral (300+ tok/s) | ✅ | ❌ |

## Setup Examples

### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
victor chat --provider anthropic --model claude-sonnet-4-5
```

### OpenAI (GPT-4)
```bash
export OPENAI_API_KEY=sk-...
victor chat --provider openai --model gpt-4o
```

### Google (Gemini)
```bash
# Install Google SDK
pip install victor-ai[google]

export GOOGLE_API_KEY=...
victor chat --provider google --model gemini-2.0-flash-exp
```

### Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
victor chat --provider azure-openai
```

## Model Recommendations

| Use Case | Provider | Model | Why |
|----------|----------|-------|-----|
| **Coding** | Anthropic | claude-sonnet-4-5 | Best code understanding |
| **Speed** | Groq | llama3.1-70b | 300+ tokens/second |
| **Reasoning** | OpenAI | o1-preview | Deep reasoning |
| **Free** | Ollama | qwen2.5-coder:7b | No API costs |
| **Vision** | Anthropic | claude-sonnet-4-5 | Best vision capabilities |

## Cost Optimization

### Multi-Provider Strategy
```python
from victor.framework import Agent

# Use different providers for different tasks
brainstormer = await Agent.create(provider="ollama")      # FREE
implementer = await Agent.create(provider="openai")       # CHEAP
reviewer = await Agent.create(provider="anthropic")       # QUALITY

ideas = await brainstormer.run("Brainstorm features")
code = await implementer.run(f"Implement: {ideas.content[0]}")
review = await reviewer.run(f"Review: {code.content}")
```

### Switch Providers Mid-Conversation
```bash
victor chat --provider ollama      # Start with free
/provider openai --model gpt-4o   # Switch for implementation
/provider anthropic               # Switch for final review
```

## Configuration

### Set Default Provider
```yaml
# ~/.victor/profiles.yaml
default_profile:
  provider: anthropic
  model: claude-sonnet-4-5
```

### Multiple Profiles
```yaml
profiles:
  fast:
    provider: groq
    model: llama3.1-70b-8192
  quality:
    provider: anthropic
    model: claude-sonnet-4-5
  free:
    provider: ollama
    model: qwen2.5-coder:7b
```

## Troubleshooting

**API key not working?**
```bash
# Verify key is set
echo $ANTHROPIC_API_KEY

# Test connection
victor chat --provider anthropic --test
```

**Rate limiting?**
- Switch to a different provider
- Use Groq for faster inference (300+ tok/s)
- Use Ollama for free local inference

## Next Steps

- [Installation](installation.md) - Install Victor
- [First Run](first-run.md) - Get started
- [Local Models](local-models.md) - Free, private AI
- [Configuration](./configuration.md) - Advanced setup

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
