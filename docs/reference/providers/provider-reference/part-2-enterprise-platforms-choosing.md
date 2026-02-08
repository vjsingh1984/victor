# Provider Reference - Part 2

**Part 2 of 2:** Enterprise Providers, Open Model Platforms, Choosing a Provider, Environment Variables, Troubleshooting

---

## Navigation

- [Part 1: Comparison, Local, Cloud](part-1-comparison-local-cloud.md)
- **[Part 2: Enterprise, Platforms, Choosing](#)** (Current)
- [**Complete Reference](../index.md)**

---

## Enterprise Providers

### Azure OpenAI

**Setup:**
```bash
export AZURE_API_KEY="your-key"
export AZURE_ENDPOINT="https://your-resource.openai.azure.com"
```

**Models:** GPT-4, GPT-3.5, Embeddings

**Features:**
- Enterprise-grade SLA
- Private deployment
- Custom models

### AWS Bedrock

**Setup:**
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
```

**Models:** Claude, Llama, Titan, Jurassic

**Features:**
- Managed service
- Private VPC
- Custom fine-tuning

### Google Vertex AI

**Setup:**
```bash
export GOOGLE_API_KEY="your-key"
export GOOGLE_PROJECT_ID="your-project"
export GOOGLE_REGION="us-central1"
```

**Models:** Gemini 2.0, PaLM

**Features:**
- Enterprise support
- Model tuning
- Vector search

---

## Open Model Platforms

### Hugging Face

**Setup:**
```bash
export HF_API_KEY="your-key"
```

**Models:** 100K+ models

**Features:**
- Largest model hub
- Free tier available
- Custom models

### Replicate

**Setup:**
```bash
export REPLICATE_API_KEY="your-key"
```

**Models:** 20K+ models

**Features:**
- Easy deployment
- Serverless
- API access

---

## Choosing a Provider

### Use Case Guide

**Best for Code:**
- Anthropic Claude 3.5 Sonnet
- OpenAI GPT-4
- Google Gemini 2.0

**Best for Cost:**
- Ollama (local, free)
- Grok (xAI)
- DeepSeek

**Best for Enterprise:**
- Azure OpenAI
- AWS Bedrock
- Google Vertex AI

**Best for Privacy:**
- Ollama (local)
- vLLM (local)
- Enterprise deployments

### Decision Tree

```
Need privacy/local?
├─ Yes → Ollama, vLLM, LM Studio
└─ No → Need enterprise features?
    ├─ Yes → Azure, AWS Bedrock, Vertex AI
    └─ No → Budget conscious?
        ├─ Yes → Grok, DeepSeek
        └─ No → Anthropic, OpenAI, Google
```

---

## Environment Variables Reference

| Provider | Environment Variables |
|----------|---------------------|
| **Anthropic** | `ANTHROPIC_API_KEY` |
| **OpenAI** | `OPENAI_API_KEY` |
| **Google** | `GOOGLE_API_KEY` |
| **Azure** | `AZURE_API_KEY`, `AZURE_ENDPOINT` |
| **AWS Bedrock** | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| **Ollama** | `OLLAMA_BASE_URL` |
| **Hugging Face** | `HF_API_KEY` |
| **Replicate** | `REPLICATE_API_KEY` |

---

## Troubleshooting

### Common Issues

**Issue:** `API key not found`

**Solution:** Set environment variable or add to `.env`

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Issue:** `Model not found`

**Solution:** Check model name for provider

```bash
# List available models
victor providers list-models anthropic
```

**Issue:** `Connection timeout`

**Solution:** Check internet connection and API status

```bash
# Test connection
victor providers test anthropic
```

---

**Last Updated:** February 01, 2026
