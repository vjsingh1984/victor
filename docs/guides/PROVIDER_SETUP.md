# Provider Setup

Pick a provider based on privacy, cost, and latency. Local providers keep data on your machine. Cloud providers are convenient but require API keys.

## Quick Start
1. Choose a provider.
2. Add credentials if needed.
3. Create a profile.
4. Run `victor chat -p <profile>`.

## Local Example (Ollama)

```bash
# Start server
ollama serve

# Pull a model
ollama pull <model-id>
```

```yaml
profiles:
  local:
    provider: ollama
    model: <model-id>

providers:
  ollama:
    base_url: http://localhost:11434
```

## Cloud Example (OpenAI)

```bash
export OPENAI_API_KEY="sk-..."
```

```yaml
profiles:
  cloud:
    provider: openai
    model: <model-id>

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

## Other Providers

Local:
- Ollama
- LM Studio
- vLLM
- llama.cpp

Cloud:
- Anthropic
- OpenAI
- Google
- xAI
- Mistral
- DeepSeek
- Together
- Fireworks
- OpenRouter
- Groq
- Moonshot
- Cerebras

Enterprise:
- Azure OpenAI
- AWS Bedrock
- Google Vertex AI

Platforms:
- Hugging Face
- Replicate

## Tips
- Use `../API_KEYS_CONFIGURATION.md` for credential options.
- If tool calling is inconsistent, try another model or switch to keyword tool selection.
- For offline use, combine local models with `../embeddings/AIRGAPPED.md`.
