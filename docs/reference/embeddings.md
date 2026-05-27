# Embeddings & Semantic Search

Embeddings power semantic tool selection and codebase search. Use local or remote providers.

## Quick Setup

```yaml
# ~/.victor/profiles.yaml
profiles:
  local:
    embedding_provider: sentence-transformers
    embedding_model: BAAI/bge-small-en-v1.5
```

Or with Ollama:
```yaml
profiles:
  local:
    embedding_provider: ollama
    embedding_model: qwen3-embedding:8b
```

## Tool Selection Strategies

| Strategy | Description |
|----------|-------------|
| `keyword` | Fast, no embeddings needed |
| `semantic` | Embedding-based matching |
| `hybrid` | Blends keyword + semantic (default) |
| `auto` | Chooses based on availability |

```yaml
tool_selection_strategy: hybrid
```

## Models

| Model | Dimensions | Size | Use Case |
|-------|------------|------|----------|
| BAAI/bge-small-en-v1.5 | 384 | ~130MB | Default |
| all-MiniLM-L12-v2 | 384 | ~120MB | Low memory |
| qwen3-embedding:8b | 4096 | ~4.7GB | High quality |

## Air-Gapped Mode

For restricted environments without network access:

```yaml
profiles:
  airgapped:
    provider: ollama
    model: qwen2.5-coder:7b
    airgapped_mode: true
    embedding_provider: sentence-transformers
    embedding_model: BAAI/bge-small-en-v1.5
```

**Behavior:**
- Web tools disabled
- Only local providers allowed
- Falls back to keyword selection if embeddings unavailable

## Architecture

```
User Request → Embedding Provider → Vector Comparison → Tool Selection
                     ↓
              Embedding Cache
```

**Providers:**
- Local: sentence-transformers, Ollama, vLLM, LM Studio
- Remote: Cloud embedding APIs

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tools not triggering | Use `keyword` strategy |
| Slow cold starts | Pre-cache embeddings |
| Memory issues | Use smaller model (bge-small) |
