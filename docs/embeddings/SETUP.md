# Embeddings Setup

Embeddings power semantic tool selection and codebase search. You can use local or remote providers depending on your environment.

## Quick Start (Local)
1. Install `sentence-transformers`.
2. Set the embedding provider and model in your profile.

```yaml
embedding_provider: sentence-transformers
embedding_model: BAAI/bge-small-en-v1.5
codebase_embedding_provider: sentence-transformers
codebase_embedding_model: BAAI/bge-small-en-v1.5
```

## Local Server Example (Ollama)
1. Install Ollama and pull an embedding model.
2. Set provider and model in your profile.

```yaml
embedding_provider: ollama
embedding_model: qwen3-embedding:8b
codebase_embedding_provider: ollama
codebase_embedding_model: qwen3-embedding:8b
```

## Notes
- The defaults are defined in `victor/config/settings.py` and can be overridden per profile.
- Local providers enable offline use when models are installed.
- Larger models can improve recall but cost more RAM.
