# Embedding Architecture

Victor supports pluggable embedding providers for semantic tool selection and codebase search. The design keeps providers swappable while sharing a common configuration model.

## Components
- Embedding configuration (provider, model, dimensions)
- Embedding service (generates vectors)
- Cache (stores tool embeddings)
- Vector store (optional, for codebase search)

## Provider Types
- Local: sentence-transformers, Ollama, vLLM, LM Studio
- Remote: cloud embedding APIs

## Data Flow
1. Text input is embedded by the provider.
2. Embeddings are cached for reuse.
3. Similarity search selects tools or documents.

## Notes
- Local providers enable offline use when models are installed.
- Latency and quality depend on model choice and hardware.
- You can override providers per profile.

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
