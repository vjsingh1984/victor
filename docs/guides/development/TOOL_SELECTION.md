# Semantic Tool Selection

Victor can select tools by comparing embeddings of the user request to tool descriptions. This reduces tool noise and
  keeps prompts smaller when you have many tools.

## When To Use
- Large tool catalog
- Ambiguous or multi-step requests
- You want fewer irrelevant tools per turn

## Requirements
- An embedding provider (local or remote)
- Optional: cached tool embeddings for faster selection

## Strategy Options
- keyword: available without embeddings, fast, deterministic
- semantic: embedding-based matching
- hybrid: blends keyword and semantic
- auto: chooses based on availability and settings

## Configuration

Profiles or environment variables:

```yaml
# Modern strategy setting (preferred)
tool_selection_strategy: auto  # keyword, semantic, hybrid, auto

# Legacy toggle (if your config still uses it)
use_semantic_tool_selection: true

# Embedding settings
embedding_provider: sentence-transformers
embedding_model: BAAI/bge-small-en-v1.5
```

## Notes
- Offline use is possible with a local embedding model installed.
- Cold starts are slower until embeddings are cached.
- Quality depends on model choice and tool descriptions.

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
