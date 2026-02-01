# Air-Gapped Mode

Air-gapped mode runs Victor without external network calls by using local models, local embeddings, and local storage. It is designed for restricted environments and privacy-sensitive work.

## Requirements
- Local LLM provider (Ollama, LM Studio, vLLM, llama.cpp)
- Local embedding model for semantic search (optional but recommended)
- Web tools disabled

## Quick Setup
1. Configure a local provider in your profile.
2. Set `airgapped_mode: true`.
3. Use a local embedding provider (for example, sentence-transformers or a local server).
4. Run a small request to confirm.

## Behavior
- Web search and remote providers are disabled.
- Tool selection falls back to keyword mode if embeddings are unavailable.
- Performance depends on hardware and model size.

## Troubleshooting
- Missing tools: try keyword or hybrid tool selection.
- Tool calling format issues: see `TOOL_CALLING_FORMATS.md`.

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
