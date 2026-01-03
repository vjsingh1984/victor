# Local Models

Run Victor with local LLMs for privacy, offline use, and predictable cost.

## Quick Start
1. Run a local server (Ollama, LM Studio, vLLM, or llama.cpp).
2. Create a local profile.
3. Run `victor chat -p local`.

## Example Profile

```yaml
profiles:
  local:
    provider: ollama
    model: <model-id>

providers:
  ollama:
    base_url: http://localhost:11434
```

## Provider Notes

Ollama:
```bash
ollama serve
ollama pull <model-id>
```

LM Studio:
- Start the local server from the app UI.

vLLM:
```bash
python -m vllm.entrypoints.openai.api_server --model <model-id>
```

llama.cpp:
```bash
./llama-server -m <model-path> --port 8080
```

## Air-Gapped Mode

If you want to disable network tools:

```bash
victor init --airgapped
```

See `docs/embeddings/AIRGAPPED.md` for offline workflows.
