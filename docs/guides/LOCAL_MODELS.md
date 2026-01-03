# Local Models Guide

Run Victor with local LLMs for privacy, offline use, and cost control.

## Quick Picks

| Provider | Best For | Default Port |
|---------|----------|--------------|
| Ollama | Fast setup, great defaults | 11434 |
| LM Studio | Desktop GUI and GGUF models | 1234 |
| vLLM | High-throughput local serving | 8000 |
| llama.cpp | CPU-friendly GGUF inference | 8080 |

For deeper setup, see:
- `docs/guides/OLLAMA_TOOL_SUPPORT.md`
- `docs/guides/LLAMACPP_SETUP.md`
- `docs/guides/VLLM_SETUP.md`

## Profiles Presets

Add a local profile in `~/.victor/profiles.yaml`:

```yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

providers:
  ollama:
    base_url: http://localhost:11434
```

Switch to it with:

```bash
victor chat --profile local
```

Shortcut:

```bash
victor init --local
```

## Ollama

```bash
ollama pull qwen2.5-coder:7b
victor chat -p ollama
```

## LM Studio

1) Start the LM Studio local server  
2) Then run:

```bash
victor chat -p lmstudio
```

## vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct

victor chat -p vllm
```

## llama.cpp

```bash
./llama-server -m models/qwen2.5-coder-7b-instruct-q4_k_m.gguf --port 8080

victor chat --provider llamacpp --endpoint http://localhost:8080
```

## Air-Gapped Mode

If you want to disable network-dependent tools:

```bash
victor init --airgapped
```

This writes `AIRGAPPED_MODE=true` to `.env` and adds an `airgapped` profile.

```bash
AIRGAPPED_MODE=true
```

See `docs/embeddings/AIRGAPPED.md` for offline workflows.

## Troubleshooting

- `Connection refused`: the local server is not running or the port is wrong.
- Slow output: use a smaller model or reduce context length.
- Tool calls missing: check `docs/guides/OLLAMA_TOOL_SUPPORT.md`.
