# Local Models

**Run Victor with local LLMs for privacy, offline use, and predictable cost.**

## Provider Comparison

| Provider      | Setup                 | Air-Gapped | Features                | Best For     |
|--------------|-----------------------|------------|-------------------------|--------------|
| **Ollama**    | `brew install ollama` | ✅         | Tool calling, streaming | Quick start  |
| **LM Studio** | Download app          | ✅         | GUI model management    | Desktop users |
| **vLLM**      | `pip install vllm`    | ✅         | High throughput         | Production   |
| **llama.cpp** | Compile from source   | ✅         | CPU inference           | No GPU       |

## Quick Start

```bash
# Option 1: Ollama (recommended)
victor init --local

# Option 2: Manual setup
ollama serve && ollama pull llama3
victor chat --provider ollama --model llama3
```

## Configuration

```yaml
# ~/.victor/config.yaml
profiles:
  local:
    provider: ollama
    model: llama3

providers:
  ollama:
    base_url: http://localhost:11434
```

## Recommended Models

| Provider | Model     | Size | Use Case         |
|----------|-----------|------|------------------|
| Ollama   | llama3    | 8B   | General coding   |
| Ollama   | codellama | 13B  | Code completion  |
| Ollama   | mistral   | 7B   | Fast responses   |
| vLLM     | mixtral   | 8x7B | Best quality     |

## Air-Gapped Mode

```bash
# Disable all network tools
victor init --airgapped

# Or enable per-session
victor chat --airgapped
```

## Commands Reference

| Provider      | Start Server                                                 | Pull Model            |
|---------------|--------------------------------------------------------------|-----------------------|
| Ollama        | `ollama serve`                                               | `ollama pull <model>` |
| LM Studio     | Start in app                                                 | Download via GUI      |
| vLLM          | `python -m vllm.entrypoints.openai.api_server --model <id>`  | Manual                |

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
