# vLLM Setup Guide

vLLM is a high-throughput LLM inference server. This guide covers setup for both GPU (production) and CPU (testing) environments.

## Prerequisites

Install vLLM:
```bash
pip install vllm
```

## GPU Setup (Production)

For NVIDIA GPU with CUDA:

```bash
# Basic setup with tool calling
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# With custom GPU memory utilization
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.9

# Multi-GPU setup
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --tensor-parallel-size 2
```

## CPU Setup (Testing/Development)

For macOS or Linux without GPU (slower but functional):

```bash
# Minimal context (faster, less memory)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype float32 \
    --enforce-eager \
    --max-model-len 2048

# Larger context (for Victor's full system prompt ~6.5K tokens)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype float32 \
    --enforce-eager \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384

# Increase KV cache for better performance
VLLM_CPU_KVCACHE_SPACE=8 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype float32 \
    --enforce-eager \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384
```

## Tool Call Parsers

Different models require different parsers:

| Parser | Models |
|--------|--------|
| `hermes` | Qwen, Hermes-style models |
| `mistral` | Mistral models |
| `llama3_json` | Llama 3.1+ models |
| `llama4_json` | Llama 4 models |
| `deepseek_v3` | DeepSeek V3 models |
| `qwen3_coder` | Qwen3 Coder models |

## Recommended Models

### For GPU (Production)

| Model | VRAM | Context | Notes |
|-------|------|---------|-------|
| Qwen/Qwen2.5-Coder-7B-Instruct | 16GB | 32K | Best balance |
| Qwen/Qwen2.5-Coder-14B-Instruct | 32GB | 32K | Higher quality |
| Qwen/Qwen2.5-Coder-32B-Instruct | 64GB | 32K | Best quality |
| deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | 32GB | 128K | Long context |
| mistralai/Codestral-22B-v0.1 | 48GB | 32K | Fast |

### For CPU (Testing)

| Model | RAM | Context | Notes |
|-------|-----|---------|-------|
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 8GB | 16K | Fast for testing |
| Qwen/Qwen2.5-Coder-3B-Instruct | 12GB | 16K | Better quality |

## Using with Victor

### List available models
```bash
victor models list -p vllm
victor models list -p vllm -e http://remote-server:8000
```

### Chat with vLLM
```bash
# Default (localhost:8000)
victor chat --provider vllm

# Specify model
victor chat --provider vllm --model Qwen/Qwen2.5-Coder-7B-Instruct

# Remote server
victor chat --provider vllm --endpoint http://gpu-server:8000
```

### Configuration in profiles.yaml
```yaml
vllm:
  base_url: http://localhost:8000/v1
  timeout: 300  # Longer timeout for CPU
```

## Tool Calling

### Native Tool Calling (Recommended)
For improved tool calling performance, start vLLM with these flags:
```bash
--enable-auto-tool-choice --tool-call-parser hermes
```

This enables vLLM to parse tool calls natively using the OpenAI-compatible format.

### Fallback Tool Parsing
If vLLM is started **without** `--enable-auto-tool-choice`, Victor will automatically:
1. Detect when models output tool calls as JSON in the response text
2. Parse JSON code blocks, `<TOOL_OUTPUT>` tags, or inline JSON
3. Execute the extracted tool calls normally

This fallback works but is slower and less reliable than native parsing.

### Tool Call Formats Supported
Victor's fallback parser handles:
```
# JSON code blocks
```json
{"name": "list_directory", "arguments": {"path": "."}}
```

# TOOL_OUTPUT tags
<TOOL_OUTPUT>{"name": "tool_name", "arguments": {...}}</TOOL_OUTPUT>

# Inline JSON
{"name": "tool_name", "arguments": {...}}
```

## Troubleshooting

### Context length error
If you see "max_num_batched_tokens is smaller than max_model_len":
```bash
# Add matching --max-num-batched-tokens
--max-model-len 8192 --max-num-batched-tokens 8192
```

### Victor system prompt too long
Victor's full system prompt is ~6.5K tokens. Use at least 16K context:
```bash
--max-model-len 16384 --max-num-batched-tokens 16384
```

### CPU memory issues
Increase KV cache space:
```bash
VLLM_CPU_KVCACHE_SPACE=8 python -m vllm.entrypoints.openai.api_server ...
```

### Slow CPU inference
- Use smaller models (1.5B-3B)
- Reduce max_tokens in requests
- Use lower context length

### Duplicate content in responses
Some models may output content twice. Victor enables output deduplication
for vLLM by default to handle this. If you still see duplicates, the model
may need different sampling parameters (lower temperature, repetition penalty).

## API Endpoints

vLLM exposes OpenAI-compatible endpoints:

| Endpoint | Description |
|----------|-------------|
| `/v1/models` | List loaded models |
| `/v1/chat/completions` | Chat completions |
| `/v1/completions` | Text completions |
| `/health` | Server health check |
| `/docs` | OpenAPI documentation |

## Docker Setup

```bash
# GPU
docker run --gpus all -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# With model cache
docker run --gpus all -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

## Performance Tips

1. **GPU**: Use `--gpu-memory-utilization 0.9` for maximum performance
2. **Multi-GPU**: Use `--tensor-parallel-size N` for large models
3. **Batch processing**: vLLM automatically batches requests for throughput
4. **Streaming**: Enabled by default, reduces time-to-first-token
5. **KV Cache**: vLLM uses PagedAttention for efficient memory usage
