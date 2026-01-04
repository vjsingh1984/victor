# llama.cpp Setup Guide

llama.cpp provides efficient CPU inference with GGUF quantized models. Ideal for local development on macOS, Linux, or Windows without GPU.

## Prerequisites

### Option 1: llama-server (Recommended)

Build from source:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# With Metal support (macOS)
make -j LLAMA_METAL=1
```

### Option 2: llama-cpp-python

Install via pip:
```bash
# CPU only
pip install llama-cpp-python[server]

# With Metal (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python[server]
```

## Download GGUF Models

Download quantized models from HuggingFace:

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download \
    Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --local-dir ./models

# Or direct download
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

## Recommended GGUF Models

| Model | Size | RAM | Quality | Use Case |
|-------|------|-----|---------|----------|
| qwen2.5-coder-1.5b-instruct.Q4_K_M | 1.0GB | 4GB | Good | Quick testing |
| qwen2.5-coder-3b-instruct.Q4_K_M | 2.0GB | 6GB | Better | Development |
| qwen2.5-coder-7b-instruct.Q4_K_M | 4.4GB | 10GB | Great | Production-ready |
| qwen2.5-coder-14b-instruct.Q4_K_M | 8.4GB | 18GB | Excellent | Best quality |
| codellama-7b-instruct.Q4_K_M | 4.2GB | 10GB | Great | Alternative |
| deepseek-coder-6.7b-instruct.Q4_K_M | 4.0GB | 10GB | Great | Alternative |

### Quantization Levels

| Quant | Size | Quality | Speed | Recommended |
|-------|------|---------|-------|-------------|
| Q8_0 | Largest | Best | Slowest | High quality needed |
| Q6_K | Large | Very Good | Slow | Good balance |
| Q5_K_M | Medium | Good | Medium | Development |
| Q4_K_M | Small | Good | Fast | **Recommended** |
| Q4_K_S | Smaller | Decent | Faster | Low memory |
| Q3_K_M | Smallest | Lower | Fastest | Very low memory |

## Starting the Server

### Using llama-server

```bash
# Basic setup
./llama-server -m models/qwen2.5-coder-7b-instruct-q4_k_m.gguf --port 8080

# With more context
./llama-server -m models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --port 8080 \
    --ctx-size 8192 \
    --n-gpu-layers 0

# With chat template
./llama-server -m models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --port 8080 \
    --ctx-size 8192 \
    --chat-template chatml
```

### Using llama-cpp-python

```bash
# Basic setup
python -m llama_cpp.server \
    --model models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --port 8080

# With configuration
python -m llama_cpp.server \
    --model models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --port 8080 \
    --n_ctx 8192 \
    --n_threads 8 \
    --chat_format chatml
```

## Using with Victor

### List server status
```bash
victor models list -p llamacpp
victor models list -p llamacpp -e http://localhost:8080
```

### Chat with llama.cpp
```bash
# Default (localhost:8080)
victor chat --provider llamacpp

# Custom endpoint
victor chat --provider llamacpp --endpoint http://localhost:8080
```

### Configuration in profiles.yaml
```yaml
llamacpp:
  base_url: http://localhost:8080/v1
  timeout: 300  # CPU inference is slower
```

## API Endpoints

llama.cpp exposes OpenAI-compatible endpoints:

| Endpoint | Description |
|----------|-------------|
| `/v1/models` | List loaded model |
| `/v1/chat/completions` | Chat completions |
| `/v1/completions` | Text completions |
| `/health` | Server health |
| `/props` | Server properties |

## Performance Tips

### CPU Optimization

```bash
# Set thread count (use physical cores, not hyperthreads)
./llama-server -m model.gguf --threads 8

# Use batch processing
./llama-server -m model.gguf --batch-size 512
```

### Memory Optimization

```bash
# Enable memory mapping (reduces RAM)
./llama-server -m model.gguf --mmap

# Use smaller quantization
# Q4_K_S instead of Q4_K_M
```

### macOS Metal Acceleration

```bash
# Build with Metal
make -j LLAMA_METAL=1

# Use GPU layers (Metal)
./llama-server -m model.gguf --n-gpu-layers 32
```

## Troubleshooting

### Server not responding
Check if server is running:
```bash
curl http://localhost:8080/health
```

### Out of memory
- Use smaller model (3B instead of 7B)
- Use more aggressive quantization (Q3_K_M)
- Reduce context size (--ctx-size 4096)

### Slow inference
- Use more threads (--threads N)
- Use smaller quantization
- Reduce max_tokens in requests
- Enable Metal on macOS

### Model format errors
- Ensure GGUF format (not GGML)
- Check model compatibility with llama.cpp version

## Docker Setup

```bash
# Build image
docker build -t llamacpp-server .

# Run with model volume
docker run -p 8080:8080 \
    -v ./models:/models \
    llamacpp-server \
    -m /models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

## Comparison with Other Local Providers

| Feature | llama.cpp | Ollama | LMStudio | vLLM |
|---------|-----------|--------|----------|------|
| GGUF Support | Yes | Yes | Yes | No |
| CPU Optimized | Yes | Yes | Yes | Limited |
| GPU (CUDA) | Yes | Yes | Yes | Yes |
| GPU (Metal) | Yes | Yes | Yes | No |
| Tool Calling | Partial | Yes | Partial | Yes |
| Memory Efficient | Yes | Yes | Yes | Yes |
| Ease of Setup | Medium | Easy | Easy | Medium |
