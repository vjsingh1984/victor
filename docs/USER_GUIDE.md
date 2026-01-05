# Victor User Guide

A short, practical guide for day-to-day CLI use.

## Start
```bash
pipx install victor-ai
victor init
victor chat
```

One-shot:
```bash
victor "review src/main.py for bugs"
```

## Choose a Model
- Local: Ollama / LM Studio / vLLM (no API key)
- Cloud: Anthropic / OpenAI / Google / Groq / DeepSeek / others

Profiles live in `~/.victor/profiles.yaml`:
```yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
```

Use a profile:
```bash
victor --profile local
```

## Modes
- **BUILD**: make edits
- **PLAN**: analyze without edits
- **EXPLORE**: understand and take notes

```bash
victor chat --mode plan "Plan the auth refactor"
```

## Tools (What the Agent Can Do)
Tools are enabled by your profile, mode, and permissions. See the full list in `TOOL_CATALOG.md`.

Examples:
```
Read README.md and summarize it
Create tests for src/api.py
Show git status and recent commits
```

## Workflows
- CLI workflows: `../examples/workflows/README.md`
- YAML workflows: `guides/WORKFLOW_DSL.md`
- Scheduling & Versioning: `guides/WORKFLOW_SCHEDULER.md`

### Quick Workflow Commands

```bash
# Validate a workflow file
victor workflow validate my_workflow.yaml

# Render workflow as diagram
victor workflow render my_workflow.yaml --format ascii
```

### Scheduling Workflows

Schedule workflows to run on a cron schedule:

```bash
# Add a scheduled workflow
victor scheduler add daily_report --cron "0 9 * * *"

# Start the scheduler daemon
victor scheduler start

# List scheduled workflows
victor scheduler list

# View execution history
victor scheduler history
```

YAML configuration:
```yaml
workflows:
  daily_report:
    schedule:
      cron: "0 9 * * *"
      timezone: "UTC"
    nodes:
      - id: generate
        type: agent
        role: analyst
        goal: "Generate daily metrics report"
        timeout: 300
        next: []
```

**Note**: The built-in scheduler is for single-instance deployments. For production HA, use Airflow or Temporal.io.

See `guides/WORKFLOW_SCHEDULER.md` for complete documentation.

## Troubleshooting

### Local Providers

#### Ollama
```bash
# Install (macOS)
brew install ollama

# Install (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start server
ollama serve

# Pull a model
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull codellama:13b

# Verify
curl http://localhost:11434/api/tags

# Use with Victor
victor chat --provider ollama --model qwen2.5-coder:7b
```

**Common issues**:
- "Connection refused": Run `ollama serve` first
- "Model not found": Run `ollama pull <model>`
- Custom port: Set `OLLAMA_HOST=http://localhost:11435`

#### LM Studio
```bash
# 1. Download from https://lmstudio.ai
# 2. Launch LM Studio
# 3. Download a model (e.g., Qwen2.5-Coder, CodeLlama)
# 4. Go to "Local Server" tab → Start Server

# Default endpoint: http://localhost:1234/v1

# Use with Victor
victor chat --provider lmstudio --model local-model

# Or set in profile
```

Profile config:
```yaml
profiles:
  lmstudio:
    provider: lmstudio
    model: qwen2.5-coder-7b    # Model name from LM Studio
    base_url: http://localhost:1234/v1
```

**Common issues**:
- "Server not running": Start server in LM Studio UI
- Wrong model name: Check model name in LM Studio sidebar

#### vLLM
```bash
# Install
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --port 8000

# Or with Docker
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-Coder-7B-Instruct

# Use with Victor
victor chat --provider vllm --model Qwen/Qwen2.5-Coder-7B-Instruct
```

Profile config:
```yaml
profiles:
  vllm:
    provider: vllm
    model: Qwen/Qwen2.5-Coder-7B-Instruct
    base_url: http://localhost:8000/v1
```

**Common issues**:
- "CUDA out of memory": Use smaller model or add `--max-model-len 4096`
- Slow startup: First run downloads model (~15GB for 7B)

#### llama.cpp
```bash
# Install llama-cpp-python with server
pip install llama-cpp-python[server]

# Download a GGUF model
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf

# Start server
python -m llama_cpp.server \
    --model qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n_ctx 4096

# Use with Victor
victor chat --provider llamacpp --model qwen2.5-coder
```

Profile config:
```yaml
profiles:
  llamacpp:
    provider: llamacpp
    model: qwen2.5-coder
    base_url: http://localhost:8080/v1
```

**Common issues**:
- "Model not found": Use full path to .gguf file
- Slow on CPU: Use Q4_K_M quantization, reduce context length

### Cloud Providers

#### Setting API Keys
```bash
# Store in system keyring (recommended)
victor keys --set anthropic --keyring
victor keys --set openai --keyring
victor keys --set google --keyring

# Or environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# List configured keys
victor keys --list
```

#### Anthropic (Claude)
```bash
# Get key from https://console.anthropic.com/
victor keys --set anthropic --keyring

# Use
victor chat --provider anthropic --model claude-sonnet-4-5
victor chat --provider anthropic --model claude-opus-4
```

#### OpenAI
```bash
# Get key from https://platform.openai.com/api-keys
victor keys --set openai --keyring

# Use
victor chat --provider openai --model gpt-4o
victor chat --provider openai --model gpt-4-turbo
```

#### Google (Gemini)
```bash
# Get key from https://aistudio.google.com/apikey
victor keys --set google --keyring

# Use
victor chat --provider google --model gemini-2.0-flash
victor chat --provider google --model gemini-1.5-pro
```

#### Groq
```bash
# Get key from https://console.groq.com/keys
victor keys --set groq --keyring

# Use (fast inference)
victor chat --provider groq --model llama-3.1-70b-versatile
victor chat --provider groq --model mixtral-8x7b-32768
```

#### DeepSeek
```bash
# Get key from https://platform.deepseek.com/
victor keys --set deepseek --keyring

# Use
victor chat --provider deepseek --model deepseek-coder
victor chat --provider deepseek --model deepseek-chat
```

#### Mistral
```bash
# Get key from https://console.mistral.ai/
victor keys --set mistral --keyring

# Use
victor chat --provider mistral --model mistral-large-latest
victor chat --provider mistral --model codestral-latest
```

#### xAI (Grok)
```bash
# Get key from https://console.x.ai/
victor keys --set xai --keyring

# Use
victor chat --provider xai --model grok-2
```

### General Issues

```bash
# Check provider status
victor providers --list

# Test a provider
victor test-provider anthropic

# Debug mode
victor chat --debug

# Reset configuration
victor init --force
```

## More Help
- Quick Start: `guides/QUICKSTART.md`
- Provider setup: `guides/PROVIDER_SETUP.md`
- Examples: `../examples/README.md`
