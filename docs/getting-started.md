# Getting Started

Install Victor and run your first conversation in under 2 minutes.

## Installation

```bash
# Recommended (isolated environment)
pipx install victor-ai

# Alternative
pip install victor-ai

# Container
docker pull vjsingh1984/victor
```

**Optional extras:**
```bash
pip install victor-ai[api]        # HTTP server
pip install victor-ai[google]     # Gemini provider
pip install victor-ai[native]     # Rust extensions
```

## Choose Your Model

Victor works with local or cloud models. Pick based on privacy, cost, and capability needs.

### Local Models (No API Key)

**Ollama** (recommended for beginners):
```bash
# Install Ollama
brew install ollama    # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Start and pull a model
ollama serve
ollama pull qwen2.5-coder:7b

# Use with Victor
victor init
victor chat
```

**Other local options:**
- **LM Studio**: GUI-based, download from [lmstudio.ai](https://lmstudio.ai)
- **vLLM**: High-throughput, requires GPU
- **llama.cpp**: CPU-friendly, GGUF models

See [reference/PROVIDERS.md](reference/PROVIDERS.md) for detailed setup.

### Cloud Providers

```bash
# Set API key (stored in system keyring)
victor keys --set anthropic --keyring
victor keys --set openai --keyring
victor keys --set google --keyring

# Use
victor chat --provider anthropic --model claude-sonnet-4-5
victor chat --provider openai --model gpt-4o
victor chat --provider google --model gemini-2.0-flash
```

**Supported cloud providers:** Anthropic, OpenAI, Google, Groq, DeepSeek, Mistral, xAI, OpenRouter, Together, Cohere, Replicate, Bedrock, Azure, Vertex AI, Fireworks, Perplexity, Cerebras.

## Initialize Victor

```bash
victor init
```

This creates `~/.victor/` with default configuration. Edit `~/.victor/profiles.yaml` to customize:

```yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
  fast:
    provider: groq
    model: llama-3.1-70b-versatile
```

Use profiles:
```bash
victor --profile local
victor --profile claude
```

## First Conversation

```bash
# Interactive mode
victor chat

# One-shot task
victor "explain this codebase structure"

# TUI mode (default)
victor
```

## Agent Modes

| Mode | Edits | Use Case |
|------|:-----:|----------|
| **BUILD** | Yes | Implementation, refactoring |
| **PLAN** | Sandbox only | Analysis, planning |
| **EXPLORE** | Notes only | Understanding code |

```bash
victor chat --mode plan "Analyze the auth system"
victor chat --mode explore "How does routing work?"
```

## Next Steps

- [User Guide](user-guide.md) - Daily usage, tools, workflows
- [Provider Reference](reference/PROVIDERS.md) - All 21 providers
- [Tool Catalog](TOOL_CATALOG.md) - 55+ available tools
- [Workflow DSL](guides/WORKFLOW_DSL.md) - YAML workflows
