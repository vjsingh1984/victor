<div align="center">

![Victor Banner](../../assets/victor-banner.png)

# Quick Start Guide

</div>

> Canonical reference: Docker users see `docker/README.md` and `docker/QUICKREF.md`. This guide focuses on local install. Reality check: semantic tool selection is enabled by default and `security_scan` is regex-only (no CVE/IaC).

Get up and running with Victor in minutes!

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** running locally (since you already have this!)
   - Verify: `ollama list` should show your installed models
3. A code editor (VS Code, PyCharm, etc.)

## Installation

### Step 1: Create Virtual Environment

```bash
cd /path/to/victor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Victor

```bash
# Install in editable mode with all dependencies
pip install -e ".[dev]"
```

### Step 3: Initialize Configuration

```bash
# This creates ~/.victor/profiles.yaml with default settings
victor init
```

## Verify Ollama is Ready

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull a recommended model if you don't have one
ollama pull qwen2.5-coder:7b
# or
ollama pull deepseek-coder:6.7b
# or
ollama pull codellama:13b
```

## First Run

### Streaming CLI (Default)

```bash
# Start streaming chat (Rich output by default)
victor chat
```

### Plain/Debug Output

```bash
# Force plain text and show debug logs
victor chat --renderer text --log-level DEBUG
```

**When to use plain mode:**
- Debugging issues (see all log output in console)
- Running in non-interactive terminals
- Piping output to files
- CI/CD environments

### Provider/Model Overrides

```bash
# Override provider/model (bypass profiles.yaml) and set endpoint for local providers
victor chat --provider ollama --model qwen3-coder:30b --endpoint http://localhost:11434
```

### Modes and Budgets

```bash
# Start in explore mode with tighter budgets
victor chat --mode explore --tool-budget 20 --max-iterations 60
```

### One-Shot Command

```bash
# Execute a single command (always uses CLI mode)
victor chat "Write a hello world FastAPI application"

# Or simply
victor "Write a hello world FastAPI application"
```

### Example Session

```
You > Write a Python function to calculate the factorial of a number
Assistant: [Generates code and can even write it to a file for you]

You > Now write unit tests for that function
Assistant: [Creates test file with pytest tests]
```

### Run Example Script

```bash
# Test with the example script
python examples/simple_chat.py
```

## Configure Your Profile

Edit `~/.victor/profiles.yaml`:

```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b  # Change to your preferred model
    temperature: 0.7
    max_tokens: 4096

providers:
  ollama:
    base_url: http://localhost:11434
```

## Usage Examples

### 1. Code Generation

```bash
victor "Create a REST API endpoint for user authentication with FastAPI"
```

### 2. Code Review

```bash
victor "Review the code in src/main.py and suggest improvements"
```

### 3. Debugging

```bash
victor "Analyze the error in error.log and suggest fixes"
```

### 4. File Operations

The agent can read and write files:
```
You: Read the contents of README.md and summarize it
You: Create a new file called hello.py with a simple greeting function
You: List all Python files in the current directory
```

### 5. Execute Commands

The agent can run bash commands:
```
You: Run pytest and show me the results
You: Check the git status and tell me what changed
You: Install the requests library using pip
```

## Available Commands in Interactive Mode

- `exit` or `quit` - Exit the REPL
- `clear` - Reset conversation history
- Just type your message and press Enter to chat

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test
pytest tests/unit/test_ollama_provider.py
```

## Troubleshooting

### "Connection refused" error

**Problem**: Can't connect to Ollama
**Solution**: Make sure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### Model not found

**Problem**: "Model not found: qwen2.5-coder:7b"
**Solution**: Pull the model first:
```bash
ollama pull qwen2.5-coder:7b
# or choose a different model you have installed
ollama list
```

### Import errors

**Problem**: `ModuleNotFoundError`
**Solution**: Make sure you installed in editable mode:
```bash
pip install -e ".[dev]"
```

## Next Steps

### 1. Add More Models

```bash
# Pull additional models
ollama pull llama3:8b
ollama pull deepseek-coder:6.7b
ollama pull mistral:7b
```

Then add profiles in `~/.victor/profiles.yaml`:
```yaml
profiles:
  llama:
    provider: ollama
    model: llama3:8b

  deepseek:
    provider: ollama
    model: deepseek-coder:6.7b
```

Use with: `victor --profile llama`

### 2. Add API Keys for Cloud Providers

**Option A: Secure Keyring Storage (Recommended)**
```bash
# Store keys in system keyring (macOS Keychain, Windows Credential Manager, Linux Secret Service)
victor keys --set anthropic --keyring
victor keys --set openai --keyring
victor keys --set google --keyring
victor keys --set groqcloud --keyring
victor keys --set mistral --keyring
victor keys --set openrouter --keyring

# List configured providers
victor keys
```

**Option B: Environment Variables**
```bash
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
```

**Free Tier Providers** (no credit card required):
```bash
# Groq - Ultra-fast inference, Llama 3.3 70B
victor keys --set groqcloud --keyring
victor chat --provider groqcloud --model llama-3.3-70b-versatile

# Cerebras - Fastest inference (1000+ tok/s)
victor keys --set cerebras --keyring
victor chat --provider cerebras --model llama-3.3-70b

# Mistral - 500K tokens/min free
victor keys --set mistral --keyring
victor chat --provider mistral --model mistral-large-latest

# OpenRouter - 350+ models, free tier
victor keys --set openrouter --keyring
victor chat --provider openrouter --model nvidia/nemotron-3-nano-30b-a3b:free
```

Update profiles.yaml for quick access:
```yaml
profiles:
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
  groq:
    provider: groqcloud
    model: llama-3.3-70b-versatile
```

### 3. Explore Examples

Victor includes many working example demos:

```bash
# Check out the examples directory
ls examples/

# Run the simple chat example (uses Ollama)
python examples/simple_chat.py

# Local/Free demos (no API keys required):
python examples/caching_demo.py           # Tiered caching system
python examples/context_management_demo.py # Context window management
python examples/multi_file_editing_demo.py # Atomic multi-file operations
python examples/codebase_indexing_demo.py  # Code symbol indexing
python examples/batch_processing_demo.py   # Multi-file search/replace
python examples/git_tool_demo.py           # Git operations
python examples/testing_demo.py            # Pytest integration
python examples/cicd_demo.py               # CI/CD workflow generation
python examples/refactoring_demo.py        # Code refactoring
python examples/dependency_demo.py         # Dependency management
python examples/documentation_demo.py      # Docstring generation
python examples/enterprise_tools_demo.py   # Code review, scaffolding, security
python examples/metrics_demo.py            # Code metrics analysis

# Semantic search (uses local sentence-transformers):
python examples/airgapped_codebase_search.py  # 100% offline semantic search
python examples/qwen3_embedding_demo.py       # Qwen3 embeddings with Ollama
```

### 4. Contribute

Check out [CONTRIBUTING.md](CONTRIBUTING.md) to:
- Add new providers (OpenAI, Google, etc.)
- Create custom tools
- Improve documentation
- Fix bugs

## Common Use Cases

### Create a New Project

```bash
victor "Create a new FastAPI project structure with:
- User authentication
- Database models with SQLAlchemy
- API endpoints for CRUD operations
- Pytest test suite
- Docker configuration"
```

### Refactor Code

```bash
victor "Refactor the code in src/legacy.py to use modern Python 3.11 features and improve performance"
```

### Generate Documentation

```bash
victor "Generate comprehensive docstrings for all functions in src/utils.py"
```

### Debug Issues

```bash
victor "Analyze the test failures in test_results.txt and fix the issues"
```

## Tips for Best Results

1. **Be Specific**: The more details you provide, the better the results
2. **Iterate**: Ask follow-up questions to refine the output
3. **Use Tools**: The agent can read/write files and execute commands
4. **Context**: Provide file paths or code snippets for context
5. **Test Locally**: Use Ollama for development to save costs

## Get Help

- **Documentation**: Check [README.md](README.md) and [DESIGN.md](DESIGN.md)
- **Issues**: Open an issue on GitHub
- **Examples**: Look at `examples/` directory

## What's Implemented

Victor includes these production-ready features:

**Cloud Providers (Tested)**
- [x] **Anthropic** - Claude 3.5/4, Sonnet/Opus/Haiku
- [x] **OpenAI** - GPT-4.1, GPT-4o, o1/o3 reasoning models
- [x] **Google** - Gemini 2.5/3.0 Pro/Flash
- [x] **xAI** - Grok 2/3 with vision
- [x] **DeepSeek** - DeepSeek-V3, affordable at $0.14/1M tokens
- [x] **Groq** - Ultra-fast LPU inference (free tier)
- [x] **Cerebras** - Fastest inference 1000+ tok/s (free tier)
- [x] **Mistral** - Mistral Large/Codestral (500K tokens/min free)
- [x] **Moonshot** - Kimi K2 with 256K context
- [x] **OpenRouter** - 350+ models, unified gateway (free tier)

**Local Inference**
- [x] **Ollama** - 100+ models, tool calling support
- [x] **LMStudio** - GUI-based local inference
- [x] **vLLM** - High-throughput production serving
- [x] **llama.cpp** - CPU/GPU GGUF models

**Core Features**
- [x] **Secure API Key Storage** - System keyring (Keychain/Credential Manager)
- [x] **MCP Protocol** - Client and server support
- [x] **Context caching** - Tiered memory + disk caching
- [x] **45+ Enterprise tools** - File ops, git, testing, CI/CD, refactoring, security
- [x] **Semantic code search** - Air-gapped mode with local embeddings
- [x] **Multi-file editing** - Atomic transactions with rollback

## Roadmap

Planned features:
- [ ] Web UI
- [ ] Multi-agent collaboration
- [ ] More provider integrations (Together, Fireworks, HuggingFace)

Contributions are welcome!

---

Happy coding with Victor! 🤖
