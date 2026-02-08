# Victor AI FAQ - Part 1

**Part 1 of 2:** General Questions, Installation & Setup, Providers & Models, Usage & Features, and Performance & Cost

---

## Navigation

- **[Part 1: Basic Questions](#)** (Current)
- [Part 2: Advanced Topics](part-2-advanced-topics.md)
- [**Complete Guide](../faq.md)**

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation--setup)
3. [Providers & Models](#providers--models)
4. [Usage & Features](#usage--features)
5. [Performance & Cost](#performance--cost)
6. [Troubleshooting](#troubleshooting) *(in Part 2)*
7. [Advanced Usage](#advanced-usage) *(in Part 2)*
8. [Community & Support](#community--support) *(in Part 2)*

---

# Victor AI FAQ

**Frequently Asked Questions about Victor AI**

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation--setup)
3. [Providers & Models](#providers--models)
4. [Usage & Features](#usage--features)
5. [Performance & Cost](#performance--cost)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)
8. [Community & Support](#community--support)

---

## General Questions

### What is Victor AI?

Victor AI is an open-source AI coding assistant that supports multiple LLM providers, tools, and domain verticals (e.g., Coding, DevOps, RAG, Data Analysis, Research). It provides both local and cloud LLM support through a unified CLI and Python API.

**Key differentiators:**
- Provider agnostic - switch models without losing context
- Multi-agent teams - coordinate specialized agents
- Air-gapped mode - local/offline operation with supported providers
- Semantic codebase search - context-aware understanding
- YAML workflows - automate repetitive tasks

### Is Victor AI free?

Yes. Victor AI is:
- **Open source** - No licensing fees
- **Free to use** - With local models (compute costs apply)
- **Bring your own key** - Use your own API keys for cloud providers
- **No subscription** - Pay only for provider usage where applicable

### How is Victor different from GitHub Copilot?

| Feature | Victor AI | GitHub Copilot |
|---------|-----------|----------------|
| **Providers** | Multiple providers | OpenAI only |
| **Local Support** | Local/offline options | None |
| **Multi-Agent** | Team workflows | Single agent |
| **Provider Switching** | Yes, mid-conversation | No |
| **Extensibility** | Custom verticals | Limited |
| **Cost** | Open-source + provider costs | Subscription only |
| **Privacy** | Local-only possible with supported providers | Cloud-only |

### Can I use Victor offline?

Yes. Victor supports **air-gapped mode** with local providers:
- Use local models (Ollama, LM Studio, vLLM)
- No internet connection required
- Full tool functionality
- Data stays on your machine with local-only configurations

```bash
export VICTOR_AIRGAPPED_MODE=true
victor chat --provider ollama
```

---

## Installation & Setup

### How do I install Victor AI?

**Recommended (pipx):**
```bash
pipx install victor-ai
```

**Alternative (pip):**
```bash
pip install victor-ai
```

**Verify:**
```bash
victor --version
```

### What are the requirements?

**Minimum:**
- Python 3.10 or higher
- 4GB RAM
- 1GB disk space

**Recommended for local models:**
- Python 3.10+
- 16GB RAM (for 7B models)
- 20GB disk space
- GPU (optional, but faster)

### How do I update Victor?

```bash
pipx upgrade victor-ai
# or
pip install --upgrade victor-ai
```

### Installation fails with "Permission denied"

**Solution:**
```bash
# Use user directory
pip install --user victor-ai

# Or use virtual environment
python -m venv .venv
source .venv/bin/activate
pip install victor-ai
```

---

## Providers & Models

### Which provider should I use?

**For beginners:**
- **Ollama** - Free, easy to set up, good models
- **Claude Sonnet** - Balanced capability/cost
- **GPT-4o** - Fast, capable

**For specific tasks:**
- **Brainstorming**: Ollama (free)
- **Implementation**: GPT-4o (fast)
- **Review**: Claude Sonnet (thorough)
- **Complex analysis**: Claude Opus (best)

### How do I set up Anthropic Claude?

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-your-key-here

# Use Victor
victor chat --provider anthropic --model claude-sonnet-4-5
```

Get API key: https://console.anthropic.com/

### How do I set up OpenAI GPT?

```bash
# Set API key
export OPENAI_API_KEY=sk-your-key-here

# Use Victor
victor chat --provider openai --model gpt-4o
```

Get API key: https://platform.openai.com/api-keys

### How do I set up Ollama (local)?

```bash
# Install Ollama
brew install ollama  # macOS
# or
curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Pull a model
ollama pull qwen2.5-coder:7b

# Use Victor
victor chat --provider ollama
```

### Can I switch providers mid-conversation?

Yes! Victor's unique feature:
```bash
# Start with Ollama
victor chat --provider ollama "Brainstorm ideas"

# Switch to Claude (in TUI mode)
/provider anthropic --model claude-sonnet-4-5
"Implement the first idea"

# Context is preserved!
```

### Which models are supported?

Models depend on your configured providers. Examples include:
- **Anthropic**: Claude Sonnet 4, Opus, Haiku
- **OpenAI**: GPT-4o, GPT-4, o1-preview
- **Google**: Gemini 2.0, Gemini 1.5 Pro
- **Local (Ollama)**: Qwen2.5, DeepSeek, CodeLlama, etc.
- And more...

[Full provider list](reference/providers/index.md)

---

## Usage & Features

### How do I start using Victor?

```bash
# Interactive TUI mode
victor

# CLI mode
victor chat "Your question here"

# With specific provider
victor chat --provider anthropic "Explain async/await"
```

### What's the difference between TUI and CLI mode?

**TUI Mode** (`victor chat`):
- Interactive terminal UI
- Syntax highlighting
- Multi-line input
- Session history
- Rich formatting

**CLI Mode** (`victor chat --no-tui`):
- Quick, one-off queries
- Script-friendly
- Pipe-able output
- Minimal overhead

### How do I save conversations?

```bash
# In TUI mode
/save "My Session Title"

# From CLI
victor chat --session my-session-name

# List sessions
victor sessions

# Resume session
victor resume my-session-name
```

### What are agent modes?

Three modes for different use cases:

| Mode | Description | Use When |
|------|-------------|----------|
| **build** | Full capabilities, can edit files | Implementing features |
| **plan** | Analysis only, no edits | Architecture review |
| **explore** | Deep exploration, no edits | Learning codebase |

```bash
victor chat --mode plan "Analyze this architecture"
```

### What tools does Victor have?

Tools across multiple domains (see the catalog for the current list):

**Coding:**
- File operations (read, write, search)
- AST parsing and analysis
- Code review
- Test generation

**DevOps:**
- Git operations
- Docker management
- CI/CD configuration

**Research:**
- Web search
- Document analysis
- Citation generation

**Data Analysis:**
- Pandas operations
- Visualization
- Statistics

**RAG:**
- Vector search
- Document indexing
- Semantic retrieval

[List all tools](user-guide/tools.md)

### How do I use workflows?

```bash
# List available workflows
victor workflow list

# Run a workflow
victor workflow run code-review --file myfile.py

# Create custom workflow
victor workflow create my-workflow.yaml
```

[Learn more about workflows](user-guide/workflows.md)

### What are multi-agent teams?

Coordinate multiple specialized agents:
```bash
# Create team
victor team create review-team.yaml

# Run team
victor team run review-team "Review this PR"
```

[Learn more about teams](guides/MULTI_AGENT_TEAMS.md)

---

## Performance & Cost

### How much does Victor cost?

**Victor itself**: Free (open source)

**Usage costs**:
- **Local models**: $0 (Ollama, LM Studio)
- **Cloud models**: Pay for API usage only
  - Claude Sonnet: ~$3/M input, ~$15/M output
  - GPT-4o: ~$2.50/M input, ~$10/M output
  - GPT-4o-mini: ~$0.15/M input, ~$0.60/M output

### How can I reduce costs?

**1. Use local models for brainstorming:**
```bash
victor chat --provider ollama "Brainstorm ideas"
```

**2. Use cheaper models for drafts:**
```bash
victor chat --provider openai --model gpt-4o-mini "Draft this"
```

**3. Use best models only for final polish:**
```bash
/provider anthropic --model claude-sonnet-4-5
"Refine this implementation"
```

**4. Enable caching:**
```yaml
# In ~/.victor/config.yaml
cache_enabled: true
cache_ttl: 3600  # 1 hour
```

### Victor is slow, how can I speed it up?

**1. Use faster models:**
```bash
victor chat --provider groq "Quick question"  # 300+ tok/s
```

**2. Use caching:**
```bash
export VICTOR_CACHE_ENABLED=true
```

**3. Reduce context:**
```bash
victor chat --max-tokens 1000 "Brief answer"
```

**4. Use local model for offline:**
```bash
victor chat --provider ollama
```

### How many tokens will this cost?

Estimate: 1 token ≈ 4 characters for English text

**Example costs (GPT-4o):**
- 1000 words (prompt): ~$0.006
- 1000 words (response): ~$0.024
- Typical coding session: ~$0.05-0.15

**Use local models (Ollama) for $0**

