# Victor AI - Quick Reference

**Version**: 0.5.0
**Last Updated**: January 24, 2026

---

## 📚 Documentation Structure

```
docs/
├── diagrams/              # 30 Mermaid diagrams ✨
├── getting-started/       # Start here!
├── user-guide/            # Daily usage
├── architecture/          # System design
├── reference/             # API & configuration
├── development/           # Contributing
└── archive/               # Historical docs (202 files)
```

---

## 🚀 Quick Start

### 1. Install

```bash
# Local model (free, offline)
pipx install victor-ai
ollama pull qwen2.5-coder:7b
victor chat "Hello"

# Cloud model (full capability)
pipx install victor-ai
export ANTHROPIC_API_KEY=sk-...
victor chat --provider anthropic
```

### 2. Essential Commands

```bash
victor chat                    # Start interactive chat
victor chat "Help me code"     # One-shot command
victor                          # TUI mode
victor --version               # Show version
victor providers               # List all 21 providers
```

### 3. Provider Switching

```bash
# Switch mid-conversation (context preserved!)
/provider openai --model gpt-4o
/provider ollama --model qwen2.5-coder:7b
```

---

## 📖 Documentation Guide

### For New Users

1. **[Installation](docs/getting-started/installation.md)** - 5 min
   - pipx, pip, Docker options

2. **[First Run](docs/getting-started/first-run.md)** - 3 min
   - Initialize Victor
   - First conversation

3. **[Local Models](docs/getting-started/local-models.md)** - 10 min
   - Ollama setup
   - LM Studio, vLLM

4. **[Cloud Models](docs/getting-started/cloud-models.md)** - 10 min
   - 17 cloud providers
   - API key setup

5. **[Troubleshooting](docs/getting-started/troubleshooting.md)** - As needed
   - Common issues and solutions

### For Daily Use

- **[User Guide Index](docs/user-guide/index.md)** - Complete usage guide
- **[CLI Reference](docs/user-guide/cli-reference.md)** - All commands
- **[Providers](docs/user-guide/providers.md)** - Provider configuration
- **[Tools](docs/user-guide/tools.md)** - 55+ tools
- **[FAQ](docs/FAQ.md)** - Frequently asked questions

### For Understanding Architecture

- **[Architecture Overview](docs/architecture/overview.md)** - System design
- **[30 Mermaid Diagrams](docs/diagrams/)** - Visual documentation
  - System architecture
  - Coordinator design
  - Protocols and events
  - Verticals and deployment
  - Developer workflows

- **[Design Patterns](docs/architecture/DESIGN_PATTERNS.md)** - SOLID patterns
- **[Protocols Reference](docs/architecture/PROTOCOLS_REFERENCE.md)** - 98 protocols
- **[Component Reference](docs/architecture/COMPONENT_REFERENCE.md)** - All components

### For Developers

- **[Contributing Guide](docs/development/CONTRIBUTING.md)** - How to contribute
- **[API Reference](docs/reference/api.md)** - Python API
- **[Internal APIs](docs/reference/internals/)** - Protocol & component APIs
- **[Configuration Reference](docs/reference/configuration/)** - All settings

### For Historical Reference

- **[Archive Index](docs/archive/README.md)** - 202 archived files
  - Implementation reports (70 files)
  - Development reports (10 files)
  - Quickstart guides (6 files)
  - Troubleshooting archives (3 files)
  - Migration guides (2 files)
  - Root-level reports (111 files)

---

## 🎯 Common Tasks

### Code Refactoring

```bash
victor chat --mode build "Refactor this function for clarity"
```

### Code Analysis

```bash
victor chat "Analyze this codebase for performance issues"
victor chat --mode explore "Summarize the architecture"
```

### Testing

```bash
victor chat "Write tests for src/auth.py"
victor chat "Run tests and fix failures"
```

### Multi-Provider Cost Optimization

```python
from victor.framework import Agent

# FREE - Brainstorming
brainstormer = await Agent.create(provider="ollama")

# CHEAP - Implementation
implementer = await Agent.create(provider="openai")

# QUALITY - Review
reviewer = await Agent.create(provider="anthropic")
```

---

## 🔧 Configuration

### Minimal Setup

**Local model (no config needed):**
```bash
victor chat  # Auto-detects Ollama
```

**Cloud provider (environment variable):**
```bash
export ANTHROPIC_API_KEY=sk-...
victor chat --provider anthropic
```

### Profile Configuration

**Create `~/.victor/profiles.yaml`:**
```yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    tools: [read, write, edit]
    mode: build

  free:
    provider: ollama
    model: qwen2.5-coder:7b

  fast:
    provider: groq
    model: llama3.1-70b
```

**Use profile:**
```bash
victor chat --profile free
```

---

## 📊 Provider Quick Reference

| Type | Providers | Best For |
|------|-----------|----------|
| **Local** | Ollama, LM Studio, vLLM, llama.cpp | Privacy, offline |
| **Cloud** | Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Groq, Together, Fireworks, OpenRouter, Moonshot, Cerebras, Replicate, Hugging Face | Quality, speed |
| **Enterprise** | Azure, AWS Bedrock, Vertex AI | Compliance |

**Switch mid-conversation:**
```
/provider openai --model gpt-4
/provider ollama --model qwen2.5-coder:7b
```

---

## 🎓 Learning Path

### Beginner (0-1 day)
1. Install Victor
2. Run first conversation
3. Explore basic commands
4. Read [FAQ](docs/FAQ.md)

### Intermediate (1-3 days)
1. Try different providers
2. Learn tool usage
3. Explore workflows
4. Read [User Guide](docs/user-guide/index.md)

### Advanced (3-7 days)
1. Study architecture (30 diagrams)
2. Create custom workflows
3. Develop custom tools
4. Contribute to project

---

## 💡 Tips

### Cost Optimization
- Use Ollama for brainstorming (FREE)
- Use Groq for rapid iteration (300+ tok/s)
- Use OpenAI mini for implementation (CHEAP)
- Use Anthropic for review (QUALITY)

### Mode Selection
- `build` (default): Full edits, active development
- `plan`: 2.5x exploration, sandbox mode
- `explore`: 3.0x exploration, no edits

### Tool Selection
- All tools: `victor chat --tools all`
- Specific: `victor chat --tools read,write,edit`
- Budget: `victor chat --tool-budget 20`

---

## 📞 Help

- **[Documentation Index](docs/)** - Complete documentation hub
- **[GitHub Issues](https://github.com/vjsingh1984/victor/issues)** - Report bugs
- **[GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)** - Ask questions

---

<div align="center">

**[← Back to Documentation](docs/)**

**Quick Reference**

*Essential commands and patterns*

</div>
