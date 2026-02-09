# Beginner User Journey

**Target Audience:** New users who want to get started with Victor
**Time Commitment:** 30 minutes
**Prerequisites:** Basic Python knowledge, terminal comfort

## Journey Overview

This journey will take you from zero to having your first productive conversation with Victor. By the end, you'll be
  able to:
- Install Victor on your machine
- Have conversations with AI coding assistants
- Switch between different LLM providers
- Save and restore conversation sessions

## Visual Guide

```mermaid
flowchart LR
    A([Install<br/>5 min]) --> B([First Run<br/>3 min])
    B --> C([Learn Basics<br/>10 min])
    C --> D([Explore Features<br/>7 min])
    D --> E([Ready for Daily Use])

    style A fill:#e8f5e9,stroke:#2e7d32
    style B fill:#e3f2fd,stroke:#1565c0
    style C fill:#fff3e0,stroke:#e65100
    style D fill:#f3e5f5,stroke:#6a1b9a
    style E fill:#c8e6c9,stroke:#1b5e20,stroke-width:3px
```

See [Beginner Onboarding Diagram](../diagrams/user-journeys/beginner-onboarding.mmd) for detailed interactive flow.

## Step 1: Installation (5 minutes)

Choose your installation method:

### Option A: Quick Install (Recommended)

```bash
# Using pipx (isolated environment)
pipx install victor-ai

# Or using pip
pip install victor-ai
```

### Option B: Docker

```bash
docker pull ghcr.io/vjsingh1984/victor:latest
```

### Option C: Development Install

```bash
git clone https://github.com/vjsingh1984/victor.git
cd victor
pip install -e ".[dev]"
```

**📖 Full Guide:** [Installation Guide](../getting-started/installation.md)

## Step 2: First Run (3 minutes)

### Choose Your Provider

**Local (No API Key Required):**
```bash
# Install Ollama
brew install ollama  # macOS
ollama serve
ollama pull qwen2.5-coder:7b

# Start Victor
victor chat --provider ollama
```

**Cloud (API Key Required):**
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-your-key

# Start Victor
victor chat --provider anthropic
```

**📖 Full Guide:** [First Run](../getting-started/first-run.md)

### Your First Conversation

```
You: Hello, Victor! What can you do?

Victor: Hello! I'm Victor, your AI coding assistant. I can help you with:
- Code generation and editing
- Refactoring and optimization
- Testing and debugging
- Git operations
- Web search and research

What would you like to work on today?
```

## Step 3: Learn the Basics (10 minutes)

### Essential Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `victor chat` | Start interactive chat | `victor chat` |
| `victor chat --provider <name>` | Use specific provider | `victor chat --provider openai` |
| `victor chat --mode plan` | Enter planning mode | `victor chat --mode plan` |
| `victor chat --session <name>` | Save/load session | `victor chat --session my-project` |

### Common Tasks

**Read a file:**
```
You: Read the file src/main.py and summarize what it does.
```

**Edit code:**
```
You: Add error handling to the parse_config function.
```

**Run tests:**
```
You: Run the tests and fix any failures.
```

**Git operations:**
```
You: Create a new branch feature/add-auth and commit the changes.
```

**📖 Full Guide:** [User Guide](../user-guide/)

## Step 4: Explore Key Features (7 minutes)

### Provider Switching

Switch providers mid-conversation without losing context:

```
You: switch to openai
[Victor switches to OpenAI, maintaining your conversation history]
```

Supported providers:
- **Cloud:** Anthropic, OpenAI, Google, Azure, AWS Bedrock
- **Local:** Ollama, LM Studio, vLLM
- **Total:** 21 providers

### Conversation Modes

- **BUILD** (default): Full editing capabilities
- **PLAN:** 2.5x more exploration, sandbox mode
- **EXPLORE:** 3.0x exploration, no edits

```
You: switch to plan mode
[Victor enters exploration mode, analyzing your codebase]
```

### Workflows (Automation)

Multi-step automation with YAML workflows:

```yaml
# test-and-fix.yaml
name: Test and Fix
steps:
  - name: Run tests
    tool: pytest
  - name: Fix failures
    tool: code_edit
    depends_on: Run tests
```

```
You: Run workflow test-and-fix.yaml
[Victor executes the workflow steps]
```

**📖 Full Guide:** [Workflows Guide](../guides/workflows/)

## What's Next?

Congratulations! 🎉 You're ready for daily use with Victor.

### Continue Your Journey

**For Daily Users:**
- → [Intermediate User Journey](intermediate.md)
- Learn advanced workflows
- Customize profiles
- Use multi-agent teams

**For Contributors:**
- → [Developer Journey](developer.md)
- Set up development environment
- Write extensions
- Contribute to core

**For DevOps/SRE:**
- → [Operations Journey](operations.md)
- Deploy Victor in production
- Configure monitoring
- Set up security

### Quick Reference

- **FAQ:** [Frequently Asked Questions](../user-guide/faq.md)
- **Troubleshooting:** [Common Issues](../getting-started/troubleshooting.md)
- **Configuration:** [Config Options](../getting-started/configuration.md)
- **CLI Reference:** [Command Reference](../user-guide/cli-reference.md)

### Need Help?

- **GitHub Issues:** [Report Bugs](https://github.com/vjsingh1984/victor/issues)
- **Discussions:** [Ask Questions](https://github.com/vjsingh1984/victor/discussions)
- **Documentation:** [Full Docs Index](../)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** January 31, 2026
**Reading Time:** 5 minutes
**Next Journey:** [Intermediate User Journey](intermediate.md)
