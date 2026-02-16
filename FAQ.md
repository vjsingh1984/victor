# Frequently Asked Questions

Comprehensive FAQ for Victor — agentic AI framework. Last updated: February 2026

## Table of Contents

- [Getting Started](#getting-started)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Features](#features)
- [Troubleshooting](#troubleshooting)
- [Advanced](#advanced)

---

## Getting Started

### What is Victor?

**Victor** is an open-source agentic AI framework that supports 22 LLM providers (both local and cloud) through a unified CLI/TUI interface. Unlike ChatGPT or Claude's web interfaces, Victor:

- **Works offline** with local models (Ollama, LM Studio, vLLM)
- **Switches providers mid-conversation** without losing context
- **Integrates with your codebase** through 33 tool modules
- **Automates workflows** via YAML-based StateGraph DSL
- **Supports 9 domain verticals**: Coding, DevOps, RAG, Data Analysis, Research, Security, IaC, Classification, Benchmark

### How does Victor differ from ChatGPT/Claude?

| Feature | Victor | ChatGPT/Claude Web |
|---------|--------|-------------------|
| **Provider Choice** | 22 providers, switchable | 1 provider |
| **Local Models** | Full support (Ollama, vLLM) | No |
| **Code Integration** | Deep (AST, LSP, git, tests) | Upload files only |
| **Offline Use** | Yes (air-gapped mode) | No |
| **Context Window** | Up to 2M tokens | Varies by provider |
| **Workflows** | YAML automation, scheduling | Manual |
| **Multi-Agent** | Built-in team coordination | No |
| **Cost Control** | Use cheapest provider per task | Fixed pricing |

### What providers are supported?

Victor supports **22 LLM providers**:

**Local (No API Key):**
- Ollama, LM Studio, vLLM, llama.cpp

**Cloud:**
- Anthropic (Claude), OpenAI (GPT-4), Google (Gemini), xAI (Grok)
- DeepSeek, Mistral, Groq, Cerebras, Together AI, Fireworks
- OpenRouter, Moonshot, Hugging Face, Replicate

**Enterprise:**
- Azure OpenAI, AWS Bedrock, Google Vertex AI

### Can I use Victor offline?

**Yes!** Victor's **air-gapped mode** enables full offline functionality:

```bash
# Enable air-gapped mode
victor --airgapped chat "Help me refactor this code"

# Or set in config
export VICTOR_AIRGAPPED=true
```

**Air-gapped mode:**
- ✅ Uses only local providers (Ollama, LM Studio, vLLM)
- ✅ Disables web tools (web_search, web_fetch)
- ✅ Uses local embeddings
- ✅ All file operations work normally
- ❌ No internet access required

**Requirements:**
- Local model installed (e.g., `ollama pull qwen2.5-coder:7b`)
- Model supports tool calling (most Qwen, Llama 3, Mistral models do)

### Is Victor free?

**Victor itself is free and open-source** (Apache 2.0 license). However:

- **Local models**: 100% free (run on your hardware)
- **Cloud providers**: Pay-per-use from provider (e.g., Anthropic, OpenAI)
- **Some providers offer free tiers**:
  - Google Gemini 2.0 Flash (experimental models)
  - Groq (free developer tier)
  - Mistral (500K tokens/minute free tier)

**Cost optimization tips:**
1. Use local models for simple tasks
2. Switch to cloud for complex reasoning (`/provider anthropic`)
3. Use cheaper providers (DeepSeek, Groq) when appropriate
4. Enable caching to avoid repeated API calls

---

## Installation & Setup

### How do I install Victor?

**Recommended: pipx**
```bash
pip install pipx
pipx ensurepath
pipx install victor-ai
```

**Alternative: pip**
```bash
pip install victor-ai
```

**Docker:**
```bash
docker pull ghcr.io/vjsingh1984/victor:latest
docker run -it -v ~/.victor:/root/.victor ghcr.io/vjsingh1984/victor:latest chat
```

**System Requirements:**
- Python 3.10 or higher
- 4GB RAM minimum (8GB+ recommended for local models)
- Linux, macOS, or Windows (WSL2)

### Why do I get "command not found: victor"?

**Cause**: Victor is not in your PATH.

**Solutions:**

1. **If using pipx:**
   ```bash
   pipx ensurepath
   source ~/.bashrc  # or ~/.zshrc
   ```

2. **If using pip:**
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify installation:**
   ```bash
   which victor
   victor --version
   ```

### How do I set up API keys?

**Method 1: Environment Variables (Recommended)**
```bash
# Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (GPT-4)
export OPENAI_API_KEY=sk-proj-...

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
```

**Method 2: System Keyring (Most Secure)**
```bash
victor keys --set anthropic --keyring
victor keys --set openai --keyring
```

**Method 3: profiles.yaml**
```yaml
# ~/.victor/profiles.yaml
profiles:
  claude:
    provider: anthropic
    api_key_env: ANTHROPIC_API_KEY
```

**Never hardcode API keys** in configuration files or commit to git.

### How do I set up Ollama?

**Installation:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve
```

**Pull a model:**
```bash
# Recommended for coding
ollama pull qwen2.5-coder:7b

# Other good options
ollama pull llama3.2:3b      # Lightweight
ollama pull mistral:7b       # General purpose
```

**Use with Victor:**
```bash
# Victor auto-detects Ollama
victor chat "Hello, Victor!"
```

**Troubleshooting Ollama:**
```bash
# Check if running
ollama list

# View logs
ollama logs

# Test connection
curl http://localhost:11434/api/version

# Custom host
export OLLAMA_HOST=127.0.0.1:11434
```

### How do I set up LM Studio?

**Installation:**
1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Install and launch LM Studio
3. Download a model (search for "Qwen 2.5 Coder" or "Llama 3.2")
4. Start the local server (Developer tab → Start Server)

**Use with Victor:**
```bash
# Set host if different from default
export LMSTUDIO_ENDPOINTS="http://127.0.0.1:1234"

# Use with Victor
victor chat --provider lmstudio --model qwen2.5-coder-7b
```

### How do I configure profiles?

**Create profiles.yaml:**
```yaml
# ~/.victor/profiles.yaml
profiles:
  # Default profile
  default:
    provider: ollama
    model: qwen2.5-coder:7b

  # Cloud for complex tasks
  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096

  # Fast for simple tasks
  fast:
    provider: groqcloud
    model: llama-3.3-70b-versatile

  # Code review (lower temperature)
  review:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.3

default_profile: default
```

**Use profiles:**
```bash
victor --profile claude chat "Design a REST API"
victor --profile fast chat "Quick question"
victor --profile review chat "Review this PR"
```

---

## Usage

### How do I switch providers mid-conversation?

**Use the `/provider` command:**
```bash
# Start with Claude
victor chat --provider anthropic "Design a REST API"

# Continue with GPT-4 (context preserved)
/provider openai --model gpt-4o

# Finish with local model (privacy)
/provider ollama --model qwen2.5-coder:7b
```

**Benefits:**
- Leverage different model strengths
- Cost optimization (cheaper for simple tasks)
- Privacy (local models for sensitive data)
- Redundancy (fallback if provider is down)

**Context is always preserved** when switching providers.

### What are agent modes (BUILD/PLAN/EXPLORE)?

Victor has three execution modes:

| Mode | File Edits | Tool Budget | Use When |
|------|------------|-------------|----------|
| **BUILD** | Yes | 1x (default) | Making actual changes |
| **PLAN** | Sandbox only | 2.5x | Analysis, planning, code review |
| **EXPLORE** | No | 3x | Understanding code, learning |

**Using modes:**
```bash
# CLI flag
victor chat --mode plan "Analyze this architecture"

# In-chat command
/mode build
/mode plan
/mode explore
```

**Mode-specific behavior:**
- **BUILD**: Full file access, can edit anything
- **PLAN**: Edits restricted to `.victor/sandbox/`, more exploration
- **EXPLORE**: Read-only, maximum exploration (3x budget)

### What's the difference between TUI and CLI mode?

**TUI Mode (Terminal User Interface):**
```bash
victor  # or just 'vic'
```
- Rich terminal UI with syntax highlighting
- Interactive session management
- Tool execution status display
- Best for interactive development

**CLI Mode:**
```bash
victor chat "your message"
```
- Simple command-line interface
- Script-friendly
- Quick one-shot queries
- Pipe content: `cat error.log | victor "diagnose this"`

### How do I save and restore sessions?

**Save session:**
```bash
/save "My Session Title"
```

**List sessions:**
```bash
/sessions
victor sessions list
```

**Restore session:**
```bash
# Interactive restore
/resume

# Restore specific session
/resume 20250110_153045

# From CLI
victor chat --resume 20250110_153045
```

**Sessions are stored in:** `~/.victor/conversations/`

### How do I use workflows?

**Run a workflow:**
```bash
victor workflow run code-review
victor workflow run deep-research
```

**Validate workflow:**
```bash
victor workflow validate path/to/workflow.yaml
```

**List workflows:**
```bash
victor workflow list
```

**Example workflow:**
```yaml
workflows:
  code_review:
    nodes:
      - id: analyze
        type: agent
        role: reviewer
        goal: "Analyze this PR for issues"
        tool_budget: 30
        next: [test]

      - id: test
        type: compute
        handler: run_tests
        inputs:
          path: tests/
```

### Can I use Victor with existing tools (git, pytest, etc.)?

**Yes!** Victor has 33 built-in tool modules:

**Git:**
```bash
victor "Show git status and commit changes"
victor "Create a branch for feature X"
victor "Generate a commit message for these changes"
```

**Testing:**
```bash
victor "Run pytest and summarize failures"
victor "Write unit tests for auth.py"
```

**File Operations:**
```bash
victor "Find all TODO comments"
victor "Search for authentication code"
victor "Read lines 50-100 of main.py"
```

**Commands:**
```bash
victor "Run make test and show results"
victor "Execute docker-compose up"
```

---

## Features

### What is semantic code search?

Semantic search finds code by **meaning**, not just text:

```bash
# Semantic (finds patterns, concepts)
victor "Find error handling patterns"
victor "Show authentication flow"
victor "Classes implementing Observer pattern"

# Literal (finds exact text)
victor "Find all uses of BaseProvider"
victor "Search for TODO comments"
```

**How it works:**
- Code is embedded into vectors
- Your query is also embedded
- Similarity search finds semantically matching code
- Works even when variable names differ

**Enable/configure:**
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    tool_selection_strategy: hybrid  # keyword, semantic, or hybrid
    semantic_similarity_threshold: 0.15
```

### What are multi-agent teams?

Victor can coordinate **multiple specialized AI agents**:

**Team formations:**
- **Hierarchical**: Lead agent delegates to sub-agents
- **Flat**: Peer agents with shared memory
- **Pipeline**: Sequential processing
- **Consensus**: Agents vote on decisions
- **Debate**: Agents debate to reach agreement

**Example:**
```python
from victor.framework import Agent, AgentTeam

frontend = Agent(role="Frontend developer")
backend = Agent(role="Backend developer")
tester = Agent(role="QA engineer")

team = AgentTeam.hierarchical(
    lead="senior-developer",
    subagents=[frontend, backend, tester]
)

result = await team.run("Implement user registration feature")
```

**Use cases:**
- Complex feature implementation
- Code review with multiple perspectives
- Parallel development (frontend + backend)
- Testing + implementation teams

### What is MCP support?

**MCP (Model Context Protocol)** enables IDE integration:

**Run Victor as MCP server:**
```bash
victor mcp --stdio
```

**Configure in Claude Desktop:**
```json
{
  "mcpServers": {
    "victor": {
      "command": "victor",
      "args": ["mcp"]
    }
  }
}
```

**Connect to MCP servers:**
```yaml
# ~/.victor/mcp.yaml
servers:
  - name: filesystem
    command: npx
    args: ["-y", "@anthropic/mcp-server-filesystem"]

  - name: github
    command: npx
    args: ["-y", "@anthropic/mcp-server-github"]
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
```

### What are RAG capabilities?

**RAG (Retrieval-Augmented Generation)** for document QA:

**Ingest documents:**
```bash
victor rag ingest ./docs --recursive --pattern "*.md"
victor rag ingest https://example.com/docs
```

**Query knowledge base:**
```bash
victor rag search "authentication"
victor rag query "How do I add a provider?" --synthesize
```

**Use cases:**
- Documentation search
- Technical knowledge base
- Code documentation Q&A
- Project-specific knowledge

---

## Troubleshooting

### Why does Victor say "API key not found"?

**Cause**: Environment variable not set or incorrect.

**Solutions:**

1. **Check environment variable:**
   ```bash
   echo $ANTHROPIC_API_KEY
   ```

2. **Set in current shell:**
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **Add to shell profile:**
   ```bash
   echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Use keyring:**
   ```bash
   victor keys --set anthropic --keyring
   ```

### Why is Victor slow?

**Possible causes:**

1. **Using local model on CPU**:
   - Switch to GPU if available
   - Use smaller model (3B instead of 7B)
   - Or use cloud provider

2. **Large context window**:
   - Start new conversation: `victor chat --new`
   - Use model with larger context (Claude, Gemini)

3. **Network latency** (cloud providers):
   - Switch to faster provider (Groq, Cerebras)
   - Check your internet connection

4. **Disable caching**:
   ```yaml
   # ~/.victor/config.yaml
   cache:
     enabled: true
     ttl: 3600
   ```

### Why do I get "model not found"?

**Cause**: Incorrect model name or model not pulled.

**Solutions:**

1. **Check model name:**
   ```bash
   victor providers --provider anthropic
   ```

2. **Use correct model format:**
   ```bash
   # Correct
   victor chat --provider anthropic --model claude-sonnet-4-20250514

   # Wrong (too generic)
   victor chat --provider anthropic --model claude-3
   ```

3. **Pull local model:**
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

### How do I fix Ollama connection errors?

**Symptoms:**
```
ERROR: Ollama not responding
```

**Solutions:**

1. **Check if Ollama is running:**
   ```bash
   ollama list
   ```

2. **Restart Ollama:**
   ```bash
   ollama stop
   ollama serve
   ```

3. **Test connection:**
   ```bash
   curl http://localhost:11434/api/version
   ```

4. **Check host:**
   ```bash
   export OLLAMA_HOST=127.0.0.1:11434
   ```

### How do I fix rate limiting errors?

**Symptoms:**
```
ERROR: Rate limit exceeded
```

**Solutions:**

1. **Wait and retry:**
   ```bash
   sleep 60
   victor chat "Continue"
   ```

2. **Switch providers mid-conversation:**
   ```bash
   /provider openai  # or /provider ollama
   ```

3. **Use local provider** (no rate limits):
   ```bash
   /provider ollama
   ```

4. **Check your rate limit:**
   - Anthropic: https://console.anthropic.com/settings/limits
   - OpenAI: https://platform.openai.com/usage

### How do I reduce memory usage?

**Symptoms:**
```
Victor consuming >1GB memory
```

**Solutions:**

1. **Use smaller model:**
   ```bash
   # 3B model: ~4GB RAM
   # 7B model: ~8GB RAM
   victor chat --provider ollama --model llama3.2:3b
   ```

2. **Reduce context size:**
   ```bash
   victor chat --max-tokens 1024 "Hello"
   ```

3. **Clear cache:**
   ```bash
   rm -rf ~/.victor/cache/
   ```

4. **Restart Victor:**
   ```bash
   # Stop and restart
   victor quit
   victor chat
   ```

### How do I debug tool execution failures?

**Enable debug logging:**
```bash
victor --debug chat "Read auth.py"
```

**View logs:**
```bash
victor logs --tail 50
victor logs --follow
victor logs | grep ERROR
```

**Common issues:**

1. **File not found**:
   - Use absolute path
   - Check current directory: `pwd`

2. **Permission denied**:
   - Check file permissions: `ls -la file.py`
   - Fix: `chmod +r file.py`

3. **Tool not available**:
   - List tools: `victor tools`
   - Check if tool requires external dependencies

---

## Advanced

### How do I create custom tools?

**1. Create tool file:**
```python
# my_tools/custom_tool.py
from victor.tools.base import BaseTool, ToolResult, CostTier

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input data"}
        },
        "required": ["input"]
    }
    cost_tier = CostTier.LOW

    async def execute(self, _exec_ctx, **kwargs):
        result = process(kwargs["input"])
        return ToolResult(success=True, output=result)
```

**2. Register in pyproject.toml:**
```toml
[project.entry-points."victor.tools"]
my_tool = "my_tools.custom_tool:MyCustomTool"
```

**3. Install package:**
```bash
pip install -e .
```

### How do I create custom workflows?

**1. Create workflow YAML:**
```yaml
# workflows/my_workflow.yaml
workflows:
  my_workflow:
    nodes:
      - id: analyze
        type: agent
        role: researcher
        goal: "Analyze the requirements"
        next: [implement]

      - id: implement
        type: agent
        role: developer
        goal: "Implement the feature"
        next: [test]

      - id: test
        type: compute
        handler: run_tests
        inputs:
          path: tests/
```

**2. Validate workflow:**
```bash
victor workflow validate workflows/my_workflow.yaml
```

**3. Run workflow:**
```bash
victor workflow run my_workflow
```

### How do I integrate a new provider?

**1. Create provider file:**
```python
# victor/providers/my_provider.py
from victor.providers.base import BaseProvider

class MyProvider(BaseProvider):
    name = "my_provider"

    async def chat(self, messages, **kwargs):
        # Implement chat logic
        pass

    async def stream_chat(self, messages, **kwargs):
        # Implement streaming
        pass

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True
```

**2. Register provider:**
```python
# victor/providers/registry.py
register_provider(MyProvider)
```

**3. Add tests:**
```python
# tests/unit/providers/test_my_provider.py
async def test_my_provider():
    provider = MyProvider(api_key="test")
    assert provider.name == "my_provider"
```

### How do I extend Victor with a vertical?

**Verticals** are domain-specific plugins:

**1. Create vertical package:**
```python
# victor_security/__init__.py
from victor.core.verticals.base import VerticalBase

class SecurityAssistant(VerticalBase):
    name = "security"
    description = "Security analysis tools"

    def get_tools(self):
        return [scan_tool, audit_tool]

    def get_system_prompt(self):
        return "You are a security expert..."
```

**2. Register in pyproject.toml:**
```toml
[project.entry-points."victor.verticals"]
security = "victor_security:SecurityAssistant"
```

**3. Install and use:**
```bash
pip install victor-security
victor --vertical security "Scan for vulnerabilities"
```

### How do I configure Victor for CI/CD?

**GitHub Actions:**
```yaml
name: Code Review
on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Victor
        run: pipx install victor-ai
      - name: Run Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: victor chat --mode plan "Review this PR"
```

**GitLab CI:**
```yaml
code-review:
  script:
    - pipx install victor-ai
    - victor chat "Review this MR"
  variables:
    ANTHROPIC_API_KEY: $ANTHROPIC_API_KEY
```

### How do I use the Python API?

**Simple use case:**
```python
from victor.framework import Agent

agent = await Agent.create(provider="anthropic")
result = await agent.run("Explain this codebase")
print(result.content)
```

**Streaming:**
```python
from victor.framework import Agent, EventType

agent = await Agent.create(provider="openai")

async for event in agent.stream("Refactor this function"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")
    elif event.type == EventType.TOOL_CALL:
        print(f"\nUsing tool: {event.tool_name}")
```

**Multi-turn conversation:**
```python
session = agent.chat()
await session.send("What files are in this project?")
await session.send("Now explain the main entry point")
```

**StateGraph workflows:**
```python
from victor.framework import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    query: str
    result: str

graph = StateGraph(MyState)
graph.add_node("research", research_fn)
graph.add_node("synthesize", synthesize_fn)
graph.add_edge("research", "synthesize")
graph.add_edge("synthesize", END)

compiled = graph.compile()
result = await compiled.invoke({"query": "AI trends"})
```

---

## Getting Help

### Documentation

- **Full Docs**: [docs/](docs/)
- **Getting Started**: [docs/getting-started/](docs/getting-started/)
- **User Guide**: [docs/user-guide/](docs/user-guide/)
- **Reference**: [docs/reference/](docs/reference/)

### Community

- **GitHub Issues**: [Report bugs](https://github.com/vjsingh1984/victor/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/vjsingh1984/victor/discussions)
- **Discord**: [Join Discord](https://discord.gg/)

### Diagnostic Commands

```bash
# Full diagnostic report
victor doctor

# Check version
victor --version

# List providers
victor providers

# View configuration
victor config show

# View logs
victor logs --tail 50
```

### When to Open an Issue

**Open an issue for:**
- ✅ Bug: Victor crashes or behaves incorrectly
- ✅ Feature request: New functionality
- ✅ Documentation: Unclear or missing docs
- ✅ Performance: Unexpected slowness

**Use Discussions for:**
- ❌ Usage questions
- ❌ "How do I..." questions
- ❌ General discussion

---

## Quick Reference

### Essential Commands

```bash
# Installation
pipx install victor-ai          # Install Victor
victor init                      # Initialize config

# Running Victor
victor                           # TUI mode
victor chat                      # CLI mode
victor chat --provider X         # Use specific provider

# In-Chat Commands
/help                            # Show help
/provider <name>                 # Switch provider
/mode <mode>                     # Switch mode (build/plan/explore)
/save "title"                    # Save session
/resume                          # Restore session
/quit                            # Exit

# Management
victor keys --set anthropic      # Set API key
victor config show               # Show config
victor tools                     # List tools
victor workflow run <name>       # Run workflow
```

### Environment Variables

```bash
# API Keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
export GOOGLE_API_KEY=...

# Local Providers
export OLLAMA_HOST=127.0.0.1:11434
export VICTOR_VLLM_HOST=127.0.0.1:8000

# Victor Settings
export VICTOR_LOG_LEVEL=INFO
export VICTOR_AIRGAPPED=true
```

### Configuration Files

```bash
~/.victor/profiles.yaml   # Provider/model profiles
~/.victor/config.yaml      # Global settings
.vvictor.md                # Project context
CLAUDE.md                  # AI instructions
```

---

**Still have questions?** Check the [full documentation](docs/) or [open a discussion](https://github.com/vjsingh1984/victor/discussions).
