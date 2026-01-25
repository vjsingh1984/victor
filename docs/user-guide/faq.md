# Victor AI FAQ

**Frequently asked questions about Victor AI**

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation and Setup](#installation-and-setup)
3. [Providers and Models](#providers-and-models)
4. [Team Coordination](#team-coordination)
5. [Tools and Capabilities](#tools-and-capabilities)
6. [Performance and Scaling](#performance-and-scaling)
7. [Configuration and Customization](#configuration-and-customization)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [Community and Support](#community-and-support)

---

## General Questions

### What is Victor AI?

Victor AI is an open-source coding assistant that supports multiple LLM providers and tools across domain verticals (Coding, DevOps, RAG, Data Analysis, Research). It provides both local and cloud LLM support through a unified CLI and Python API.

**Key features**:
- Provider agnostic (multiple providers)
- Air-gapped mode support
- Multi-agent team coordination
- Semantic codebase search
- YAML workflow DSL

### How is Victor different from other coding assistants?

| Feature | Victor | Others |
|---------|--------|--------|
| **Providers** | Multiple providers | Varies |
| **Local Support** | Local/offline options | Varies |
| **Multi-Agent** | Team workflows | Often single agent |
| **Extensibility** | Vertical + plugin model | Often closed |
| **Cost** | Open-source + BYO provider | Often subscription |
| **Privacy** | Local-only possible with supported providers | Often cloud-only |

### Is Victor free?

Victor is open-source and free to use. Costs may come from model providers or local compute. You can:
- Use local models if you have the hardware
- Bring your own API keys for cloud providers
- Use the source code without a subscription

### What can Victor do?

Victor can help with:
- **Code review**: Identify issues, suggest improvements
- **Test generation**: Create unit tests, integration tests
- **Refactoring**: Restructure code safely
- **Documentation**: Generate docs, add docstrings
- **Debugging**: Find and fix bugs
- **Security**: Audit code for vulnerabilities
- **Performance**: Identify bottlenecks
- **And much more**: See the tool catalog for the current list

### Is Victor suitable for production use?

It depends on your requirements. Review the production runbooks, test coverage, and known issues, and validate in staging before rollout.
- [Production runbooks](../production/README.md)
- [Testing guide](../testing/README.md)
- [Known issues](../known_issues/README.md)

---

## Installation and Setup

### How do I install Victor?

```bash
# Recommended: Using pipx
pipx install victor-ai

# Or using pip
pip install victor-ai

# Verify installation
victor --version
```

### What are the system requirements?

**Minimum**:
- Python 3.10 or higher
- 4GB RAM
- 1GB disk space

**Recommended for local models**:
- Python 3.10+
- 16GB RAM
- 10GB disk space
- macOS/Linux/Windows

**For cloud models**:
- Python 3.10+
- 4GB RAM
- 1GB disk space
- Internet connection

### Can I use Victor offline?

Yes! Victor supports air-gapped mode:
```bash
# Use local provider (Ollama)
victor chat --provider ollama "Review this code"

# Enable air-gapped mode explicitly
export VICTOR_AIRGAPPED=true
victor chat "Review this code"
```

### How do I update Victor?

```bash
# Update to latest version
pip install --upgrade victor-ai

# Or using pipx
pipx upgrade victor-ai

# Check version after update
victor --version
```

### How do I uninstall Victor?

```bash
# Uninstall Victor
pip uninstall victor-ai

# Remove configuration
rm -rf ~/.victor

# Or using pipx
pipx uninstall victor-ai
```

---

## Providers and Models

### Which providers does Victor support?

Victor supports 21 providers:

**Cloud Providers**: Anthropic, OpenAI, Google, Azure, AWS Bedrock, xAI, DeepSeek, Mistral, Cohere, Groq, Together AI, Fireworks AI, OpenRouter, Replicate, Hugging Face, Moonshot, Cerebras

**Local Providers**: Ollama, LM Studio, vLLM, llama.cpp

### Which provider should I use?

| Use Case | Recommended Provider |
|----------|---------------------|
| **Best capability** | Anthropic (Claude Sonnet 4) |
| **Fastest cloud** | Groq (300+ tok/s) |
| **Free & private** | Ollama (local) |
| **Best value** | DeepSeek (low cost) |
| **Large context** | Google (2M tokens) |
| **Production** | Azure OpenAI |

### How do I set up a provider?

```bash
# Step 1: Set API key (cloud providers)
export ANTHROPIC_API_KEY=sk-ant-...

# Step 2: Or add to config file
cat > ~/.victor/config.yaml << EOF
provider:
  name: anthropic
  model: claude-sonnet-4-5
  api_key: sk-ant-...
EOF

# Step 3: Test provider
victor provider test
```

### How do I use local models?

```bash
# Step 1: Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Step 2: Pull a model
ollama pull qwen2.5-coder:7b

# Step 3: Use with Victor
victor chat --provider ollama "Review this code"
```

### Can I switch providers mid-conversation?

Yes! Victor supports provider switching:
```bash
# Start with one provider
victor chat --provider anthropic "Review auth.py"

# Switch in CLI mode
/provider openai --model gpt-4o
Continue reviewing the code

# Or use Python API
await orchestrator.provider.switch_provider("openai", "gpt-4o")
```

### Which models support tool calling?

Most modern models support tool calling:
- **Anthropic**: Claude Sonnet 4, Claude 3.5 Sonnet, Opus
- **OpenAI**: GPT-4o, GPT-4, o1-preview
- **Google**: Gemini 2.0 Flash, Gemini 1.5 Pro
- **Local**: Qwen 2.5 Coder, Llama 3.1

Check model capabilities:
```bash
victor provider capabilities
```

---

## Team Coordination

### What are team formations?

Team formations define how multiple AI agents work together:

1. **SEQUENTIAL**: Agents work one after another
2. **PARALLEL**: Agents work simultaneously
3. **HIERARCHICAL**: Manager delegates to workers
4. **PIPELINE**: Output flows through agents
5. **CONSENSUS**: Agents discuss until agreement

### How do I create a team?

```python
from victor.teams import create_coordinator, TeamFormation
from victor.agent.subagents.base import SubAgentRole

coordinator = create_coordinator(orchestrator)

# Add members
coordinator.add_member(
    role=SubAgentRole.RESEARCHER,
    name="Researcher",
    goal="Find security issues"
)

coordinator.add_member(
    role=SubAgentRole.EXECUTOR,
    name="Implementer",
    goal="Implement fixes"
)

# Set formation
coordinator.set_formation(TeamFormation.PIPELINE)

# Execute
result = await coordinator.execute_task(
    "Review and fix auth.py",
    {}
)
```

### When should I use teams?

Use teams for:
- **Complex reviews**: Multiple perspectives
- **Multi-stage tasks**: Research -> Review -> Execute
- **Cross-domain work**: Security + Performance + Testing
- **Quality assurance**: Consensus on critical decisions

**Single agent is better for**:
- Simple tasks
- Quick questions
- One-off changes

### What are rich personas?

Rich personas add depth to team members:
```python
TeamMember(
    id="security_expert",
    role=SubAgentRole.RESEARCHER,
    name="Security Expert",
    goal="Ensure security",
    backstory="15 years security experience...",
    expertise=["security", "oauth", "jwt"],
    personality="methodical and thorough",
)
```

### How does team memory work?

Team members can remember across tasks:
```python
from victor.teams.types import MemoryConfig

TeamMember(
    memory=True,
    memory_config=MemoryConfig(
        enabled=True,
        persist_across_sessions=True,
        memory_types={"entity", "semantic"},
    )
)

# Use memory
await member.remember("auth_patterns", patterns)
memories = await member.recall("authentication")
```

---

## Tools and Capabilities

### How many tools does Victor have?

Victor has 55+ tools across 5 verticals:
- **Coding**: File operations, AST analysis, review, testing
- **DevOps**: Docker, Kubernetes, CI/CD, Terraform
- **RAG**: Document ingestion, vector search, retrieval
- **Data Analysis**: Pandas, visualization, statistics
- **Research**: Web search, citations, synthesis

### How do I list available tools?

```bash
# List all tools
victor tools list

# Show tool details
victor tools show read_file

# Search tools
victor tools search git

# Test tool
victor tools test execute_command
```

### What is tool budget?

Tool budget limits tool calls per session:
```bash
# Set budget
victor chat --tool-budget 100 "Review code"

# Check usage
victor tools usage

# Increase if needed
victor chat --tool-budget 200 "Comprehensive review"
```

### How does tool selection work?

Victor uses intelligent tool selection:
- **Keyword**: Matches task keywords to tools
- **Semantic**: Uses embeddings to find relevant tools
- **Hybrid**: Combines both (default)

```bash
# Change strategy
export VICTOR_TOOL_SELECTION_STRATEGY=keyword
```

### Can I add custom tools?

Yes! Create tools by inheriting from BaseTool:
```python
from victor.tools.base import BaseTool

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Does something custom"

    async def execute(self, **kwargs):
        # Your logic here
        return result

# Register tool
from victor.tools.registry import register_tool
register_tool(MyCustomTool())
```

---

## Performance and Scaling

### Is Victor fast?

Yes! Victor is optimized for performance:
- 3-5% overhead from coordinator architecture
- Parallel tool execution
- Intelligent caching
- Streaming responses

### How can I improve performance?

```bash
# Use faster provider
victor chat --provider groq "Test"  # 300+ tok/s

# Use local provider
victor chat --provider ollama "Test"

# Reduce context
victor chat --max-context-size 50000 "Test"

# Disable observability
export VICTOR_OBSERVABILITY_ENABLED=false

# Use streaming
victor chat --stream "Test"
```

### Can Victor handle large codebases?

Yes! Victor is designed for large codebases:
- Semantic search for relevant code
- Context compaction
- Incremental analysis
- Parallel processing

```bash
# Analyze large project
victor chat "Review entire codebase" --max-context-size 100000
```

### What are the resource limits?

| Resource | Default | Maximum |
|----------|---------|---------|
| **Context size** | 100K tokens | 2M tokens (Gemini) |
| **Tool budget** | 100 calls | Unlimited |
| **Team size** | 3 members | 10 members |
| **Session time** | 1 hour | Unlimited |

---

## Configuration and Customization

### Where is the config file?

Config file location:
- **Linux/macOS**: `~/.victor/config.yaml`
- **Windows**: `%USERPROFILE%\.victor\config.yaml`

### How do I configure Victor?

```yaml
# ~/.victor/config.yaml
provider:
  name: anthropic
  model: claude-sonnet-4-5
  api_key: ${ANTHROPIC_API_KEY}

mode:
  default: build
  exploration_factor: 1.0

tools:
  budget: 100
  selection_strategy: hybrid

teams:
  formation: pipeline
  max_iterations: 50
```

### Can I use environment variables?

Yes! Victor supports environment variables:
```bash
# Provider
export VICTOR_PROVIDER=anthropic
export VICTOR_MODEL=claude-sonnet-4-5
export ANTHROPIC_API_KEY=sk-ant-...

# Mode
export VICTOR_MODE=build

# Tools
export VICTOR_TOOL_BUDGET=100

# Debug
export VICTOR_LOG_LEVEL=DEBUG
export VICTOR_DEBUG=true
```

### How do I create custom workflows?

Create YAML workflow:
```yaml
# ~/.victor/workflows/my_workflow.yaml
workflows:
  security_review:
    nodes:
      - id: analyze
        type: agent
        role: researcher
        goal: "Analyze security"
        tool_budget: 20
        next: [report]

      - id: report
        type: agent
        role: reviewer
        goal: "Generate report"
```

Use workflow:
```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

compiler = UnifiedWorkflowCompiler()
workflow = compiler.compile("my_workflow")
result = await workflow.invoke({"query": "Review auth.py"})
```

---

## Troubleshooting

### Victor says "command not found"

```bash
# Solution: Install Victor
pipx install victor-ai

# Or add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

### "API key not found" error

```bash
# Solution: Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Or add to config
cat > ~/.victor/config.yaml << EOF
provider:
  api_key: sk-ant-...
EOF
```

### Victor is slow

```bash
# Solution: Use faster provider
victor chat --provider ollama "Test"

# Or reduce context
victor chat --max-context-size 50000 "Test"
```

### Team execution hangs

```python
# Solution: Add timeout
result = await coordinator.execute_task(
    "Review code",
    {},
    timeout=300  # 5 minutes
)
```

### More troubleshooting?

See the [Troubleshooting Guide](troubleshooting.md)

---

## Advanced Usage

### What is the Python API?

Victor provides a Python API for programmatic use:
```python
from victor import Victor

vic = Victor(provider="anthropic")
response = vic.chat("Review this code")
print(response)
```

### Can I extend Victor with plugins?

Yes! Create external verticals:
```python
# my_vertical/__init__.py
from victor.core.verticals import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"

    def get_tools(self):
        return [MyTool()]

    def get_system_prompt(self):
        return "You are a specialized assistant..."
```

### How does observability work?

Victor emits events for monitoring:
```python
from victor.observability.event_bus import EventBus

bus = EventBus.get_instance()

@bus.subscribe("tool.execution")
async def on_tool_exec(event):
    print(f"Tool executed: {event.data}")

@bus.subscribe("team.task_complete")
async def on_task_complete(event):
    print(f"Task complete: {event.data}")
```

### What is reinforcement learning?

Victor can learn from outcomes:
```python
from victor.framework.rl import RLManager

rl = RLManager()

# Record outcome
await rl.record_outcome(
    task="security_review",
    outcome="success",
    metrics={"quality": 0.9}
)

# Get recommendations
recommendation = await rl.get_recommendation(context)
```

---

## Community and Support

### Where can I get help?

- **Documentation**: https://github.com/vjsingh1984/victor/docs
- **Issues**: https://github.com/vjsingh1984/victor/issues
- **Discussions**: https://github.com/vjsingh1984/victor/discussions
- **Email**: singhvjd@gmail.com

### How can I contribute?

We welcome contributions!
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See: [Development Guide](../contributing/index.md)

### Is Victor production-ready?

It depends on your requirements and deployment context. Review the production docs, validate in staging, and assess against your own SLOs.
- [Production runbooks](../production/README.md)
- [Testing guide](../testing/README.md)
- [Known issues](../known_issues/README.md)

### What's the roadmap?

See: [ROADMAP.md](../roadmap/future_roadmap.md)

Key areas:
- Enhanced team coordination
- More providers
- Better performance
- New tools
- Improved UX

### How do I report a bug?

Create an issue with:
1. Victor version (`victor --version`)
2. Python version (`python --version`)
3. Error message
4. Steps to reproduce
5. Expected vs actual behavior
6. Diagnostics (`victor doctor`)

### Can I use Victor commercially?

Yes! Victor is Apache 2.0 licensed:
- Free commercial use
- No restrictions
- No attribution required
- See: [LICENSE](https://github.com/vjsingh1984/victor/blob/main/LICENSE)

---

## Quick Tips

### Best Practices

1. **Start with PLAN mode**: Explore before making changes
2. **Use version control**: Commit before major changes
3. **Review before applying**: Victor explains changes first
4. **Set appropriate budgets**: Don't waste tokens
5. **Use teams for complex tasks**: Multi-agent coordination
6. **Enable memory for long sessions**: Persistent context
7. **Monitor tool usage**: Stay within budget
8. **Choose the right formation**: Match task to formation

### Common Patterns

```python
# Code review
victor chat "Review auth.py for security issues"

# Add tests
victor chat "Generate unit tests for user service"

# Refactor
victor chat --mode plan "Plan refactoring of data pipeline"
victor chat --mode build "Execute refactoring"

# Team review
coordinator.set_formation(TeamFormation.PIPELINE)
await coordinator.execute_task("Review and fix auth.py", {})
```

### Keyboard Shortcuts (TUI Mode)

| Key | Action |
|-----|--------|
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit Victor |
| `Ctrl+L` | Clear screen |
| `↑/↓` | Navigate history |
| `Tab` | Auto-complete |

---

**Still have questions?** Check the [Troubleshooting Guide](troubleshooting.md) or [create an issue](https://github.com/vjsingh1984/victor/issues/new)
