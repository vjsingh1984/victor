# Common Workflows - Victor AI Examples

**Version**: 0.5.0
**Last Updated**: January 24, 2026

---

## Quick Reference for Common Tasks

### 1. First-Time Setup (5 minutes)

```bash
# Install Victor
pipx install victor-ai

# Local model (free, offline)
ollama pull qwen2.5-coder:7b
victor chat "Hello, Victor!"

# Cloud model (full capability)
export ANTHROPIC_API_KEY=sk-ant-...
victor chat --provider anthropic "Hello, Victor!"
```

**Documentation:** [Installation Guide](../getting-started/installation.md)

---

### 2. Daily Development Workflow

```bash
# Morning - Start coding session
victor chat

# Throughout the day - Quick commands
victor chat "Refactor this function for clarity"
victor chat "Write tests for auth.py"
victor chat --mode explore "Analyze this codebase"

# Switch providers as needed (context preserved!)
/provider openai --model gpt-4o
/provider ollama --model qwen2.5-coder:7b

# End of day - Save conversation
# (automatically saved in ~/.victor/conversations/)
```

**Documentation:** [User Guide Index](../user-guide/index.md)

---

### 3. Code Refactoring Workflow

```bash
# 1. Analyze current code
victor chat --mode explore "Review src/auth.py for bugs and improvements"

# 2. Plan the refactoring
victor chat --mode plan "Design a refactoring strategy for auth.py"

# 3. Implement the refactoring
victor chat --mode build "Refactor auth.py based on the strategy"

# 4. Review the changes
victor chat --provider anthropic "Review the refactored code"
```

**Modes:**
- `build` (default): Full edits
- `plan`: 2.5x exploration, sandbox mode
- `explore`: 3.0x exploration, no edits

**Documentation:** [Modes Explanation](../reference/quick-reference.md#agent-modes)

---

### 4. Multi-Provider Cost Optimization

```python
from victor.framework import Agent

# FREE - Brainstorming with local model
brainstormer = await Agent.create(provider="ollama")
ideas = await brainstormer.run("Generate 5 feature ideas for a blog API")

# CHEAP - Implement with fast cloud model
implementer = await Agent.create(
    provider="groq",
    model="llama3.1-70b"  # Fast!
)
code = await implementer.run(f"Implement: {ideas.content[0]}")

# QUALITY - Review with best model
reviewer = await Agent.create(
    provider="anthropic",
    model="claude-sonnet-4-5"
)
review = await reviewer.run(f"Review:\n{code.content}")
```

**Cost savings:**
- Free brainstorming (Ollama)
- Fast iteration (Groq: 300+ tok/s)
- Quality assurance (Anthropic: best results)

**Documentation:** [Provider Comparison](../reference/providers/comparison.md)

---

### 5. Testing Workflow

```bash
# 1. Generate tests
victor chat --mode build "Write comprehensive tests for src/auth.py"

# 2. Run tests
pytest tests/unit/auth/test_auth.py -v

# 3. Fix failures
victor chat "Fix the failing tests in tests/unit/auth/test_auth.py"

# 4. Re-run
pytest tests/unit/auth/test_auth.py -v

# 5. Coverage report
pytest tests/unit/auth/ --cov=src/auth --cov-report=html
```

**Documentation:** [Testing Guide](../contributing/testing.md)

---

### 6. Provider Switching Mid-Conversation

```bash
# Start with Claude for planning
victor chat --provider anthropic "Design a REST API"

# Switch to GPT-4 for implementation (context preserved!)
/provider openai --model gpt-4o

# Switch to local model for privacy (context preserved!)
/provider ollama --model qwen2.5-coder:7b
```

**Key benefit:** Conversation context is preserved when switching providers!

**Documentation:** [Provider Guide](../user-guide/providers.md)

---

### 7. Workflow Automation

```yaml
# workflows/code-review.yaml
name: code_review
description: Automated code review workflow

nodes:
  - id: analyze
    type: agent
    provider: anthropic
    tools: [read, grep]
    action: Analyze code for issues

  - id: tests
    type: agent
    provider: anthropic
    tools: [run_tests]
    depends_on: [analyze]

  - id: review
    type: agent
    provider: anthropic
    depends_on: [tests]
```

```bash
# Run workflow
victor workflow run code-review.yaml --target src/auth.py
```

**Documentation:** [Workflow Reference](../reference/internals/workflows-api.md)

---

### 8. Error Resolution Workflow

```bash
# 1. Check error
victor chat "I'm getting a connection error"

# 2. Try troubleshooting
victor chat " diagnose the Ollama connection issue"

# 3. Check documentation
# docs/getting-started/troubleshooting.md

# 4. Use diagnostics
victor doctor  # (if available)
```

**Documentation:** [Troubleshooting](../getting-started/troubleshooting.md)

---

### 9. Architecture Exploration

```bash
# 1. View diagrams
ls docs/diagrams/

# 2. Open system architecture diagram
# docs/diagrams/system-architecture.mmd
# (Renders automatically on GitHub)

# 3. Read architecture overview
victor chat "Explain the two-layer coordinator architecture"

# 4. Deep dive into specific component
victor chat "How does the ToolCoordinator work?"
```

**Documentation:**
- [Architecture Overview](../architecture/overview.md)
- [30 Mermaid Diagrams](../diagrams/index.md)
- [Component Reference](../architecture/COMPONENT_REFERENCE.md)

---

### 10. Contributing Workflow

```bash
# 1. Set up development environment
pip install -e ".[dev]"
make test

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
victor chat "Help me implement feature X"

# 4. Test changes
make test
make lint

# 5. Commit
git commit -m "feat: add feature X"

# 6. Push and create PR
git push origin feature/my-feature
```

**Documentation:** [Contributing Guide](../contributing/index.md)

---

## Advanced Workflows

### Multi-Agent Team Workflow

```python
from victor.framework import Agent

# Create specialist agents
architect = await Agent.create(
    provider="anthropic",
    system_prompt="You are a software architect."
)

developer = await Agent.create(
    provider="openai",
    system_prompt="You are a developer."
)

reviewer = await Agent.create(
    provider="anthropic",
    system_prompt="You are a code reviewer."
)

# Orchestrate team
task = "Design and implement a REST API"
architecture = await architect.run(f"Design: {task}")
implementation = await developer.run(f"Implement: {architecture}")
review = await reviewer.run(f"Review: {implementation}")
```

### Custom Workflow Creation

```yaml
# workflows/automated-review.yaml
name: automated_review

nodes:
  - id: checkout
    type: tools
    tool: git_checkout

  - id: analyze
    type: agent
    provider: anthropic
    tools: [read, grep]

  - id: fix
    type: agent
    provider: openai
    tools: [write, edit]

  - id: verify
    type: agent
    provider: anthropic
    tools: [run_tests]
```

### Event-Driven Integration

```python
from victor.core.events import EventBus, AgentExecutionEvent

# Subscribe to events
async def on_event(event: AgentExecutionEvent):
    if event.type == EventType.CONTENT:
        print(event.content, end="")

# Stream with events
async for event in agent.stream("Analyze code"):
    if event.type == EventType.TOOL_CALL:
        print(f"Tool called: {event.tool_name}")
```

---

## Tips and Best Practices

### 1. Always Use Appropriate Modes

| Task | Mode | Why |
|-----|------|-----|
| Active development | `build` | Full editing capability |
| Research & analysis | `explore` | 3.0x exploration, no edits |
| Planning & strategy | `plan` | 2.5x exploration, sandbox |
| Quick questions | `build` | Default, fastest |

### 2. Optimize for Cost

- **Brainstorming**: Ollama (free)
- **Fast iteration**: Groq (300+ tok/s)
- **Implementation**: OpenAI (gpt-4o-mini)
- **Quality review**: Anthropic (claude-sonnet-4)

### 3. Leverage Provider Switching

Switch providers mid-conversation without losing context to optimize for:
- Cost (switch to free provider)
- Speed (switch to faster provider)
- Quality (switch to better model)
- Privacy (switch to local provider)

### 4. Use Workflows for Repetitive Tasks

Create YAML workflows for:
- Code review
- Testing
- Deployment
- Documentation generation

### 5. Archive Important Conversations

Victor automatically saves conversations to `~/.victor/conversations/`. Reference them for:
- Important decisions
- Complex troubleshooting
- Multi-step solutions

---

## Common Patterns

### Pattern 1: Progressive Refinement

```bash
# 1. High-level exploration (explore mode)
victor chat --mode explore "What does this codebase do?"

# 2. Focused analysis (plan mode)
victor chat --mode plan "How should I refactor this module?"

# 3. Implementation (build mode)
victor chat --mode build "Refactor the module"
```

### Pattern 2: Multi-Provider Review

```bash
# 1. Quick review with fast model
victor chat --provider groq "Quick review of this PR"

# 2. Deep review with best model
victor chat --provider anthropic "Deep review of this PR"
```

### Pattern 3: Tool Selection Strategy

```bash
# All tools available
victor chat --tools all "Comprehensive analysis"

# Specific tools for task
victor chat --tools read,write "Just modify this file"

# Tool budget for cost control
victor chat --tool-budget 10 "Analyze with max 10 tool calls"
```

---

## Troubleshooting Common Issues

### Issue: "command not found: victor"

```bash
# Fix: Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or use python module
python -m victor chat "Hello"
```

### Issue: "Ollama connection error"

```bash
# Fix: Start Ollama
ollama serve

# Verify connection
ollama ps
```

### Issue: "API key not found"

```bash
# Fix: Set environment variable
export ANTHROPIC_API_KEY=sk-ant-...

# Or use keyring
victor keys --set anthropic --keyring
```

**Documentation:** [Troubleshooting Guide](../getting-started/troubleshooting.md)

---

## Learning Resources

### For Beginners
1. [Installation](../getting-started/installation.md) - 5 min
2. [First Run](../getting-started/first-run.md) - 3 min
3. [Local Models](../getting-started/local-models.md) - 10 min
4. [FAQ](../FAQ.md) - 5 min

### For Intermediate Users
1. [User Guide](../user-guide/index.md) - 30 min
2. [Provider Guide](../user-guide/providers.md) - All 21 providers
3. [Tool Catalog](../reference/tools/catalog.md) - 55+ tools
4. [Workflows](../user-guide/workflows.md) - Automation

### For Advanced Users
1. [Architecture Overview](../architecture/overview.md) - System design
2. [30 Mermaid Diagrams](../diagrams/index.md) - Visual documentation
3. [API Reference](../reference/api.md) - Python API
4. [Contributing Guide](../contributing/index.md) - Development

---

## Conclusion

Victor AI is designed for flexibility and power. Use these workflows to:
- Optimize costs (multi-provider)
- Save time (automation)
- Improve quality (provider switching)
- Maintain privacy (local models)

**Experiment and find what works best for your workflow!**

---

<div align="center">

**[← Back to Documentation](../index.md)**

**Common Workflows**

*Practical examples for everyday use*

</div>

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
