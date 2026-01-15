# Victor AI Quick Start Guide

**Get up and running with Victor AI in 5 minutes**

---

## What's New in Victor 0.5

Victor 0.5 represents a major architectural refactoring that delivers:

- **93% reduction in core complexity** - Coordinator-based architecture for better maintainability
- **5 team formations** - Coordinate specialized AI agents for complex tasks
- **85% test coverage** - More reliable and stable
- **100% backward compatibility** - No breaking changes, upgrade with confidence
- **3-5% performance overhead** - Well below our 10% goal

### Key Benefits

| Before | After |
|--------|-------|
| Monolithic orchestrator | Coordinator-based architecture |
| Complex debugging | Clear separation of concerns |
| Hard to test | 85% test coverage |
| Difficult to extend | Easy to add features |

---

## 5-Minute Quick Start

### Step 1: Install Victor (30 seconds)

```bash
# Using pipx (recommended)
pipx install victor-ai

# Or using pip
pip install victor-ai
```

### Step 2: Choose Your Provider (1 minute)

**Option A: Local (Free, Private)**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a code model
ollama pull qwen2.5-coder:7b

# Run Victor
victor chat
```

**Option B: Cloud (Maximum Capability)**

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run Victor with cloud model
victor chat --provider anthropic --model claude-sonnet-4-5
```

### Step 3: First Task (2 minutes)

Try one of these example tasks:

```bash
# Code review
victor chat "Review the authentication code in src/auth.py"

# Add tests
victor chat "Add unit tests for the user service"

# Fix a bug
victor chat "Fix the memory leak in the data processor"

# Documentation
victor chat "Generate API docs for the REST endpoints"
```

### Step 4: Try Multi-Agent Coordination (1.5 minutes)

```python
# Create a Python script: team_task.py
from victor import Victor

# Initialize Victor
vic = Victor()

# Define a team task
task = """
Review the authentication system and:
1. Identify security vulnerabilities
2. Suggest fixes
3. Generate test cases
"""

# Run with team coordination
result = vic.chat(task, mode="team")
print(result)
```

```bash
python team_task.py
```

---

## Basic Usage Examples

### Interactive Chat (TUI Mode)

```bash
# Start interactive terminal UI
victor

# Start with specific provider
victor chat --provider openai --model gpt-4o

# Start with local model
victor chat --provider ollama --model qwen2.5-coder:7b
```

### Command Line (One-Shot)

```bash
# Single command execution
victor chat "Add error handling to the API client"

# With file context
victor chat "Refactor this file" src/api/client.py

# With specific mode
victor chat "Plan the refactoring" --mode plan

# Generate and apply changes
victor chat "Add type hints to all functions" --mode build
```

### Python API

```python
from victor import Victor

# Simple chat
vic = Victor(provider="anthropic")
response = vic.chat("Explain this code")
print(response)

# With file context
with open("src/auth.py") as f:
    code = f.read()

response = vic.chat(f"Review this code:\n\n{code}")
print(response)

# Multi-agent team
from victor.teams import create_coordinator, TeamFormation

coordinator = create_coordinator(vic.orchestrator)
coordinator.set_formation(TeamFormation.PIPELINE)

# Add team members (researcher, reviewer, executor)
result = await coordinator.execute_task(
    "Review and fix authentication issues",
    {"files": ["src/auth.py"]}
)
```

---

## Common Tasks

### Code Review

```bash
# Review entire codebase
victor chat "Review the entire codebase for security issues"

# Review specific file
victor chat "Review auth.py for vulnerabilities" src/auth.py

# Review with team (multi-agent)
victor chat "Comprehensive review of payment system" --mode team
```

### Generate Tests

```bash
# Generate unit tests
victor chat "Generate unit tests for user service"

# Generate integration tests
victor chat "Add integration tests for API endpoints"

# Generate with coverage goals
victor chat "Achieve 90% coverage for auth module"
```

### Refactoring

```bash
# Plan refactoring (exploration mode)
victor chat "Plan refactoring of data pipeline" --mode plan

# Execute refactoring (build mode)
victor chat "Refactor to use dependency injection" --mode build

# Review refactoring
victor chat "Review the refactored code for issues"
```

### Documentation

```bash
# Generate API docs
victor chat "Generate OpenAPI documentation for REST endpoints"

# Add docstrings
victor chat "Add comprehensive docstrings to all functions"

# Create README
victor chat "Create a comprehensive README for this project"
```

### Debugging

```bash
# Debug an error
victor chat "Fix the authentication error in login()"

# Investigate performance
victor chat "Identify performance bottlenecks in the API"

# Memory leak
victor chat "Find and fix memory leaks in data processor"
```

---

## Understanding Coordinators

### What Are Coordinators?

Coordinators are specialized components that handle specific aspects of Victor's operation:

| Coordinator | Responsibility |
|-------------|---------------|
| **ConversationController** | Message history, context tracking, stage management |
| **ToolPipeline** | Tool validation, execution coordination, budget enforcement |
| **ProviderManager** | Provider initialization, switching, health checks |
| **StreamingController** | Session lifecycle, metrics collection |
| **WorkflowCoordinator** | YAML workflow compilation and execution |

### Why Coordinators Matter

**Before (Monolithic)**:
```python
# One 6000+ line orchestrator
class AgentOrchestrator:
    def chat(self, message):
        # 500+ lines of logic
        pass
```

**After (Coordinator-based)**:
```python
# Clean facade coordinating specialists
class AgentOrchestrator:
    def __init__(self):
        self.conversation = ConversationController()
        self.tools = ToolPipeline()
        self.provider = ProviderManager()
        self.streaming = StreamingController()

    def chat(self, message):
        # 50 lines of coordination
        pass
```

**Benefits**:
- Easier to understand (605 lines vs 6000+)
- Easier to test (85% coverage)
- Easier to extend (add new features faster)
- Easier to debug (clear separation)

---

## Agent Modes

Victor has three modes that control exploration vs. exploitation:

| Mode | Exploration | Tool Budget | Best For |
|------|-------------|-------------|----------|
| **BUILD** (default) | 1.0x | Standard | Making changes, writing code |
| **PLAN** | 2.5x | Sandbox only | Planning, exploration, understanding |
| **EXPLORE** | 3.0x | No edits | Deep investigation, research |

### Usage Examples

```bash
# Build mode (default) - make changes
victor chat "Add error handling"

# Plan mode - explore without changes
victor chat "Plan the refactoring" --mode plan

# Explore mode - deep investigation
victor chat "Investigate the architecture" --mode explore

# Switch modes mid-session
/mode plan    # Switch to planning
/mode build   # Switch to building
```

---

## Team Formations

Victor can coordinate multiple AI agents working together:

### Sequential (Default)

```python
# Agents work one after another
coordinator.set_formation(TeamFormation.SEQUENTIAL)
# Researcher → Reviewer → Executor
```

### Parallel

```python
# Agents work simultaneously
coordinator.set_formation(TeamFormation.PARALLEL)
# All agents work independently, results combined
```

### Hierarchical

```python
# Manager delegates to workers
coordinator.set_formation(TeamFormation.HIERARCHICAL)
# Planner delegates to specialized workers
```

### Pipeline

```python
# Output flows through agents
coordinator.set_formation(TeamFormation.PIPELINE)
# Researcher → Analyst → Reviewer → Executor
```

### Consensus

```python
# Agents must agree
coordinator.set_formation(TeamFormation.CONSENSUS)
# Multiple rounds until agreement reached
```

### Example Usage

```python
from victor.teams import create_coordinator, TeamFormation
from victor.agent.subagents.base import SubAgentRole

# Create coordinator
coordinator = create_coordinator(orchestrator)

# Add specialized agents
coordinator.add_member(
    role=SubAgentRole.RESEARCHER,
    goal="Find security vulnerabilities"
)
coordinator.add_member(
    role=SubAgentRole.REVIEWER,
    goal="Review and validate findings"
)
coordinator.add_member(
    role=SubAgentRole.EXECUTOR,
    goal="Implement fixes"
)

# Set formation
coordinator.set_formation(TeamFormation.PIPELINE)

# Execute task
result = await coordinator.execute_task(
    "Secure the authentication system",
    context={"files": ["src/auth.py"]}
)
```

---

## Next Steps

### Learn More

- **[Coordinator Guide](COORDINATOR_GUIDE.md)** - Deep dive into coordinator architecture
- **[Migration Guide](MIGRATION_GUIDE.md)** - Upgrading from previous versions
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[FAQ](FAQ.md)** - Frequently asked questions

### Advanced Features

- **[Workflows](../user-guide/workflows.md)** - YAML workflow DSL
- **[Session Management](../user-guide/session-management.md)** - Context and state management
- **[Tools Reference](../user-guide/tools.md)** - All 55+ tools explained
- **[Provider Guide](../user-guide/providers.md)** - All 21 providers configured

### Configuration

```bash
# View configuration
victor config show

# Edit configuration
victor config edit

# Set provider
victor config set provider anthropic

# Set model
victor config set model claude-sonnet-4-5
```

### Getting Help

```bash
# General help
victor --help

# Command help
victor chat --help

# Check version
victor --version

# Debug mode
victor chat --debug
```

---

## Quick Reference

### Essential Commands

```bash
victor chat                    # Start interactive chat
victor chat "task"             # One-shot command
victor chat --mode plan        # Use specific mode
victor chat --provider X       # Use specific provider
victor config show             # View configuration
victor --version               # Check version
```

### Common Modes

```bash
--mode build                   # Make changes (default)
--mode plan                    # Explore without changes
--mode explore                 # Deep investigation
```

### Popular Providers

```bash
--provider anthropic           # Claude (cloud)
--provider openai              # GPT-4 (cloud)
--provider ollama              # Local models
--provider google              # Gemini (cloud)
```

### Environment Variables

```bash
ANTHROPIC_API_KEY              # Anthropic API key
OPENAI_API_KEY                 # OpenAI API key
GOOGLE_API_KEY                 # Google API key
VICTOR_MODE                    # Default mode (build/plan/explore)
VICTOR_PROVIDER                # Default provider
VICTOR_LOG_LEVEL               # Log level (DEBUG/INFO/WARNING/ERROR)
```

---

## Tips for Success

1. **Start with the right mode** - Use `--mode plan` for exploration, `--mode build` for changes
2. **Be specific** - Clear, detailed tasks get better results
3. **Provide context** - Include file paths, error messages, requirements
4. **Use teams for complex tasks** - Multi-agent coordination for reviews, refactoring
5. **Check provider health** - Use `victor provider check` before important tasks
6. **Review before applying** - Victor explains changes before making them
7. **Use version control** - Always commit before major changes
8. **Start local, go cloud** - Test with local models, use cloud for production

---

## What's Next?

You're now ready to use Victor AI! Here are some suggestions:

1. **Try a simple task** - Review a file, add tests, fix a bug
2. **Explore the codebase** - Use `--mode plan` to understand your project
3. **Set up your preferred provider** - Configure your default in `~/.victor/config.yaml`
4. **Read the detailed guides** - Deep dive into coordinators, workflows, teams
5. **Join the community** - Contribute, report issues, share feedback

**Happy coding with Victor AI!**
