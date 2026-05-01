# Victor API Reference

Complete API documentation for Victor - Open-source agentic AI framework.

## Overview

Victor provides a comprehensive Python API for building agentic AI applications:

- **Agent API**: High-level agent interface for conversations and tasks
- **StateGraph API**: Stateful workflow engine with conditional logic
- **Tools API**: Tool registry and execution framework
- **Configuration API**: Settings, profiles, and environment management
- **Provider API**: 24 LLM provider adapters with unified interface
- **Core APIs**: Events, State, Workflows, Multi-agent teams

## Quick Start

```python
from victor.framework import Agent
from victor.config import ProfileConfig

# Simple agent usage
agent = Agent(
    profile="default",
    tools="default",  # or tools=["filesystem", "search"]
)

# Single-turn execution
response = agent.run("What files are in the current directory?")
print(response.content)

# Multi-turn conversation
agent.chat()  # Interactive chat loop
```

## API Modules

| Module | Description | Link |
|--------|-------------|------|
| **Agent** | High-level agent interface | [agent.md](agent.md) |
| **StateGraph** | Stateful workflow engine | [graph.md](graph.md) |
| **Tools** | Tool registry and execution | [tools.md](tools.md) |
| **Configuration** | Settings and profiles | [../users/reference/config.md](../users/reference/config.md) |
| **Providers** | LLM provider adapters | [providers.md](providers.md) |
| **Core** | Events, state, teams, errors | [core.md](core.md) |

## Key Concepts

### Agents

Agents are the primary interface for interacting with LLMs. They handle:
- Conversation management
- Tool routing and execution
- Provider selection
- Error handling
- State persistence

```python
from victor.framework import Agent

agent = Agent(
    profile="default",
    tools=["filesystem", "search"],
    system_prompt="You are a helpful coding assistant."
)

response = agent.run("Help me debug this code")
```

### StateGraphs

StateGraphs provide stateful workflow execution with:
- Typed state management
- Conditional routing
- Checkpointing and resumption
- Human-in-the-loop interactions

```python
from victor.framework import StateGraph
from victor.framework.graph import StateGraph, Node, Edge

def route_intent(state):
    if state.get("intent") == "code":
        return "generate_code"
    return "answer_question"

graph = StateGraph(
    nodes={
        "start": Node(handler=process_input),
        "generate_code": Node(handler=write_code),
        "answer_question": Node(handler=answer),
    },
    edges={
        "start": Edge(condition=route_intent),
    }
)
```

### Tools

Tools provide extensible capabilities:
- File operations
- Code execution
- Web search
- Database queries
- Git operations
- And 25+ more

```python
from victor.framework import ToolRegistry
from victor.tools.filesystem import read_file

# Register custom tool
registry = ToolRegistry.get_instance()
registry.register_tool(
    name="read_file",
    tool=read_file,
    metadata={"description": "Read file contents"}
)
```

## Common Patterns

### Agent with Custom Tools

```python
from victor.framework import Agent

agent = Agent(
    profile="default",
    tools=["filesystem", "git", "search"],
)

# Use the agent
response = agent.run("What changed in the last commit?")
```

### Streaming Responses

```python
from victor.framework import Agent

agent = Agent(profile="default")

# Stream events in real-time
for event in agent.stream("Explain this code"):
    if event.type == "content":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[Tool: {event.tool_name}]")
```

### Workflow Execution

```python
from victor.framework import Agent

agent = Agent(profile="default")

# Run a YAML workflow
result = agent.run_workflow(
    workflow_path="workflow.yaml",
    input_data={"query": "test user login"}
)
```

### Multi-Agent Teams

```python
from victor.framework import Agent
from victor.teams import AgentTeam, TeamMemberSpec

# Create specialized agents
researcher = Agent(profile="research", tools=["web", "search"])
coder = Agent(profile="coding", tools=["filesystem", "git"])
tester = Agent(profile="coding", tools=["test"])

# Create team
team = AgentTeam(
    name="development",
    formation="sequential",
    members=[
        TeamMemberSpec(agent=researcher, name="researcher"),
        TeamMemberSpec(agent=coder, name="coder"),
        TeamMemberSpec(agent=tester, name="tester"),
    ]
)

# Run team
result = team.run("Implement user authentication feature")
```

## Configuration

### Using Profiles

```python
from victor.framework import Agent

# Use default profile
agent = Agent(profile="default")

# Use custom profile
agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-5-20250514",
    temperature=0.7,
    tools=["filesystem", "search"]
)
```

### Environment Variables

```bash
# Provider selection
export VICTOR_DEFAULT_PROVIDER=anthropic
export VICTOR_DEFAULT_MODEL=claude-sonnet-4-5-20250514

# API keys
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here

# Configuration directory
export VICTOR_CONFIG_DIR=/custom/config/path
```

## Error Handling

```python
from victor.framework import Agent
from victor.framework.contextual_errors import (
    ProviderConnectionError,
    ToolExecutionError,
)

try:
    agent = Agent(profile="default")
    response = agent.run("Hello")
except ProviderConnectionError as e:
    print(f"Provider error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
except ToolExecutionError as e:
    print(f"Tool error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
```

## Type Hints

Victor uses Python type hints throughout:

```python
from victor.framework import Agent
from victor.framework.graph import StateGraph
from victor.tools import ToolRegistry

# Type-safe agent creation
agent: Agent = Agent(profile="default")

# Type-safe graph creation
graph: StateGraph = StateGraph(nodes={...}, edges={...})

# Type-safe tool registry
registry: ToolRegistry = ToolRegistry.get_instance()
```

## Best Practices

### 1. Use Profiles for Common Configurations

```python
# Good - Use predefined profiles
agent = Agent(profile="coding")

# Avoid - Hardcoded configuration (unless necessary)
agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-5-20250514",
    temperature=0.7,
    max_tokens=4096,
    tools=[...],
    system_prompt="..."
)
```

### 2. Let Victor Handle Tool Selection

```python
# Good - Use tool presets
agent = Agent(tools="default")

# Specify tools when needed
agent = Agent(tools=["filesystem", "git", "search"])
```

### 3. Use Streaming for Long Responses

```python
# Good - Stream for better UX
for event in agent.stream("Generate documentation"):
    if event.type == "content":
        print(event.content, end="", flush=True)
```

### 4. Handle Errors Gracefully

```python
# Good - Contextual error handling
try:
    response = agent.run("Execute code")
except ProviderConnectionError as e:
    logger.error(f"Provider error: {e}")
    # Suggestion is included in error message
except ToolExecutionError as e:
    logger.warning(f"Tool failed: {e}")
```

### 5. Use Appropriate Tool Budgets

```python
# Good - Configure tool budgets in profile
agent = Agent(
    profile="expert",  # Higher tool budget
)
```

## Next Steps

- Explore specific API modules:
  - [Agent API](agent.md) - Agent creation and usage
  - [StateGraph API](graph.md) - Workflow creation
  - [Tools API](tools.md) - Tool registration and usage
  - [Configuration API](../users/reference/config.md) - Settings and profiles
  - [Provider API](providers.md) - Provider customization
  - [Core APIs](core.md) - Events, state, workflows

- See examples:
  - [Quickstart Guide](../guides/quickstart.md)
  - [First Agent](../guides/first-agent.md)
  - [Multi-Agent Teams](../guides/MULTI_AGENT_TEAMS.md)

- Get help:
  - `victor doctor` - Run diagnostics
  - `victor docs` - View all documentation
  - GitHub Issues - Report problems
