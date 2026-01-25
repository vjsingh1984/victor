# Python API Reference

Python package API for integrating Victor into Python applications.

## Installation

```bash
# Install Victor
pip install victor-ai

# With dev dependencies
pip install "victor-ai[dev]"

# From source
pip install -e .
```

## Quick Start

### Basic Usage

```python
import asyncio
from victor import Agent

async def main():
    # Create agent
    agent = await Agent.create(provider="anthropic")

    # Run task
    result = await agent.run("Write a REST API with FastAPI")
    print(result)

asyncio.run(main())
```

### With Custom Configuration

```python
from victor import Agent

async def main():
    agent = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=4096
    )

    result = await agent.run("Explain recursion")
    print(result)

asyncio.run(main())
```

---

## Core API

### Agent

The main interface for interacting with Victor.

#### Agent.create()

Create an agent instance.

**Signature**:
```python
async def Agent.create(
    provider: str = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs
) -> Agent
```

**Parameters**:
- `provider` (str): Provider name (default: "ollama")
- `model` (str, optional): Model name
- `temperature` (float): Sampling temperature (0.0-1.0, default: 0.7)
- `max_tokens` (int): Maximum tokens to generate (default: 4096)
- `**kwargs`: Additional provider-specific parameters

**Returns**: `Agent` instance

**Example**:
```python
# Default (Ollama)
agent = await Agent.create()

# Specific provider and model
agent = await Agent.create(
    provider="anthropic",
    model="claude-sonnet-4-20250514"
)

# With parameters
agent = await Agent.create(
    provider="openai",
    model="gpt-4o",
    temperature=0.3,
    max_tokens=8192
)
```

#### agent.run()

Execute a task and return the result.

**Signature**:
```python
async def agent.run(
    task: str,
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None
) -> str
```

**Parameters**:
- `task` (str): Task description
- `context` (dict, optional): Additional context
- `tools` (list, optional): List of tool names to enable

**Returns**: `str` - Agent response

**Example**:
```python
agent = await Agent.create()

# Simple task
result = await agent.run("Write a function to sort an array")
print(result)

# With context
result = await agent.run(
    "Optimize this function",
    context={"code": "def slow_func(): ..."}
)

# With specific tools
result = await agent.run(
    "Run tests",
    tools=["pytest", "coverage"]
)
```

#### agent.astream()

Stream agent response in real-time.

**Signature**:
```python
async def agent.astream(
    task: str,
    context: Optional[Dict[str, Any]] = None
) -> AsyncIterator[str]
```

**Parameters**:
- `task` (str): Task description
- `context` (dict, optional): Additional context

**Yields**: `str` - Response tokens

**Example**:
```python
agent = await Agent.create()

async for token in agent.astream("Write a REST API"):
    print(token, end="", flush=True)

# Output streams in real-time
```

#### agent.chat()

Multi-turn conversation.

**Signature**:
```python
async def agent.chat(
    message: str
) -> str
```

**Parameters**:
- `message` (str): Message content

**Returns**: `str` - Agent response

**Example**:
```python
agent = await Agent.create()

# First message
response1 = await agent.chat("Create a class for User")

# Continue conversation
response2 = await agent.chat("Add authentication")

# Continue
response3 = await agent.chat("Add tests")

# Agent maintains conversation context
```

---

## Advanced API

### Task Management

#### Task

Structured task with lifecycle management.

```python
from victor import Task

async def main():
    agent = await Agent.create()

    # Create task
    task = Task(
        name="code_review",
        description="Review the authentication module",
        agent=agent
    )

    # Execute task
    result = await task.execute()

    # Check status
    print(task.status)  # pending, running, completed, failed

    # Get result
    print(result)

asyncio.run(main())
```

**Task Lifecycle**:
1. **pending**: Task created, not started
2. **running**: Task is executing
3. **completed**: Task finished successfully
4. **failed**: Task failed

#### Task Hooks

Add custom hooks to task lifecycle.

```python
from victor import Task, hooks

async def on_start(task):
    print(f"Task {task.name} started")

async def on_complete(task, result):
    print(f"Task {task.name} completed: {result}")

async def on_error(task, error):
    print(f"Task {task.name} failed: {error}")

# Use hooks
task = Task(
    name="my_task",
    description="Do something",
    agent=agent,
    hooks={
        "on_start": on_start,
        "on_complete": on_complete,
        "on_error": on_error
    }
)

await task.execute()
```

### Tools

#### Tool Composition

Compose tools for complex operations.

```python
from victor.tools import pipe, parallel, branch

# Sequential execution
result = await pipe(
    "read_file",
    "analyze_code",
    "suggest_improvements"
)

# Parallel execution
results = await parallel(
    "run_tests",
    "run_linter",
    "run_coverage"
)

# Conditional branching
result = await branch(
    condition=lambda: has_tests(),
    if_true="run_tests",
    if_false="write_tests"
)
```

#### Custom Tools

Create custom tools.

```python
from victor.tools import BaseTool, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    parameters = {
        "input": {
            "type": "string",
            "description": "Input to process",
            "required": True
        }
    }

    async def execute(self, **kwargs) -> ToolResult:
        result = process(kwargs["input"])
        return ToolResult(
            success=True,
            output=result
        )

# Register tool
from victor.tools import registry
registry.register(MyTool())

# Use
agent = await Agent.create(tools=["my_tool"])
```

### State Management

#### State

Manage agent state.

```python
from victor import State

# Create state
state = State()

# Set values
state["user"] = "Alice"
state["context"] = "Building a REST API"

# Get values
user = state["user"]

# Check existence
if "context" in state:
    print(f"Context: {state['context']}")

# Clear state
state.clear()
```

#### Persistence

Save and load state.

```python
from victor import State

# Save state to file
state = State()
state["data"] = "important data"
state.save("state.json")

# Load state from file
loaded_state = State.load("state.json")
print(loaded_state["data"])
```

---

## Multi-Agent API

### Agent Team

Coordinate multiple agents.

```python
from victor import Agent, AgentTeam

async def main():
    # Create specialized agents
    frontend = Agent(role="Frontend developer", tools=["react", "typescript"])
    backend = Agent(role="Backend developer", tools=["fastapi", "sqlalchemy"])
    tester = Agent(role="QA engineer", tools=["pytest", "selenium"])

    # Create team
    team = AgentTeam.hierarchical(
        lead="senior-developer",
        subagents=[frontend, backend, tester]
    )

    # Execute task with team
    result = await team.run("Implement user registration")
    print(result)

asyncio.run(main())
```

### Team Formations

**Hierarchical**:
```python
team = AgentTeam.hierarchical(
    lead="lead_agent",
    subagents=["agent1", "agent2", "agent3"]
)
```

**Flat** (all agents equal):
```python
team = AgentTeam.flat(
    agents=["agent1", "agent2", "agent3"],
    shared_memory=True
)
```

**Pipeline** (sequential):
```python
team = AgentTeam.pipeline(
    agents=["analyzer", "implementer", "tester"]
)
```

**Consensus** (voting):
```python
team = AgentTeam.consensus(
    agents=["agent1", "agent2", "agent3"],
    min_agreement=2
)
```

**Debate** (agents discuss):
```python
team = AgentTeam.debate(
    agents=["agent1", "agent2", "agent3"],
    rounds=3
)
```

---

## Event System

### EventBus

Subscribe to events.

```python
from victor.core.events import EventBus

async def on_tool_execution(event):
    print(f"Tool {event.data['tool_name']} executed")

async def on_provider_error(event):
    print(f"Provider error: {event.data['error']}")

# Subscribe to events
EventBus.subscribe("tool.execution", on_tool_execution)
EventBus.subscribe("provider.error", on_provider_error)

# Use agent
agent = await Agent.create()
result = await agent.run("Execute a tool")
# Events will be triggered automatically
```

### Event Types

| Event | Description | Data |
|-------|-------------|------|
| `agent.start` | Agent started task | `{"task": "..."}` |
| `agent.complete` | Agent completed task | `{"result": "..."}` |
| `agent.error` | Agent encountered error | `{"error": "..."}` |
| `tool.execution` | Tool executed | `{"tool_name": "...", "result": "..."}` |
| `tool.error` | Tool execution failed | `{"tool_name": "...", "error": "..."}` |
| `provider.request` | Provider API request | `{"provider": "...", "model": "..."}` |
| `provider.response` | Provider API response | `{"provider": "...", "duration_ms": ...}` |
| `provider.error` | Provider API error | `{"provider": "...", "error": "..."}` |

---

## Configuration API

### Profiles

Load configuration from profiles.

```python
from victor import Agent, load_profile

# Load profile
profile = load_profile("development")

# Create agent from profile
agent = await Agent.from_profile(profile)

# Or directly
agent = await Agent.from_profile_name("development")
```

### Environment Variables

Set configuration programmatically.

```python
import os
from victor import Agent

# Set environment variables
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
os.environ['VICTOR_LOG_LEVEL'] = 'DEBUG'

# Create agent
agent = await Agent.create(provider="anthropic")
```

---

## Error Handling

### Exceptions

```python
from victor import Agent, VictorError, ProviderError, ToolError

async def main():
    try:
        agent = await Agent.create(provider="anthropic")
        result = await agent.run("Do something")
    except ProviderError as e:
        print(f"Provider error: {e}")
    except ToolError as e:
        print(f"Tool error: {e}")
    except VictorError as e:
        print(f"Victor error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

asyncio.run(main())
```

### Retry Logic

```python
from victor import Agent
import asyncio

async def run_with_retry(agent, task, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await agent.run(task)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Use
agent = await Agent.create()
result = await run_with_retry(agent, "Complex task")
```

---

## Testing

### Unit Testing

```python
import pytest
from victor import Agent

@pytest.mark.asyncio
async def test_agent_creation():
    agent = await Agent.create(provider="ollama")
    assert agent is not None

@pytest.mark.asyncio
async def test_agent_run():
    agent = await Agent.create(provider="ollama")
    result = await agent.run("Say hello")
    assert "hello" in result.lower()
```

### Mocking Providers

```python
from unittest.mock import AsyncMock, patch
from victor import Agent

@pytest.mark.asyncio
async def test_agent_with_mock():
    # Mock provider response
    with patch('victor.providers.anthropic.AnthropicProvider.chat') as mock:
        mock.return_value = AgentResponse(
            content="Mocked response",
            tokens_used=10
        )

        agent = await Agent.create(provider="anthropic")
        result = await agent.run("Test")
        assert result == "Mocked response"
```

---

## Examples

### Example 1: Code Generation

```python
from victor import Agent

async def generate_code():
    agent = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-20250514"
    )

    code = await agent.run("""
    Create a FastAPI application with:
    - User model (id, name, email)
    - CRUD endpoints
    - Pydantic validation
    - SQLAlchemy database
    """)

    # Save to file
    with open("app.py", "w") as f:
        f.write(code)

    print("Code generated successfully!")

asyncio.run(generate_code())
```

### Example 2: Code Review

```python
from victor import Agent

async def review_code(file_path: str):
    agent = await Agent.create(provider="anthropic")

    # Read file
    with open(file_path) as f:
        code = f.read()

    # Review code
    review = await agent.run(f"""
    Review this code for:
    - Bugs
    - Security issues
    - Performance problems
    - Code style

    Code:
    {code}
    """)

    print(review)

asyncio.run(review_code("auth.py"))
```

### Example 3: Workflow Execution

```python
from victor import Agent

async def run_workflow():
    agent = await Agent.create()

    # Execute workflow
    result = await agent.run_workflow(
        workflow="code-review",
        input={"pr_number": 123}
    )

    print(f"Review complete: {result}")

asyncio.run(run_workflow())
```

### Example 4: Multi-Agent Task

```python
from victor import Agent, AgentTeam

async def multi_agent_development():
    # Create specialized agents
    architect = Agent(role="System architect")
    frontend = Agent(role="Frontend developer", tools=["react", "typescript"])
    backend = Agent(role="Backend developer", tools=["fastapi", "python"])
    tester = Agent(role="QA engineer", tools=["pytest", "selenium"])

    # Coordinate team
    team = AgentTeam.hierarchical(
        lead=architect,
        subagents=[frontend, backend, tester]
    )

    # Execute complex task
    result = await team.run("""
    Design and implement a user authentication system with:
    - Frontend login form
    - Backend API with JWT
    - Database integration
    - Comprehensive tests
    """)

    print(result)

asyncio.run(multi_agent_development())
```

### Example 5: Streaming Response

```python
from victor import Agent

async def stream_response():
    agent = await Agent.create(provider="anthropic")

    print("Response: ", end="", flush=True)

    async for token in agent.astream("Explain microservices"):
        print(token, end="", flush=True)

    print("\nDone!")

asyncio.run(stream_response())
```

### Example 6: Tool Composition

```python
from victor import Agent
from victor.tools import pipe, parallel

async def composed_pipeline():
    agent = await Agent.create()

    # Define pipeline
    pipeline = pipe(
        "read_file",      # Read source code
        "analyze_code",   # Analyze
        "suggest_improvements",  # Suggest changes
        "write_file"      # Write improvements
    )

    # Execute pipeline
    result = await pipeline(
        agent,
        file_path="auth.py"
    )

    print(result)

asyncio.run(composed_pipeline())
```

---

## Best Practices

### 1. Use Async/Await

**DO**:
```python
async def main():
    agent = await Agent.create()
    result = await agent.run("Task")
    print(result)

asyncio.run(main())
```

**DON'T**:
```python
# Wrong: No async
agent = Agent.create()  # This will fail!
result = agent.run("Task")
```

### 2. Handle Errors

**DO**:
```python
try:
    result = await agent.run("Task")
except VictorError as e:
    logger.error(f"Victor error: {e}")
    raise
```

**DON'T**:
```python
# Wrong: No error handling
result = await agent.run("Task")  # May crash!
```

### 3. Use Streaming for Long Responses

**DO**:
```python
async for token in agent.astream("Long task"):
    print(token, end="", flush=True)
```

**DON'T**:
```python
# Wrong: Waits for entire response
result = await agent.run("Long task")  # Slow
```

### 4. Reuse Agent Instances

**DO**:
```python
agent = await Agent.create()

result1 = await agent.run("Task 1")
result2 = await agent.run("Task 2")
result3 = await agent.run("Task 3")
```

**DON'T**:
```python
# Wrong: Creates new instance each time
agent1 = await Agent.create()
result1 = await agent1.run("Task 1")

agent2 = await Agent.create()
result2 = await agent2.run("Task 2")
```

### 5. Clean Up Resources

**DO**:
```python
agent = await Agent.create()
try:
    result = await agent.run("Task")
finally:
    await agent.cleanup()  # Release resources
```

**DON'T**:
```python
# Wrong: No cleanup
agent = await Agent.create()
result = await agent.run("Task")
# Resources not released!
```

---

## Reference

### Classes

**Agent**: Main agent interface
- `create()`: Create agent instance
- `run()`: Execute task
- `astream()`: Stream response
- `chat()`: Multi-turn conversation
- `cleanup()`: Release resources

**Task**: Structured task
- `execute()`: Execute task
- `status`: Task status

**State**: State management
- `save()`: Save state to file
- `load()`: Load state from file

**AgentTeam**: Multi-agent coordination
- `hierarchical()`: Hierarchical team
- `flat()`: Flat team
- `pipeline()`: Pipeline team
- `consensus()`: Consensus team
- `debate()`: Debate team

**EventBus**: Event system
- `subscribe()`: Subscribe to events
- `unsubscribe()`: Unsubscribe from events
- `emit()`: Emit event

### Exceptions

**VictorError**: Base exception
**ProviderError**: Provider-related errors
**ToolError**: Tool-related errors
**ConfigurationError**: Configuration errors

---

## Additional Resources

- **API Overview**: [API Reference →](index.md)
- **HTTP API**: [HTTP API →](http-api.md)
- **MCP Server**: [MCP Server →](mcp-server.md)
- **Configuration**: [Configuration →](../configuration/index.md)
- **Examples**: [Examples →](../../tutorials/README.md)

---

**Next**: [API Overview →](index.md) | [HTTP API →](http-api.md) | [MCP Server →](mcp-server.md)
