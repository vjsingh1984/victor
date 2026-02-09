# Python API Reference - Part 2

**Part 2 of 2:** Testing, Examples, and Best Practices

---

## Navigation

- [Part 1: Core & Advanced API](part-1-core-advanced-api.md)
- **[Part 2: Testing, Examples & Best Practices](#)** (Current)
- [**API Reference Home](../README.md)**

---

## Testing

### Unit Testing

Test Victor agent interactions.

```python
import pytest
from victor import Agent
from victor.testing import mock_provider

@pytest.mark.asyncio
async def test_agent_creation():
    agent = await Agent.create(provider="anthropic")
    assert agent is not None
    assert agent.provider == "anthropic"

@pytest.mark.asyncio
async def test_task_execution():
    agent = await Agent.create(provider="anthropic")
    result = await agent.run("Say hello")
    assert "hello" in result.lower()
```text

### Mocking Providers

Mock LLM providers for testing.

```python
from victor.testing import MockProvider
from victor import Agent

# Create mock provider
mock = MockProvider(responses={
    "default": "Mocked response"
})

agent = await Agent.create(provider=mock)
result = await agent.run("Any task")
assert result == "Mocked response"
```text

### Integration Tests

Test with real providers (use sparingly).

```python
import pytest
from victor import Agent

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.requires_ollama
async def test_ollama_integration():
    agent = await Agent.create(
        provider="ollama",
        model="llama2"
    )
    result = await agent.run("Test prompt")
    assert len(result) > 0
```text

---

## Examples

### Example 1: Code Generation

Generate complete code implementations.

```python
from victor import Agent

async def generate_code():
    agent = await Agent.create(
        provider="anthropic",
        temperature=0.3
    )

    result = await agent.run("""
    Create a FastAPI endpoint that:
    - Accepts POST requests to /users
    - Validates user email and password
    - Returns user data with ID
    - Uses SQLAlchemy for database
    """)

    print(result)

asyncio.run(generate_code())
```text

### Example 2: Code Review

Review and improve existing code.

```python
from victor import Agent

async def review_code():
    agent = await Agent.create(provider="anthropic")

    code = """
def calculate(a,b):
return a+b
    """

    result = await agent.run(f"""
    Review this code for:
    - Style issues
    - Potential bugs
    - Performance improvements

    Code:
    {code}
    """)

    print(result)

asyncio.run(review_code())
```text

### Example 3: Workflow Execution

Execute complex multi-step workflows.

```python
from victor import Agent, Task

async def execute_workflow():
    agent = await Agent.create(provider="anthropic")

    # Create workflow
    workflow = Task(
        description="Build a microservice",
        steps=[
            "Design API schema",
            "Implement models",
            "Create endpoints",
            "Write tests",
            "Generate documentation"
        ]
    )

    result = await agent.run(workflow)
    print(result)

asyncio.run(execute_workflow())
```text

### Example 4: Multi-Agent Task

Coordinate multiple specialized agents.

```python
from victor import Agent, Team

async def multi_agent_task():
    # Create specialized agents
    architect = await Agent.create(
        provider="anthropic",
        role="architect",
        system_prompt="You design software architecture"
    )

    developer = await Agent.create(
        provider="openai",
        role="developer",
        system_prompt="You implement code"
    )

    tester = await Agent.create(
        provider="anthropic",
        role="tester",
        system_prompt="You write tests"
    )

    # Create team
    team = Team(agents=[architect, developer, tester])

    result = await team.execute("""
    Build a REST API for user management
    with authentication and testing
    """)

    print(result)

asyncio.run(multi_agent_task())
```text

### Example 5: Streaming Response

Stream responses in real-time.

```python
from victor import Agent
import asyncio

async def stream_response():
    agent = await Agent.create(provider="anthropic")

    print("Response: ", end="", flush=True)

    async for chunk in agent.astream(
        "Explain how neural networks work"
    ):
        print(chunk, end="", flush=True)

    print()  # New line

asyncio.run(stream_response())
```text

### Example 6: Tool Composition

Combine multiple tools.

```python
from victor import Agent
from victor.tools import ReadFile, WriteFile, SearchCode

async def tool_composition():
    agent = await Agent.create(
        provider="anthropic",
        tools=[ReadFile, WriteFile, SearchCode]
    )

    result = await agent.run("""
    Read main.py, find the calculate function,
    optimize it, and write back to the file
    """)

    print(result)

asyncio.run(tool_composition())
```text

---

## Best Practices

### 1. Use Async/Await

Always use async/await for I/O operations.

```python
# ✅ Good
async def main():
    agent = await Agent.create()
    result = await agent.run("Task")
    return result

# ❌ Bad
def main():
    agent = Agent.create()  # Missing await
    result = agent.run("Task")  # Missing await
    return result
```text

### 2. Handle Errors

Implement proper error handling.

```python
from victor.errors import VictorError, ProviderError

async def safe_execution():
    try:
        agent = await Agent.create(provider="anthropic")
        result = await agent.run("Task")
        return result
    except ProviderError as e:
        print(f"Provider failed: {e}")
        # Retry with different provider
        return await fallback_provider()
    except VictorError as e:
        print(f"Victor error: {e}")
        raise
```text

### 3. Use Streaming for Long Responses

Use streaming for responses that take time.

```python
# For long responses
async for chunk in agent.astream("Write a novel"):
    print(chunk, end="", flush=True)

# Instead of
result = await agent.run("Write a novel")  # Blocks until complete
```text

### 4. Reuse Agent Instances

Reuse agents instead of creating new ones.

```python
# ✅ Good - Reuse agent
agent = await Agent.create()

for task in tasks:
    result = await agent.run(task)

# ❌ Bad - Create new agent each time
for task in tasks:
    agent = await Agent.create()  # Wasteful
    result = await agent.run(task)
```text

### 5. Clean Up Resources

Close agents and connections properly.

```python
async def main():
    agent = await Agent.create()
    try:
        result = await agent.run("Task")
        return result
    finally:
        await agent.close()  # Clean up
```text

### 6. Use Appropriate Temperature

Adjust temperature based on task.

```python
# For creative tasks
agent = await Agent.create(temperature=1.0)

# For analytical tasks
agent = await Agent.create(temperature=0.2)

# For balanced tasks
agent = await Agent.create(temperature=0.7)
```text

### 7. Monitor Token Usage

Track token usage for cost management.

```python
from victor import Agent

agent = await Agent.create()
result = await agent.run("Task")

# Check token usage
usage = agent.get_token_usage()
print(f"Input tokens: {usage['input_tokens']}")
print(f"Output tokens: {usage['output_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
```text

---

## Reference

### Classes

| Class | Description | Module |
|-------|-------------|--------|
| `Agent` | Main agent interface | `victor` |
| `Task` | Structured task representation | `victor` |
| `Team` | Multi-agent coordination | `victor` |
| `State` | State management | `victor` |
| `EventBus` | Event system | `victor.events` |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `VictorError` | Base Victor exception |
| `ProviderError` | LLM provider errors |
| `ToolError` | Tool execution errors |
| `ValidationError` | Input validation errors |
| `ConfigurationError` | Configuration errors |

---

## Additional Resources

- [Architecture Guide](../../../architecture/README.md)
- [Provider Reference](../internals/providers-api.md)
- [Tools Reference](../internals/tools-api.md)
- [Configuration Guide](../configuration/README.md)

---

## See Also

- [Core & Advanced API](part-1-core-advanced-api.md)
- [Provider API Reference](../internals/providers-api.md)
- [Tools API Reference](../internals/tools-api.md)
- [Documentation Home](../../README.md)


**Reading Time:** 12 min
**Last Updated:** February 08, 2026**
