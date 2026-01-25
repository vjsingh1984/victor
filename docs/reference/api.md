# API Reference

Complete reference for Victor AI's APIs and interfaces.

## Quick Links

| API | Description | Documentation |
|-----|-------------|--------------|
| **HTTP API** | REST endpoints for HTTP clients | [HTTP API ->](api/http-api.md) |
| **MCP Server** | Model Context Protocol server | [MCP ->](api/mcp-server.md) |
| **Python API** | Programmatic access | [Python ->](api/python-api.md) |
| **CLI Commands** | Command-line interface | [CLI ->](../user-guide/cli-reference.md) |

---

## Python API

### Quick Start

```python
from victor.framework import Agent

# Create agent
agent = await Agent.create(provider="anthropic")

# Run task
result = await agent.run("Write tests for auth.py")

# Stream events
async for event in agent.stream("Analyze code"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")
```

### Core Classes

#### Agent
```python
class Agent:
    async def create(
        provider: str = "anthropic",
        model: Optional[str] = None,
        tools: Optional[ToolSet] = None,
    ) -> Agent:
        """Create agent instance."""

    async def run(
        self,
        prompt: str,
        task_type: Optional[FrameworkTaskType] = None,
    ) -> TaskResult:
        """Run task synchronously."""

    async def stream(
        self,
        prompt: str,
    ) -> AsyncIterator[AgentExecutionEvent]:
        """Stream task execution events."""
```

#### ToolSet
```python
class ToolSet:
    @staticmethod
    def minimal() -> ToolSet:
        """Minimal tool set (read, write, edit)."""

    @staticmethod
    def default() -> ToolSet:
        """Default balanced tool set."""

    @staticmethod
    def full() -> ToolSet:
        """All available tools."""

    @staticmethod
    def custom(tools: List[str]) -> ToolSet:
        """Custom tool selection."""
```

### Example: Multi-Provider Workflow

```python
from victor.framework import Agent

# Use different providers for cost optimization
brainstormer = await Agent.create(provider="ollama")      # FREE
implementer = await Agent.create(provider="openai")       # CHEAP
reviewer = await Agent.create(provider="anthropic")       # QUALITY

ideas = await brainstormer.run("Brainstorm features")
code = await implementer.run(f"Implement: {ideas.content[0]}")
review = await reviewer.run(f"Review: {code.content}")
```

---

## HTTP API

### Start Server

```bash
victor serve --port 8000
```

### Endpoints

#### POST /chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain this code",
    "provider": "anthropic",
    "model": "claude-sonnet-4-5"
  }'
```

#### POST /chat/stream
Server-Sent Events (SSE) streaming endpoint.

#### POST /workflow/run
Execute YAML workflows.

```bash
curl -X POST http://localhost:8000/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "code-review",
    "input": {"target": "src/auth.py"}
  }'
```

### Authentication

```bash
# API key in header
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/chat

# Or environment variable
export VICTOR_API_KEY=your-key
```

[Full HTTP API Documentation ->](api/http-api.md)

---

## MCP Server

### Start MCP Server

```bash
# stdio transport
victor mcp --stdio

# SSE transport
victor mcp --sse --port 3000
```

### Configuration

```json
{
  "mcpServers": {
    "victor": {
      "command": "victor",
      "args": ["mcp", "--stdio"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-..."
      }
    }
  }
}
```

### Capabilities

- **Tools**: Exposes Victor's tool suite
- **Prompts**: System prompt templates
- **Resources**: File access, project context

[Full MCP Documentation ->](api/mcp-server.md)

---

## CLI Commands

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `victor` | Start TUI | `victor` |
| `victor chat` | CLI mode | `victor chat "Hello"` |
| `victor serve` | HTTP API server | `victor serve --port 8000` |
| `victor init` | Initialize config | `victor init` |
| `victor workflow` | Workflow commands | `victor workflow run review` |

### Provider Options

```bash
# Specify provider
victor chat --provider anthropic

# Specify model
victor chat --provider openai --model gpt-4

# Switch mid-conversation
/provider ollama --model qwen2.5-coder:7b
```

### Mode Selection

```bash
# Build (default, full edits)
victor chat --mode build

# Plan (2.5x exploration, sandbox)
victor chat --mode plan

# Explore (3.0x exploration, no edits)
victor chat --mode explore
```

### Tool Control

```bash
# All tools
victor chat --tools all

# Specific tools
victor chat --tools read,write,edit

# Tool budget
victor chat --tool-budget 20
```

[Full CLI Documentation ->](../user-guide/cli-reference.md)

---

## Protocols Reference

Victor uses **98 protocols** for loose coupling and testability.

### Key Protocols

| Protocol | Purpose |
|----------|---------|
| `ProviderProtocol` | LLM provider interface |
| `ToolProtocol` | Tool interface |
| `ToolExecutorProtocol` | Tool execution |
| `ChatCoordinatorProtocol` | Chat coordination |
| `ToolCoordinatorProtocol` | Tool coordination |
| `CacheProtocol` | Caching operations |
| `MetricsProtocol` | Metrics collection |

### Example: Using Protocols

```python
from victor.agent.protocols import ToolExecutorProtocol
from victor.core.container import ServiceContainer

container = ServiceContainer()
executor = container.get(ToolExecutorProtocol)

result = await executor.execute_tool(tool, arguments)
```

[Full Protocol Documentation ->](internals/protocols-api.md)

---

## Configuration API

### Profile Configuration

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    tools: [read, write, edit]
    mode: build
```

### Programmatic Configuration

```python
from victor.framework import Agent, AgentConfig

config = AgentConfig(
    provider="anthropic",
    model="claude-sonnet-4-5",
    tools=ToolSet.default(),
    system_prompt="You are a Python expert.",
)

agent = await Agent.create(config=config)
```

[Full Configuration Documentation ->](configuration/index.md)

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ProviderNotFoundError` | Invalid provider name | Check available providers |
| `ToolExecutionError` | Tool execution failed | Check tool permissions |
| `APIKeyError` | Missing/invalid API key | Set environment variable |
| `ContextWindowExceeded` | Input too long | Reduce context or use larger model |

### Error Handling Example

```python
from victor.framework import Agent
from victor.framework.errors import ToolExecutionError

try:
    result = await agent.run("Execute tool")
except ToolExecutionError as e:
    print(f"Tool failed: {e}")
    # Handle error
```

[Full Error Documentation ->](errors.md)

---

## See Also

- [Provider Reference](providers/index.md) - Current provider list
- [Tool Reference](tools/catalog.md) - Current tool list
- [Workflow Reference](internals/workflows-api.md) - YAML workflows
- [Configuration](configuration/index.md) - Complete configuration guide

---

<div align="center">

**[<- Back to Reference](index.md)**

**API Reference**

*Programmatic access to Victor*

</div>
