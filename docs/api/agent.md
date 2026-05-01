# Agent API Reference

Core agent interface for conversational AI interactions with Victor.

## Overview

The `Agent` class provides a high-level interface for:
- Single-turn execution (`run()`)
- Multi-turn conversations (`chat()`)
- Streaming responses (`stream()`)
- Workflow execution (`run_workflow()`)
- Multi-agent teams (`run_team()`)

## Quick Example

```python
from victor.framework import Agent

# Create an agent
agent = Agent(
    profile="default",
    tools="default",
)

# Execute a single query
response = agent.run("What files are in this directory?")
print(response.content)

# Start interactive chat
agent.chat()
```

## Agent Class

### Constructor

```python
Agent(
    profile: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    tools: str | list[str] | ToolRegistry | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> Agent
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `profile` | `str \| None` | `None` | Profile name from `profiles.yaml` (uses default profile if None) |
| `provider` | `str \| None` | `None` | Provider name (e.g., "anthropic", "ollama"). Overrides profile setting. |
| `model` | `str \| None` | `None` | Model identifier (e.g., "claude-sonnet-4-5-20250514"). Overrides profile setting. |
| `tools` | `str \| list[str] \| ToolRegistry \| None` | `None` | Tool preset ("default", "minimal", "full") or list of tool names, or ToolRegistry instance. |
| `system_prompt` | `str \| None` | `None` | Custom system prompt. Overrides profile system prompt. |
| `**kwargs` | `Any` | `{}` | Additional settings passed to configuration |

**Returns**: `Agent` instance

**Raises**:
- `ConfigurationError`: If configuration is invalid
- `ProviderConnectionError`: If provider is unavailable

**Examples**:

```python
# Using profile
agent = Agent(profile="coding")

# Using explicit provider/model
agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-5-20250514"
)

# Using tools preset
agent = Agent(tools="default")

# Using specific tools
agent = Agent(tools=["filesystem", "git", "search"])

# With custom system prompt
agent = Agent(
    profile="default",
    system_prompt="You are a Python expert. Always explain your reasoning."
)
```

### Methods

#### run()

```python
def run(
    self,
    query: str,
    **kwargs: Any,
) -> AgentResponse:
    """Execute a single-turn query.

    Args:
        query: User query or instruction
        **kwargs: Additional arguments (session_id, context, etc.)

    Returns:
        AgentResponse with content and metadata

    Raises:
        ProviderConnectionError: If provider connection fails
        ToolExecutionError: If tool execution fails
    """
```

**Examples**:

```python
agent = Agent(profile="default")

# Simple query
response = agent.run("What files are in src/?")
print(response.content)

# With session context
response = agent.run(
    "Continue from where we left off",
    session_id="my-session"
)

# Access metadata
print(response.content)      # Response text
print(response.tool_calls)    # List of tool calls
print(response.metadata)     # Timing, tokens, etc.
```

**AgentResponse**:

```python
@dataclass
class AgentResponse:
    """Response from agent execution."""

    content: str                          # Response text
    tool_calls: List[ToolCall]             # Tools executed
    metadata: Dict[str, Any]               # Timing, tokens, etc.

    @property
    def token_count(self) -> int:           # Total tokens used
        """Return total tokens (prompt + completion)."""
```

#### chat()

```python
def chat(
    self,
    message: str | None = None,
) -> None:
    """Start interactive chat session.

    If message is provided, sends it before starting interactive loop.
    Otherwise, starts interactive chat immediately.

    Args:
        message: Optional initial message to send
    """
```

**Examples**:

```python
agent = Agent(profile="default")

# Start chat immediately
agent.chat()

# Send initial message then chat
agent.chat("Hello! I need help with Python.")
```

**Interactive Commands** (during chat):
- `/quit` or `Ctrl+C` - Exit chat
- `/clear` - Clear conversation history
- `/save <name>` - Save conversation
- `/load <name>` - Load conversation
- `/undo` - Undo last action

#### stream()

```python
def stream(
    self,
    query: str,
    **kwargs: Any,
) -> Iterator[Event]:
    """Stream response as events.

    Yields Event objects with real-time updates:
    - THINKING: Agent is thinking
    - TOOL_CALL: Tool is being called
    - TOOL_RESULT: Tool result received
    - CONTENT: Text content chunk
    - STREAM_END: Stream complete
    - ERROR: Error occurred

    Args:
        query: User query or instruction
        **kwargs: Additional arguments

    Yields:
        Event objects with type and data

    Raises:
        ProviderConnectionError: If provider connection fails
        ToolExecutionError: If tool execution fails
    """
```

**Event Types**:

```python
class EventType(Enum):
    THINKING = "thinking"           # Agent is reasoning
    TOOL_CALL = "tool_call"           # Tool being called
    TOOL_RESULT = "tool_result"       # Tool result received
    CONTENT = "content"               # Text content
    STREAM_END = "stream_end"         # Stream complete
    ERROR = "error"                   # Error occurred
```

**Examples**:

```python
agent = Agent(profile="default")

# Stream with real-time updates
for event in agent.stream("Explain this code"):
    if event.type == "thinking":
        print("Thinking...")
    elif event.type == "content":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[Tool: {event.tool_name}]")
    elif event.type == "error":
        print(f"\nError: {event.error}")
```

**Event Object**:

```python
@dataclass
class Event:
    """Event from agent stream."""

    type: EventType
    data: Dict[str, Any]
    correlation_id: str

    @property
    def content(self) -> str | None:
        """Content for CONTENT events."""

    @property
    def tool_name(self) -> str | None:
        """Tool name for TOOL_CALL events."""

    @property
    def tool_input(self) -> Dict[str, Any] | None:
        """Tool input for TOOL_CALL events."""

    @property
    def error(self) -> str | None:
        """Error message for ERROR events."""
```

#### run_workflow()

```python
def run_workflow(
    self,
    workflow_path: str | Path,
    input_data: Dict[str, Any] | None = None,
) -> Any:
    """Execute a YAML workflow file.

    Args:
        workflow_path: Path to workflow YAML file
        input_data: Input data for workflow

    Returns:
        Workflow result

    Raises:
        YAMLWorkflowError: If workflow is invalid
        WorkflowExecutionError: If execution fails
    """
```

**Examples**:

```python
agent = Agent(profile="default")

# Run workflow
result = agent.run_workflow(
    workflow_path="workflows/analyze.yaml",
    input_data={"file_path": "src/main.py"}
)

print(result)
```

**Example Workflow**:

```yaml
# workflows/analyze.yaml
name: Code Analysis

nodes:
  load:
    handler: load_file
  analyze:
    handler: analyze_code
  report:
    handler: generate_report

edges:
  load: analyze
  analyze: report

input_schema:
  file_path: str

output_schema:
  analysis: str
  report: str
```

#### run_team()

```python
def run_team(
    self,
    team: AgentTeam,
    query: str,
    **kwargs: Any,
) -> TeamResult:
    """Execute query with multi-agent team.

    Args:
        team: AgentTeam to execute
        query: Query for the team
        **kwargs: Additional arguments

    Returns:
        TeamResult with member responses and final result

    Raises:
        TeamExecutionError: If team execution fails
    """
```

**Examples**:

```python
from victor.framework import Agent
from victor.teams import AgentTeam, TeamMemberSpec, TeamFormation

# Create agents
researcher = Agent(profile="research", name="researcher")
coder = Agent(profile="coding", name="coder")
tester = Agent(profile="coding", name="tester")

# Create team
team = AgentTeam(
    name="development",
    formation=TeamFormation.SEQUENTIAL,
    members=[
        TeamMemberSpec(agent=researcher, name="researcher"),
        TeamMemberSpec(agent=coder, name="coder"),
        TeamMemberSpec(agent=tester, name="tester"),
    ],
)

# Execute with agent
agent = Agent(profile="default")
result = agent.run_team(team, "Implement a REST API endpoint")

print(result.final_answer)
for member_result in result.member_results:
    print(f"{member_name}: {member_result.response}")
```

## AgentBuilder

For fluent agent configuration:

```python
from victor.framework import AgentBuilder

agent = (
    AgentBuilder()
    .with_profile("coding")
    .with_tools(["filesystem", "git", "test"])
    .with_system_prompt("You are a Python expert.")
    .with_max_tokens(8192)
    .build()
)
```

**AgentBuilder Methods**:

```python
class AgentBuilder:
    """Fluent builder for Agent configuration."""

    def with_profile(self, profile: str) -> AgentBuilder:
        """Set profile."""

    def with_provider(self, provider: str) -> AgentBuilder:
        """Set provider."""

    def with_model(self, model: str) -> AgentBuilder:
        """Set model."""

    def with_temperature(self, temperature: float) -> AgentBuilder:
        """Set temperature."""

    def with_max_tokens(self, max_tokens: int) -> AgentBuilder:
        """Set max tokens."""

    def with_tools(self, tools: list[str] | ToolRegistry) -> AgentBuilder:
        """Set tools."""

    def with_system_prompt(self, prompt: str) -> AgentBuilder:
        """Set system prompt."""

    def build(self) -> Agent:
        """Build and return Agent instance."""
```

## Tool Presets

Victor provides tool presets for common use cases:

| Preset | Tools Included | Use Case |
|--------|----------------|----------|
| `"minimal"` | filesystem | Simple file operations |
| `"default"` | filesystem, search | General purpose |
| `"full"` | All 34 tools | Maximum capabilities |

**Examples**:

```python
# Minimal - file operations only
agent = Agent(tools="minimal")

# Default - general purpose
agent = Agent(tools="default")

# Full - all tools available
agent = Agent(tools="full")

# Custom tool selection
agent = Agent(tools=["filesystem", "git", "web_search", "test"])
```

## Session Management

Agents support session-based conversations:

```python
# Continue previous session
agent = Agent(profile="default")

response = agent.run(
    "Continue our discussion",
    session_id="my-session-123"
)
```

## Error Handling

```python
from victor.framework import Agent
from victor.framework.contextual_errors import (
    ProviderConnectionError,
    ToolExecutionError,
    ConfigurationError,
)

try:
    agent = Agent(profile="default")
    response = agent.run("Execute this code")
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Field: {e.field}")
    print(f"Suggestion: {e.suggestion}")
except ProviderConnectionError as e:
    print(f"Provider error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
except ToolExecutionError as e:
    print(f"Tool error: {e.message}")
    print(f"Tool: {e.tool_name}")
    print(f"Suggestion: {e.suggestion}")
```

## Best Practices

### 1. Use Profiles

```python
# Good - Use profile
agent = Agent(profile="coding")

# Avoid - Unless custom config needed
agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-5-20250514",
    temperature=0.7,
    max_tokens=4096,
    tools=[...]
)
```

### 2. Stream Long Responses

```python
# Good - Stream for better UX
for event in agent.stream("Generate comprehensive documentation"):
    if event.type == "content":
        print(event.content, end="", flush=True)
```

### 3. Handle Errors

```python
# Good - Contextual error handling
try:
    response = agent.run("Execute code")
except ToolExecutionError as e:
    logger.warning(f"Tool failed: {e.message}")
    # Suggestion is included
except ProviderConnectionError as e:
    logger.error(f"Provider error: {e.message}")
    # Suggestion is included
```

### 4. Use Appropriate Tool Budgets

```python
# Basic profile - 5 tools
agent = Agent(profile="basic")

# Advanced profile - 10 tools
agent = Agent(profile="advanced")

# Expert profile - 20 tools
agent = Agent(profile="expert")
```

### 5. Leverage System Prompts

```python
# Add domain expertise
agent = Agent(
    profile="coding",
    system_prompt=(
        "You are a Python expert. "
        "Always explain your reasoning step by step. "
        "Follow PEP 8 style guidelines."
    )
)
```

## See Also

- [Configuration API](../users/reference/config.md) - Profiles and settings
- [Tools API](tools.md) - Tool registration
- [Provider API](providers.md) - Provider customization
- [StateGraph API](graph.md) - Workflow creation
- [Core APIs](core.md) - Events, state, workflows
