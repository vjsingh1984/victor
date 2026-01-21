# API Reference

Complete API documentation for Victor AI, including public APIs, function signatures, parameters, return types, and usage examples.

## Table of Contents

- [Overview](#overview)
- [AgentOrchestrator API](#agentorchestrator-api)
- [Coordinator APIs](#coordinator-apis)
- [Provider APIs](#provider-apis)
- [Tool APIs](#tool-apis)
- [Framework APIs](#framework-apis)
- [Event Bus APIs](#event-bus-apis)
- [Configuration APIs](#configuration-apis)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)

## Overview

Victor AI provides multiple APIs for different use cases:

- **AgentOrchestrator**: High-level API for chat and tool execution
- **Coordinators**: Specialized APIs for complex operations
- **Providers**: LLM provider abstraction
- **Tools**: Tool system for extending capabilities
- **Framework**: Low-level APIs for custom workflows
- **Event Bus**: Event-driven communication
- **Configuration**: Settings and profiles management

**API Design Principles:**
- **Type-Safe**: Full type hints with mypy validation
- **Async-First**: All I/O operations are async
- **Protocol-Based**: Loose coupling via interfaces
- **Extensible**: Easy to extend with custom implementations

## AgentOrchestrator API

### Main API

**Location**: `victor.agent.orchestrator.AgentOrchestrator`

**Purpose**: High-level facade for all agent operations

#### Initialization

```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings

# Load settings
settings = load_settings()

# Create orchestrator
orchestrator = AgentOrchestrator(settings=settings)
```

#### Methods

##### chat()

Process a single query and return complete response.

```python
async def chat(
    self,
    query: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Process a query and return complete response.

    Args:
        query: User query to process
        provider: LLM provider name (overrides default)
        model: Model name (overrides default)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Returns:
        Complete response text

    Raises:
        ProviderError: Provider unavailable or error
        ToolExecutionError: Tool execution failed
        BudgetExceededError: Tool budget exceeded

    Example:
        >>> response = await orchestrator.chat("What is 2+2?")
        >>> print(response)
        "2 + 2 equals 4."
    """
```

##### stream_chat()

Stream response chunks in real-time.

```python
async def stream_chat(
    self,
    query: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[StreamChunk]:
    """Stream response chunks in real-time.

    Args:
        query: User query to process
        provider: LLM provider name (overrides default)
        model: Model name (overrides default)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Yields:
        StreamChunk objects with:
        - content: Text chunk
        - is_complete: True if final chunk
        - metadata: Optional metadata (tool calls, etc.)

    Example:
        >>> async for chunk in orchestrator.stream_chat("Explain async/await"):
        ...     print(chunk.content, end="", flush=True)
    """
```

##### execute_tool()

Execute a specific tool with arguments.

```python
async def execute_tool(
    self,
    tool_name: str,
    **kwargs: Any,
) -> Any:
    """Execute a specific tool.

    Args:
        tool_name: Name of tool to execute
        **kwargs: Tool arguments

    Returns:
        Tool execution result

    Raises:
        ToolNotFoundError: Tool not found
        ToolExecutionError: Tool execution failed
        ValidationError: Invalid arguments

    Example:
        >>> result = await orchestrator.execute_tool(
        ...     "read_file",
        ...     file_path="main.py",
        ... )
        >>> print(result)
        "def main():\\n    ..."
    """
```

##### switch_provider()

Switch to different LLM provider.

```python
async def switch_provider(
    self,
    provider: str,
    model: Optional[str] = None,
) -> None:
    """Switch to different LLM provider.

    Args:
        provider: Provider name (anthropic, openai, ollama, etc.)
        model: Model name (uses provider default if None)

    Raises:
        ProviderNotFoundError: Provider not found
        ProviderError: Provider initialization failed

    Example:
        >>> await orchestrator.switch_provider("anthropic", "claude-sonnet-4-5")
    """
```

##### get_state()

Get current conversation state.

```python
def get_state(self) -> ConversationState:
    """Get current conversation state.

    Returns:
        ConversationState with:
        - stage: Current conversation stage
        - messages: Message history
        - context: Current context
        - metadata: Additional metadata

    Example:
        >>> state = orchestrator.get_state()
        >>> print(state.stage)
        ConversationStage.BUILD
    """
```

##### reset()

Reset conversation state.

```python
def reset(self) -> None:
    """Reset conversation state.

    Clears message history, context, and metadata.
    Useful for starting fresh conversation.

    Example:
        >>> orchestrator.reset()
    """
```

## Coordinator APIs

### ToolCoordinator

**Location**: `victor.agent.coordinators.tool_coordinator.ToolCoordinator`

**Purpose**: Coordinate tool selection and execution

#### Initialization

```python
from victor.agent.coordinators import ToolCoordinator
from victor.agent.protocols import ToolPipelineProtocol, IBudgetManager

tool_coordinator = ToolCoordinator(
    tool_pipeline=tool_pipeline,
    budget_manager=budget_manager,
)
```

#### Methods

##### select_and_execute()

Select and execute tools for a query.

```python
async def select_and_execute(
    self,
    query: str,
    context: AgentToolSelectionContext,
) -> ToolResult:
    """Select and execute tools.

    Args:
        query: User query
        context: Selection context with:
            - max_tools: Maximum tools to select
            - budget: Tool budget
            - stage: Conversation stage
            - tools: Available tools

    Returns:
        ToolResult with:
        - outputs: List of tool outputs
        - total_cost: Total cost (tokens/time)
        - execution_time: Execution time in seconds

    Example:
        >>> from victor.agent.protocols import AgentToolSelectionContext
        >>> context = AgentToolSelectionContext(
        ...     max_tools=5,
        ...     budget=ToolBudget(max_calls=100),
        ... )
        >>> result = await coordinator.select_and_execute(
        ...     "Read main.py",
        ...     context,
        ... )
    """
```

### PromptCoordinator

**Location**: `victor.agent.coordinators.prompt_coordinator.PromptCoordinator`

**Purpose**: Coordinate prompt building and assembly

#### Methods

##### build_prompt()

Build complete prompt for LLM.

```python
async def build_prompt(
    self,
    query: str,
    tools: List[BaseTool],
    context: PromptContext,
) -> str:
    """Build complete prompt.

    Args:
        query: User query
        tools: Available tools
        context: Prompt context with:
            - conversation_history: Past messages
            - system_prompt: System instructions
            - project_context: Project-specific context
            - metadata: Additional metadata

    Returns:
        Complete prompt string

    Example:
        >>> from victor.agent.protocols import PromptContext
        >>> context = PromptContext(
        ...     system_prompt="You are a helpful assistant",
        ...     conversation_history=[],
        ... )
        >>> prompt = await coordinator.build_prompt(
        ...     "Hello",
        ...     tools,
        ...     context,
        ... )
    """
```

### StateCoordinator

**Location**: `victor.agent.coordinators.state_coordinator.StateCoordinator`

**Purpose**: Coordinate conversation state management

#### Methods

##### transition_stage()

Transition to new conversation stage.

```python
async def transition_stage(
    self,
    new_stage: ConversationStage,
) -> None:
    """Transition to new stage.

    Args:
        new_stage: New stage (INITIAL, PLANNING, READING, etc.)

    Raises:
        InvalidTransitionError: Invalid stage transition

    Example:
        >>> from victor.agent.conversation_state import ConversationStage
        >>> await coordinator.transition_stage(
        ...     ConversationStage.BUILD,
        ... )
    """
```

##### get_state()

Get current conversation state.

```python
async def get_state(self) -> ConversationState:
    """Get current state.

    Returns:
        ConversationState with current state information
    """
```

## Provider APIs

### BaseProvider

**Location**: `victor.providers.base.BaseProvider`

**Purpose**: Abstract base for all LLM providers

#### Methods

##### chat()

Send message and get response.

```python
async def chat(
    self,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> str:
    """Send message and get response.

    Args:
        prompt: Prompt to send
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Provider-specific parameters

    Returns:
        Complete response text

    Raises:
        ProviderError: Provider error
        RateLimitError: Rate limit exceeded
        ValidationError: Invalid parameters
    """
```

##### stream_chat()

Stream response chunks.

```python
async def stream_chat(
    self,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]:
    """Stream response chunks.

    Args:
        prompt: Prompt to send
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Provider-specific parameters

    Yields:
        StreamChunk objects with content and metadata
    """
```

##### supports_tools()

Check if provider supports tool calling.

```python
def supports_tools(self) -> bool:
    """Check tool calling support.

    Returns:
        True if provider supports tool calling
    """
```

### ProviderManager

**Location**: `victor.agent.provider_manager.ProviderManager`

**Purpose**: Manage provider lifecycle

#### Methods

##### get_provider()

Get provider instance.

```python
def get_provider(
    self,
    provider_name: str,
) -> BaseProvider:
    """Get provider instance.

    Args:
        provider_name: Provider name

    Returns:
        Provider instance

    Raises:
        ProviderNotFoundError: Provider not found
    """
```

##### switch_provider()

Switch to different provider.

```python
async def switch_provider(
    self,
    new_provider: str,
    new_model: Optional[str] = None,
) -> None:
    """Switch to different provider.

    Args:
        new_provider: New provider name
        new_model: New model name (optional)
    """
```

## Tool APIs

### BaseTool

**Location**: `victor.tools.base.BaseTool`

**Purpose**: Abstract base for all tools

#### Attributes

```python
class BaseTool:
    name: str
    """Tool identifier (unique)"""

    description: str
    """Tool description for LLM"""

    parameters: Dict[str, Any]
    """JSON Schema for parameters"""

    cost_tier: CostTier
    """Cost tier: FREE, LOW, MEDIUM, HIGH"""

    category: str
    """Tool category (filesystem, search, etc.)"""
```

#### Methods

##### execute()

Execute tool with arguments.

```python
async def execute(self, **kwargs: Any) -> str:
    """Execute tool.

    Args:
        **kwargs: Tool arguments (validated against parameters schema)

    Returns:
        Tool execution result as string

    Raises:
        ValidationError: Invalid arguments
        ToolExecutionError: Execution failed

    Example:
        >>> result = await tool.execute(
        ...     file_path="main.py",
        ...     start_line=1,
        ...     end_line=10,
        ... )
    """
```

### SharedToolRegistry

**Location**: `victor.agent.tool_registrar.SharedToolRegistry`

**Purpose**: Register and discover tools

#### Methods

##### register_tool()

Register a new tool.

```python
def register_tool(self, tool: BaseTool) -> None:
    """Register tool.

    Args:
        tool: Tool instance to register

    Raises:
        ToolRegistrationError: Registration failed (duplicate, etc.)

    Example:
        >>> from victor.tools.base import BaseTool, CostTier
        >>> from victor.agent.tool_registrar import SharedToolRegistry
        >>>
        >>> class MyTool(BaseTool):
        ...     name = "my_tool"
        ...     description = "Does something useful"
        ...     parameters = {"type": "object", "properties": {}}
        ...     cost_tier = CostTier.LOW
        ...
        ...     async def execute(self, **kwargs) -> str:
        ...         return "result"
        >>>
        >>> SharedToolRegistry.register_tool(MyTool())
    """
```

##### get_tool()

Get tool by name.

```python
def get_tool(self, name: str) -> BaseTool:
    """Get tool.

    Args:
        name: Tool name

    Returns:
        Tool instance

    Raises:
        ToolNotFoundError: Tool not found
    """
```

##### get_tools_by_category()

Get all tools in category.

```python
def get_tools_by_category(
    self,
    category: str,
) -> List[BaseTool]:
    """Get tools by category.

    Args:
        category: Category name

    Returns:
        List of tools in category
    """
```

## Framework APIs

### StateGraph

**Location**: `victor.framework.graph.StateGraph`

**Purpose**: LangGraph-compatible workflow DSL

#### Initialization

```python
from victor.framework import StateGraph, START, END

# Define state structure
class AgentState(TypedDict):
    query: str
    context: Dict[str, Any]
    response: Optional[str]

# Create workflow
workflow = StateGraph(AgentState)
```

#### Methods

##### add_node()

Add a node to the workflow.

```python
def add_node(
    self,
    name: str,
    func: Callable[[AgentState], TaskResult],
) -> None:
    """Add node.

    Args:
        name: Node name (unique)
        func: Node function

    Example:
        >>> async def research_node(state: AgentState) -> TaskResult:
        ...     # Research logic
        ...     return TaskResult(status="success", data={})
        >>>
        >>> workflow.add_node("researcher", research_node)
    """
```

##### add_edge()

Add edge between nodes.

```python
def add_edge(
    self,
    start: str,
    end: str,
) -> None:
    """Add edge.

    Args:
        start: Start node name
        end: End node name

    Example:
        >>> workflow.add_edge(START, "researcher")
        >>> workflow.add_edge("researcher", "writer")
        >>> workflow.add_edge("writer", END)
    """
```

##### compile()

Compile workflow for execution.

```python
def compile(self) -> CompiledGraph:
    """Compile workflow.

    Returns:
        CompiledGraph ready for execution

    Example:
        >>> compiled = workflow.compile()
        >>> result = await compiled.invoke({"query": "Hello"})
    """
```

### Multi-Agent Teams

**Location**: `victor.teams`

**Purpose**: Coordinate multiple agents

#### Initialization

```python
from victor.teams import create_coordinator, TeamFormation, AgentRole

# Create team
coordinator = create_coordinator(
    formation=TeamFormation.PARALLEL,
    agents=[
        AgentRole(
            name="security_reviewer",
            persona="security_expert",
            goal="Review code for security issues",
        ),
        AgentRole(
            name="quality_reviewer",
            persona="code_reviewer",
            goal="Review code for quality issues",
        ),
    ],
)
```

#### Methods

##### execute()

Execute task with team.

```python
async def execute(
    self,
    task: str,
) -> Dict[str, Any]:
    """Execute task with team.

    Args:
        task: Task description

    Returns:
        Aggregated results from all agents

    Example:
        >>> result = await coordinator.execute(
        ...     "Review this authentication code",
        ... )
        >>> print(result["security_reviewer"]["analysis"])
        >>> print(result["quality_reviewer"]["analysis"])
    """
```

## Event Bus APIs

### IEventBackend

**Location**: `victor.core.events.protocols.IEventBackend`

**Purpose**: Event backend interface

#### Methods

##### publish()

Publish event.

```python
async def publish(
    self,
    event: MessagingEvent,
) -> None:
    """Publish event.

    Args:
        event: Event with:
            - topic: Event topic (e.g., "tool.complete")
            - data: Event data
            - timestamp: Event timestamp
            - metadata: Optional metadata

    Example:
        >>> from victor.core.events import MessagingEvent
        >>> await backend.publish(
        ...     MessagingEvent(
        ...         topic="tool.complete",
        ...         data={"tool": "read_file", "result": "..."},
        ...     ),
        ... )
    """
```

##### subscribe()

Subscribe to events.

```python
async def subscribe(
    self,
    topic: str,
    handler: Callable[[MessagingEvent], Awaitable[None]],
) -> None:
    """Subscribe to events.

    Args:
        topic: Topic pattern (supports wildcards, e.g., "tool.*")
        handler: Async event handler

    Example:
        >>> async def my_handler(event: MessagingEvent):
        ...     print(f"Event: {event.topic} - {event.data}")
        >>>
        >>> await backend.subscribe("tool.*", my_handler)
    """
```

### Event Backend Factory

**Location**: `victor.core.events.create_event_backend`

**Purpose**: Create event backend

#### Function

```python
def create_event_backend(
    config: Optional[BackendConfig] = None,
) -> IEventBackend:
    """Create event backend.

    Args:
        config: Backend configuration with:
            - backend_type: Backend type (in_memory, sqlite, redis, etc.)
            - connection_string: Connection string (for persistent backends)

    Returns:
        Event backend instance

    Example:
        >>> from victor.core.events import create_event_backend, BackendConfig
        >>>
        >>> # In-memory backend (default)
        >>> backend = create_event_backend()
        >>>
        >>> # SQLite backend
        >>> config = BackendConfig.for_observability()
        >>> backend = create_event_backend(config)
        >>>
        >>> # Redis backend
        >>> config = BackendConfig(
        ...     backend_type="redis",
        ...     connection_string="redis://localhost:6379",
        ... )
        >>> backend = create_event_backend(config)
    """
```

## Configuration APIs

### Settings

**Location**: `victor.config.settings.Settings`

**Purpose**: Application settings

#### Loading

```python
from victor.config.settings import load_settings

# Load from environment variables and .env
settings = load_settings()

# Access settings
provider = settings.default_provider
model = settings.default_model
api_key = settings.anthropic_api_key
```

#### Profiles

```python
# Load provider profiles
from victor.config.settings import Settings

profiles = Settings.load_profiles()

# Get profile
profile = profiles["default"]
print(profile.provider)  # "anthropic"
print(profile.model)  # "claude-sonnet-4-5"
print(profile.temperature)  # 0.7
```

### ProjectPaths

**Location**: `victor.config.settings.ProjectPaths`

**Purpose**: Manage project and global paths

#### Usage

```python
from victor.config.settings import get_project_paths

# Get paths for current project
paths = get_project_paths()

# Project-local paths
print(paths.project_victor_dir)  # ./.victor/
print(paths.conversation_db)  # ./.victor/conversation.db
print(paths.embeddings_dir)  # ./embeddings/

# Global paths
print(paths.global_victor_dir)  # ~/.victor/
print(paths.global_cache_dir)  # ~/.victor/cache/
print(paths.global_logs_dir)  # ~/.victor/logs/

# Ensure directories exist
paths.ensure_project_dirs()
paths.ensure_global_dirs()
```

## Error Handling

### Exception Hierarchy

```python
# Base exception
class VictorError(Exception):
    """Base exception for all Victor errors."""

# Provider errors
class ProviderError(VictorError):
    """Base provider error."""

class RateLimitError(ProviderError):
    """Rate limit exceeded."""

class ProviderUnavailableError(ProviderError):
    """Provider unavailable."""

# Tool errors
class ToolError(VictorError):
    """Base tool error."""

class ToolNotFoundError(ToolError):
    """Tool not found."""

class ToolExecutionError(ToolError):
    """Tool execution failed."""

class ValidationError(ToolError):
    """Invalid tool arguments."""

# Budget errors
class BudgetExceededError(VictorError):
    """Tool budget exceeded."""

# State errors
class InvalidStateError(VictorError):
    """Invalid conversation state."""

class InvalidTransitionError(InvalidStateError):
    """Invalid state transition."""
```

### Error Handling Patterns

```python
# Basic error handling
from victor.agent.orchestrator import AgentOrchestrator
from victor.providers import ProviderError, RateLimitError

orchestrator = AgentOrchestrator()

try:
    response = await orchestrator.chat("Hello")
except RateLimitError as e:
    # Handle rate limit
    print(f"Rate limited: {e}")
    await asyncio.sleep(60)
    response = await orchestrator.chat("Hello")
except ProviderError as e:
    # Handle provider error
    print(f"Provider error: {e}")
    # Try fallback provider
    await orchestrator.switch_provider("ollama")
    response = await orchestrator.chat("Hello")
```

```python
# Tool error handling
from victor.tools import ToolNotFoundError, ToolExecutionError

try:
    result = await orchestrator.execute_tool("my_tool", param="value")
except ToolNotFoundError:
    print("Tool not found")
except ToolExecutionError as e:
    print(f"Tool execution failed: {e}")
except ValidationError as e:
    print(f"Invalid arguments: {e}")
```

```python
# Budget error handling
from victor.agent.budget_manager import BudgetExceededError

try:
    result = await orchestrator.chat("Complex task requiring many tools")
except BudgetExceededError:
    print("Tool budget exceeded")
    # Increase budget or optimize tool usage
```

## Usage Examples

### Example 1: Basic Chat

```python
import asyncio
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings

async def main():
    # Initialize
    settings = load_settings()
    orchestrator = AgentOrchestrator(settings=settings)

    # Chat
    response = await orchestrator.chat("What is 2+2?")
    print(response)  # "2 + 2 equals 4."

asyncio.run(main())
```

### Example 2: Streaming Chat

```python
import asyncio
from victor.agent.orchestrator import AgentOrchestrator

async def main():
    orchestrator = AgentOrchestrator()

    # Stream response
    async for chunk in orchestrator.stream_chat("Explain async/await"):
        print(chunk.content, end="", flush=True)
    print()  # New line

asyncio.run(main())
```

### Example 3: Tool Execution

```python
import asyncio
from victor.agent.orchestrator import AgentOrchestrator

async def main():
    orchestrator = AgentOrchestrator()

    # Execute tool
    result = await orchestrator.execute_tool(
        "read_file",
        file_path="main.py",
    )
    print(result)

asyncio.run(main())
```

### Example 4: Provider Switching

```python
import asyncio
from victor.agent.orchestrator import AgentOrchestrator

async def main():
    orchestrator = AgentOrchestrator()

    # Start with Anthropic
    response1 = await orchestrator.chat("Generate code")
    print(response1)

    # Switch to Ollama
    await orchestrator.switch_provider("ollama", "qwen3-coder:30b")

    # Continue with Ollama
    response2 = await orchestrator.chat("Refactor this code")
    print(response2)

asyncio.run(main())
```

### Example 5: Custom Workflow

```python
import asyncio
from victor.framework import StateGraph, START, END, Task, TaskResult
from typing import TypedDict, Annotated
from operator import add

# Define state
class GraphState(TypedDict):
    query: str
    research: str
    draft: str
    final_answer: str
    steps: Annotated[list[int], add]

# Define nodes
async def research(state: GraphState) -> TaskResult:
    print("Researching...")
    return TaskResult(
        status="success",
        data={"research": f"Research for: {state['query']}"},
    )

async def draft(state: GraphState) -> TaskResult:
    print("Drafting...")
    return TaskResult(
        status="success",
        data={"draft": f"Draft based on: {state['research']}"},
    )

async def finalize(state: GraphState) -> TaskResult:
    print("Finalizing...")
    return TaskResult(
        status="success",
        data={"final_answer": f"Final: {state['draft']}"},
    )

# Build workflow
workflow = StateGraph(GraphState)
workflow.add_node("research", research)
workflow.add_node("draft", draft)
workflow.add_node("finalize", finalize)

workflow.add_edge(START, "research")
workflow.add_edge("research", "draft")
workflow.add_edge("draft", "finalize")
workflow.add_edge("finalize", END)

# Compile and run
compiled = workflow.compile()

async def main():
    result = await compiled.invoke({"query": "What is AI?"})
    print(result)

asyncio.run(main())
```

### Example 6: Event Handling

```python
import asyncio
from victor.core.events import create_event_backend, MessagingEvent, BackendConfig

async def main():
    # Create event backend
    config = BackendConfig.for_observability()
    backend = create_event_backend(config)
    await backend.connect()

    # Subscribe to events
    async def handle_tool_complete(event: MessagingEvent):
        print(f"Tool completed: {event.data['tool']}")

    await backend.subscribe("tool.complete", handle_tool_complete)

    # Publish event
    await backend.publish(
        MessagingEvent(
            topic="tool.complete",
            data={"tool": "read_file", "result": "..."},
        ),
    )

    # Keep running to receive events
    await asyncio.sleep(1)

asyncio.run(main())
```

### Example 7: Tool Registration

```python
from victor.tools.base import BaseTool, CostTier
from victor.agent.tool_registrar import SharedToolRegistry

class MyCustomTool(BaseTool):
    name = "my_custom_tool"
    description = "Does something custom"
    parameters = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Input string",
            },
        },
        "required": ["input"],
    }
    cost_tier = CostTier.LOW

    async def execute(self, **kwargs) -> str:
        input_str = kwargs["input"]
        return f"Processed: {input_str}"

# Register tool
tool = MyCustomTool()
SharedToolRegistry.register_tool(tool)

# Use tool
from victor.agent.orchestrator import AgentOrchestrator
orchestrator = AgentOrchestrator()

result = await orchestrator.execute_tool("my_custom_tool", input="Hello")
print(result)  # "Processed: Hello"
```

## Additional Resources

- **Production Deployment**: See [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)
- **Feature Guide**: See [FEATURE_GUIDE.md](FEATURE_GUIDE.md)
- **Migration Guide**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)

## Support

- **GitHub Issues**: https://github.com/vijayksingh/victor/issues
- **Discord Community**: https://discord.gg/victor-ai
- **Documentation**: https://docs.victor-ai.com
- **API Reference**: https://docs.victor-ai.com/api
