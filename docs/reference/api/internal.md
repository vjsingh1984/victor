# Internal API Reference

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: API documentation for internal systems

---

## Provider System

### BaseProvider

**Purpose**: Abstract base class for all LLM providers

**File Location**: `/Users/vijaysingh/code/codingagent/victor/providers/base.py`

**Key Methods**:

```python
class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Current model."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> str:
        """
        Execute chat completion.

        Args:
            messages: List of messages
            **kwargs: Additional arguments

        Returns:
            Response string
        """

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream chat completion.

        Args:
            messages: List of messages
            **kwargs: Additional arguments

        Yields:
            Stream chunks
        """

    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
```

**Supported Providers**: 21 providers (Anthropic, OpenAI, Google, Azure, etc.)

**Usage Example**:

```python
# Create provider
provider = AnthropicProvider(api_key="...", model="claude-3-5-sonnet-20241022")

# Use provider
messages = [
    Message(role="user", content="Hello!")
]
response = await provider.chat(messages)

# Stream
async for chunk in provider.stream_chat(messages):
    print(chunk.content, end="")
```

### ProviderPool

**Purpose**: Manage multiple provider instances with load balancing

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/provider_pool.py`

**Key Methods**:

```python
class ProviderPool:
    """Pool of provider instances."""

    def __init__(
        self,
        providers: List[BaseProvider],
        strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN,
    ):
        """
        Initialize provider pool.

        Args:
            providers: List of providers
            strategy: Load balancing strategy
        """

    async def route_request(
        self,
        request: Any,
    ) -> Any:
        """
        Route request to provider using strategy.

        Args:
            request: Request to route

        Returns:
            Response from provider

        Example:
            >>> pool = ProviderPool([prov1, prov2, prov3])
            >>> response = await pool.route_request(messages)
        """

    def add_provider(self, provider: BaseProvider) -> None:
        """Add provider to pool."""

    def remove_provider(self, provider: BaseProvider) -> None:
        """Remove provider from pool."""
```

**Load Balancing Strategies**:

- **ROUND_ROBIN**: Rotate through providers
- **LEAST_LOADED**: Route to least busy provider
- **PRIORITY**: Route to highest priority provider
- **HEALTH_AWARE**: Route to healthiest provider

---


## Tool System

### BaseTool

**Purpose**: Abstract base class for all tools

**File Location**: `/Users/vijaysingh/code/codingagent/victor/tools/base.py`

**Key Methods**:

```python
class BaseTool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters (JSON Schema)."""

    @property
    @abstractmethod
    def cost_tier(self) -> CostTier:
        """Tool cost tier (FREE, LOW, MEDIUM, HIGH)."""

    @abstractmethod
    async def execute(
        self,
        **kwargs,
    ) -> Any:
        """
        Execute tool.

        Args:
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
```

**Tool Categories**: 55 tools across 5 verticals

**Cost Tiers**:

| Tier | Description | Examples |
|------|-------------|----------|
| **FREE** | No cost | read_file, list_files |
| **LOW** | Minimal cost | search_code, parse_ast |
| **MEDIUM** | Moderate cost | execute_command, test_generation |
| **HIGH** | High cost | refactoring, code_review |

### ToolRegistry

**Purpose**: Registry of available tools

**File Location**: `/Users/vijaysingh/code/codingagent/victor/tools/registry.py`

**Key Methods**:

```python
class ToolRegistry:
    """Registry of available tools."""

    def register_tool(self, tool: BaseTool) -> None:
        """
        Register tool.

        Args:
            tool: Tool to register

        Example:
            >>> registry.register_tool(my_tool)
        """

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None

        Example:
            >>> tool = registry.get_tool("read_file")
        """

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.

        Returns:
            List of all tools

        Example:
            >>> tools = registry.get_all_tools()
        """

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        Get tools by category.

        Args:
            category: Tool category

        Returns:
            List of tools in category

        Example:
            >>> tools = registry.get_tools_by_category("coding")
        """
```

---


## Event System

### IEventBackend

**Purpose**: Protocol for event backends

**File Location**: `/Users/vijaysingh/code/codingagent/victor/core/events/base.py`

**Key Methods**:

```python
@runtime_checkable
class IEventBackend(Protocol):
    """Protocol for event backends."""

    async def publish(
        self,
        event: MessagingEvent,
    ) -> bool:
        """
        Publish event.

        Args:
            event: Event to publish

        Returns:
            True if published successfully

        Example:
            >>> await backend.publish(
            ...     MessagingEvent(
            ...         topic="tool.complete",
            ...         data={"tool": "read_file"}
            ...     )
            ... )
        """

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """
        Subscribe to events.

        Args:
            pattern: Topic pattern (e.g., "tool.*")
            handler: Event handler function

        Returns:
            Subscription handle

        Example:
            >>> await backend.subscribe(
            ...     "tool.*",
            ...     my_handler
            ... )
        """

    async def connect(self) -> None:
        """Connect to backend."""

    async def disconnect(self) -> None:
        """Disconnect from backend."""
```

**Supported Backends**: In-Memory, Kafka, SQS, RabbitMQ, Redis

**Usage Example**:

```python
from victor.core.events import create_event_backend, MessagingEvent

# Create backend
backend = create_event_backend(BackendConfig.for_observability())
await backend.connect()

# Publish event
await backend.publish(
    MessagingEvent(
        topic="tool.complete",
        data={"tool": "read_file", "result": "..."},
    )
)

# Subscribe to events
await backend.subscribe("tool.*", my_handler)
```

---


## Workflow System

### StateGraph

**Purpose**: State machine DSL for defining workflows

**File Location**: `/Users/vijaysingh/code/codingagent/victor/framework/graph.py`

**Key Methods**:

```python
class StateGraph:
    """State graph for workflow definition."""

    def add_node(
        self,
        name: str,
        func: Callable,
    ) -> "StateGraph":
        """
        Add node to graph.

        Args:
            name: Node name
            func: Node function

        Returns:
            Self for chaining

        Example:
            >>> graph = StateGraph(State)
            >>> graph.add_node("read", read_function)
        """

    def add_edge(
        self,
        from_node: str,
        to_node: str,
    ) -> "StateGraph":
        """
        Add edge between nodes.

        Args:
            from_node: Source node
            to_node: Target node

        Returns:
            Self for chaining

        Example:
            >>> graph.add_edge("read", "process")
        """

    def compile(
        self,
    ) -> CompiledGraph:
        """
        Compile graph for execution.

        Returns:
            Compiled graph ready for execution

        Example:
            >>> compiled = graph.compile()
            >>> result = await compiled.invoke(input_data)
        """
```

**Usage Example**:

```python
# Define workflow
workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge("agent", "tools")
workflow.add_edge("tools", END)

# Compile and execute
compiled = workflow.compile()
result = await compiled.invoke({"query": "refactor code"})
```

---


## Framework Components

Framework components provide reusable, domain-agnostic capabilities.

### Key Framework Modules

| Module | Purpose | File Location |
|--------|---------|--------------|
| **AgentBuilder** | Build agents flexibly | `framework/agent_components.py` |
| **ValidationPipeline** | Generic validation | `framework/validation/` |
| **ServiceManager** | Service lifecycle | `framework/services/` |
| **HealthChecker** | Health monitoring | `framework/health.py` |
| **MetricsRegistry** | Metrics collection | `framework/metrics.py` |
| **ExponentialBackoff** | Retry logic | `framework/resilience/` |
| **CircuitBreaker** | Circuit breaking | `framework/resilience/` |
| **PromptBuilder** | Prompt construction | `framework/prompt_builder.py` |
| **ParallelExecutor** | Parallel execution | `framework/parallel/` |
| **YAMLWorkflowCoordinator** | YAML workflows | `framework/coordinators/` |

**Usage Example**:

```python
from victor.framework import AgentBuilder, PromptBuilder
from victor.framework.validation import ValidationPipeline

# Build agent
agent = (AgentBuilder()
    .with_tools([read_tool, write_tool])
    .with_prompt("You are a coding assistant.")
    .build())

# Build prompt
prompt = (PromptBuilder()
    .add_system_message("You are helpful.")
    .add_context({"project": "victor"})
    .build())

# Validate input
pipeline = ValidationPipeline()
pipeline.add_validator(ThresholdValidator("max_length", 1000))
result = await pipeline.validate({"text": "hello"})
```

---

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 2 min
