# Victor AI Component Reference

**Version**: 0.5.0
**Last Updated**: January 18, 2026
**Audience**: Developers, Contributors
**Purpose**: Comprehensive reference for all major Victor AI components

---

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Coordinators](#coordinators)
4. [Adapters](#adapters)
5. [Mixins](#mixins)
6. [Provider System](#provider-system)
7. [Tool System](#tool-system)
8. [Event System](#event-system)
9. [Workflow System](#workflow-system)
10. [Framework Components](#framework-components)
11. [Component Interactions](#component-interactions)
12. [Extension Points](#extension-points)

---

## Overview

Victor AI is built from modular, loosely-coupled components that work together through well-defined protocols. This reference provides detailed documentation for each major component.

### Component Hierarchy

```
Victor AI
├── Core Components
│   ├── AgentOrchestrator (Facade)
│   ├── OrchestratorFactory (Factory)
│   └── ServiceContainer (DI)
├── Coordinators (15 specialized coordinators)
├── Adapters (Result and interface adapters)
├── Mixins (Cross-cutting concerns)
├── Provider System (21 LLM providers)
├── Tool System (55 tools)
├── Event System (5 backends)
├── Workflow System (StateGraph DSL)
└── Framework Components (Reusable capabilities)
```

---

## Core Components

### AgentOrchestrator

**Purpose**: Facade for coordinator layer, provides unified API

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/orchestrator.py`

**Design Pattern**: Facade Pattern

**Key Responsibilities**:
- Initialize and coordinate all coordinators
- Route client requests to appropriate coordinators
- Maintain backward compatibility with legacy code
- Provide high-level API for common operations
- Coordinate complex multi-coordinator workflows

**Dependencies**:
- ConfigCoordinator
- PromptCoordinator
- ContextCoordinator
- ChatCoordinator
- ToolCoordinator
- ProviderCoordinator
- SessionCoordinator
- MetricsCoordinator
- AnalyticsCoordinator

**Key Methods**:

```python
class AgentOrchestrator:
    """Main orchestrator for Victor AI."""

    async def chat(
        self,
        message: str,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Main chat interface.

        Args:
            message: User message
            stream: Whether to stream response

        Returns:
            Response string or async iterator of chunks

        Example:
            >>> response = await orchestrator.chat("Hello, Victor!")
            >>> print(response)
        """

    async def execute_tools(
        self,
        tool_calls: List[ToolCall],
    ) -> List[ToolCallResult]:
        """
        Execute tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool call results

        Example:
            >>> results = await orchestrator.execute_tools([
            ...     ToolCall(name="read_file", arguments={"path": "test.py"})
            ... ])
        """

    def switch_provider(
        self,
        provider: BaseProvider,
    ) -> None:
        """
        Switch to different provider.

        Args:
            provider: New provider to use

        Example:
            >>> orchestrator.switch_provider(openai_provider)
        """

    def switch_mode(
        self,
        mode: AgentMode,
    ) -> None:
        """
        Switch agent mode.

        Args:
            mode: New mode (BUILD, PLAN, EXPLORE)

        Example:
            >>> orchestrator.switch_mode(AgentMode.PLAN)
        """

    async def save_checkpoint(
        self,
        name: str,
    ) -> None:
        """
        Save current state as checkpoint.

        Args:
            name: Checkpoint name

        Example:
            >>> await orchestrator.save_checkpoint("before_refactor")
        """

    async def load_checkpoint(
        self,
        name: str,
    ) -> None:
        """
        Load state from checkpoint.

        Args:
            name: Checkpoint name

        Example:
            >>> await orchestrator.load_checkpoint("before_refactor")
        """
```

**Usage Example**:

```python
from victor.agent.orchestrator_factory import OrchestratorFactory

# Create orchestrator
factory = OrchestratorFactory(settings, provider, model)
orchestrator = factory.create_orchestrator()

# Use orchestrator
response = await orchestrator.chat("Help me refactor this code")

# Switch provider
orchestrator.switch_provider(another_provider)

# Save/restore state
await orchestrator.save_checkpoint("backup")
# ... do work ...
await orchestrator.load_checkpoint("backup")
```

**Component Size**: ~5,997 lines (93% reduction from monolithic 6,082 lines)

### OrchestratorFactory

**Purpose**: Factory for creating AgentOrchestrator instances with proper dependency injection

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/orchestrator_factory.py`

**Design Pattern**: Factory Pattern

**Key Responsibilities**:
- Create orchestrator components
- Manage provider initialization
- Create core services with DI
- Provide test-specific component creation

**Dependencies**:
- Settings
- BaseProvider
- ServiceContainer

**Key Methods**:

```python
class OrchestratorFactory:
    """Factory for creating AgentOrchestrator components."""

    def create_orchestrator(
        self,
        provider: Optional[BaseProvider] = None,
        mode: Optional[AgentMode] = None,
    ) -> AgentOrchestrator:
        """
        Create fully configured AgentOrchestrator.

        Args:
            provider: Optional provider override
            mode: Optional agent mode

        Returns:
            Configured AgentOrchestrator instance

        Example:
            >>> factory = OrchestratorFactory(settings, provider, model)
            >>> orchestrator = factory.create_orchestrator()
        """

    def create_provider_components(
        self,
        provider: Optional[BaseProvider] = None,
    ) -> ProviderComponents:
        """
        Create provider-related components.

        Returns:
            ProviderComponents dataclass with:
            - provider: BaseProvider instance
            - model: Model name
            - provider_name: Provider name
            - tool_adapter: Tool calling adapter
            - tool_calling_caps: Tool calling capabilities
        """

    def create_core_services(
        self,
    ) -> CoreServices:
        """
        Create core service components.

        Returns:
            CoreServices dataclass with:
            - sanitizer: ResponseSanitizer
            - prompt_builder: SystemPromptBuilder
            - project_context: ProjectContext
            - complexity_classifier: ComplexityClassifier
            - action_authorizer: ActionAuthorizer
            - search_router: SearchRouter
        """

    def create_conversation_components(
        self,
        provider_components: ProviderComponents,
    ) -> ConversationComponents:
        """
        Create conversation management components.

        Args:
            provider_components: Provider components from create_provider_components()

        Returns:
            ConversationComponents dataclass with:
            - conversation_controller: ConversationController
            - tool_pipeline: ToolPipeline
            - streaming_controller: StreamingController
            - context_compactor: ContextCompactor
            - usage_analytics: UsageAnalytics
            - tool_sequence_tracker: ToolSequenceTracker
            - tool_output_formatter: ToolOutputFormatter
            - recovery_handler: RecoveryHandler
            - observability: ObservabilityIntegration
        """
```

**Usage Example**:

```python
# For production use
factory = OrchestratorFactory(settings, provider, model)
orchestrator = factory.create_orchestrator()

# For testing
factory = OrchestratorFactory(settings, provider, model)
sanitizer = factory.create_sanitizer()
prompt_builder = factory.create_prompt_builder()
# Test individual components
```

### ServiceContainer

**Purpose**: Dependency injection container managing 55+ services

**File Location**: `/Users/vijaysingh/code/codingagent/victor/core/container.py`

**Design Pattern**: Dependency Injection Container

**Key Responsibilities**:
- Register services with lifetime management
- Resolve service dependencies
- Manage service lifecycle (singleton, scoped, transient)
- Provide service location

**Key Methods**:

```python
class ServiceContainer:
    """Dependency injection container."""

    def register(
        self,
        protocol: Type[T],
        factory: Callable[[ServiceContainer], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> None:
        """
        Register service.

        Args:
            protocol: Protocol type (interface)
            factory: Factory function to create service
            lifetime: Service lifetime (singleton, scoped, transient)

        Example:
            >>> container.register(
            ...     ToolRegistryProtocol,
            ...     lambda c: ToolRegistry(),
            ...     ServiceLifetime.SINGLETON
            ... )
        """

    def get(
        self,
        protocol: Type[T],
    ) -> T:
        """
        Resolve service.

        Args:
            protocol: Protocol type to resolve

        Returns:
            Service instance

        Raises:
            ServiceNotFoundError: If service not registered

        Example:
            >>> tool_registry = container.get(ToolRegistryProtocol)
        """

    def create_scope(
        self,
    ) -> ServiceContainer:
        """
        Create scoped container.

        Returns:
            New container with scoped lifetime

        Example:
            >>> scoped = container.create_scope()
            >>> state_machine = scoped.get(ConversationStateMachineProtocol)
        """
```

**Service Lifetime**:

| Lifetime | Description | Use Cases |
|----------|-------------|-----------|
| **Singleton** | One instance for entire application | ToolRegistry, EventBus, Settings |
| **Scoped** | One instance per scope/session | ConversationStateMachine, TaskTracker |
| **Transient** | New instance every time | Not currently used |

**Usage Example**:

```python
from victor.core.container import ServiceContainer, ServiceLifetime

# Create container
container = ServiceContainer()

# Register singleton service
container.register(
    ToolRegistryProtocol,
    lambda c: ToolRegistry(),
    ServiceLifetime.SINGLETON,
)

# Register scoped service
container.register(
    ConversationStateMachineProtocol,
    lambda c: ConversationStateMachine(),
    ServiceLifetime.SCOPED,
)

# Resolve services
tool_registry = container.get(ToolRegistryProtocol)
tools = tool_registry.get_all_tools()

# Create scope
scoped = container.create_scope()
state_machine = scoped.get(ConversationStateMachineProtocol)
```

---

## Coordinators

Coordinators are specialized components that encapsulate complex operations. Each coordinator has a single, well-defined responsibility.

### ConfigCoordinator

**Purpose**: Configuration loading, validation, and management

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/config_coordinator.py`

**Dependencies**: Settings

**Key Methods**:

```python
class ConfigCoordinator:
    """Configuration management coordinator."""

    def get_config(
        self,
    ) -> OrchestratorConfig:
        """
        Get current configuration.

        Returns:
            OrchestratorConfig with all settings

        Example:
            >>> config = config_coord.get_config()
            >>> print(config.max_tokens)
        """

    def validate_config(
        self,
        config: OrchestratorConfig,
    ) -> ValidationResult:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            ValidationResult with is_valid and errors

        Example:
            >>> result = config_coord.validate_config(config)
            >>> if not result.is_valid:
            ...     print(result.errors)
        """

    def reload_config(
        self,
    ) -> None:
        """
        Reload configuration from source.

        Example:
            >>> config_coord.reload_config()
        """

    def update_config(
        self,
        updates: Dict[str, Any],
    ) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of config updates

        Example:
            >>> config_coord.update_config({"max_tokens": 4000})
        """
```

**Usage Example**:

```python
# Get config
config = orchestrator._config_coordinator.get_config()

# Validate config
result = orchestrator._config_coordinator.validate_config(config)
if not result.is_valid:
    print(f"Invalid config: {result.errors}")

# Update config
orchestrator._config_coordinator.update_config({"temperature": 0.7})
```

### PromptCoordinator

**Purpose**: Prompt building from multiple contributors

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/prompt_coordinator.py`

**Dependencies**: ContextCoordinator, List<IPromptContributor>

**Key Methods**:

```python
class PromptCoordinator:
    """Prompt building coordinator."""

    async def build_prompt(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build system prompt from contributors.

        Args:
            message: User message
            context: Optional context for prompt building

        Returns:
            Complete system prompt

        Example:
            >>> prompt = await prompt_coord.build_prompt("Help me code")
            >>> print(prompt)
        """

    def add_contributor(
        self,
        contributor: IPromptContributor,
    ) -> None:
        """
        Add prompt contributor.

        Args:
            contributor: Contributor to add

        Example:
            >>> prompt_coord.add_contributor(MyCustomContributor())
        """

    def remove_contributor(
        self,
        contributor: IPromptContributor,
    ) -> None:
        """
        Remove prompt contributor.

        Args:
            contributor: Contributor to remove

        Example:
            >>> prompt_coord.remove_contributor(old_contributor)
        """

    def list_contributors(
        self,
    ) -> List[IPromptContributor]:
        """
        List all contributors.

        Returns:
            List of contributors

        Example:
            >>> contributors = prompt_coord.list_contributors()
        """
```

**Usage Example**:

```python
# Build prompt
prompt = await orchestrator._prompt_coordinator.build_prompt(
    message="Help me refactor this code",
    context={"language": "python"},
)

# Add custom contributor
class MyContributor:
    def get_contribution(self, context):
        return "\nAlways write clean, documented code."

orchestrator._prompt_coordinator.add_contributor(MyContributor())
```

### ContextCoordinator

**Purpose**: Context management and compaction

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/context_coordinator.py`

**Dependencies**: ConfigCoordinator

**Key Methods**:

```python
class ContextCoordinator:
    """Context management coordinator."""

    def get_context(
        self,
    ) -> Context:
        """
        Get current context.

        Returns:
            Current context object

        Example:
            >>> context = context_coord.get_context()
            >>> print(context.tokens_used)
        """

    async def compact_context(
        self,
        strategy: CompactionStrategy = CompactionStrategy.SUMMARIZATION,
    ) -> Context:
        """
        Compact context using strategy.

        Args:
            strategy: Compaction strategy to use

        Returns:
            Compacted context

        Example:
            >>> compacted = await context_coord.compact_context(
            ...     CompactionStrategy.SUMMARIZATION
            ... )
        """

    def update_context(
        self,
        updates: Dict[str, Any],
    ) -> None:
        """
        Update context with new data.

        Args:
            updates: Dictionary of context updates

        Example:
            >>> context_coord.update_context({
            ...     "current_file": "main.py",
            ...     "language": "python"
            ... })
        """

    def reset_context(
        self,
    ) -> None:
        """
        Reset context to initial state.

        Example:
            >>> context_coord.reset_context()
        """

    def get_token_count(
        self,
    ) -> int:
        """
        Get current token count.

        Returns:
            Number of tokens in context

        Example:
            >>> tokens = context_coord.get_token_count()
            >>> print(f"Using {tokens} tokens")
        """
```

**Compaction Strategies**:

| Strategy | Description | Speed | Information Loss |
|----------|-------------|-------|------------------|
| **TRUNCATION** | Remove oldest messages | Fast | High |
| **SUMMARIZATION** | Summarize using LLM | Medium | Low |
| **SEMANTIC** | Use embeddings to select important content | Slow | Very Low |
| **HYBRID** | Adaptive strategy based on context | Variable | Variable |

**Usage Example**:

```python
# Get context
context = orchestrator._context_coordinator.get_context()

# Update context
orchestrator._context_coordinator.update_context({
    "current_file": "main.py",
    "language": "python",
})

# Compact when threshold exceeded
if context.get_token_count() > context.max_tokens:
    compacted = await orchestrator._context_coordinator.compact_context(
        CompactionStrategy.SUMMARIZATION
    )
```

### ChatCoordinator

**Purpose**: Chat and streaming operations

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/chat_coordinator.py`

**Dependencies**: All coordinators

**Key Methods**:

```python
class ChatCoordinator:
    """Chat operations coordinator."""

    async def chat(
        self,
        prompt: str,
        context: Context,
    ) -> str:
        """
        Execute chat and return response.

        Args:
            prompt: Complete prompt including system message
            context: Conversation context

        Returns:
            Response string

        Example:
            >>> response = await chat_coord.chat(prompt, context)
            >>> print(response)
        """

    async def stream_chat(
        self,
        prompt: str,
        context: Context,
    ) -> AsyncIterator[str]:
        """
        Stream chat response.

        Args:
            prompt: Complete prompt including system message
            context: Conversation context

        Yields:
            Response chunks

        Example:
            >>> async for chunk in chat_coord.stream_chat(prompt, context):
            ...     print(chunk, end="")
        """

    async def chat_with_tools(
        self,
        prompt: str,
        context: Context,
    ) -> ChatResult:
        """
        Execute chat with tool calling.

        Args:
            prompt: Complete prompt
            context: Conversation context

        Returns:
            ChatResult with response and tool calls

        Example:
            >>> result = await chat_coord.chat_with_tools(prompt, context)
            >>> print(result.response)
            >>> for call in result.tool_calls:
            ...     print(f"Called: {call.name}")
        """
```

**Usage Example**:

```python
# Simple chat
response = await orchestrator._chat_coordinator.chat(prompt, context)

# Streaming chat
async for chunk in orchestrator._chat_coordinator.stream_chat(prompt, context):
    print(chunk, end="")

# Chat with tools
result = await orchestrator._chat_coordinator.chat_with_tools(prompt, context)
print(f"Response: {result.response}")
print(f"Tools called: {len(result.tool_calls)}")
```

### ToolCoordinator

**Purpose**: Tool execution coordination

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/tool_execution_coordinator.py`

**Dependencies**: ContextCoordinator, ToolRegistry

**Key Methods**:

```python
class ToolCoordinator:
    """Tool execution coordinator."""

    async def execute_tools(
        self,
        tool_calls: List[ToolCall],
    ) -> List[ToolCallResult]:
        """
        Execute tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool call results

        Example:
            >>> results = await tool_coord.execute_tools([
            ...     ToolCall(name="read_file", arguments={"path": "test.py"})
            ... ])
            >>> for result in results:
            ...     print(f"{result.tool}: {result.output}")
        """

    async def execute_tool(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> ToolCallResult:
        """
        Execute a single tool.

        Args:
            tool: Tool to execute
            arguments: Tool arguments

        Returns:
            Tool call result

        Example:
            >>> result = await tool_coord.execute_tool(
            ...     read_tool,
            ...     {"path": "test.py"}
            ... )
            >>> print(result.output)
        """

    async def select_tools(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[BaseTool]:
        """
        Select tools for execution.

        Args:
            query: Query describing what tools are needed
            context: Tool selection context

        Returns:
            List of selected tools

        Example:
            >>> context = AgentToolSelectionContext(max_tools=5)
            >>> tools = await tool_coord.select_tools(
            ...     "Read Python files",
            ...     context
            ... )
        """
```

**Usage Example**:

```python
# Execute tools
results = await orchestrator._tool_coordinator.execute_tools([
    ToolCall(name="read_file", arguments={"path": "main.py"}),
    ToolCall(name="search_code", arguments={"query": "class User"}),
])

# Select and execute
context = AgentToolSelectionContext(max_tools=3)
tools = await orchestrator._tool_coordinator.select_tools(
    "Find all test files",
    context,
)
results = await orchestrator._tool_coordinator.execute_tools([
    ToolCall(name=tool.name, arguments={})
    for tool in tools
])
```

### ProviderCoordinator

**Purpose**: Provider switching and health monitoring

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/provider_coordinator.py`

**Dependencies**: ProviderManager

**Key Methods**:

```python
class ProviderCoordinator:
    """Provider management coordinator."""

    def switch_provider(
        self,
        provider: BaseProvider,
    ) -> None:
        """
        Switch to different provider.

        Args:
            provider: New provider to use

        Example:
            >>> provider_coord.switch_provider(openai_provider)
        """

    def get_current_provider(
        self,
    ) -> BaseProvider:
        """
        Get current provider.

        Returns:
            Current provider instance

        Example:
            >>> provider = provider_coord.get_current_provider()
            >>> print(provider.name)
        """

    async def check_provider_health(
        self,
        provider: Optional[BaseProvider] = None,
    ) -> HealthStatus:
        """
        Check provider health.

        Args:
            provider: Provider to check (default: current)

        Returns:
            Health status

        Example:
            >>> health = await provider_coord.check_provider_health()
            >>> print(f"Status: {health.status}")
        """

    def list_available_providers(
        self,
    ) -> List[BaseProvider]:
        """
        List all available providers.

        Returns:
            List of provider instances

        Example:
            >>> providers = provider_coord.list_available_providers()
            >>> for p in providers:
            ...     print(f"{p.name}: {p.model}")
        """
```

**Usage Example**:

```python
# Switch provider
new_provider = OpenAIProvider(api_key="...")
orchestrator._provider_coordinator.switch_provider(new_provider)

# Check health
health = await orchestrator._provider_coordinator.check_provider_health()
if health.status != "healthy":
    print(f"Provider unhealthy: {health.message}")

# List providers
providers = orchestrator._provider_coordinator.list_available_providers()
```

### SessionCoordinator

**Purpose**: Session lifecycle management

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/session_coordinator.py`

**Dependencies**: MetricsCoordinator

**Key Methods**:

```python
class SessionCoordinator:
    """Session management coordinator."""

    def create_session(
        self,
        session_id: Optional[str] = None,
    ) -> Session:
        """
        Create new session.

        Args:
            session_id: Optional session ID (auto-generated if not provided)

        Returns:
            New session instance

        Example:
            >>> session = session_coord.create_session()
            >>> print(f"Session: {session.session_id}")
        """

    def get_session(
        self,
        session_id: str,
    ) -> Optional[Session]:
        """
        Get existing session.

        Args:
            session_id: Session ID

        Returns:
            Session or None if not found

        Example:
            >>> session = session_coord.get_session("abc123")
            >>> if session:
            ...     print(f"Found session: {session.session_id}")
        """

    def close_session(
        self,
        session_id: str,
    ) -> None:
        """
        Close session.

        Args:
            session_id: Session ID

        Example:
            >>> session_coord.close_session("abc123")
        """

    def list_sessions(
        self,
    ) -> List[Session]:
        """
        List all active sessions.

        Returns:
            List of sessions

        Example:
            >>> sessions = session_coord.list_sessions()
            >>> for session in sessions:
            ...     print(f"{session.session_id}: {session.created_at}")
        """
```

**Usage Example**:

```python
# Create session
session = orchestrator._session_coordinator.create_session()
print(f"Created session: {session.session_id}")

# Get session
session = orchestrator._session_coordinator.get_session(session.session_id)

# Close session
orchestrator._session_coordinator.close_session(session.session_id)
```

### MetricsCoordinator

**Purpose**: Metrics collection and export

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/metrics_coordinator.py`

**Dependencies**: None

**Key Methods**:

```python
class MetricsCoordinator:
    """Metrics collection coordinator."""

    def track_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Track metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags

        Example:
            >>> metrics_coord.track_metric(
            ...     "chat_latency",
            ...     0.5,
            ...     tags={"model": "gpt-4"}
            ... )
        """

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment counter.

        Args:
            name: Counter name
            value: Increment value (default: 1)
            tags: Optional tags

        Example:
            >>> metrics_coord.increment_counter(
            ...     "tool_executions",
            ...     tags={"tool": "read_file"}
            ... )
        """

    def record_timing(
        self,
        name: str,
        duration: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record timing.

        Args:
            name: Timing name
            duration: Duration in seconds
            tags: Optional tags

        Example:
            >>> metrics_coord.record_timing(
            ...     "chat_duration",
            ...     1.5,
            ...     tags={"stream": "true"}
            ... )
        """

    def export_metrics(
        self,
        format: str = "json",
    ) -> Union[str, Dict[str, Any]]:
        """
        Export metrics.

        Args:
            format: Export format (json, prometheus)

        Returns:
            Formatted metrics

        Example:
            >>> metrics = metrics_coord.export_metrics("json")
            >>> print(metrics)
        """
```

**Usage Example**:

```python
# Track metric
orchestrator._metrics_coordinator.track_metric(
    "tokens_used",
    1500,
    tags={"model": "gpt-4"}
)

# Increment counter
orchestrator._metrics_coordinator.increment_counter(
    "tool_calls",
    tags={"tool": "read_file"}
)

# Export metrics
metrics = orchestrator._metrics_coordinator.export_metrics("json")
```

### WorkflowCoordinator

**Purpose**: Workflow execution and management

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/coordinators/workflow_coordinator.py`

**Dependencies**: ChatCoordinator

**Key Methods**:

```python
class WorkflowCoordinator:
    """Workflow execution coordinator."""

    async def execute_workflow(
        self,
        workflow: StateGraph,
        input_data: Dict[str, Any],
    ) -> WorkflowResult:
        """
        Execute workflow.

        Args:
            workflow: StateGraph workflow
            input_data: Input data for workflow

        Returns:
            Workflow execution result

        Example:
            >>> result = await workflow_coord.execute_workflow(
            ...     my_workflow,
            ...     {"query": "refactor code"}
            ... )
            >>> print(result.output)
        """

    def validate_workflow(
        self,
        workflow: StateGraph,
    ) -> ValidationResult:
        """
        Validate workflow.

        Args:
            workflow: StateGraph to validate

        Returns:
            Validation result

        Example:
            >>> result = workflow_coord.validate_workflow(my_workflow)
            >>> if not result.is_valid:
            ...     print(f"Invalid: {result.errors}")
        """

    def list_workflows(
        self,
    ) -> List[StateGraph]:
        """
        List available workflows.

        Returns:
            List of registered workflows

        Example:
            >>> workflows = workflow_coord.list_workflows()
            >>> for wf in workflows:
            ...     print(wf.name)
        """
```

**Usage Example**:

```python
# Execute workflow
result = await orchestrator._workflow_coordinator.execute_workflow(
    refactor_workflow,
    {"file_path": "main.py", "goal": "improve readability"}
)

print(f"Workflow result: {result.output}")
print(f"Steps executed: {len(result.trace)}")
```

### Other Coordinators

| Coordinator | Purpose | File Location |
|-------------|---------|--------------|
| **ModeCoordinator** | Agent mode management | `coordinators/mode_coordinator.py` |
| **AnalyticsCoordinator** | Analytics tracking | `coordinators/analytics_coordinator.py` |
| **CheckpointCoordinator** | Checkpoint persistence | `coordinators/checkpoint_coordinator.py` |
| **SearchCoordinator** | Search operations | `coordinators/search_coordinator.py` |
| **MemoryCoordinator** | Memory management | `coordinators/memory_coordinator.py` |
| **MiddlewareCoordinator** | Middleware execution | `coordinators/middleware_coordinator.py` |
| **ValidationCoordinator** | Validation operations | `coordinators/validation_coordinator.py` |
| **ToolBudgetCoordinator** | Tool budget management | `coordinators/tool_budget_coordinator.py` |
| **FileContextCoordinator** | File context management | `coordinators/file_context_coordinator.py` |
| **IntelligentFeatureCoordinator** | Intelligent features | `coordinators/intelligent_feature_coordinator.py` |

---

## Adapters

### IntelligentPipelineAdapter

**Purpose**: Adapt results from tool pipeline to expected format

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/adapters/intelligent_pipeline_adapter.py`

**Key Methods**:

```python
class IntelligentPipelineAdapter:
    """Adapt pipeline results."""

    async def adapt_result(
        self,
        result: Any,
        target_format: str,
    ) -> Any:
        """
        Adapt result to target format.

        Args:
            result: Original result
            target_format: Target format (json, string, dict, etc.)

        Returns:
            Adapted result

        Example:
            >>> adapted = await adapter.adapt_result(
            ...     tool_result,
            ...     "json"
            ... )
        """

    async def adapt_stream(
        self,
        stream: AsyncIterator[Any],
    ) -> AsyncIterator[Any]:
        """
        Adapt streaming result.

        Args:
            stream: Original stream

        Yields:
            Adapted stream chunks

        Example:
            >>> async for chunk in adapter.adapt_stream(stream):
            ...     print(chunk)
        """
```

### CoordinatorAdapter

**Purpose**: Adapt legacy orchestrator calls to coordinator-based architecture

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/adapters/coordinator_adapter.py`

**Key Methods**:

```python
class CoordinatorAdapter:
    """Adapt legacy calls to coordinator architecture."""

    async def adapt_chat_call(
        self,
        orchestrator: AgentOrchestrator,
        message: str,
    ) -> str:
        """
        Adapt chat call to coordinator architecture.

        Args:
            orchestrator: AgentOrchestrator instance
            message: User message

        Returns:
            Chat response

        Example:
            >>> response = await adapter.adapt_chat_call(
            ...     orchestrator,
            ...     "Hello"
            ... )
        """

    async def adapt_tool_execution(
        self,
        orchestrator: AgentOrchestrator,
        tool_calls: List[ToolCall],
    ) -> List[ToolCallResult]:
        """
        Adapt tool execution to coordinator architecture.

        Args:
            orchestrator: AgentOrchestrator instance
            tool_calls: Tool calls to execute

        Returns:
            Tool call results

        Example:
            >>> results = await adapter.adapt_tool_execution(
            ...     orchestrator,
            ...     tool_calls
            ... )
        """
```

---

## Mixins

### ComponentAccessor

**Purpose**: Provide access to orchestrator components

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/mixins/component_accessor.py`

**Key Methods**:

```python
class ComponentAccessor:
    """Mixin for accessing orchestrator components."""

    def get_config_coordinator(self) -> ConfigCoordinator:
        """Get config coordinator."""

    def get_prompt_coordinator(self) -> PromptCoordinator:
        """Get prompt coordinator."""

    def get_context_coordinator(self) -> ContextCoordinator:
        """Get context coordinator."""

    # ... other getters
```

**Usage Example**:

```python
class MyComponent(ComponentAccessor):
    """Component with accessor mixin."""

    def __init__(self, orchestrator: AgentOrchestrator):
        super().__init__(orchestrator)

    async def do_work(self):
        """Use mixin methods."""
        config = self.get_config_coordinator().get_config()
        prompt = await self.get_prompt_coordinator().build_prompt("...")
        # ...
```

### StateDelegation

**Purpose**: Delegate state operations to StateCoordinator

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/mixins/state_delegation.py`

**Key Methods**:

```python
class StateDelegation:
    """Mixin for state delegation."""

    async def get_state(self) -> State:
        """Get current state."""

    async def update_state(self, updates: Dict[str, Any]) -> None:
        """Update state."""

    async def reset_state(self) -> None:
        """Reset state."""
```

**Usage Example**:

```python
class MyComponent(StateDelegation):
    """Component with state delegation."""

    async def process_request(self, request):
        """Update state based on request."""
        state = await self.get_state()
        # ... process ...
        await self.update_state({"last_request": request})
```

### LegacyAPI

**Purpose**: Maintain backward compatibility with legacy code

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/mixins/legacy_api.py`

**Key Methods**:

```python
class LegacyAPI:
    """Mixin for legacy API compatibility."""

    def legacy_method(self, *args, **kwargs) -> Any:
        """Legacy method implementation."""
```

**Usage Example**:

```python
class ModernComponent(LegacyAPI):
    """Modern component with legacy compatibility."""

    def new_method(self):
        """New implementation."""

    def legacy_method(self):
        """Legacy compatibility."""
        return self.new_method()
```

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

## Component Interactions

### Chat Flow

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant PromptCoord
    participant ContextCoord
    participant ChatCoord
    participant Provider

    User->>Orchestrator: chat("Hello")
    Orchestrator->>PromptCoord: build_prompt()
    PromptCoord->>ContextCoord: get_context()
    ContextCoord-->>PromptCoord: context
    PromptCoord-->>Orchestrator: prompt
    Orchestrator->>ChatCoord: chat(prompt, context)
    ChatCoord->>Provider: chat(messages)
    Provider-->>ChatCoord: response
    ChatCoord-->>Orchestrator: result
    Orchestrator-->>User: "Hi there!"
```

### Tool Execution Flow

```mermaid
sequenceDiagram
    participant ChatCoord
    participant ToolCoord
    participant ToolRegistry
    participant Tool
    participant EventBus

    ChatCoord->>ToolCoord: execute_tools(tool_calls)
    ToolCoord->>ToolRegistry: get_tool(name)
    ToolRegistry-->>ToolCoord: tool
    ToolCoord->>EventBus: publish("tool.start")
    ToolCoord->>Tool: execute(**args)
    Tool-->>ToolCoord: result
    ToolCoord->>EventBus: publish("tool.complete")
    ToolCoord-->>ChatCoord: results
```

---

## Extension Points

### Custom Coordinators

Create specialized coordinators:

```python
class MyCoordinator:
    """Custom coordinator."""

    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator

    async def do_specialized_work(self):
        """Do specialized work."""
        # Implementation
```

### Custom Prompt Contributors

Add prompt building blocks:

```python
class MyContributor:
    """Custom prompt contributor."""

    def get_contribution(self, context: Dict[str, Any]) -> str:
        """Get prompt contribution."""
        return "\nCustom instruction"

# Register
orchestrator._prompt_coordinator.add_contributor(MyContributor())
```

### Custom Tools

Create new tools:

```python
class MyTool(BaseTool):
    """Custom tool."""

    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Does something useful"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            }
        }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.LOW

    async def execute(self, **kwargs):
        # Implementation
        return result

# Register
tool_registry.register_tool(MyTool())
```

### Custom Event Backends

Implement event backend:

```python
class MyEventBackend:
    """Custom event backend."""

    async def publish(self, event: MessagingEvent) -> bool:
        # Implementation
        pass

    async def subscribe(self, pattern: str, handler: EventHandler):
        # Implementation
        pass

    async def connect(self):
        # Implementation
        pass

    async def disconnect(self):
        # Implementation
        pass
```

---

## Conclusion

This reference provides comprehensive documentation for all major Victor AI components. For more detailed information:

- See [ARCHITECTURE.md](./ARCHITECTURE.md) for architecture overview
- See [DESIGN_PATTERNS.md](./DESIGN_PATTERNS.md) for pattern documentation
- See [MIGRATION_GUIDES.md](./MIGRATION_GUIDES.md) for migration guidance

---

**Document Version**: 1.0
**Last Updated**: January 18, 2026
**Maintainers**: Victor AI Development Team
