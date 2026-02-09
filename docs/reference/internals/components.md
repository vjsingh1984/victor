# Component Overview

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Brief overview of key Victor AI components

---

## Overview

Victor AI is built from modular,
  loosely-coupled components that work together through well-defined protocols. This reference provides detailed
  documentation for each major component.

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

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 1 min
