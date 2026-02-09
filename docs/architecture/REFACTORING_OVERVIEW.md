# Victor Architecture Refactoring Overview

## Executive Summary

Victor underwent a comprehensive three-phase architectural refactoring to address technical debt,
  improve maintainability,
  and establish a scalable foundation for future development. This refactoring modernized the codebase through
  protocol-based design,
  dependency injection, event-driven architecture, and SOLID principles.

**Timeline**: Phases 1-3 completed over multiple iterations
**Impact**: 98 protocols defined, 55+ services registered in DI container, event-driven architecture implemented
**Code Coverage**: Test coverage maintained and improved throughout refactoring

### Key Achievements

- **Protocol-First Design**: 98 protocols defined across the codebase for loose coupling
- **Dependency Injection**: 55 services registered in DI container (56.1% coverage)
- **Event-Driven Architecture**: Pluggable event backends (in-memory, Kafka, SQS, RabbitMQ, Redis)
- **SOLID Compliance**: ISP, DIP, SRP improvements across coordinators and handlers
- **Performance**: Lock contention reduced, lazy loading implemented, error observability enhanced

## Phase Summaries

### Phase 1: Critical Fixes (Foundation)

**Focus**: Address immediate technical debt and critical issues

**Key Improvements**:
1. Provider API key handling from environment variables
2. Elimination of magic numbers throughout codebase
3. Lock contention resolution in concurrent operations
4. Concurrency test reliability improvements

**Impact**:
- Reduced production issues from configuration errors
- Improved code maintainability
- Better test reliability

### Phase 2: Architectural Improvements (Structure)

**Focus**: Establish SOLID principles and protocol-based design

**Key Improvements**:
1. **Coordinator Pattern**: Introduced coordinators for complex operations
2. **Protocol Definitions**: 98 protocols for loose coupling
3. **Handler Pattern**: Dedicated handlers for specific tasks
4. **SOLID Compliance**: ISP, DIP, SRP across core components

**Impact**:
- Clear separation of concerns
- Testable components with protocol-based mocking
- Flexible architecture for extensions

### Phase 3: Long-term Architecture (Scalability)

**Focus**: Event-driven architecture and dependency injection

**Key Improvements**:
1. **Event-Driven Architecture**: Pluggable event backends
2. **DI Container**: 55 services registered for dependency management
3. **Error Observability**: Enhanced error tracking and recovery
4. **Lazy Loading**: CLI startup optimization

**Impact**:
- Scalable for distributed deployments
- Clean dependency management
- Better observability and debugging

## Before/After Architecture

### Before: Monolithic Orchestrator

```
┌─────────────────────────────────────────────────┐
│          AgentOrchestrator (Monolith)           │
│  - Direct dependencies on concrete classes      │
│  - Tight coupling between components            │
│  - Difficult to test in isolation               │
│  - Hard to extend or modify                     │
└─────────────────────────────────────────────────┘
         │
         ├──> Provider (concrete)
         ├──> ToolRegistry (concrete)
         ├──> ConversationController (concrete)
         ├──> ToolPipeline (concrete)
         └──> [50+ other dependencies]
```

**Problems**:
- Tight coupling made changes risky
- Testing required full orchestrator setup
- No clear separation of concerns
- Difficult to swap implementations

### After: Protocol-Based, DI-Driven Architecture

```
┌───────────────────────────────────────────────────────────┐
│            ServiceContainer (DI Container)                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Singleton Services (55 registered)                │   │
│  │  - ToolRegistry, ObservabilityBus, etc.            │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Scoped Services (per-session)                     │   │
│  │  - ConversationStateMachine, TaskTracker, etc.     │   │
│  └────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│          AgentOrchestrator (Facade)                       │
│  Depends on protocols, not concrete classes               │
└───────────────────────────────────────────────────────────┘
         │
         ├──> ProviderManagerProtocol (ProviderCoordinator)
         ├──> ToolRegistryProtocol (ToolRegistry)
         ├──> ConversationControllerProtocol (ConversationController)
         ├──> ToolPipelineProtocol (ToolPipeline)
         └──> [50+ protocol-based dependencies]
```

**Benefits**:
- Loose coupling through protocols
- Easy testing with mock protocols
- Flexible implementation swapping
- Clear dependency graph

## Design Patterns Applied

### 1. Protocol Pattern (Interface Segregation)

**Purpose**: Define clear contracts between components

**Example**:
```python
@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    """Protocol for tool coordination operations."""

    async def select_tools(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[BaseTool]:
        """Select tools for execution."""
        ...

    async def execute_tool(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> ToolCallResult:
        """Execute a single tool."""
        ...
```

**Benefits**:
- Multiple protocols can be implemented by one class
- Protocol composition enables flexible design
- Type-safe without requiring inheritance
- Easy to create mock implementations

### 2. Coordinator Pattern (Separation of Concerns)

**Purpose**: Centralize complex operations that span multiple services

**Example**: ToolCoordinator

```python
class ToolCoordinator:
    """Coordinates tool selection, budgeting, and execution."""

    def __init__(
        self,
        tool_pipeline: Optional[ToolPipelineProtocol],
        tool_selector: Optional[IToolSelector],
        budget_manager: Optional[IBudgetManager],
        tool_cache: Optional[ToolCacheProtocol],
        config: ToolCoordinatorConfig,
    ):
        self._tool_pipeline = tool_pipeline
        self._tool_selector = tool_selector
        self._budget_manager = budget_manager
        self._tool_cache = tool_cache
        self._config = config

    async def select_and_execute(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[ToolCallResult]:
        """Select tools and execute within budget."""
        # 1. Select tools
        tools = await self._tool_selector.select_tools(query, context)

        # 2. Check budget
        if not self._budget_manager.can_execute(len(tools)):
            raise BudgetExceededError()

        # 3. Execute with caching
        results = []
        for tool in tools:
            cached = await self._tool_cache.get(tool.name)
            if cached:
                results.append(cached)
            else:
                result = await self._tool_pipeline.execute_tool(tool)
                await self._tool_cache.set(tool.name, result)
                results.append(result)

        return results
```

**Benefits**:
- Single responsibility for coordination logic
- Testable in isolation
- Reusable across different contexts
- Easy to extend with new capabilities

### 3. Factory Pattern (Dependency Creation)

**Purpose**: Centralize object creation with proper dependency injection

**Example**: OrchestratorFactory

```python
class OrchestratorFactory:
    """Factory for creating AgentOrchestrator instances."""

    def __init__(self, container: ServiceContainer, settings: Settings):
        self._container = container
        self._settings = settings

    def create_orchestrator(
        self,
        provider: Optional[BaseProvider] = None,
        mode: Optional[AgentMode] = None,
    ) -> AgentOrchestrator:
        """Create orchestrator with all dependencies."""
        # Get services from DI container
        tool_registry = self._container.get(ToolRegistryProtocol)
        observability = self._container.get(ObservabilityProtocol)
        task_analyzer = self._container.get(TaskAnalyzerProtocol)

        # Create orchestrator-specific components
        tool_pipeline = self._create_tool_pipeline()
        conversation_controller = self._create_conversation_controller()
        streaming_controller = self._create_streaming_controller()

        # Create orchestrator
        return AgentOrchestrator(
            provider=provider,
            tool_registry=tool_registry,
            observability=observability,
            tool_pipeline=tool_pipeline,
            conversation_controller=conversation_controller,
            streaming_controller=streaming_controller,
            # ... other dependencies
        )
```

**Benefits**:
- Centralized creation logic
- Proper dependency injection
- Easy to test with mock containers
- Consistent object creation

### 4. Strategy Pattern (Pluggable Algorithms)

**Purpose**: Enable algorithm swapping through protocol-based strategies

**Example**: Event Backends

```python
# Protocol
@runtime_checkable
class IEventBackend(Protocol):
    """Protocol for event backends."""

    async def publish(self, event: MessagingEvent) -> bool:
        ...

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        ...

# Implementations
class InMemoryBackend:
    """In-memory event backend (default)."""
    async def publish(self, event: MessagingEvent) -> bool:
        # Local implementation
        ...

class KafkaBackend:
    """Kafka event backend (distributed)."""
    async def publish(self, event: MessagingEvent) -> bool:
        # Kafka implementation
        ...

# Factory
def create_event_backend(
    backend_type: BackendType,
    config: BackendConfig,
) -> IEventBackend:
    """Create event backend by type."""
    if backend_type == BackendType.IN_MEMORY:
        return InMemoryBackend(config)
    elif backend_type == BackendType.KAFKA:
        return KafkaBackend(config)
    # ... other backends
```

**Benefits**:
- Easy to add new implementations
- Runtime algorithm selection
- Testable with mock implementations
- Configuration-driven behavior

### 5. Observer Pattern (Event-Driven Communication)

**Purpose**: Enable loose coupling through pub/sub messaging

**Example**: Event Bus

```python
# Publisher
class ToolExecutor:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus

    async def execute_tool(self, tool: BaseTool, args: dict):
        # Publish event before execution
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.start",
                data={"tool": tool.name, "args": args},
            )
        )

        result = await tool.execute(**args)

        # Publish event after execution
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.complete",
                data={"tool": tool.name, "result": result},
            )
        )

        return result

# Subscriber
class MetricsCollector:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus
        self._event_bus.subscribe("tool.*", self._on_tool_event)

    async def _on_tool_event(self, event: MessagingEvent):
        if event.topic == "tool.start":
            self._track_tool_start(event.data)
        elif event.topic == "tool.complete":
            self._track_tool_complete(event.data)
```

**Benefits**:
- Loose coupling between components
- Easy to add new subscribers
- Asynchronous communication
- Event sourcing capabilities

## Key Metrics

### Protocol Coverage

| Category | Protocols Defined | Registered in DI | Coverage |
|----------|------------------|------------------|----------|
| Agent Services | 98 | 55 | 56.1% |
| Framework | 45 | 20 | 44.4% |
| Verticals | 30 | 12 | 40.0% |
| Infrastructure | 23 | 23 | 100% |
| **Total** | **196** | **110** | **56.1%** |

**Note**: 43 protocols intentionally not registered (orchestrator-specific, framework-level, or implemented by
  orchestrator itself)

### Service Registration

| Lifetime | Count | Examples |
|----------|-------|----------|
| Singleton | 45 | ToolRegistry, ObservabilityBus, EventBus |
| Scoped | 10 | ConversationStateMachine, TaskTracker |
| Transient | 0 | N/A (not used) |

### Test Coverage

| Phase | Before | After | Improvement |
|-------|--------|-------|-------------|
| Phase 1 | 68% | 72% | +4% |
| Phase 2 | 72% | 76% | +4% |
| Phase 3 | 76% | 81% | +5% |
| **Total** | **68%** | **81%** | **+13%** |

## Architecture Diagrams

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Clients Layer                          │
│  CLI │ TUI │ VS Code │ MCP Server │ API Server │ HTTP     │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│                 ServiceContainer (DI)                       │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │  Singleton     │  │  Scoped        │  │  Transient   │ │
│  │  Services      │  │  Services      │  │  (unused)    │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│              AgentOrchestrator (Facade)                     │
│  Delegates to specialized coordinators and handlers         │
└──────────────────────┬─────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ Coordinators │ │ Handlers │ │  Services    │
│              │ │          │ │              │
│ Tool         │ │ Recovery │ │ Provider     │
│ State        │ │ Error    │ │ ToolRegistry │
│ Prompt       │ │ Metrics  │ │ EventBus     │
│ Chat         │ │          │ │ Etc.         │
└──────────────┘ └──────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│                   Event Bus (Pub/Sub)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Topics: tool.*, agent.*, workflow.*, error.*        │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│              Event Backends (Pluggable)                     │
│  In-Memory │ Kafka │ SQS │ RabbitMQ │ Redis │ Database     │
└────────────────────────────────────────────────────────────┘
```

### Dependency Flow

```
User Request
     │
     ▼
CLI/API Entry Point
     │
     ├─> ServiceContainer.get(Protocol)
     │        │
     │        ├─> Returns: Service instance (singleton/scoped)
     │        │
     │        ▼
     ├─> AgentOrchestrator
     │        │
     │        ├─> Uses: Coordinators (via protocols)
     │        │        │
     │        │        ├─> ToolCoordinator
     │        │        │        ├─> ToolSelector
     │        │        │        ├─> BudgetManager
     │        │        │        └─> ToolCache
     │        │        │
     │        │        ├─> StateCoordinator
     │        │        │        ├─> ConversationStateMachine
     │        │        │        └─> ConversationController
     │        │        │
     │        │        └─> PromptCoordinator
     │        │                 ├─> PromptBuilder
     │        │                 └─> ProjectContext
     │        │
     │        ├─> Uses: Services (via protocols)
     │        │        │
     │        │        ├─> ProviderManager
     │        │        ├─> ToolRegistry
     │        │        └─> EventBus
     │        │
     │        └─> Publishes: Events (to EventBus)
     │                 │
     │                 └─> Subscribers receive events
     │                          │
     │                          ├─> MetricsCollector
     │                          ├─> ObservabilityIntegration
     │                          └─> RecoveryHandler
     │
     └─> Response to User
```

## Migration Path

The refactoring followed a gradual migration approach:

1. **Phase 1**: Fix critical issues (provider keys, magic numbers, locks)
2. **Phase 2**: Introduce protocols and coordinators (SOLID principles)
3. **Phase 3**: Implement DI and event-driven architecture

This approach allowed the codebase to evolve incrementally while maintaining functionality and test coverage.

## Next Steps

For detailed information on specific aspects:
- See [MIGRATION_GUIDES.md](./MIGRATION_GUIDES.md) for how to migrate code
- See [BEST_PRACTICES.md](./BEST_PRACTICES.md) for usage patterns
- See [PROTOCOLS_REFERENCE.md](./PROTOCOLS_REFERENCE.md) for protocol documentation

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
