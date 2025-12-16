# ADR 0004: Phase 10 Dependency Injection Migration

Date: 2025-12-16

## Status

Accepted

## Context

The AgentOrchestrator in Victor had grown to over 7,000 lines with direct instantiation of 18+ service components, creating tight coupling and testability challenges:

### Problems with Direct Instantiation

1. **Tight Coupling**: Orchestrator directly imported and instantiated concrete classes
   ```python
   from victor.tools.base import ToolRegistry
   from victor.agent.task_analyzer import TaskAnalyzer
   from victor.agent.usage_analytics import UsageAnalytics

   self.tool_registry = ToolRegistry()
   self.task_analyzer = TaskAnalyzer(...)
   self._usage_analytics = UsageAnalytics.get_instance(...)
   ```

2. **Testing Difficulty**: Tests required extensive mocking and patching
   ```python
   @patch('victor.agent.orchestrator.TaskAnalyzer')
   @patch('victor.agent.orchestrator.ToolRegistry')
   @patch('victor.agent.orchestrator.UsageAnalytics')
   def test_orchestrator(mock_analytics, mock_registry, mock_analyzer):
       # Complex test setup with multiple patches
       ...
   ```

3. **Singleton Proliferation**: Multiple singleton patterns (instance variables, class methods, module globals)
   ```python
   # Pattern 1: Class method singleton
   UsageAnalytics.get_instance(config)

   # Pattern 2: Module-level singleton
   from victor.agent.task_analyzer import get_task_analyzer

   # Pattern 3: Instance singleton
   if not hasattr(self, '_task_analyzer'):
       self._task_analyzer = ...
   ```

4. **Service Lifecycle Confusion**: Unclear which services are singletons vs per-session
5. **Configuration Sprawl**: Service configuration scattered across orchestrator init
6. **Substitution Difficulty**: Replacing implementations required forking orchestrator code

### Alternative Approaches Considered

- **Service Locator Pattern**: Global registry for service lookup
  - **Con**: Hidden dependencies, harder to test
  - **Con**: Runtime errors if service not registered

- **Factory Pattern**: Factory methods for each service
  - **Con**: Still requires orchestrator to know concrete types
  - **Con**: Factories become complex with many dependencies

- **Constructor Injection Only**: Pass all services via __init__
  - **Con**: 18+ constructor parameters unwieldy
  - **Con**: Breaks backward compatibility

## Decision

We implement **Incremental Dependency Injection Migration** using a DI container with protocol-based service contracts. The migration follows a phased approach over multiple phases (Phase 10 being the latest).

### Architecture

#### Three-Component System

```
┌────────────────────────────────────────────────────────────┐
│  SERVICE PROVIDER (victor/agent/service_provider.py)      │
│  Registers all services with DI container                 │
├────────────────────────────────────────────────────────────┤
│  - OrchestratorServiceProvider class                       │
│  - register_singleton_services()                           │
│  - register_scoped_services()                              │
│  - Factory methods for service creation                    │
└──────────────┬─────────────────────────────────────────────┘
               │ Registers in
               ▼
┌────────────────────────────────────────────────────────────┐
│  DI CONTAINER (victor/core/container.py)                   │
│  Manages service lifecycles and resolution                 │
├────────────────────────────────────────────────────────────┤
│  - ServiceContainer class                                  │
│  - ServiceLifetime enum (SINGLETON, SCOPED, TRANSIENT)    │
│  - ServiceScope for per-session services                   │
│  - Type-safe resolution with protocols                     │
└──────────────┬─────────────────────────────────────────────┘
               │ Injected into
               ▼
┌────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (victor/agent/orchestrator.py)               │
│  Consumes services via protocol types                      │
├────────────────────────────────────────────────────────────┤
│  - Accepts optional ServiceContainer in __init__           │
│  - Uses get_optional() for DI resolution                   │
│  - Falls back to direct instantiation if not in container │
└────────────────────────────────────────────────────────────┘
```

### Service Lifetimes

We define three service lifetimes:

#### SINGLETON (Application Lifetime)
Shared across all orchestrator sessions:
- ToolRegistry
- ObservabilityIntegration
- TaskAnalyzer
- IntentClassifier
- ComplexityClassifier
- ActionAuthorizer
- SearchRouter
- ResponseSanitizer
- ArgumentNormalizer
- ProjectContext
- RecoveryHandler
- CodeExecutionManager
- WorkflowRegistry
- UsageAnalytics
- ToolSequenceTracker

#### SCOPED (Per Session)
Fresh instance for each orchestrator session:
- ConversationStateMachine
- UnifiedTaskTracker
- MessageHistory

#### TRANSIENT (Per Request)
Created fresh every time requested (not currently used)

### Migration Pattern: DI with Fallback

All migrations use a safe "DI with fallback" pattern:

```python
# Step 1: Import protocol (not concrete class)
from victor.agent.protocols import ComponentProtocol

# Step 2: Try DI container resolution
self.component = self._container.get_optional(ComponentProtocol)

# Step 3: Fallback to direct instantiation
if self.component is None:
    self.component = ConcreteComponent(...)
```

This ensures:
- **Zero Breaking Changes**: Existing code without DI container continues working
- **Testability**: Tests can inject mocks via container
- **Flexibility**: Can use DI when beneficial, direct instantiation when simpler

### Service Provider Pattern

`OrchestratorServiceProvider` centralizes service registration:

```python
class OrchestratorServiceProvider:
    def __init__(self, settings: Settings):
        self._settings = settings

    def register_services(self, container: ServiceContainer) -> None:
        """Register all orchestrator services."""
        self.register_singleton_services(container)
        self.register_scoped_services(container)

    def register_singleton_services(self, container: ServiceContainer) -> None:
        """Register application-lifetime services."""
        # ToolRegistry
        container.register(
            ToolRegistryProtocol,
            lambda c: ToolRegistry(),
            ServiceLifetime.SINGLETON,
        )

        # TaskAnalyzer
        container.register(
            TaskAnalyzerProtocol,
            lambda c: get_task_analyzer(),
            ServiceLifetime.SINGLETON,
        )

        # ... 13 more singleton services
```

### Bootstrap Pattern

Application bootstrap wires everything together:

```python
# In application startup (e.g., cli.py)
from victor.core.container import ServiceContainer
from victor.agent.service_provider import configure_orchestrator_services

# Create container and register services
container = ServiceContainer()
configure_orchestrator_services(container, settings)

# Create orchestrator with DI
orchestrator = AgentOrchestrator(
    provider=provider,
    settings=settings,
    container=container,  # Optional: uses DI if provided
)
```

### Key Design Principles

1. **Incremental Migration**: Migrate high-value services first, not all at once
2. **Backward Compatibility**: Always provide fallback to direct instantiation
3. **Protocol-Based**: Depend on protocols, not concrete implementations
4. **Centralized Configuration**: Single place (service_provider.py) to see all services
5. **Explicit Lifecycles**: Clear distinction between singleton, scoped, transient
6. **Type Safety**: Full type checking via protocols

## Consequences

### Positive Consequences

- **Reduced Coupling**: Orchestrator depends on protocols, not concrete classes
- **Improved Testability**: Inject mock implementations without patching
- **Centralized Service Config**: All service creation in one file
- **Clear Lifecycles**: Explicit singleton vs per-session distinction
- **Easy Substitution**: Swap implementations by changing container registration
- **Better Testing Isolation**: Test services independently
- **Backward Compatible**: Existing code continues working
- **Progressive Enhancement**: Adopt DI incrementally

### Negative Consequences

- **Indirection**: One more layer (container) to understand
- **Initialization Complexity**: Need to bootstrap container correctly
- **Memory Overhead**: Container maintains service instances (~5KB)
- **Debugging Complexity**: Need to check both DI and fallback paths

### Risks and Mitigations

- **Risk**: Developers might mix DI and direct instantiation inconsistently
  - **Mitigation**: Linting rules enforce DI usage for migrated services
  - **Mitigation**: Code review guidelines

- **Risk**: Container could be in inconsistent state if partially configured
  - **Mitigation**: Use service provider pattern for atomic registration
  - **Mitigation**: Validate container state at startup

- **Risk**: Scoped services might leak across sessions
  - **Mitigation**: ServiceScope manages cleanup via context manager
  - **Mitigation**: Tests verify service isolation

## Implementation Notes

### Migration Phases

Phase 10 was completed in three sub-phases:

**Sub-Phase 10.1** (Initial DI Infrastructure):
- Created ServiceContainer in victor/core/container.py
- Added 19 protocols to victor/agent/protocols.py
- Created OrchestratorServiceProvider
- Migrated 10 singleton services

**Sub-Phase 10.2** (Scoped Services):
- Added scoped service support
- Migrated ConversationStateMachine, UnifiedTaskTracker, MessageHistory
- Implemented ServiceScope for per-session isolation

**Sub-Phase 10.3** (Final Services):
- Added 6 new protocols (CodeExecutionManager, WorkflowRegistry, etc.)
- Migrated 4 more singleton services
- Fixed singleton registration bugs
- **Result**: 18 components now using DI

### Components Intentionally NOT Migrated

These remain directly instantiated due to tight coupling with orchestrator:
- **ConversationController**: Requires orchestrator callbacks
- **ToolPipeline**: Requires orchestrator callbacks
- **StreamingController**: Requires orchestrator callbacks
- **StreamingChatHandler**: Circular dependency on orchestrator
- **ContextCompactor**: Depends on ConversationController
- **ToolOutputFormatter**: Depends on ContextCompactor

These are "child components" properly scoped to orchestrator instance.

### Testing Pattern

```python
# Test with mock services
def test_orchestrator_with_mocks():
    container = ServiceContainer()

    # Inject mocks
    mock_registry = MagicMock(spec=ToolRegistryProtocol)
    container.register_instance(ToolRegistryProtocol, mock_registry)

    mock_analyzer = MagicMock(spec=TaskAnalyzerProtocol)
    container.register_instance(TaskAnalyzerProtocol, mock_analyzer)

    # Create orchestrator with mocked dependencies
    orchestrator = AgentOrchestrator(
        provider=provider,
        settings=settings,
        container=container,
    )

    # Test behavior without patching
    assert orchestrator.tool_registry is mock_registry
    assert orchestrator.task_analyzer is mock_analyzer
```

### Service Registration Example

```python
# In OrchestratorServiceProvider
def register_singleton_services(self, container: ServiceContainer) -> None:
    """Register singleton services."""

    # UsageAnalytics - cross-session analytics
    container.register(
        UsageAnalyticsProtocol,
        lambda c: self._create_usage_analytics(),
        ServiceLifetime.SINGLETON,
    )

def _create_usage_analytics(self) -> Any:
    """Factory method for UsageAnalytics."""
    from victor.agent.usage_analytics import UsageAnalytics, AnalyticsConfig

    return UsageAnalytics.get_instance(
        AnalyticsConfig(
            cache_dir=Path(self._settings.cache_dir),
            enable_prometheus_export=self._settings.enable_prometheus_export,
        )
    )
```

### Key Files

**DI Infrastructure**:
- `victor/core/container.py` - ServiceContainer, ServiceScope, lifecycles (538 lines)
- `victor/agent/service_provider.py` - OrchestratorServiceProvider (605 lines)
- `victor/agent/protocols.py` - 25 service protocols (805 lines)

**Consumers**:
- `victor/agent/orchestrator.py` - Uses DI with fallback for 18 services
- Tests throughout codebase inject mock services

### Performance Characteristics

- **Container Initialization**: ~5ms to register 18 services
- **Service Resolution**: O(1) dictionary lookup, ~0.01ms per get()
- **Memory Overhead**: ~5KB for container + service instances
- **Scoped Service Cleanup**: ~0.1ms per scope disposal

### Migration Success Metrics

**Phase 10 Results**:
- ✅ 18 components migrated to DI
- ✅ 25 protocols defined
- ✅ 0 breaking changes
- ✅ 7565/7566 tests passing (99.99%)
- ✅ Backward compatible fallback for all services

## References

- Related: [ADR-0001: Protocol-First Architecture](0001-protocol-first-architecture.md)
- Implementation: victor/core/container.py, victor/agent/service_provider.py
- Summary: PHASE_10_DI_MIGRATION_SUMMARY.md
- ROADMAP.md: P1.2 - Migrate to DI Container
- CLAUDE.md: Search for "Dependency Injection" section
