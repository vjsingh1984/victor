# Victor AI: Migration Guides

**Version**: 0.5.0
**Last Updated**: January 18, 2026
**Audience**: Developers, Architects

---

## Table of Contents

1. [Overview](#overview)
2. [Migration to Coordinators](#migration-to-coordinators)
3. [Migration to Protocol-Based Verticals](#migration-to-protocol-based-verticals)
4. [Migration to Vertical Template System](#migration-to-vertical-template-system)
5. [Migration to Event-Driven Architecture](#migration-to-event-driven-architecture)
6. [Migration to Dependency Injection](#migration-to-dependency-injection)
7. [Common Migration Scenarios](#common-migration-scenarios)
8. [Testing Migrated Code](#testing-migrated-code)
9. [Rollback Strategies](#rollback-strategies)

---

## Overview

### What Changed in Victor 0.5.0

Victor 0.5.0 introduces major architectural improvements:

- **Protocol-Based Design**: 98 protocols for loose coupling
- **Dependency Injection**: ServiceContainer with 55+ services
- **Event-Driven Architecture**: 5 pluggable event backends
- **Coordinator Pattern**: 20 specialized coordinators
- **Vertical Template System**: YAML-first configuration
- **Universal Registry**: Unified entity management

### Migration Philosophy

**Key Principles**:
1. **Backward Compatibility**: All existing code continues to work
2. **Incremental Migration**: Migrate gradually, component by component
3. **Clear Migration Paths**: Well-documented steps for each scenario
4. **Testing**: Comprehensive test coverage for migrations
5. **No Breaking Changes**: Deprecated but not removed

### Migration Levels

| Level | Effort | Impact | Recommended |
|-------|--------|--------|-------------|
| **Level 1**: Continue using existing code | None | None | For stable code |
| **Level 2**: Use new features in new code | Low | Medium | For new development |
| **Level 3**: Migrate existing code gradually | Medium | High | For active development |
| **Level 4**: Full migration | High | Very High | For major refactors |

---

## Migration to Coordinators

### Understanding Coordinators

Coordinators encapsulate complex operations that span multiple services:

**Application Layer** (Victor-specific):
- ChatCoordinator, ToolCoordinator, StateCoordinator
- ProviderCoordinator, SessionCoordinator, etc.

**Framework Layer** (Domain-agnostic):
- YAMLWorkflowCoordinator, GraphExecutionCoordinator
- HITLCoordinator, CacheCoordinator

### Scenario: Migrating from Orchestrator to Coordinators

#### Before: Direct Orchestrator Usage

```python
from victor.agent.orchestrator import AgentOrchestrator

class MyComponent:
    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator

    async def do_tool_work(self, query: str):
        # Direct dependency on orchestrator
        tools = await self._orchestrator.select_tools(query)
        results = await self._orchestrator.execute_tools(tools)
        return results
```

**Problems**:
- Tight coupling to orchestrator
- Hard to test (need full orchestrator)
- Violates SRP (component does too much)

#### After: Using Coordinators

```python
from victor.agent.protocols import ToolCoordinatorProtocol

class MyComponent:
    def __init__(self, tool_coordinator: ToolCoordinatorProtocol):
        self._tool_coordinator = tool_coordinator

    async def do_tool_work(self, query: str):
        # Depends only on tool coordination
        context = AgentToolSelectionContext(max_tools=5)
        results = await self._tool_coordinator.select_and_execute(
            query=query,
            context=context,
        )
        return results
```

**Benefits**:
- Loose coupling (depends on protocol)
- Easy to test (mock protocol)
- Single responsibility

### Step-by-Step Migration

#### Step 1: Identify Coordinator Dependencies

**What coordination does your code need?**

| Your Code Does | Use Coordinator |
|----------------|-----------------|
| Tool selection/execution | ToolCoordinator |
| Chat/streaming | ChatCoordinator |
| State management | StateCoordinator |
| Provider switching | ProviderCoordinator |
| Prompt building | PromptCoordinator |

#### Step 2: Update Dependencies

```python
# Before
from victor.agent.orchestrator import AgentOrchestrator

class MyComponent:
    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator

# After
from victor.agent.protocols import ToolCoordinatorProtocol

class MyComponent:
    def __init__(self, tool_coordinator: ToolCoordinatorProtocol):
        self._tool_coordinator = tool_coordinator
```

#### Step 3: Update Method Calls

```python
# Before
tools = await self._orchestrator.select_tools(query)

# After
from victor.agent.tool_selection import AgentToolSelectionContext

context = AgentToolSelectionContext(
    max_tools=5,
    allowed_tools=["read", "write"],
)
tools = await self._tool_coordinator.select_tools(query, context)
```

#### Step 4: Update Tests

```python
# Before: Complex orchestrator mock
def test_my_component():
    mock_orchestrator = Mock(spec=AgentOrchestrator)
    mock_orchestrator.select_tools.return_value = []
    component = MyComponent(mock_orchestrator)

# After: Simple protocol mock
def test_my_component():
    mock_coordinator = Mock(spec=ToolCoordinatorProtocol)
    mock_coordinator.select_tools.return_value = []
    component = MyComponent(mock_coordinator)
```

### Complete Example: Migrating Tool Usage

#### Before Code

```python
class CodeAnalyzer:
    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator

    async def analyze_file(self, filepath: str):
        # Tight coupling to orchestrator
        content = await self._orchestrator.read_file(filepath)
        ast = await self._orchestrator.parse_ast(content)
        issues = await self._orchestrator.check_quality(ast)
        return issues
```

#### After Code

```python
from victor.agent.protocols import ToolCoordinatorProtocol

class CodeAnalyzer:
    def __init__(self, tool_coordinator: ToolCoordinatorProtocol):
        self._tool_coordinator = tool_coordinator

    async def analyze_file(self, filepath: str):
        # Use tool coordinator
        context = AgentToolSelectionContext(
            max_tools=3,
            allowed_tools=["read_file", "parse_ast", "check_quality"],
        )

        # Single coordinated call
        results = await self._tool_coordinator.select_and_execute(
            query=f"Analyze {filepath}",
            context=context,
        )

        return results
```

---

## Migration to Protocol-Based Verticals

### Understanding ISP Compliance

The Interface Segregation Principle (ISP) states that clients should not depend on interfaces they don't use. In Victor, verticals now register only the protocols they implement.

### Scenario: Minimal Vertical Migration

#### Before: Legacy Vertical

```python
from victor.core.verticals import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"
    description = "My custom vertical"

    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "You are a helpful assistant..."

    # Inherits ALL extension methods (20+)
    # Even if most return None or empty defaults
```

#### After: ISP-Compliant Vertical

```python
from victor.core.verticals import VerticalBase
from victor.core.verticals.protocols.providers import (
    ToolProvider,
    PromptContributorProvider,
)

class MyVertical(VerticalBase):
    name = "my_vertical"
    description = "My custom vertical"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register ONLY implemented protocols
        cls.register_protocol(ToolProvider)
        cls.register_protocol(PromptContributorProvider)

    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "You are a helpful assistant..."

    @classmethod
    def get_task_type_hints(cls):
        return {
            "edit": {
                "hint": "Read file first, then modify",
                "tool_budget": 10,
            }
        }
```

### Step-by-Step Migration

#### Step 1: Identify Implemented Protocols

Check which extension methods your vertical overrides:

```python
# Check each method:
# - get_tools() → ToolProvider
# - get_task_type_hints() → PromptContributorProvider
# - get_safety_extension() → SafetyProvider
# - get_middleware() → MiddlewareProvider
# - get_workflow_provider() → WorkflowProvider
# - get_team_spec_provider() → TeamProvider
# - get_rl_config_provider() → RLProvider
# - get_handlers() → HandlerProvider
```

#### Step 2: Add Protocol Registration

```python
from victor.core.verticals.protocols.providers import (
    ToolProvider,
    PromptContributorProvider,
)

class MyVertical(VerticalBase):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register implemented protocols
        cls.register_protocol(ToolProvider)
        cls.register_protocol(PromptContributorProvider)
```

#### Step 3: Verify Protocol Conformance

```python
# Test protocol conformance
from victor.core.verticals.protocols.providers import ToolProvider

assert MyVertical.implements_protocol(ToolProvider)
print("MyVertical implements ToolProvider: ✓")

# List all implemented protocols
protocols = MyVertical.list_implemented_protocols()
print(f"Implemented protocols: {[p.__name__ for p in protocols]}")
```

#### Step 4: Update Framework Code

```python
# Before: Check for method presence
if hasattr(vertical, 'get_tools'):
    tools = vertical.get_tools()

# After: Check protocol conformance
from victor.core.verticals.protocols.providers import ToolProvider

if isinstance(vertical, ToolProvider):
    tools = vertical.get_tools()
```

### Complete Example: Research Vertical Migration

See [ISP_MIGRATION_GUIDE.md](../ISP_MIGRATION_GUIDE.md) for complete Research vertical migration example.

---

## Migration to Vertical Template System

### Understanding Template System

The vertical template system provides YAML-first configuration with automatic code generation:

**Before**: 500-800 lines of manual Python code
**After**: 50-100 lines of YAML template
**Reduction**: 65-70% code duplication eliminated

### Scenario: Creating New Vertical

#### Before: Manual Vertical Creation

```python
# victor/my_vertical/assistant.py
from victor.core.verticals import VerticalBase

class MyVerticalAssistant(VerticalBase):
    name = "my_vertical"
    description = "My custom vertical"

    @classmethod
    def get_tools(cls):
        return ["read", "write", "search"]

    @classmethod
    def get_system_prompt(cls):
        return """You are a helpful assistant..."""

    # ... 500+ more lines of boilerplate
```

#### After: Template-Based Creation

**Step 1: Create YAML Template**

```yaml
# victor/config/templates/my_vertical.yaml
metadata:
  name: "my_vertical"
  description: "My custom vertical"
  version: "0.5.0"
  author: "Your Name"

tools:
  - name: read
    cost_tier: FREE
    enabled: true
  - name: write
    cost_tier: FREE
    enabled: true
  - name: search
    cost_tier: LOW
    enabled: true

prompts:
  system: |
    You are a helpful assistant for {vertical_name}.
    You have access to tools: {tools}.

stages:
  - name: INITIAL
    goal: "Understand the user's request"
    tool_budget: 5
  - name: EXECUTION
    goal: "Execute the requested task"
    tool_budget: 10
  - name: COMPLETION
    goal: "Provide final response"
    tool_budget: 0
```

**Step 2: Generate Vertical**

```bash
python scripts/generate_vertical.py \
  --template victor/config/templates/my_vertical.yaml \
  --output victor/my_vertical
```

**Step 3: Use Generated Vertical**

```python
from victor.my_vertical import MyVerticalAssistant

config = MyVerticalAssistant.get_config()
agent = await Agent.create(
    tools=config.tools,
    vertical=MyVerticalAssistant,
)
```

### Step-by-Step Migration

#### Step 1: Create Template

```bash
# Copy base template
cp victor/config/templates/base_vertical_template.yaml my_vertical.yaml

# Edit my_vertical.yaml
```

#### Step 2: Validate Template

```bash
python scripts/generate_vertical.py \
  --validate my_vertical.yaml \
  --output /tmp/test
```

#### Step 3: Generate Vertical

```bash
python scripts/generate_vertical.py \
  --template my_vertical.yaml \
  --output victor/my_vertical
```

#### Step 4: Test Generated Code

```python
# Test imports
from victor.my_vertical import MyVerticalAssistant

# Test configuration
config = MyVerticalAssistant.get_config()
assert len(config.tools) > 0

# Test agent creation
agent = await Agent.create(vertical=MyVerticalAssistant)
```

### Complete Example: Security Vertical

See [vertical_template_guide.md](../vertical_template_guide.md) for complete example.

---

## Migration to Event-Driven Architecture

### Understanding Event-Driven Design

Victor 0.5.0 supports 5 pluggable event backends:
- In-Memory (default)
- Kafka (distributed)
- SQS (serverless)
- RabbitMQ (reliable)
- Redis (fast)

### Scenario: Adding Event Publishing

#### Before: Direct Method Calls

```python
class ToolExecutor:
    async def execute_tool(self, tool: BaseTool, args: dict):
        # Execute tool
        result = await tool.execute(**args)

        # Track metrics (tight coupling)
        self.metrics_collector.track_execution(tool.name)

        # Log execution (tight coupling)
        self.logger.info(f"Executed {tool.name}")

        return result
```

#### After: Event-Driven

```python
from victor.core.events import create_event_backend, MessagingEvent

class ToolExecutor:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus

    async def execute_tool(self, tool: BaseTool, args: dict):
        # Publish start event
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.start",
                data={"tool": tool.name, "args": args},
                correlation_id=str(uuid.uuid4()),
            )
        )

        # Execute tool
        result = await tool.execute(**args)

        # Publish complete event
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.complete",
                data={"tool": tool.name, "result": result},
                correlation_id=correlation_id,
            )
        )

        return result
```

### Step-by-Step Migration

#### Step 1: Choose Event Backend

```python
from victor.core.events import create_event_backend, BackendConfig

# For development/testing
backend = create_event_backend(BackendConfig.for_observability())

# For distributed production
backend = create_event_backend(
    BackendConfig(
        backend_type=BackendType.KAFKA,
        delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        extra={"bootstrap_servers": "localhost:9092"},
    )
)

await backend.connect()
```

#### Step 2: Publish Events

```python
# Import event types
from victor.core.events import MessagingEvent

# Publish event
await event_bus.publish(
    MessagingEvent(
        topic="tool.complete",
        data={"tool": "read_file", "result": "..."},
        correlation_id=str(uuid.uuid4()),
    )
)
```

#### Step 3: Subscribe to Events

```python
# Subscribe to topics
await event_bus.subscribe("tool.*", self._on_tool_event)
await event_bus.subscribe("agent.*", self._on_agent_event)

# Handle events
async def _on_tool_event(self, event: MessagingEvent):
    if event.topic == "tool.start":
        self._track_start(event.data)
    elif event.topic == "tool.complete":
        self._track_complete(event.data)
```

#### Step 4: Handle Errors

```python
async def on_event(self, event: MessagingEvent):
    try:
        result = await process_event(event)
        await event.ack()
    except TemporaryError as e:
        # Requeue for retry
        await event.nack(requeue=True)
    except PermanentError as e:
        # Don't retry
        await event.nack(requeue=False)
```

### Complete Example: Metrics Collector

#### Before: Tight Coupling

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {}

    def track_tool_execution(self, tool_name: str):
        self.metrics[tool_name] = self.metrics.get(tool_name, 0) + 1

# Usage in ToolExecutor
executor = ToolExecutor()
executor.metrics_collector = MetricsCollector()
executor.execute_tool(tool)  # Calls metrics_collector internally
```

#### After: Event-Driven

```python
class MetricsCollector:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus
        self.metrics = {}

        # Subscribe to events
        self._event_bus.subscribe("tool.*", self._on_tool_event)

    async def _on_tool_event(self, event: MessagingEvent):
        if event.topic == "tool.complete":
            tool_name = event.data["tool"]
            self.metrics[tool_name] = self.metrics.get(tool_name, 0) + 1

# Usage: Just create collector, it subscribes automatically
event_bus = create_event_backend(BackendConfig.for_observability())
collector = MetricsCollector(event_bus)
executor = ToolExecutor(event_bus)  # No coupling!
```

---

## Migration to Dependency Injection

### Understanding ServiceContainer

Victor 0.5.0 includes a dependency injection container with 55+ registered services.

### Scenario: Using ServiceContainer

#### Before: Manual Dependency Management

```python
class MyComponent:
    def __init__(self):
        # Manual dependency creation
        self.tool_registry = ToolRegistry()
        self.event_bus = EventBus()
        self.config = load_config()

    async def do_work(self):
        tools = self.tool_registry.get_all_tools()
        ...
```

**Problems**:
- Tight coupling to concrete implementations
- Hard to test (can't easily mock)
- Duplicate service instances
- Complex dependency graphs

#### After: Dependency Injection

```python
class MyComponent:
    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        event_bus: IEventBackend,
        config: Settings,
    ):
        # Dependencies injected
        self._tool_registry = tool_registry
        self._event_bus = event_bus
        self._config = config

    async def do_work(self):
        tools = self._tool_registry.get_all_tools()
        ...
```

**Benefits**:
- Loose coupling (depends on protocols)
- Easy to test (inject mocks)
- Shared service instances
- Clear dependency graph

### Step-by-Step Migration

#### Step 1: Register Services

```python
from victor.core.container import ServiceContainer, ServiceLifetime

container = ServiceContainer()

# Register singleton services
container.register(
    ToolRegistryProtocol,
    lambda c: ToolRegistry(),
    ServiceLifetime.SINGLETON,
)

# Register scoped services
container.register(
    ConversationStateMachineProtocol,
    lambda c: ConversationStateMachine(),
    ServiceLifetime.SCOPED,
)
```

#### Step 2: Update Component Constructors

```python
# Before
class MyComponent:
    def __init__(self):
        self.tool_registry = ToolRegistry()

# After
class MyComponent:
    def __init__(self, tool_registry: ToolRegistryProtocol):
        self._tool_registry = tool_registry
```

#### Step 3: Resolve Services

```python
# Resolve from container
tool_registry = container.get(ToolRegistryProtocol)
event_bus = container.get(IEventBackend)

# Create component with dependencies
component = MyComponent(
    tool_registry=tool_registry,
    event_bus=event_bus,
)
```

#### Step 4: Use Factory for Complex Creation

```python
class MyComponentFactory:
    def __init__(self, container: ServiceContainer):
        self._container = container

    def create_component(self) -> MyComponent:
        return MyComponent(
            tool_registry=self._container.get(ToolRegistryProtocol),
            event_bus=self._container.get(IEventBackend),
            config=self._container.get(Settings),
        )
```

### Complete Example: Orchestrator Factory

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
        event_bus = self._container.get(IEventBackend)

        # Create orchestrator-specific components
        tool_pipeline = self._create_tool_pipeline(provider)
        conversation_controller = self._create_conversation_controller(provider)

        # Create orchestrator
        return AgentOrchestrator(
            provider=provider,
            tool_registry=tool_registry,
            observability=observability,
            tool_pipeline=tool_pipeline,
            conversation_controller=conversation_controller,
            # ... other dependencies
        )
```

---

## Common Migration Scenarios

### Scenario 1: Adding New Feature

**Approach**: Use new architecture from the start

```python
# 1. Define protocol
@runtime_checkable
class MyFeatureProtocol(Protocol):
    async def do_something(self, data: str) -> dict:
        ...

# 2. Implement protocol
class MyFeature:
    async def do_something(self, data: str) -> dict:
        return {"result": data.upper()}

# 3. Register in DI
container.register(
    MyFeatureProtocol,
    lambda c: MyFeature(),
    ServiceLifetime.SINGLETON,
)

# 4. Use in coordinators
class MyCoordinator:
    def __init__(self, feature: MyFeatureProtocol):
        self._feature = feature

    async def execute(self, data: str):
        return await self._feature.do_something(data)
```

### Scenario 2: Refactoring Existing Component

**Approach**: Gradual migration

```python
# Phase 1: Extract protocol (keep old code working)
@runtime_checkable
class MyComponentProtocol(Protocol):
    async def process(self, data: str) -> dict:
        ...

class MyComponent:
    async def process(self, data: str) -> dict:
        # Old implementation
        return {"result": data.upper()}

# Phase 2: Update constructor to use protocol
class MyComponent:
    def __init__(self, dependency: MyDependencyProtocol):
        self._dependency = dependency

# Phase 3: Update tests to use mocks
def test_my_component():
    mock_dependency = Mock(spec=MyDependencyProtocol)
    component = MyComponent(mock_dependency)
    assert await component.process("test") == {"result": "TEST"}

# Phase 4: Register in DI and update factory
container.register(
    MyComponentProtocol,
    lambda c: MyComponent(c.get(MyDependencyProtocol)),
    ServiceLifetime.SINGLETON,
)
```

### Scenario 3: Migrating Vertical

**Approach**: Use template system

```bash
# 1. Extract existing vertical to template
python scripts/migrate_vertical_to_yaml.py \
  --vertical victor/my_vertical \
  --output my_vertical_template.yaml

# 2. Review and edit template
vim my_vertical_template.yaml

# 3. Generate new vertical from template
python scripts/generate_vertical.py \
  --template my_vertical_template.yaml \
  --output victor/my_vertical_new

# 4. Test new vertical
pytest tests/unit/my_vertical/

# 5. Replace old vertical
mv victor/my_vertical victor/my_vertical_old
mv victor/my_vertical_new victor/my_vertical
```

### Scenario 4: Adding Event Publishing

**Approach**: Add events without breaking existing code

```python
# Phase 1: Add event bus (optional)
class MyComponent:
    def __init__(
        self,
        event_bus: Optional[IEventBackend] = None,
    ):
        self._event_bus = event_bus

    async def do_work(self, data: str):
        result = await self._process(data)

        # Publish event if bus available
        if self._event_bus:
            await self._event_bus.publish(
                MessagingEvent(
                    topic="work.complete",
                    data={"result": result},
                )
            )

        return result

# Phase 2: Make event bus required
class MyComponent:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus

# Phase 3: Subscribers can listen to events
class WorkMonitor:
    def __init__(self, event_bus: IEventBackend):
        event_bus.subscribe("work.*", self._on_work_event)
```

---

## Testing Migrated Code

### Unit Testing with Protocols

```python
import pytest
from unittest.mock import Mock

def test_my_component():
    # Create protocol mocks
    mock_tool_registry = Mock(spec=ToolRegistryProtocol)
    mock_tool_registry.get_all_tools.return_value = []

    mock_event_bus = Mock(spec=IEventBackend)

    # Inject mocks
    component = MyComponent(
        tool_registry=mock_tool_registry,
        event_bus=mock_event_bus,
    )

    # Test
    result = component.do_work()

    # Verify
    mock_tool_registry.get_all_tools.assert_called_once()
```

### Integration Testing with DI

```python
@pytest.fixture
def container():
    container = ServiceContainer()
    configure_orchestrator_services(container, settings)
    return container

def test_integration(container):
    # Resolve real services
    tool_registry = container.get(ToolRegistryProtocol)
    event_bus = container.get(IEventBackend)

    # Create component with real services
    component = MyComponent(
        tool_registry=tool_registry,
        event_bus=event_bus,
    )

    # Test integration
    result = await component.do_work()
    assert result is not None
```

### Testing Event-Driven Code

```python
@pytest.mark.asyncio
async def test_event_publishing():
    # Create event bus
    event_bus = create_event_backend(BackendConfig.for_observability())
    await event_bus.connect()

    # Track events
    events_received = []

    async def event_handler(event: MessagingEvent):
        events_received.append(event)

    # Subscribe
    await event_bus.subscribe("test.*", event_handler)

    # Publish event
    await event_bus.publish(
        MessagingEvent(
            topic="test.event",
            data={"message": "hello"},
        )
    )

    # Wait for processing
    await asyncio.sleep(0.1)

    # Verify
    assert len(events_received) == 1
    assert events_received[0].data["message"] == "hello"
```

---

## Rollback Strategies

### Strategy 1: Feature Flags

```python
# Use feature flags for new architecture
USE_NEW_COORDINATORS = os.getenv("USE_NEW_COORDINATORS", "false") == "true"

class MyComponent:
    def __init__(
        self,
        old_orchestrator: AgentOrchestrator,
        new_coordinator: ToolCoordinatorProtocol,
    ):
        self._old_orchestrator = old_orchestrator
        self._new_coordinator = new_coordinator

    async def do_work(self, data: str):
        if USE_NEW_COORDINATORS:
            return await self._new_coordinator.process(data)
        else:
            return await self._old_orchestrator.process(data)
```

### Strategy 2: Gradual Rollout

```python
# Gradually migrate users to new architecture
MIGRATION_PERCENTAGE = int(os.getenv("MIGRATION_PERCENTAGE", "0"))

class MyComponent:
    async def do_work(self, user_id: str, data: str):
        # Hash user ID to determine if they get new architecture
        if hash(user_id) % 100 < MIGRATION_PERCENTAGE:
            return await self._new_coordinator.process(data)
        else:
            return await self._old_orchestrator.process(data)
```

### Strategy 3: AB Testing

```python
# Compare old vs new architecture
class MyComponent:
    async def do_work(self, data: str):
        # Run both
        old_result = await self._old_orchestrator.process(data)
        new_result = await self._new_coordinator.process(data)

        # Compare results
        if old_result != new_result:
            logger.warning(f"Results differ: {old_result} vs {new_result}")

        # Return old result during testing
        return old_result
```

---

## Summary

### Migration Checklist

- [ ] Identify migration scope (component, vertical, architecture)
- [ ] Choose migration level (1-4)
- [ ] Create migration plan
- [ ] Write tests for existing code
- [ ] Implement migration incrementally
- [ ] Test migrated code thoroughly
- [ ] Update documentation
- [ ] Monitor production
- [ ] Rollback plan if needed

### Best Practices

1. **Test First**: Write tests before migrating
2. **Incremental**: Migrate gradually, not all at once
3. **Backward Compatible**: Keep old code working during migration
4. **Monitor**: Watch production metrics after migration
5. **Rollback**: Have rollback plan ready

### Resources

- [Protocol Reference](./PROTOCOLS_REFERENCE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [ISP Migration Guide](../ISP_MIGRATION_GUIDE.md)
- [Vertical Template Guide](../vertical_template_guide.md)
- [Testing Guide](../testing/)

---

**Last Updated**: January 18, 2026
**Version**: 0.5.0
**Status**: COMPLETE
