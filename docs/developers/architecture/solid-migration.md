# Victor Framework Migration Guide

**Target Audience**: Victor framework contributors and extenders
**Purpose**: Guide for migrating to and working with the SOLID-refactored architecture
**Status**: Contributor guide for the post-refactor codebase
**Last Reviewed**: 2026-05-04
**Current agent runtime ownership**: `docs/architecture/CURRENT_STATE.md`

## Table of Contents

1. [Quick Start](#quick-start)
2. [Feature Flags](#feature-flags)
3. [Service Architecture](#service-architecture)
4. [Migration Patterns](#migration-patterns)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### For Users

The service-owned runtime is the default architecture. No feature flag is
required to use `ChatService`, `ToolService`, or the rest of the canonical
service layer.

```bash
# Default runtime
victor chat "Hello, world!"

# Optional framework-side experiment when validating AgenticLoop parity
VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=true victor chat "Hello, world!"
```

### For Contributors

#### Adding a New Tool Type (OCP Compliance)

**Before** (modifies core code):
```python
# Had to modify ToolRegistry.register()
def register(self, tool):
    if isinstance(tool, MyCustomType):
        # Custom logic here
        ...
```

**After** (add via strategy):
```python
from victor.tools.registration.strategies import ToolRegistrationStrategy
from victor.tools.registry import ToolRegistry

class MyCustomToolStrategy:
    def can_handle(self, tool):
        return isinstance(tool, MyCustomType)

    def register(self, registry, tool, enabled=True):
        wrapper = self._create_wrapper(tool)
        registry._register_direct(wrapper.name, wrapper, enabled)

    @property
    def priority(self):
        return 75  # Between FunctionDecorator (100) and BaseTool (50)

# Register the strategy (no core code modification!)
registry = ToolRegistry()
registry.add_custom_strategy(MyCustomToolStrategy())
```

#### Creating a New Vertical (Composition over Inheritance)

**Before** (inheritance):
```python
class MyVertical(VerticalBase):
    # Must implement ALL methods even if not needed
    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "You are my assistant"

    # ... many more required methods
```

**After** (composition):
```python
from victor.core.verticals.base import VerticalBase

MyVertical = (
    VerticalBase
    .compose()
    .with_metadata("my_vertical", "My assistant", "1.0.0")
    .with_tools(["read", "write"])
    .with_system_prompt("You are a helpful assistant.")
    .build()
)
```

## Feature Flags

### Available Flags

| Flag | Description | Default |
|------|-------------|---------|
| `USE_COMPOSITION_OVER_INHERITANCE` | Use composition for verticals | `false` |
| `USE_STRATEGY_BASED_TOOL_REGISTRATION` | Use strategy pattern for tool registration | `false` |
| `USE_LLM_DECISION_SERVICE` | Use LLM-based decision service | `false` |
| `USE_EDGE_MODEL` | Use fast edge model for micro-decisions | `false` |
| `USE_FUZZY_MATCHING` | Enable fuzzy matching for robust classification | `true` |
| `USE_SEMANTIC_RESPONSE_CACHE` | Enable semantic response caching | `false` |
| `USE_SMART_ROUTING` | Enable smart model routing | `false` |
| `USE_LEARNING_FROM_EXECUTION` | Enable learning from execution | `true` |
| `USE_STATEGRAPH_AGENTIC_LOOP` | Use StateGraph executor in AgenticLoop | `false` |

> **Note**: Phase 3 service layer flags (`USE_NEW_CHAT_SERVICE`, `USE_NEW_TOOL_SERVICE`, etc.) were removed in W3 cleanup. Services (ChatService, ToolService, ContextService, ProviderService, RecoveryService, SessionService) are now **mandatory** and no longer gated by feature flags.

### Configuration Methods

#### Environment Variables

```bash
export VICTOR_USE_SMART_ROUTING=true
export VICTOR_USE_EDGE_MODEL=false
export VICTOR_USE_STRATEGY_BASED_TOOL_REGISTRATION=true
```

#### YAML Configuration

Create `~/.victor/features.yaml`:

```yaml
features:
  use_smart_routing: true
  use_edge_model: false
  use_strategy_based_tool_registration: true
```

#### Runtime Control

```python
from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

manager = get_feature_flag_manager()

# Check if flag is enabled
if manager.is_enabled(FeatureFlag.USE_SMART_ROUTING):
    # Use smart routing
    pass

# Enable flag at runtime
manager.enable(FeatureFlag.USE_SMART_ROUTING)

# Disable flag at runtime
manager.disable(FeatureFlag.USE_SMART_ROUTING)
```

## Service Architecture

### Service Protocols

All services implement focused protocols (ISP compliance):

```python
from victor.agent.services.protocols import ChatServiceProtocol

class MyChatService(ChatServiceProtocol):
    async def chat(self, user_message: str, stream: bool = False, **kwargs):
        # Implementation
        pass

    async def stream_chat(self, user_message: str, **kwargs):
        # Implementation
        pass

    def reset_conversation(self):
        # Implementation
        pass
```

### Service Dependencies

Services have well-defined dependencies:

```
ChatService depends on:
├── ProviderService (required)
├── ToolService (optional)
├── ContextService (optional)
└── RecoveryService (optional)

ToolService depends on:
└── (no service dependencies)

ContextService depends on:
└── (no service dependencies)

ProviderService depends on:
└── (no service dependencies)
```

### Service Lifecycle

Services are managed by the DI container:

```python
from victor.core.bootstrap import ensure_bootstrapped
from victor.agent.services.protocols import ChatServiceProtocol

# Bootstrap container (idempotent)
container = ensure_bootstrapped(settings)

# Get service
chat_service = container.get(ChatServiceProtocol)
```

### Canonical Migration Surfaces

Prefer these entry points in new runtime code instead of reaching for legacy
coordinator shims directly.

| Concern | Preferred surface | Avoid by default |
|---------|-------------------|------------------|
| Chat turns | `ChatService` or `OrchestrationFacade.chat_service` | `ChatCoordinator` shims |
| Tool execution | `ToolService` or `OrchestrationFacade.tool_service` | `ToolCoordinator` shims |
| Session lifecycle | `SessionService` or `OrchestrationFacade.session_service` | `SessionCoordinator` shims |
| Read-only exploration | `victor.agent.coordinators.ExplorationCoordinator` | deep submodule imports when package-root works |

For deprecated chat shims, treat `ChatCoordinator.stream_chat()` as a
compatibility-only forwarding layer. Its resolution order is:
1. bound `ChatService.stream_chat()`
2. orchestrator `_get_service_streaming_runtime()`
3. legacy `_stream_chat_runtime` hook

New runtime code should bind or call the first two surfaces directly and avoid
wiring the legacy hook unless it is preserving older integration behavior.
| State-passed exploration | `victor.agent.coordinators.ExplorationStatePassedCoordinator` or `OrchestrationFacade.exploration_state_passed` | direct orchestrator-coupled exploration glue |
| State-passed system prompt logic | `victor.agent.coordinators.SystemPromptStatePassedCoordinator` or `OrchestrationFacade.system_prompt_state_passed` | reintroducing removed prompt-coordinator compatibility imports |
| State-passed safety logic | `victor.agent.coordinators.SafetyStatePassedCoordinator` or `OrchestrationFacade.safety_state_passed` | using `SafetyCoordinator` when snapshot/transition flow is intended |

```python
from victor.agent.coordinators import (
    ExplorationStatePassedCoordinator,
    SafetyStatePassedCoordinator,
    SystemPromptStatePassedCoordinator,
)
from victor.agent.facades import OrchestrationFacade

facade = OrchestrationFacade(
    chat_service=chat_service,
    tool_service=tool_service,
    session_service=session_service,
    exploration_state_passed=ExplorationStatePassedCoordinator(),
    system_prompt_state_passed=SystemPromptStatePassedCoordinator(
        task_analyzer=task_analyzer
    ),
    safety_state_passed=SafetyStatePassedCoordinator(),
)
```

## Migration Patterns

### Pattern 1: Service-First Delegation

**Scenario**: Migrate an existing feature to the canonical service surface

```python
# Old implementation
class AgentOrchestrator:
    async def chat(self, message: str):
        # Direct implementation
        return await self._chat_coordinator.chat(message)

# Current implementation (service-first)
class AgentOrchestrator:
    def __init__(self, ...):
        self._chat_service = self._container.get_optional(ChatServiceProtocol)

    async def chat(self, message: str):
        if self._chat_service is not None:
            return await self._chat_service.chat(message)
        return await self._chat_coordinator.chat(message)  # compatibility-only fallback
```

### Pattern 2: Extending Without Modification

**Scenario**: Add a new tool type without modifying core code

```python
# Define strategy
class PydanticModelStrategy:
    def __init__(self):
        from pydantic import BaseModel
        self.BaseModel = BaseModel

    def can_handle(self, tool):
        try:
            return isinstance(tool, self.BaseModel)
        except ImportError:
            return False

    def register(self, registry, tool, enabled=True):
        # Convert Pydantic model to BaseTool
        wrapper = self._create_wrapper(tool)
        registry._register_direct(wrapper.name, wrapper, enabled)

    @property
    def priority(self):
        return 75

# Register globally (once at startup)
from victor.tools.registry import ToolRegistry

ToolRegistry.add_custom_strategy(PydanticModelStrategy())
```

### Pattern 3: Vertical by Composition

**Scenario**: Create a new vertical using composition

```python
from victor.core.verticals.base import VerticalBase
from victor.core.vertical_types import StageDefinition, StageBuilder

# Define stages
builder = StageBuilder()
stages = (
    builder.workflow("MyVertical")
    .initial()
    .stage("ANALYZE")
        .description("Analyze the request")
        .tools({"read", "grep"})
    .stage("EXECUTE")
        .description("Execute the plan")
        .tools({"write", "edit"})
    .completion()
    .build_workflow()
)

# Compose vertical
MyVertical = (
    VerticalBase
    .compose()
    .with_metadata("my_vertical", "My specialized assistant", "1.0.0")
    .with_tools(["read", "write", "grep", "edit"])
    .with_system_prompt("You are a specialized assistant for my vertical.")
    .with_stages(stages)
    .with_extensions([SafetyExtension()])
    .build()
)

# Use vertical
agent = Agent(assistant=MyVertical)
```

## Testing

### Testing Canonical and Experimental Paths

```python
import pytest
from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

def test_default_service_owned_runtime():
    """Test the default service-owned runtime path."""
    response = await orchestrator.chat("test")
    assert response.content

def test_with_stategraph_agentic_loop_enabled():
    """Enable the remaining framework-side experiment explicitly."""
    manager = get_feature_flag_manager()
    manager.enable(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)

    # Test the framework-side experimental path
    response = await orchestrator.chat("test")
    assert response.content
    manager.disable(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)

@pytest.mark.parametrize("service_present", [True, False])
def test_service_and_compatibility_paths(service_present):
    """Test canonical service and compatibility fallback where both still exist."""
    response = await orchestrator.chat("test")
    assert response.content
```

### Testing Custom Strategies

```python
def test_custom_tool_strategy():
    """Test custom tool registration strategy."""
    from victor.tools.registry import ToolRegistry
    from victor.tools.registration.strategies import ToolRegistrationStrategy

    # Define test strategy
    class TestStrategy(ToolRegistrationStrategy):
        def can_handle(self, tool):
            return hasattr(tool, "_is_test_tool")

        def register(self, registry, tool, enabled=True):
            registry._register_direct(tool.name, tool, enabled)

        @property
        def priority(self):
            return 50

    # Test registration
    registry = ToolRegistry()
    registry.add_custom_strategy(TestStrategy())

    # Create test tool
    class MockTool:
        name = "test_tool"
        _is_test_tool = True

    # Register via strategy
    registry.register(MockTool())
    assert "test_tool" in registry._tools
```

### Testing Vertical Composition

```python
def test_vertical_composition():
    """Test vertical composition."""
    from victor.core.verticals.base import VerticalBase

    # Compose vertical
    TestVertical = (
        VerticalBase
        .compose()
        .with_metadata("test", "Test vertical", "1.0.0")
        .with_tools(["read", "write"])
        .with_system_prompt("Test prompt")
        .build()
    )

    # Verify
    assert TestVertical.name == "test"
    assert TestVertical.description == "Test vertical"
    assert TestVertical.version == "1.0.0"
    assert TestVertical.get_tools() == ["read", "write"]
    assert "Test prompt" in TestVertical.get_system_prompt()
```

## Troubleshooting

### Issue: Service Not Available

**Symptom**: `ServiceNotRegisteredError` when trying to get a service

**Solution**: Verify bootstrap/container registration and resolve the canonical
protocol from the bootstrapped container

```python
from victor.core.bootstrap import ensure_bootstrapped
from victor.agent.services.protocols import ChatServiceProtocol

container = ensure_bootstrapped(settings)
chat_service = container.get_optional(ChatServiceProtocol)
if chat_service is None:
    print("ChatService is not registered - check bootstrap wiring")
```

### Issue: Custom Strategy Not Working

**Symptom**: Custom strategy not being used for tool registration

**Solution**: Check priority and `can_handle()` implementation

```python
# Debug strategy selection
strategy = registry._strategy_registry.get_strategy_for(my_tool)
print(f"Selected strategy: {type(strategy).__name__}")

# Check if can_handle works
print(f"Strategy can_handle: {strategy.can_handle(my_tool)}")
```

### Issue: Vertical Build Fails

**Symptom**: `ValueError: Metadata capability is required`

**Solution**: Ensure metadata is provided before `build()`

```python
# Wrong - missing metadata
MyVertical = VerticalBase.compose().build()  # ERROR!

# Correct - metadata provided
MyVertical = (
    VerticalBase
    .compose()
    .with_metadata("name", "description", "version")  # Required!
    .build()
)
```

### Issue: Runtime Experiment Flag Not Taking Effect

**Symptom**: `VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=true` is set but the framework
path you expected is not active

**Solution**: Check flag loading order

```python
# Flags loaded from:
# 1. Environment variables
# 2. YAML config (~/.victor/features.yaml)
# 3. Runtime overrides

# Check flag state
from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

manager = get_feature_flag_manager()
print(
    "Flag enabled:",
    manager.is_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP),
)

# Enable explicitly
manager.enable(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)
```

## Best Practices

### 1. Use Protocols for Type Hints

```python
from victor.agent.services.protocols import ChatServiceProtocol

def process_chat(service: ChatServiceProtocol, message: str):
    """Process chat with protocol-typed service."""
    return await service.chat(message)
```

### 2. Do Not Reintroduce Service-Gating Flags

```python
from victor.core.bootstrap import ensure_bootstrapped
from victor.agent.services.protocols import ToolServiceProtocol

def use_tool_service():
    container = ensure_bootstrapped(settings)
    tool_service = container.get_optional(ToolServiceProtocol)
    if tool_service is None:
        raise RuntimeError("ToolService is not registered")
    return tool_service
```

### 3. Provide Fallbacks for Optional Dependencies

```python
class MyService:
    def __init__(self, container):
        # Optional dependency
        self.tool_service = container.get_optional(ToolServiceProtocol)
        if self.tool_service is None:
            logger.warning("ToolService not available, using fallback")
```

### 4. Write Tests for Both Old and New Implementations

```python
# Test canonical and compatibility paths only where compatibility still exists
@pytest.mark.parametrize("service_present", [True, False])
def test_chat_delegation_parity(service_present):
    # Ensure canonical service and compatibility fallback stay aligned
    pass
```

## Rollback

If you encounter issues with the framework-side experimental path:

1. **Disable the remaining experiment flag**:
   ```bash
   export VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=false
   ```

2. **Clear runtime overrides**:
   ```bash
   unset VICTOR_USE_STATEGRAPH_AGENTIC_LOOP
   ```

3. **For service-owned runtime issues, fix bootstrap or registration rather than toggling removed service flags**:
   ```bash
   python -m pytest tests/unit/agent/services/test_service_layer_validation.py
   ```

## Support

- Issues: Report at https://github.com/anthropics/victor-ai/issues
- Questions: Use GitHub Discussions
- Documentation: See `docs/SERVICE_GUIDE.md` for service creation guide
