# Victor Framework Migration Guide

**Target Audience**: Victor framework contributors and extenders
**Purpose**: Guide for migrating to and working with the SOLID-refactored architecture
**Status**: Phases 1-5 Complete, Phase 6 In Progress

## Table of Contents

1. [Quick Start](#quick-start)
2. [Feature Flags](#feature-flags)
3. [Service Architecture](#service-architecture)
4. [Migration Patterns](#migration-patterns)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### For Users

The refactored architecture is **opt-in via feature flags**. No changes are required to existing code.

```bash
# Current architecture (default)
victor chat "Hello, world!"

# New architecture (when flags enabled)
VICTOR_USE_NEW_CHAT_SERVICE=true victor chat "Hello, world!"
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
| `USE_NEW_CHAT_SERVICE` | Use ChatService for chat operations | `false` |
| `USE_NEW_TOOL_SERVICE` | Use ToolService for tool operations | `false` |
| `USE_NEW_CONTEXT_SERVICE` | Use ContextService for context management | `false` |
| `USE_NEW_PROVIDER_SERVICE` | Use ProviderService for provider management | `false` |
| `USE_NEW_RECOVERY_SERVICE` | Use RecoveryService for error recovery | `false` |
| `USE_NEW_SESSION_SERVICE` | Use SessionService for session lifecycle | `false` |
| `USE_STRATEGY_BASED_TOOL_REGISTRATION` | Use strategy pattern for tool registration | `false` |
| `USE_COMPOSITION_OVER_INHERITANCE` | Use composition for verticals | `false` |

### Configuration Methods

#### Environment Variables

```bash
export VICTOR_USE_NEW_CHAT_SERVICE=true
export VICTOR_USE_NEW_TOOL_SERVICE=true
export VICTOR_USE_STRATEGY_BASED_TOOL_REGISTRATION=true
```

#### YAML Configuration

Create `~/.victor/features.yaml`:

```yaml
feature_flags:
  use_new_chat_service: true
  use_new_tool_service: true
  use_strategy_based_tool_registration: true
```

#### Runtime Control

```python
from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

manager = get_feature_flag_manager()

# Check if flag is enabled
if manager.is_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE):
    # Use new ChatService
    pass

# Enable flag at runtime
manager.enable(FeatureFlag.USE_NEW_CHAT_SERVICE)

# Disable flag at runtime
manager.disable(FeatureFlag.USE_NEW_CHAT_SERVICE)
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

## Migration Patterns

### Pattern 1: Gradual Feature Migration

**Scenario**: Migrate an existing feature to use new services

```python
# Old implementation
class AgentOrchestrator:
    async def chat(self, message: str):
        # Direct implementation
        return await self._chat_coordinator.chat(message)

# New implementation (feature flag controlled)
class AgentOrchestrator:
    def __init__(self, ...):
        # Get feature flag manager
        self._feature_flags = get_feature_flag_manager()

        # Initialize new services if flag enabled
        if self._feature_flags.is_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE):
            self._new_chat_service = self._container.get(ChatServiceProtocol)

    async def chat(self, message: str):
        if self._feature_flags.is_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE):
            return await self._new_chat_service.chat(message)
        return await self._chat_coordinator.chat(message)
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

### Testing with Feature Flags

```python
import pytest
from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

def test_with_old_implementation():
    """Test with old implementation (default)."""
    manager = get_feature_flag_manager()
    manager.disable(FeatureFlag.USE_NEW_CHAT_SERVICE)

    # Test old implementation
    response = await orchestrator.chat("test")
    assert response.content

def test_with_new_implementation():
    """Test with new implementation."""
    manager = get_feature_flag_manager()
    manager.enable(FeatureFlag.USE_NEW_CHAT_SERVICE)

    # Test new implementation
    response = await orchestrator.chat("test")
    assert response.content

@pytest.mark.parametrize("use_new_service", [True, False])
def test_both_implementations(use_new_service):
    """Test both implementations parametrized."""
    manager = get_feature_flag_manager()
    manager.set_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE, use_new_service)

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

**Solution**: Check if feature flag is enabled and dependencies are available

```python
from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

manager = get_feature_flag_manager()
if not manager.is_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE):
    print("ChatService not enabled - enable feature flag first")
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

### Issue: Feature Flag Not Taking Effect

**Symptom**: Feature flag enabled but old code still running

**Solution**: Check flag loading order

```python
# Flags loaded from:
# 1. Environment variables
# 2. YAML config (~/.victor/features.yaml)
# 3. Runtime overrides

# Check flag state
from victor.core.feature_flags import get_feature_flag_manager

manager = get_feature_flag_manager()
print(f"Flag enabled: {manager.is_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE)}")

# Enable explicitly
manager.enable(FeatureFlag.USE_NEW_CHAT_SERVICE)
```

## Best Practices

### 1. Use Protocols for Type Hints

```python
from victor.agent.services.protocols import ChatServiceProtocol

def process_chat(service: ChatServiceProtocol, message: str):
    """Process chat with protocol-typed service."""
    return await service.chat(message)
```

### 2. Check Feature Flags Before Using New Services

```python
from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

def use_tool_service():
    manager = get_feature_flag_manager()
    if not manager.is_enabled(FeatureFlag.USE_NEW_TOOL_SERVICE):
        raise RuntimeError("ToolService feature flag not enabled")
    # Proceed with ToolService
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
# Test both implementations to ensure compatibility
@pytest.mark.parametrize("use_new", [True, False])
def test_feature_parity(use_new):
    # Ensure old and new produce same results
    pass
```

## Rollback

If you encounter issues with the new architecture:

1. **Disable specific flags**:
   ```bash
   export VICTOR_USE_NEW_CHAT_SERVICE=false
   ```

2. **Disable all new architecture**:
   ```bash
   # Unset all flags
   unset VICTOR_USE_NEW_CHAT_SERVICE
   unset VICTOR_USE_NEW_TOOL_SERVICE
   # ... etc
   ```

3. **Remove YAML config**:
   ```bash
   rm ~/.victor/features.yaml
   ```

## Support

- Issues: Report at https://github.com/anthropics/victor-ai/issues
- Questions: Use GitHub Discussions
- Documentation: See `docs/SERVICE_GUIDE.md` for service creation guide
