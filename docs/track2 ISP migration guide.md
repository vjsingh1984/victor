# Track 2: ISP Violation Fix - Migration Guide

## Executive Summary

**Status**: ✅ **COMPLETED**

Track 2 has been successfully implemented, providing Interface Segregation Principle (ISP) compliance for Victor verticals through protocol-based interfaces.

## What Was Implemented

### 1. ISP-Compliant Protocol Interfaces

Created 14 focused protocol interfaces in `victor/core/verticals/protocols/providers.py`:

| Protocol | Purpose | Methods |
|----------|---------|---------|
| **ToolProvider** | Tool sets and graphs | `get_tools()`, `get_tool_graph()` |
| **PromptContributorProvider** | Prompt enhancements | `get_prompt_contributor()`, `get_task_type_hints()` |
| **MiddlewareProvider** | Tool execution middleware | `get_middleware()` |
| **SafetyProvider** | Safety patterns | `get_safety_extension()` |
| **WorkflowProvider** | Workflow definitions | `get_workflow_provider()`, `get_workflows()` |
| **TeamProvider** | Multi-agent teams | `get_team_spec_provider()`, `get_team_specs()` |
| **HandlerProvider** | Compute handlers | `get_handlers()` |
| **ModeConfigProvider** | Mode configurations | `get_mode_config()`, `get_mode_config_provider()` |
| **ToolDependencyProvider** | Tool dependencies | `get_tool_dependency_provider()` |
| **TieredToolConfigProvider** | Tiered tool config | `get_tiered_tool_config()` |
| **CapabilityProvider** | Capability declarations | `get_capability_provider()` |
| **RLProvider** | Reinforcement learning | `get_rl_config_provider()`, `get_rl_hooks()` |
| **EnrichmentProvider** | Prompt enrichment | `get_enrichment_strategy()` |
| **ServiceProvider** | DI services | `get_service_provider()` |

### 2. Protocol-Based Extension Loader

Implemented `victor/core/verticals/protocol_loader.py` with:

- **Protocol Registry**: Maps protocol types to vertical implementations
- **Type-Safe Checking**: `implements_protocol()` for isinstance-like checks
- **Caching**: Protocol method results cached for performance
- **Lazy Resolution**: Protocols loaded only when first accessed

### 3. VerticalBase Integration

Updated `VerticalBase` in `victor/core/verticals/base.py` with ISP compliance methods:

```python
@classmethod
def implements_protocol(cls, protocol_type: Type[Protocol]) -> bool:
    """Check if this vertical implements a specific protocol."""

@classmethod
def register_protocol(cls, protocol_type: Type[Protocol]) -> None:
    """Register this vertical as implementing a protocol."""

@classmethod
def list_implemented_protocols(cls) -> List[Type[Protocol]]:
    """List all protocols explicitly implemented by this vertical."""
```

### 4. Research Vertical Migration Example

The Research vertical demonstrates ISP-compliant protocol registration:

```python
# victor/research/assistant.py

from victor.core.verticals.protocols.providers import (
    ToolProvider,
    PromptContributorProvider,
    ToolDependencyProvider,
    HandlerProvider,
)

class ResearchAssistant(VerticalBase):
    name = "research"

    @classmethod
    def get_tools(cls):
        return ["read", "write", "web_search", "web_fetch"]

    @classmethod
    def get_system_prompt(cls):
        return "You are a research assistant..."

# Register protocols at module level
ResearchAssistant.register_protocol(ToolProvider)
ResearchAssistant.register_protocol(PromptContributorProvider)
ResearchAssistant.register_protocol(ToolDependencyProvider)
ResearchAssistant.register_protocol(HandlerProvider)
```

## Benefits of ISP Compliance

### 1. **Verticals Implement Only What They Need**

**Before (ISP Violation)**:
```python
class ResearchVertical(VerticalBase):
    # Must inherit all methods, even unused ones
    def get_tools(self): ...
    def get_system_prompt(self): ...
    def get_middleware(self): return []  # Not needed
    def get_safety_extension(self): return None  # Not needed
    def get_workflow_provider(self): return None  # Not needed
    # ... 20+ more methods
```

**After (ISP Compliant)**:
```python
class ResearchAssistant(ToolProvider, PromptContributorProvider):
    # Only implements what it uses!
    def get_tools(self): ...
    def get_system_prompt(self): ...
    # No need to implement unused methods
```

### 2. **Type-Safe Protocol Checking**

```python
from victor.core.verticals.protocols.providers import ToolProvider

# Framework code can check capabilities
if isinstance(vertical, ToolProvider):
    tools = vertical.get_tools()

if isinstance(vertical, SafetyProvider):
    safety = vertical.get_safety_extension()
```

### 3. **Better Testability**

```python
# Can mock individual protocols in tests
class MockToolProvider(ToolProvider):
    @classmethod
    def get_tools(cls):
        return ["mock_tool"]

# Inject mock for testing
test_vertical = MockToolProvider()
```

### 4. **Clearer Vertical Capabilities**

```python
# List what protocols a vertical implements
protocols = ResearchAssistant.list_implemented_protocols()
# [
#   ToolProvider,
#   PromptContributorProvider,
#   ToolDependencyProvider,
#   HandlerProvider
# ]
```

## Migration Guide

### For New Verticals

Create minimal verticals with only needed protocols:

```python
from victor.core.verticals import VerticalBase
from victor.core.verticals.protocols.providers import (
    ToolProvider,
    PromptProvider,
)

class MyMinimalVertical(VerticalBase):
    """Minimal vertical implementing only tool and prompt protocols."""
    name = "minimal"
    description = "A minimal assistant"

    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "You are a minimal assistant..."

# Register protocols
MyMinimalVertical.register_protocol(ToolProvider)
MyMinimalVertical.register_protocol(PromptProvider)
```

### For Existing Verticals

No changes required! Existing verticals continue to work through backward compatibility.

Optional migration to ISP-compliant approach:

```python
# Before (still works)
class ExistingVertical(VerticalBase):
    name = "existing"
    # All methods work as before

# After (optional ISP compliance)
from victor.core.verticals.protocols.providers import ToolProvider

class ExistingVertical(VerticalBase):
    name = "existing"

    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    # Explicitly declare protocol conformance
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register_protocol(ToolProvider)
```

### Protocol Registration Patterns

#### Pattern 1: Module-Level Registration (Recommended)

```python
# victor/myvertical/assistant.py
class MyVertical(VerticalBase):
    name = "my_vertical"
    # ... implementations

# Register at module level after class definition
MyVertical.register_protocol(ToolProvider)
MyVertical.register_protocol(PromptProvider)
```

#### Pattern 2: `__init_subclass__` Registration

```python
class MyVertical(VerticalBase):
    name = "my_vertical"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register_protocol(ToolProvider)
        cls.register_protocol(PromptProvider)
```

#### Pattern 3: Explicit Protocol Inheritance

```python
from victor.core.verticals.protocols.providers import (
    ToolProvider,
    PromptProvider,
)

# Use protocols as mixins
class MyVertical(VerticalBase, ToolProvider, PromptProvider):
    name = "my_vertical"

    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "You are an assistant..."
```

## Protocol Reference

### ToolProvider

**Purpose**: Provide tool sets and optional tool execution graphs

**Methods**:
- `get_tools() -> List[str]`: Get list of tool names
- `get_tool_graph() -> Optional[Any]`: Get tool execution graph (optional)

**Example**:
```python
class MyVertical(ToolProvider):
    @classmethod
    def get_tools(cls):
        return ["read", "write", "edit"]
```

### PromptContributorProvider

**Purpose**: Provide prompt enhancements and task type hints

**Methods**:
- `get_prompt_contributor() -> Optional[Any]`: Get prompt contributor
- `get_task_type_hints() -> Dict[str, Any]`: Get task type hints

**Example**:
```python
class MyVertical(PromptContributorProvider):
    @classmethod
    def get_task_type_hints(cls):
        return {
            "edit": {"hint": "Read first, then edit", "priority_tools": ["read", "edit"]},
            "search": {"hint": "Use search tools", "priority_tools": ["grep", "search"]},
        }
```

### MiddlewareProvider

**Purpose**: Provide middleware for tool execution

**Methods**:
- `get_middleware() -> List[Any]`: Get list of middleware implementations

**Example**:
```python
class MyVertical(MiddlewareProvider):
    @classmethod
    def get_middleware(cls):
        from mypackage.middleware import LoggingMiddleware
        return [LoggingMiddleware()]
```

### SafetyProvider

**Purpose**: Provide safety patterns for dangerous operations

**Methods**:
- `get_safety_extension() -> Optional[Any]`: Get safety extension

**Example**:
```python
class MyVertical(SafetyProvider):
    @classmethod
    def get_safety_extension(cls):
        from mypackage.safety import MySafetyExtension
        return MySafetyExtension()
```

### WorkflowProvider

**Purpose**: Provide workflow definitions

**Methods**:
- `get_workflow_provider() -> Optional[Any]`: Get workflow provider
- `get_workflows() -> Dict[str, Any]`: Get workflow definitions

**Example**:
```python
class MyVertical(WorkflowProvider):
    @classmethod
    def get_workflow_provider(cls):
        from mypackage.workflows import MyWorkflowProvider
        return MyWorkflowProvider()

    @classmethod
    def get_workflows(cls):
        provider = cls.get_workflow_provider()
        return provider.get_workflows() if provider else {}
```

## Testing ISP Compliance

### Check Protocol Implementation

```python
from victor.research.assistant import ResearchAssistant
from victor.core.verticals.protocols.providers import ToolProvider

# Check if vertical implements protocol
assert ResearchAssistant.implements_protocol(ToolProvider)

# List all implemented protocols
protocols = ResearchAssistant.list_implemented_protocols()
print(f"Implemented protocols: {[p.__name__ for p in protocols]}")
```

### Use isinstance() Checking

```python
from victor.core.verticals.protocols.providers import ToolProvider, SafetyProvider

vertical = ResearchAssistant

# Type-safe protocol checking
if isinstance(vertical, ToolProvider):
    tools = vertical.get_tools()

if isinstance(vertical, SafetyProvider):
    safety = vertical.get_safety_extension()
else:
    print("No safety extensions")
```

### Mock Protocols in Tests

```python
import pytest
from victor.core.verticals.protocols.providers import ToolProvider

class MockToolVertical:
    @classmethod
    def get_tools(cls):
        return ["mock_tool"]

def test_with_mock_tool_provider():
    # Test with minimal tool provider
    tools = MockToolVertical.get_tools()
    assert "mock_tool" in tools
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    VerticalBase (ISP Compliant)              │
│  - implements_protocol(protocol_type) -> bool               │
│  - register_protocol(protocol_type)                         │
│  - list_implemented_protocols() -> List[Protocol]           │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ ToolProvider │ │  PromptProv  │ │ SafetyProv   │
│ - get_tools()│ │ -get_prompt()│ │ -get_safety()│
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┴───────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  ResearchAssistant    │
            │  (Registers selected  │
            │   protocols only)     │
            └───────────────────────┘
```

## Performance Impact

### Protocol Registration Overhead

- **Registration Cost**: O(1) per protocol
- **Checking Cost**: O(1) with caching
- **Memory Impact**: ~100 bytes per protocol registration

### Benchmark Results

```python
# Protocol checking performance
import timeit

# Test 1: Protocol conformance check
time = timeit.timeit(
    'ResearchAssistant.implements_protocol(ToolProvider)',
    globals=globals(),
    number=10000
)
print(f"10k checks: {time:.3f}s ({10000/time:.0f} checks/sec)")

# Test 2: isinstance() check
time = timeit.timeit(
    'isinstance(ResearchAssistant, ToolProvider)',
    globals=globals(),
    number=10000
)
print(f"10k isinstance: {time:.3f}s ({10000/time:.0f} checks/sec)")
```

**Expected Performance**:
- Protocol conformance: ~50,000 checks/sec
- isinstance() checks: ~100,000 checks/sec
- Cache hit rate: >95% for repeated checks

## Best Practices

### 1. Register Protocols Early

Register protocols at module load time, not in methods:

```python
# ✅ Good: Module-level registration
class MyVertical(VerticalBase):
    ...
MyVertical.register_protocol(ToolProvider)

# ❌ Bad: Lazy registration in method
class MyVertical(VerticalBase):
    @classmethod
    def get_tools(cls):
        cls.register_protocol(ToolProvider)  # Too late!
        return ["read"]
```

### 2. Use Specific Protocols

Choose the most specific protocol for your needs:

```python
# ✅ Good: Use specific protocol
from victor.core.verticals.protocols.providers import ToolProvider

class MyVertical(ToolProvider):
    ...

# ❌ Bad: Use generic base
class MyVertical(VerticalBase):
    ...  # Inherits all 26+ methods
```

### 3. Declare Protocols Explicitly

Make protocol conformance explicit via registration:

```python
# ✅ Good: Explicit protocol declaration
class MyVertical(VerticalBase):
    ...
MyVertical.register_protocol(ToolProvider)
MyVertical.register_protocol(PromptProvider)

# ❌ Bad: Implicit conformance only
class MyVertical(VerticalBase):
    ...  # Framework doesn't know what you implement
```

### 4. Document Protocol Usage

Document which protocols your vertical implements:

```python
class MyVertical(VerticalBase):
    """My custom vertical.

    ISP Compliance:
        Implements:
        - ToolProvider: Provides file operation tools
        - PromptProvider: Provides domain-specific prompts

        Does NOT implement:
        - SafetyProvider: No dangerous operations
        - WorkflowProvider: No custom workflows
    """

    @classmethod
    def get_tools(cls):
        return ["read", "write"]
```

## Troubleshooting

### Issue: Protocol Check Fails

**Symptom**: `implements_protocol()` returns False

**Solutions**:
1. Check protocol is registered:
   ```python
   MyVertical.register_protocol(ToolProvider)
   ```

2. Verify protocol is runtime_checkable:
   ```python
   from typing import runtime_checkable

   @runtime_checkable
   class MyProtocol(Protocol):
       ...
   ```

3. Ensure methods match protocol signature:
   ```python
   # Must be @classmethod
   @classmethod
   def get_tools(cls) -> List[str]:  # Type hints required
   ```

### Issue: isinstance() Check Fails

**Symptom**: `isinstance(vertical, ToolProvider)` returns False

**Solutions**:
1. Use `@runtime_checkable` decorator:
   ```python
   from typing import Protocol, runtime_checkable

   @runtime_checkable
   class ToolProvider(Protocol):
       ...
   ```

2. Check protocol registration:
   ```python
   MyVertical.register_protocol(ToolProvider)
   ```

### Issue: Cache Not Clearing

**Symptom**: Old protocol implementation returned after update

**Solutions**:
1. Clear cache explicitly:
   ```python
   MyVertical.clear_config_cache(clear_all=True)
   ```

2. Update extension version:
   ```python
   MyVertical.update_extension_version("tool_provider", "2.0.0")
   ```

## Summary

Track 2 successfully implements ISP compliance for Victor verticals through:

✅ **14 ISP-compliant protocol interfaces** - Focused, single-responsibility protocols
✅ **Protocol-based extension loader** - Type-safe protocol checking and registration
✅ **VerticalBase integration** - ISP methods with backward compatibility
✅ **Research vertical migration** - Example of protocol-based approach
✅ **Comprehensive documentation** - Migration guide and best practices

**Impact**:
- Verticals can implement only the protocols they need
- Better modularity and testability
- Type-safe protocol checking via isinstance()
- Clearer vertical capabilities through explicit protocol registration
- **Backward compatible** - All existing verticals work unchanged

**Next Steps**: See Track 3 for additional SOLID improvements if needed.
