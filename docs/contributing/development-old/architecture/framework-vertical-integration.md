# Framework & Vertical Integration Architecture

> **Archived**: This document is kept for historical context and may be outdated. See `docs/contributing/index.md` for current guidance.


**Target Audience**: Framework developers extending Victor with new verticals or capabilities.

## Overview

Victor's architecture follows SOLID principles with a protocol-first design that separates concerns between the framework core and domain-specific verticals.

```
┌──────────────────────────────────────────────────────────────────────┐
│                           REQUEST FLOW                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CLI/TUI ──▶ AgentOrchestrator ──▶ VerticalIntegrationPipeline      │
│                                        │                              │
│                                        ▼                              │
│                              StepHandlerRegistry                     │
│                                  ┌─────┴─────┐                        │
│                                  │ Handlers │                        │
│                                  │ ┌───────┐ │                        │
│                                  │ │ Tools │ │                        │
│                                  │ │ Prompt│ │                        │
│                                  │ │ Config│ │                        │
│                                  │ │ Extend│ │                        │
│                                  │ │Framework│ │                       │
│                                  │ └───────┘ │                        │
│                                  └─────┬─────┘                        │
│                                        │                              │
│                                        ▼                              │
│                               VerticalBase                         │
│                            (Coding, Research, ...)                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. VerticalIntegrationPipeline

**Location**: [`victor/framework/vertical_integration.py`](../../../victor/framework/vertical_integration.py)

Facade that applies vertical configurations to orchestrators via step handlers.

```python
from victor.framework.vertical_integration import VerticalIntegrationPipeline

pipeline = VerticalIntegrationPipeline()
result = pipeline.apply(orchestrator, CodingAssistant)
```

**Key Features**:
- Protocol-first capability invocation (`_check_capability`, `_invoke_capability`)
- Two-level caching (definition + execution)
- Result tracking with `IntegrationResult`

### 2. StepHandler System

**Location**: [`victor/framework/step_handlers.py`](../../../victor/framework/step_handlers.py)

Single Responsibility Principle: each handler manages one integration concern.

| Handler | Order | Responsibility |
|---------|-------|----------------|
| `CapabilityConfigStepHandler` | 5 | Centralized capability config storage |
| `ToolStepHandler` | 10 | Tool filter canonicalization |
| `TieredConfigStepHandler` | 15 | Tiered tool configuration |
| `PromptStepHandler` | 20 | System prompt application |
| `ConfigStepHandler` | 40 | Stage definitions |
| `MiddlewareStepHandler` | 50 | Middleware chain |
| `SafetyStepHandler` | 30 | Safety patterns |
| `ExtensionsStepHandler` | 45 | Extension coordination |
| `FrameworkStepHandler` | 60 | Workflows, RL, teams, handlers |
| `ContextStepHandler` | 100 | Final context attachment |

**Protocol-First Design**:
```python
def _check_capability(obj: Any, capability_name: str) -> bool:
    """Check capability via protocol (DIP compliant)."""
    if isinstance(obj, CapabilityRegistryProtocol):
        return obj.has_capability(capability_name)
    # Fallback to public method only (no private attributes)
```

### 3. VerticalBase

**Location**: [`victor/core/verticals/base.py`](../../../victor/core/verticals/base.py)

Abstract base class for all verticals with built-in caching.

```python
class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "grep"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a specialized assistant..."
```

**Extension Caching**:
```python
@classmethod
def get_safety_extension(cls):
    def _create():
        from victor.myvertical.safety import MySafetyExtension
        return MySafetyExtension()
    return cls._get_cached_extension("safety_extension", _create)
```

## Protocol-First Design

### Capability Registry Protocol

**Location**: [`victor/framework/protocols.py`](../../../victor/framework/protocols.py)

Replaces `hasattr` duck-typing with explicit capability contracts.

```python
@runtime_checkable
class CapabilityRegistryProtocol(Protocol):
    def has_capability(self, name: str, min_version: Optional[str] = None) -> bool: ...
    def invoke_capability(self, name: str, *args, **kwargs) -> Any: ...
    def get_capability(self, name: str) -> Optional[OrchestratorCapability]: ...
```

### OrchestratorVerticalProtocol

```python
@runtime_checkable
class OrchestratorVerticalProtocol(Protocol):
    def set_vertical_context(self, context: VerticalContext) -> None: ...
    def set_enabled_tools(self, tools: Set[str]) -> None: ...
    def apply_vertical_middleware(self, middleware: List[Any]) -> None: ...
    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None: ...
```

## Capability System

### Framework Capabilities

**Location**: [`victor/framework/capabilities/`](../../../victor/framework/capabilities/)

Reusable capabilities across verticals following DRY:

```python
from victor.framework.capabilities import (
    FileOperationsCapability,    # read, write, edit, grep
    PromptContributionCapability, # common prompt hints
    PrivacyCapabilityProvider,    # PII management
)

# Use in vertical
class MyVertical(VerticalBase):
    _file_ops = FileOperationsCapability()

    def get_tools(cls):
        return cls._file_ops.get_tool_list()
```

### Vertical-Specific Capabilities

Each vertical can define capabilities via `BaseCapabilityProvider`:

```python
# victor/coding/capabilities.py
class CodingCapabilityProvider(BaseCapabilityProvider):
    def get_capabilities(self) -> Dict[str, Callable]:
        return {
            "git_safety": configure_git_safety,
            "code_style": configure_code_style,
        }
```

## SOLID Compliance Summary

| Principle | Implementation | File |
|-----------|----------------|------|
| **SRP** | Each StepHandler handles one concern | [`step_handlers.py`](../../../victor/framework/step_handlers.py) |
| **OCP** | ExtensionHandlerRegistry for pluggable handlers | [`vertical_integration.py:287`](../../../victor/framework/vertical_integration.py#L287) |
| **LSP** | StepHandlerProtocol ensures substitutability | [`step_handlers.py:278`](../../../victor/framework/step_handlers.py#L278) |
| **ISP** | Focused protocols (CapabilityRegistry, SubAgentContext) | [`protocols.py`](../../../victor/framework/protocols.py) |
| **DIP** | `_check_capability`/`_invoke_capability` use protocols | [`step_handlers.py:120`](../../../victor/framework/step_handlers.py#L120) |

## Extension Points

### Adding a New Step Handler

```python
from victor.framework.step_handlers import BaseStepHandler, StepHandlerRegistry

class MyCustomStepHandler(BaseStepHandler):
    @property
    def name(self) -> str:
        return "my_custom"

    @property
    def order(self) -> int:
        return 55  # Between extensions (45) and framework (60)

    def _do_apply(self, orchestrator, vertical, context, result):
        # Your logic here
        pass

# Register
registry = StepHandlerRegistry.default()
registry.add_handler(MyCustomStepHandler())
```

### Adding Extension Handlers

```python
from victor.framework.vertical_integration import register_extension_handler

def handle_my_extension(orchestrator, ext_value, extensions, context, result):
    # Handle extension type
    pass

register_extension_handler(
    name="my_extension",
    attr_name="my_extension_provider",
    handler=handle_my_extension,
    order=60,
)
```

## Data Flow Summary

```
User Request
    │
    ▼
AgentOrchestrator.set_enabled_tools()          ◀─── ToolStepHandler
    │
    ▼
AgentOrchestrator.set_system_prompt()          ◀─── PromptStepHandler
    │
    ▼
AgentOrchestrator.apply_vertical_middleware()   ◀─── MiddlewareStepHandler
    │
    ▼
AgentOrchestrator.apply_vertical_safety_patterns() ◀─── SafetyStepHandler
    │
    ▼
AgentOrchestrator.set_vertical_context()        ◀─── ContextStepHandler
    │
    ▼
Response
```

## Key Files Reference

| Component | File |
|-----------|------|
| Pipeline | [`victor/framework/vertical_integration.py`](../../../victor/framework/vertical_integration.py) |
| Step Handlers | [`victor/framework/step_handlers.py`](../../../victor/framework/step_handlers.py) |
| Protocols | [`victor/framework/protocols.py`](../../../victor/framework/protocols.py) |
| Vertical Base | [`victor/core/verticals/base.py`](../../../victor/core/verticals/base.py) |
| Framework Capabilities | [`victor/framework/capabilities/`](../../../victor/framework/capabilities/) |
| WorkflowEngine | [`victor/framework/workflow_engine.py`](../../../victor/framework/workflow_engine.py) |
