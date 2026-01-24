# Extension Handler Registry Developer Guide

This guide documents the `ExtensionHandlerRegistry`, Victor's primary extension mechanism for third-party plugins and custom integrations.

## Overview

The `ExtensionHandlerRegistry` follows the **Open/Closed Principle (OCP)** - it allows new extension types to be registered without modifying existing code. This enables third-party developers to extend Victor's behavior through custom handlers that integrate seamlessly with the vertical integration pipeline.

### Key Benefits

- **Non-invasive**: Add functionality without modifying Victor's core code
- **Priority-based**: Control execution order relative to built-in handlers
- **Type-safe**: Extension handlers receive strongly-typed context objects
- **Error-isolated**: Handler failures don't crash the integration pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  VerticalIntegrationPipeline                     │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              ExtensionsStepHandler                       │   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │         ExtensionHandlerRegistry                 │   │   │
│   │   │                                                  │   │   │
│   │   │  Priority 5:  service_provider                   │   │   │
│   │   │  Priority 10: middleware                         │   │   │
│   │   │  Priority 15: tool_selection_strategy            │   │   │
│   │   │  Priority 20: safety_extensions                  │   │   │
│   │   │  Priority 30: prompt_contributors                │   │   │
│   │   │  Priority 40: mode_config_provider               │   │   │
│   │   │  Priority 50: tool_dependency_provider           │   │   │
│   │   │  Priority 60: enrichment_strategy                │   │   │
│   │   │  Priority ??: YOUR_CUSTOM_HANDLER                │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### ExtensionHandler

A dataclass representing a single extension handler:

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class ExtensionHandler:
    """Extension handler definition.

    Attributes:
        name: Extension type name (matches VerticalExtensions field)
        handler: Callable that applies the extension
        priority: Execution order (lower runs first)
    """
    name: str
    handler: Callable[[Any, Any, Any, "VerticalContext", "IntegrationResult"], None]
    priority: int = 50
```

### Handler Signature

All extension handlers must follow this signature:

```python
def my_handler(
    orchestrator: Any,           # AgentOrchestrator instance
    extension_value: Any,        # Value from VerticalExtensions field
    extensions: VerticalExtensions,  # Full extensions container
    context: VerticalContext,    # Vertical context (name, mode, config)
    result: IntegrationResult,   # Result object for adding info/warnings
) -> None:
    """Process the extension."""
    pass
```

### ExtensionHandlerRegistry

The registry manages all extension handlers:

```python
class ExtensionHandlerRegistry:
    """Registry for extension handlers (OCP pattern)."""

    def register(self, handler: ExtensionHandler, replace: bool = False) -> None:
        """Register an extension handler."""

    def unregister(self, name: str) -> bool:
        """Unregister an extension handler."""

    def get_ordered_handlers(self) -> List[ExtensionHandler]:
        """Get handlers in priority order."""

    def apply_all(
        self,
        orchestrator: Any,
        extensions: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply all registered extension handlers."""
```

## Default Handlers and Priorities

Victor registers these default handlers with the following priorities:

| Priority | Handler Name | Description |
|----------|--------------|-------------|
| 5 | `service_provider` | DI container service registration |
| 10 | `middleware` | Tool execution middleware |
| 15 | `tool_selection_strategy` | Custom tool selection logic |
| 20 | `safety_extensions` | Safety patterns and validators |
| 30 | `prompt_contributors` | System prompt sections and task hints |
| 40 | `mode_config_provider` | Mode configurations and tool budgets |
| 50 | `tool_dependency_provider` | Tool dependency injection |
| 60 | `enrichment_strategy` | DSPy-like prompt enrichment |

### Priority Guidelines

When choosing a priority for your handler:

- **Priority 1-4**: Before service registration (rare, use with caution)
- **Priority 5-9**: After services, before middleware
- **Priority 10-14**: After middleware, before tool selection
- **Priority 15-19**: After tool selection, before safety
- **Priority 20-29**: After safety, before prompts
- **Priority 30-39**: After prompts, before mode config
- **Priority 40-49**: After mode config, before tool deps
- **Priority 50-59**: After tool deps, before enrichment
- **Priority 60+**: After enrichment (cleanup, telemetry, etc.)

## Registering Custom Handlers

### Method 1: Direct Registry Access

Access the registry through the `ExtensionsStepHandler`:

```python
from victor.framework.step_handlers import (
    ExtensionHandler,
    ExtensionsStepHandler,
)

# Get the step handler (typically from VerticalIntegrationPipeline)
extensions_handler = ExtensionsStepHandler()

# Define your handler function
def handle_my_extension(
    orchestrator,
    my_extension_value,
    extensions,
    context,
    result,
):
    """Process my custom extension."""
    if my_extension_value is None:
        return

    # Your custom logic here
    result.add_info(f"Applied my extension: {my_extension_value}")

# Register the handler
extensions_handler.extension_registry.register(
    ExtensionHandler(
        name="my_extension",  # Must match field name in VerticalExtensions
        handler=handle_my_extension,
        priority=55,  # After tool_dependency_provider, before enrichment
    )
)
```

### Method 2: Via VerticalIntegrationPipeline

For cleaner integration, access through the pipeline:

```python
from victor.framework.step_handlers import VerticalIntegrationPipeline, ExtensionHandler

# Get or create the pipeline
pipeline = VerticalIntegrationPipeline()

# Access the extensions step handler's registry
extensions_step = pipeline._steps.get("extensions")
if extensions_step:
    extensions_step.extension_registry.register(
        ExtensionHandler(
            name="analytics",
            handler=my_analytics_handler,
            priority=65,
        )
    )
```

### Method 3: Plugin Entry Point

For third-party packages, use entry points:

```python
# In your_package/__init__.py
def register_extension_handlers(registry):
    """Called by Victor to register your handlers."""
    from .handlers import my_handler

    registry.register(
        ExtensionHandler(
            name="your_extension",
            handler=my_handler,
            priority=55,
        )
    )
```

```toml
# In pyproject.toml
[project.entry-points."victor.extension_handlers"]
your_package = "your_package:register_extension_handlers"
```

## Adding Custom Extension Fields

To use a custom extension, you need to:

1. **Extend VerticalExtensions** (or use dynamic extensions)
2. **Register a handler** for the new field

### Using Dynamic Extensions

For third-party plugins, use the `_dynamic_extensions` field:

```python
from victor.core.verticals.protocols import VerticalExtensions

# In your vertical's get_extensions() method
def get_extensions(cls) -> VerticalExtensions:
    extensions = VerticalExtensions(
        middleware=[...],
        # ... other standard extensions
    )

    # Add dynamic extension
    extensions._dynamic_extensions["analytics_config"] = {
        "enabled": True,
        "sample_rate": 0.1,
    }

    return extensions
```

Then register a handler that reads from `_dynamic_extensions`:

```python
def handle_analytics_config(orchestrator, value, extensions, context, result):
    # Read from dynamic extensions
    config = extensions._dynamic_extensions.get("analytics_config")
    if config and config.get("enabled"):
        # Apply analytics configuration
        sample_rate = config.get("sample_rate", 0.01)
        result.add_info(f"Analytics enabled with sample rate: {sample_rate}")
```

## Complete Example: Custom Telemetry Handler

Here's a complete example of adding a telemetry extension:

### Step 1: Define the Handler

```python
# victor_telemetry/handlers.py

from victor.framework.step_handlers import ExtensionHandler

def handle_telemetry(
    orchestrator,
    telemetry_config,
    extensions,
    context,
    result,
):
    """Apply telemetry configuration to the orchestrator.

    Args:
        orchestrator: The agent orchestrator
        telemetry_config: Telemetry configuration dict or object
        extensions: Full VerticalExtensions container
        context: Vertical context
        result: Integration result for reporting
    """
    if telemetry_config is None:
        return

    # Extract configuration
    if isinstance(telemetry_config, dict):
        enabled = telemetry_config.get("enabled", False)
        endpoint = telemetry_config.get("endpoint")
        sample_rate = telemetry_config.get("sample_rate", 0.01)
    else:
        enabled = getattr(telemetry_config, "enabled", False)
        endpoint = getattr(telemetry_config, "endpoint", None)
        sample_rate = getattr(telemetry_config, "sample_rate", 0.01)

    if not enabled:
        result.add_info("Telemetry disabled")
        return

    # Configure telemetry (example using observability bus)
    try:
        from victor.core.events import get_observability_bus

        bus = get_observability_bus()

        # Add telemetry exporter
        from .exporters import TelemetryExporter
        exporter = TelemetryExporter(
            endpoint=endpoint,
            sample_rate=sample_rate,
        )
        bus.add_exporter(exporter)

        result.add_info(f"Telemetry enabled: {endpoint} (sample_rate={sample_rate})")

    except Exception as e:
        result.add_warning(f"Failed to configure telemetry: {e}")


def get_telemetry_handler() -> ExtensionHandler:
    """Create the telemetry extension handler."""
    return ExtensionHandler(
        name="telemetry_config",  # Matches _dynamic_extensions key
        handler=handle_telemetry,
        priority=70,  # After enrichment, near end of pipeline
    )
```

### Step 2: Create Plugin Package

```python
# victor_telemetry/__init__.py

from .handlers import get_telemetry_handler

def register_extension_handlers(registry):
    """Register telemetry handlers with Victor."""
    registry.register(get_telemetry_handler())

__all__ = ["register_extension_handlers", "get_telemetry_handler"]
```

### Step 3: Configure Entry Point

```toml
# pyproject.toml

[project]
name = "victor-telemetry"
version = "0.5.0"
dependencies = ["victor-ai>=0.5.0"]

[project.entry-points."victor.extension_handlers"]
telemetry = "victor_telemetry:register_extension_handlers"
```

### Step 4: Use in Vertical

```python
# my_vertical/vertical.py

from victor.core.verticals.base import VerticalBase
from victor.core.verticals.protocols import VerticalExtensions

class MyVertical(VerticalBase):
    name = "myvertical"

    @classmethod
    def get_extensions(cls) -> VerticalExtensions:
        extensions = VerticalExtensions(
            middleware=[...],
            safety_extensions=[...],
        )

        # Add telemetry configuration
        extensions._dynamic_extensions["telemetry_config"] = {
            "enabled": True,
            "endpoint": "https://telemetry.example.com/v1/events",
            "sample_rate": 0.1,
        }

        return extensions
```

## Error Handling

The registry isolates handler failures:

```python
# From ExtensionHandlerRegistry.apply_all():
for handler in self.get_ordered_handlers():
    ext_value = getattr(extensions, handler.name, None)
    if ext_value is not None:
        try:
            handler.handler(orchestrator, ext_value, extensions, context, result)
        except Exception as e:
            # Error is logged and added as warning, but doesn't stop other handlers
            result.add_warning(f"Extension handler '{handler.name}' failed: {e}")
```

### Best Practices for Error Handling

```python
def my_safe_handler(orchestrator, value, extensions, context, result):
    """Handler with proper error handling."""
    try:
        # Validate input
        if value is None:
            return

        # Check for required dependencies
        if not hasattr(orchestrator, "_container"):
            result.add_warning("Orchestrator lacks container, skipping my_extension")
            return

        # Your logic here
        do_something(value)
        result.add_info("my_extension applied successfully")

    except ImportError as e:
        # Missing optional dependency
        result.add_warning(f"my_extension requires additional dependency: {e}")
    except ValueError as e:
        # Invalid configuration
        result.add_warning(f"my_extension configuration error: {e}")
    except Exception as e:
        # Unexpected error - log for debugging
        import logging
        logging.getLogger(__name__).debug(f"my_extension error: {e}", exc_info=True)
        result.add_warning(f"my_extension failed: {e}")
```

## Testing Extension Handlers

### Unit Test Example

```python
# tests/test_my_extension.py

import pytest
from unittest.mock import MagicMock, AsyncMock

from victor.framework.step_handlers import (
    ExtensionHandler,
    ExtensionHandlerRegistry,
    VerticalContext,
    IntegrationResult,
)
from my_package.handlers import handle_my_extension


class TestMyExtensionHandler:
    """Tests for my custom extension handler."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        mock = MagicMock()
        mock._container = MagicMock()
        mock.settings = MagicMock()
        return mock

    @pytest.fixture
    def context(self):
        """Create test context."""
        return VerticalContext(
            vertical_name="test",
            mode="build",
            config={},
        )

    @pytest.fixture
    def result(self):
        """Create test result."""
        return IntegrationResult()

    def test_handler_applies_config(self, mock_orchestrator, context, result):
        """Test handler applies configuration correctly."""
        config = {"enabled": True, "setting": "value"}

        handle_my_extension(
            mock_orchestrator,
            config,
            MagicMock(),  # extensions
            context,
            result,
        )

        # Verify behavior
        assert result.infos  # Check info messages were added
        assert not result.warnings  # No warnings

    def test_handler_skips_when_disabled(self, mock_orchestrator, context, result):
        """Test handler skips when disabled."""
        config = {"enabled": False}

        handle_my_extension(
            mock_orchestrator,
            config,
            MagicMock(),
            context,
            result,
        )

        # Verify skipped
        assert "disabled" in str(result.infos).lower() or len(result.infos) == 0

    def test_handler_handles_none_gracefully(self, mock_orchestrator, context, result):
        """Test handler handles None value."""
        handle_my_extension(
            mock_orchestrator,
            None,  # No config
            MagicMock(),
            context,
            result,
        )

        # Should not error
        assert not result.warnings


class TestExtensionHandlerRegistry:
    """Tests for registry integration."""

    def test_handler_registration(self):
        """Test handler can be registered."""
        registry = ExtensionHandlerRegistry()

        handler = ExtensionHandler(
            name="my_extension",
            handler=handle_my_extension,
            priority=55,
        )

        registry.register(handler)

        handlers = registry.get_ordered_handlers()
        assert any(h.name == "my_extension" for h in handlers)

    def test_priority_ordering(self):
        """Test handlers are ordered by priority."""
        registry = ExtensionHandlerRegistry()

        registry.register(ExtensionHandler("high", lambda *a: None, priority=10))
        registry.register(ExtensionHandler("low", lambda *a: None, priority=90))
        registry.register(ExtensionHandler("mid", lambda *a: None, priority=50))

        handlers = registry.get_ordered_handlers()
        names = [h.name for h in handlers]

        assert names == ["high", "mid", "low"]
```

### Integration Test Example

```python
# tests/integration/test_my_extension_integration.py

import pytest
from victor.framework.step_handlers import VerticalIntegrationPipeline
from victor.core.verticals.base import VerticalBase


class TestVertical(VerticalBase):
    """Test vertical with custom extension."""
    name = "test_vertical"

    @classmethod
    def get_extensions(cls):
        from victor.core.verticals.protocols import VerticalExtensions
        extensions = VerticalExtensions()
        extensions._dynamic_extensions["my_extension"] = {"enabled": True}
        return extensions


class TestMyExtensionIntegration:
    """Integration tests for my extension."""

    @pytest.fixture
    def pipeline(self):
        """Create integration pipeline with handler registered."""
        from my_package.handlers import get_my_handler

        pipeline = VerticalIntegrationPipeline()

        # Register handler
        extensions_step = pipeline._steps.get("extensions")
        if extensions_step:
            extensions_step.extension_registry.register(get_my_handler())

        return pipeline

    def test_extension_applied_in_pipeline(self, pipeline):
        """Test extension is applied during vertical integration."""
        from unittest.mock import MagicMock

        orchestrator = MagicMock()
        orchestrator._container = MagicMock()

        result = pipeline.integrate(orchestrator, TestVertical)

        # Check extension was processed
        assert result.is_success
        # Check for info messages from handler
        assert any("my_extension" in str(info).lower() for info in result.infos)
```

## Debugging Tips

### Enable Debug Logging

```python
import logging

# Enable debug logging for step handlers
logging.getLogger("victor.framework.step_handlers").setLevel(logging.DEBUG)

# Or enable for your handler specifically
logging.getLogger("my_package.handlers").setLevel(logging.DEBUG)
```

### Inspect Registered Handlers

```python
from victor.framework.step_handlers import ExtensionsStepHandler

handler = ExtensionsStepHandler()

# List all registered handlers
for h in handler.extension_registry.get_ordered_handlers():
    print(f"Priority {h.priority}: {h.name}")
```

### Check Integration Result

```python
from victor.framework.step_handlers import VerticalIntegrationPipeline

pipeline = VerticalIntegrationPipeline()
result = pipeline.integrate(orchestrator, MyVertical)

# Check for issues
print(f"Success: {result.is_success}")
print(f"Infos: {result.infos}")
print(f"Warnings: {result.warnings}")
print(f"Errors: {result.errors}")
```

## API Reference

### ExtensionHandler

```python
@dataclass
class ExtensionHandler:
    name: str  # Extension field name
    handler: Callable[
        [Any, Any, Any, VerticalContext, IntegrationResult],
        None
    ]
    priority: int = 50  # Lower = runs first
```

### ExtensionHandlerRegistry

```python
class ExtensionHandlerRegistry:
    def register(self, handler: ExtensionHandler, replace: bool = False) -> None
    def unregister(self, name: str) -> bool
    def get_ordered_handlers(self) -> List[ExtensionHandler]
    def apply_all(
        self,
        orchestrator: Any,
        extensions: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None

    @classmethod
    def default(cls) -> ExtensionHandlerRegistry
```

### VerticalContext

```python
@dataclass
class VerticalContext:
    vertical_name: str
    mode: str
    config: Dict[str, Any]
    # Additional context fields...
```

### IntegrationResult

```python
class IntegrationResult:
    def add_info(self, message: str) -> None
    def add_warning(self, message: str) -> None
    def add_error(self, message: str) -> None

    @property
    def is_success(self) -> bool
    @property
    def infos(self) -> List[str]
    @property
    def warnings(self) -> List[str]
    @property
    def errors(self) -> List[str]
```

## Summary

The `ExtensionHandlerRegistry` provides a clean, OCP-compliant mechanism for extending Victor:

1. **Register handlers** with specific priorities to control execution order
2. **Access orchestrator and context** for full integration capabilities
3. **Report status** through the `IntegrationResult` object
4. **Fail safely** - handler errors don't crash the pipeline
5. **Use dynamic extensions** for third-party fields without modifying core types

For questions or issues, see the [Victor repository](https://github.com/anthropics/victor) or existing verticals for implementation examples.
