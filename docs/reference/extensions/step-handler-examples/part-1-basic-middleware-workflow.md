# Step Handler Examples - Part 1

**Part 1 of 3:** Basic Examples, Tool Management, Middleware Integration, and Workflow Registration

---

## Navigation

- **[Part 1: Basic through Workflow](#)** (Current)
- [Part 2: Safety & Config](part-2-safety-config.md)
- [Part 3: Advanced & Testing](part-3-advanced-testing.md)
- [**Complete Guide**](../step_handler_examples.md)

---

This document provides practical, working examples of step handlers for common vertical extension scenarios.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Tool Management](#tool-management)
3. [Middleware Integration](#middleware-integration)
4. [Workflow Registration](#workflow-registration)
5. [Safety & Security](#safety--security) *(in Part 2)*
6. [Configuration Management](#configuration-management) *(in Part 2)*
7. [Advanced Patterns](#advanced-patterns) *(in Part 3)*
8. [Testing Examples](#testing-examples) *(in Part 3)*

---

## Basic Examples

### Example 1: Minimal Step Handler

The simplest possible step handler:

```python
from victor.framework.step_handlers import BaseStepHandler
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase
    from victor.core.verticals import VerticalContext
    from victor.framework.vertical_integration import IntegrationResult

class MinimalHandler(BaseStepHandler):
    """A minimal step handler example."""

    @property
    def name(self) -> str:
        return "minimal"

    @property
    def order(self) -> int:
        return 25

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Do something simple."""
        result.add_info("Minimal handler executed")

# Usage
from victor.framework.step_handlers import StepHandlerRegistry

registry = StepHandlerRegistry.default()
registry.add_handler(MinimalHandler())
```

### Example 2: Logging Handler

Log information about the vertical being applied:

```python
import logging
from victor.framework.step_handlers import BaseStepHandler

logger = logging.getLogger(__name__)

class LoggingHandler(BaseStepHandler):
    """Logs vertical integration details."""

    @property
    def name(self) -> str:
        return "logging"

    @property
    def order(self) -> int:
        return 7  # Early, after capability_config (5)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Log vertical details."""
        logger.info(f"Applying vertical: {vertical.name}")

        # Log tools count
        tools = vertical.get_tools()
        logger.info(f"  Tools: {len(tools)}")

        # Log system prompt length
        prompt = vertical.get_system_prompt()
        logger.info(f"  Prompt length: {len(prompt)} chars")

        result.add_info(f"Logged vertical: {vertical.name}")

    def _get_step_details(self, result: IntegrationResult) -> dict:
        """Provide step details."""
        return {
            "vertical_name": result.vertical_name,
            "logged": True,
        }
```

### Example 3: Conditional Handler

Only execute for specific verticals:

```python
class ConditionalExtensionHandler(BaseStepHandler):
    """Applies extensions only to specific verticals."""

    @property
    def name(self) -> str:
        return "conditional_extensions"

    @property
    def order(self) -> int:
        return 25

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Only apply to coding vertical."""
        if vertical.name != "coding":
            result.add_info(f"Skipped {vertical.name} (coding only)")
            return

        # Apply coding-specific extensions
        self._apply_coding_extensions(orchestrator, vertical, context, result)

    def _apply_coding_extensions(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply coding-specific extensions."""
        # Custom logic for coding vertical
        result.add_info("Applied coding-specific extensions")
```

---

## Tool Management

### Example 4: Custom Tool Registration with Validation

Register tools with custom validation logic:

```python
from typing import List, Set

class CustomToolsHandler(BaseStepHandler):
    """Register tools with validation and filtering."""

    @property
    def name(self) -> str:
        return "custom_tools"

    @property
    def order(self) -> int:
        return 15  # After default tools (10)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply tools with validation."""
        tools = vertical.get_tools()

        # Validate
        if not self._validate_tools(tools):
            result.add_error("Tool validation failed")
            return

        # Filter based on settings
        filtered_tools = self._filter_tools(orchestrator, tools)

        # Apply via context
        context.apply_enabled_tools(set(filtered_tools))
        result.tools_applied = set(filtered_tools)
        result.add_info(f"Applied {len(filtered_tools)} tools")

    def _validate_tools(self, tools: List[str]) -> bool:
        """Validate tool list."""
        # Check for required tools
        required = {"read", "write"}
        return required.issubset(set(tools))

    def _filter_tools(
        self,
        orchestrator: Any,
        tools: List[str],
    ) -> List[str]:
        """Filter tools based on settings."""
        filtered = tools.copy()

        # Remove web tools in airgapped mode
        if hasattr(orchestrator, "settings"):
            if getattr(orchestrator.settings, "airgapped_mode", False):
                filtered = [t for t in filtered if not t.startswith("web_")]

        return filtered

    def _get_step_details(self, result: IntegrationResult) -> dict:
        """Provide step details."""
        return {
            "tools_count": len(result.tools_applied),
            "filtered": len(result.tools_applied) < len(result.tools_applied),
        }
```

### Example 5: Tiered Tool Configuration

Apply tiered tool access control:

```python
class TieredToolAccessHandler(BaseStepHandler):
    """Configure tiered tool access levels."""

    @property
    def name(self) -> str:
        return "tiered_tool_access"

    @property
    def order(self) -> int:
        return 16  # After tool registration (15)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply tiered access control."""
        # Get tiered config from vertical
        tiered_config = self._get_tiered_config(vertical)

        if tiered_config is None:
            result.add_info("No tiered config available")
            return

        # Apply to context
        context.apply_tiered_config(tiered_config)

        # Log tier breakdown
        mandatory_count = len(tiered_config.mandatory) if tiered_config.mandatory else 0
        core_count = len(tiered_config.vertical_core) if tiered_config.vertical_core else 0
        pool_count = len(tiered_config.semantic_pool) if tiered_config.semantic_pool else 0

        result.add_info(
            f"Tiered config: mandatory={mandatory_count}, "
            f"core={core_count}, pool={pool_count}"
        )

    def _get_tiered_config(self, vertical: Type[VerticalBase]) -> Optional[Any]:
        """Get tiered config from vertical."""
        # Try get_tiered_tool_config() first
        if hasattr(vertical, "get_tiered_tool_config"):
            return vertical.get_tiered_tool_config()

        # Fallback to get_tiered_tools()
        if hasattr(vertical, "get_tiered_tools"):
            return vertical.get_tiered_tools()

        return None
```

### Example 6: Tool Dependency Resolution

Resolve and apply tool dependencies:

```python
class ToolDependencyHandler(BaseStepHandler):
    """Resolve and apply tool dependencies."""

    @property
    def name(self) -> str:
        return "tool_dependencies"

    @property
    def order(self) -> int:
        return 17  # After tiered config (16)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply tool dependencies."""
        # Get dependency provider from vertical
        provider = self._get_dependency_provider(vertical)

        if provider is None:
            result.add_info("No tool dependency provider")
            return

        # Get dependencies and sequences
        dependencies = provider.get_dependencies()
        sequences = provider.get_tool_sequences()

        # Apply to context
        context.apply_tool_dependencies(dependencies, sequences)

        # Log details
        dep_count = len(dependencies) if dependencies else 0
        seq_count = len(sequences) if sequences else 0

        result.add_info(f"Applied {dep_count} dependencies, {seq_count} sequences")

    def _get_dependency_provider(self, vertical: Type[VerticalBase]) -> Optional[Any]:
        """Get dependency provider from vertical."""
        if hasattr(vertical, "get_tool_dependency_provider"):
            return vertical.get_tool_dependency_provider()
        return None
```

---

## Middleware Integration

### Example 7: Custom Middleware Injection

Inject custom middleware with priority ordering:

```python
from typing import List, Any

class CustomMiddlewareHandler(BaseStepHandler):
    """Inject custom middleware with priority ordering."""

    @property
    def name(self) -> str:
        return "custom_middleware"

    @property
    def order(self) -> int:
        return 50  # Same as MiddlewareStepHandler

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply custom middleware."""
        # Get middleware from vertical
        middleware_list = vertical.get_middleware()

        if not middleware_list:
            result.add_info("No custom middleware")
            return

        # Add conditional middleware
        if self._should_add_caching(orchestrator):
            middleware_list.append(CachingMiddleware())

        # Order by priority
        ordered_middleware = self._order_by_priority(middleware_list)

        # Apply via context
        context.apply_middleware(ordered_middleware)
        result.middleware_count = len(ordered_middleware)
        result.add_info(f"Applied {len(ordered_middleware)} middleware")

    def _should_add_caching(self, orchestrator: Any) -> bool:
        """Check if caching middleware should be added."""
        if hasattr(orchestrator, "settings"):
            return getattr(orchestrator.settings, "enable_caching", False)
        return False

    def _order_by_priority(self, middleware: List[Any]) -> List[Any]:
        """Order middleware by priority attribute."""
        def get_priority(mw):
            return getattr(mw, "priority", 50)

        return sorted(middleware, key=get_priority)

    def _get_step_details(self, result: IntegrationResult) -> dict:
        """Provide step details."""
        return {
            "middleware_count": result.middleware_count,
        }
```

### Example 8: Middleware Chain Validation

Validate middleware chain before application:

```python
class MiddlewareValidationHandler(BaseStepHandler):
    """Validate middleware before application."""

    @property
    def name(self) -> str:
        return "middleware_validation"

    @property
    def order(self) -> int:
        return 48  # Before middleware application (50)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate middleware chain."""
        middleware_list = vertical.get_middleware()

        if not middleware_list:
            return

        # Validate each middleware
        valid = []
        invalid = []

        for mw in middleware_list:
            if self._is_valid_middleware(mw):
                valid.append(mw)
            else:
                invalid.append(mw)

        # Report invalid middleware
        if invalid:
            result.add_warning(
                f"Found {len(invalid)} invalid middleware items"
            )

        # Store validated list in context for next step
        context.apply_validated_middleware(valid)

        result.add_info(f"Validated {len(valid)} middleware")

    def _is_valid_middleware(self, middleware: Any) -> bool:
        """Check if middleware is valid."""
        # Check for required methods
        if not hasattr(middleware, "process_request"):
            return False
        if not hasattr(middleware, "process_response"):
            return False
        return True
```

---

## Workflow Registration

### Example 9: Workflow Registration with Validation

Register workflows with structure validation:

```python
from typing import Dict, Any

class WorkflowRegistrationHandler(BaseStepHandler):
    """Register workflows with validation."""

    @property
    def name(self) -> str:
        return "workflow_registration"

    @property
    def order(self) -> int:
        return 60  # Same as FrameworkStepHandler

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Register workflows with validation."""
        workflows = vertical.get_workflows()

        if not workflows:
            result.add_info("No workflows to register")
            return

        # Validate workflows
        valid_workflows = {}
        for name, workflow in workflows.items():
            if self._validate_workflow(workflow):
                valid_workflows[name] = workflow
            else:
                result.add_warning(f"Invalid workflow: {name}")

        # Register with global registry
        try:
            from victor.workflows.registry import get_workflow_registry

            registry = get_workflow_registry()
            for name, workflow in valid_workflows.items():
                full_name = f"{vertical.name}:{name}"
                registry.register(full_name, workflow, replace=True)

            result.workflows_count = len(valid_workflows)
            result.add_info(f"Registered {len(valid_workflows)} workflows")

        except ImportError:
            result.add_warning("Workflow registry not available")

    def _validate_workflow(self, workflow: Any) -> bool:
        """Validate workflow structure."""
        # Check for required attributes
        if not hasattr(workflow, "nodes"):
            return False
        if not hasattr(workflow, "edges"):
            return False
        return True

    def _get_step_details(self, result: IntegrationResult) -> dict:
        """Provide step details."""
        return {
            "workflows_count": result.workflows_count,
        }
```

### Example 10: Workflow Trigger Registration

Register auto-workflow triggers:

```python
from typing import List, Tuple

class WorkflowTriggerHandler(BaseStepHandler):
    """Register workflow triggers."""

    @property
    def name(self) -> str:
        return "workflow_triggers"

    @property
    def order(self) -> int:
        return 61  # After workflow registration (60)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Register workflow triggers."""
        # Get triggers from vertical
        triggers = vertical.get_workflow_triggers()

        if not triggers:
            result.add_info("No workflow triggers")
            return

        # Register with trigger registry
        try:
            from victor.workflows.trigger_registry import get_trigger_registry

            trigger_registry = get_trigger_registry()
            trigger_registry.register_from_vertical(vertical.name, triggers)

            result.add_info(f"Registered {len(triggers)} workflow triggers")

        except ImportError:
            result.add_warning("Trigger registry not available")

# Vertical should provide triggers
class MyVertical(VerticalBase):
    @classmethod
    def get_workflow_triggers(cls) -> List[Tuple[str, str]]:
        """Return (pattern, workflow_name) tuples."""
        return [
            (r"analyze.*code", "code_analysis"),
            (r"refactor.*function", "function_refactor"),
        ]
```


**Reading Time:** 7 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Safety & Configuration Management](part-2-safety-config.md)**
