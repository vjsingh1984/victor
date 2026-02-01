# Step Handler Migration Guide

## Overview

This guide helps you migrate from direct vertical extension patterns to the StepHandlerRegistry-based approach. This migration improves testability, maintainability, and reusability of your vertical extensions.

## Why Migrate?

### Before: Direct Extension

**Problems:**
- Monolithic `apply_to_orchestrator()` methods
- Hard to test individual concerns
- Tight coupling between vertical and orchestrator
- Private attribute access violates SOLID principles
- No clear separation of concerns
- Difficult to extend or modify

**Example:**
```python
class MyVertical(VerticalBase):
    def apply_to_orchestrator(self, orchestrator):
        # Everything in one method
        orchestrator._enabled_tools = self.get_tools()
        orchestrator._middleware = self.get_middleware()
        orchestrator._safety_patterns = self.get_safety_patterns()
        orchestrator._mode_configs = self.get_mode_configs()
        orchestrator._workflows = self.get_workflows()
        # ... 100+ lines
        # Violates SRP, DIP, encapsulation
```

### After: Step Handlers

**Benefits:**
- Focused, testable handlers
- Protocol-based communication
- Clear separation of concerns
- Reusable across verticals
- Observable execution
- SOLID compliant

**Example:**
```python
# Handler 1: Tools
class CustomToolsHandler(BaseStepHandler):
    order = 10
    def _do_apply(self, orchestrator, vertical, context, result):
        tools = vertical.get_tools()
        context.apply_enabled_tools(tools)

# Handler 2: Middleware
class CustomMiddlewareHandler(BaseStepHandler):
    order = 50
    def _do_apply(self, orchestrator, vertical, context, result):
        middleware = vertical.get_middleware()
        context.apply_middleware(middleware)

# Register and use
registry = StepHandlerRegistry.default()
registry.add_handler(CustomToolsHandler())
registry.add_handler(CustomMiddlewareHandler())
```

---

## Migration Patterns

### Pattern 1: Custom Tool Registration

#### Before (Direct Extension)

```python
class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "custom_tool"]

    def apply_to_orchestrator(self, orchestrator):
        # Direct private attribute assignment
        orchestrator._enabled_tools = set(self.get_tools())

        # Custom validation mixed in
        if not orchestrator._enabled_tools:
            raise ValueError("No tools configured")

        # Custom filtering mixed in
        if orchestrator.settings.airgapped_mode:
            orchestrator._enabled_tools = {
                t for t in orchestrator._enabled_tools
                if not t.startswith("web_")
            }
```

**Problems:**
- Private attribute access
- Multiple concerns in one method
- Hard to test validation and filtering separately
- Tightly coupled to orchestrator internals

#### After (Step Handler)

```python
from victor.framework.step_handlers import BaseStepHandler
from victor.core.verticals.base import VerticalBase
from typing import TYPE_CHECKING, Any, List, Set, Type

if TYPE_CHECKING:
    from victor.core.verticals import VerticalContext
    from victor.framework.vertical_integration import IntegrationResult

class CustomToolsHandler(BaseStepHandler):
    """Handle tool registration with validation and filtering."""

    @property
    def name(self) -> str:
        return "custom_tools"

    @property
    def order(self) -> int:
        return 10  # Same as ToolStepHandler

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply tools with validation and filtering."""
        # Get tools from vertical
        tools = vertical.get_tools()

        # Validate
        if not tools:
            result.add_error("No tools configured")
            return

        # Filter based on settings
        if hasattr(orchestrator, "settings"):
            if getattr(orchestrator.settings, "airgapped_mode", False):
                tools = self._filter_airgapped_tools(tools)

        # Apply via context (SOLID compliant)
        context.apply_enabled_tools(set(tools))
        result.tools_applied = set(tools)
        result.add_info(f"Applied {len(tools)} tools")

    def _filter_airgapped_tools(self, tools: List[str]) -> List[str]:
        """Filter out web tools in airgapped mode."""
        return [t for t in tools if not t.startswith("web_")]

    def _get_step_details(self, result: "IntegrationResult") -> dict:
        """Provide step details for observability."""
        return {
            "tools_count": len(result.tools_applied),
            "filtered": len(result.tools_applied) < len(result.tools_applied),
        }

# Vertical is now clean
class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "custom_tool"]

    # No apply_to_orchestrator method needed!
    # Step handler handles it
```

**Benefits:**
- Testable validation and filtering
- Protocol-based (no private attributes)
- Clear separation of concerns
- Reusable handler across verticals

#### Testing the Migrated Handler

```python
import pytest
from unittest.mock import MagicMock

class TestCustomToolsHandler:
    """Tests for CustomToolsHandler."""

    def test_apply_tools_success(self):
        """Test successful tool application."""
        handler = CustomToolsHandler()

        # Mock vertical with tools
        vertical = MagicMock()
        vertical.get_tools.return_value = ["read", "write"]

        # Mock other inputs
        orchestrator = MagicMock()
        orchestrator.settings.airgapped_mode = False
        context = MagicMock()
        result = MagicMock()

        # Execute
        handler._do_apply(orchestrator, vertical, context, result)

        # Verify
        context.apply_enabled_tools.assert_called_once_with({"read", "write"})
        assert result.tools_applied == {"read", "write"}

    def test_filter_airgapped_tools(self):
        """Test airgapped mode filtering."""
        handler = CustomToolsHandler()

        tools = ["read", "web_search", "write", "web_scrape"]
        filtered = handler._filter_airgapped_tools(tools)

        assert filtered == ["read", "write"]
        assert "web_search" not in filtered
        assert "web_scrape" not in filtered

    def test_no_tools_error(self):
        """Test error when no tools configured."""
        handler = CustomToolsHandler()

        # Mock vertical without tools
        vertical = MagicMock()
        vertical.get_tools.return_value = []

        orchestrator = MagicMock()
        context = MagicMock()
        result = MagicMock()

        # Execute
        handler._do_apply(orchestrator, vertical, context, result)

        # Verify error recorded
        result.add_error.assert_called_once_with("No tools configured")
```

---

### Pattern 2: Custom Middleware Injection

#### Before (Direct Extension)

```python
class MyVertical(VerticalBase):
    name = "my_vertical"

    def get_middleware(self):
        return [
            LoggingMiddleware(),
            SecurityMiddleware(),
            CustomCachingMiddleware(),
        ]

    def apply_to_orchestrator(self, orchestrator):
        # Direct private attribute access
        if not hasattr(orchestrator, "_middleware_chain"):
            orchestrator._middleware_chain = MiddlewareChain()

        # Add middleware
        for mw in self.get_middleware():
            orchestrator._middleware_chain.add(mw)

        # Custom ordering logic mixed in
        if hasattr(orchestrator, "_enable_caching"):
            orchestrator._middleware_chain.add(CachingMiddleware(), priority=100)
```

**Problems:**
- Private attribute access
- Middleware creation mixed with application
- Custom ordering logic inlined
- Hard to test middleware selection

#### After (Step Handler)

```python
from victor.framework.step_handlers import BaseStepHandler

class CustomMiddlewareHandler(BaseStepHandler):
    """Handle middleware injection with custom ordering."""

    @property
    def name(self) -> str:
        return "custom_middleware"

    @property
    def order(self) -> int:
        return 50  # After extensions, before framework

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply middleware with custom ordering."""
        # Get middleware from vertical
        middleware_list = vertical.get_middleware()

        # Add caching if enabled
        if self._should_enable_caching(orchestrator):
            middleware_list.append(CachingMiddleware())

        # Order by priority
        middleware_list = self._order_middleware(middleware_list)

        # Apply via context
        context.apply_middleware(middleware_list)
        result.middleware_count = len(middleware_list)
        result.add_info(f"Applied {len(middleware_list)} middleware")

    def _should_enable_caching(self, orchestrator: Any) -> bool:
        """Check if caching should be enabled."""
        if hasattr(orchestrator, "settings"):
            return getattr(orchestrator.settings, "enable_caching", False)
        return False

    def _order_middleware(self, middleware_list: List[Any]) -> List[Any]:
        """Order middleware by priority attribute."""
        def get_priority(mw):
            return getattr(mw, "priority", 50)

        return sorted(middleware_list, key=get_priority)

# Vertical provides middleware, handler applies it
class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_middleware(cls) -> List[Any]:
        return [
            LoggingMiddleware(priority=10),
            SecurityMiddleware(priority=20),
            CustomCachingMiddleware(priority=30),
        ]
```

**Benefits:**
- Separation of middleware creation and application
- Testable ordering logic
- Protocol-based application
- Reusable handler

---

### Pattern 3: Custom Workflow Registration

#### Before (Direct Extension)

```python
class MyVertical(VerticalBase):
    name = "my_vertical"

    def get_workflows(self):
        return {
            "analyze": self._create_analyze_workflow(),
            "refactor": self._create_refactor_workflow(),
        }

    def apply_to_orchestrator(self, orchestrator):
        # Direct registration
        from victor.workflows.registry import get_workflow_registry

        registry = get_workflow_registry()
        for name, workflow in self.get_workflows().items():
            registry.register(f"{self.name}:{name}", workflow)

        # Auto-workflow triggers mixed in
        orchestrator._workflow_triggers = {
            r"analyze.*": "analyze",
            r"refactor.*": "refactor",
        }
```

**Problems:**
- Workflow creation mixed with registration
- Private attribute access for triggers
- No validation of workflow structure
- Hard to test registration logic

#### After (Step Handler)

```python
from victor.framework.step_handlers import BaseStepHandler

class CustomWorkflowHandler(BaseStepHandler):
    """Handle workflow registration with validation."""

    @property
    def name(self) -> str:
        return "custom_workflows"

    @property
    def order(self) -> int:
        return 60  # Same as FrameworkStepHandler

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: "VerticalContext",
        result: "IntegrationResult",
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

            # Register triggers
            self._register_workflow_triggers(
                orchestrator, vertical, valid_workflows, result
            )

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

    def _register_workflow_triggers(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        workflows: Dict[str, Any],
        result: "IntegrationResult",
    ) -> None:
        """Register auto-workflow triggers."""
        # Get triggers from vertical
        triggers = vertical.get_workflow_triggers()

        if not triggers:
            return

        try:
            from victor.workflows.trigger_registry import get_trigger_registry

            trigger_registry = get_trigger_registry()
            trigger_registry.register_from_vertical(vertical.name, triggers)

            result.add_info(f"Registered {len(triggers)} workflow triggers")

        except ImportError:
            result.add_warning("Trigger registry not available")

# Vertical defines workflows, handler registers them
class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_workflows(cls) -> Dict[str, Any]:
        return {
            "analyze": cls._create_analyze_workflow(),
            "refactor": cls._create_refactor_workflow(),
        }

    @classmethod
    def get_workflow_triggers(cls) -> List[Tuple[str, str]]:
        """Return (pattern, workflow_name) tuples."""
        return [
            (r"analyze.*", "analyze"),
            (r"refactor.*", "refactor"),
        ]
```

**Benefits:**
- Workflow validation before registration
- Separation of creation and registration
- Proper trigger registry integration
- Testable validation logic

---

### Pattern 4: Custom Safety Patterns

#### Before (Direct Extension)

```python
class MyVertical(VerticalBase):
    name = "my_vertical"

    def get_safety_extensions(self):
        return [
            BashSafetyExtension(),
            FileSafetyExtension(),
            CustomSecurityExtension(),
        ]

    def apply_to_orchestrator(self, orchestrator):
        # Direct safety checker access
        if not hasattr(orchestrator, "safety_checker"):
            orchestrator.safety_checker = SafetyChecker()

        # Collect patterns
        all_patterns = []
        for ext in self.get_safety_extensions():
            all_patterns.extend(ext.get_bash_patterns())
            all_patterns.extend(ext.get_file_patterns())

        # Apply patterns
        orchestrator.safety_checker.add_patterns(all_patterns)

        # Custom validation mixed in
        if orchestrator.settings.strict_mode:
            orchestrator.safety_checker.enable_strict_validation()
```

**Problems:**
- Safety checker creation mixed with pattern application
- Custom validation inlined
- No pattern validation
- Hard to test safety logic

#### After (Step Handler)

```python
from victor.framework.step_handlers import BaseStepHandler

class CustomSafetyHandler(BaseStepHandler):
    """Handle safety pattern application with validation."""

    @property
    def name(self) -> str:
        return "custom_safety"

    @property
    def order(self) -> int:
        return 30  # Same as SafetyStepHandler

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply safety patterns with validation."""
        # Get safety extensions
        safety_extensions = vertical.get_safety_extensions()

        if not safety_extensions:
            result.add_info("No safety extensions")
            return

        # Collect and validate patterns
        all_patterns = []
        for ext in safety_extensions:
            patterns = self._get_extension_patterns(ext)
            validated = self._validate_patterns(patterns)
            all_patterns.extend(validated)

        # Apply via context
        context.apply_safety_patterns(all_patterns)
        result.safety_patterns_count = len(all_patterns)
        result.add_info(f"Applied {len(all_patterns)} safety patterns")

        # Enable strict mode if needed
        if self._should_enable_strict(orchestrator):
            self._enable_strict_validation(orchestrator)

    def _get_extension_patterns(self, extension: Any) -> List[Any]:
        """Get patterns from a single extension."""
        patterns = []
        if hasattr(extension, "get_bash_patterns"):
            patterns.extend(extension.get_bash_patterns())
        if hasattr(extension, "get_file_patterns"):
            patterns.extend(extension.get_file_patterns())
        return patterns

    def _validate_patterns(self, patterns: List[Any]) -> List[Any]:
        """Validate individual patterns."""
        valid = []
        for pattern in patterns:
            if self._is_valid_pattern(pattern):
                valid.append(pattern)
            else:
                logger.warning(f"Invalid safety pattern: {pattern}")
        return valid

    def _is_valid_pattern(self, pattern: Any) -> bool:
        """Check if pattern has required structure."""
        # Check for required attributes
        if not hasattr(pattern, "pattern"):
            return False
        if not hasattr(pattern, "type"):
            return False
        return True

    def _should_enable_strict(self, orchestrator: Any) -> bool:
        """Check if strict mode should be enabled."""
        if hasattr(orchestrator, "settings"):
            return getattr(orchestrator.settings, "strict_mode", False)
        return False

    def _enable_strict_validation(self, orchestrator: Any) -> None:
        """Enable strict validation."""
        # Use capability-based approach
        if _check_capability(orchestrator, "strict_safety"):
            _invoke_capability(orchestrator, "strict_safety", True)
```

**Benefits:**
- Pattern validation before application
- Testable validation logic
- Protocol-based strict mode
- Clear separation of concerns

---

## Testing Migrated Extensions

### Unit Testing Strategy

After migration, each handler should have dedicated unit tests:

```python
# tests/unit/framework/test_custom_handlers.py

import pytest
from unittest.mock import MagicMock, PropertyMock
from victor.framework.step_handlers import BaseStepHandler

class TestCustomToolsHandler:
    """Test suite for CustomToolsHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return CustomToolsHandler()

    @pytest.fixture
    def mock_vertical(self):
        """Create mock vertical."""
        vertical = MagicMock()
        vertical.get_tools.return_value = ["read", "write"]
        return vertical

    def test_apply_tools_success(self, handler, mock_vertical):
        """Test successful tool application."""
        orchestrator = MagicMock()
        context = MagicMock()
        result = MagicMock()

        handler._do_apply(orchestrator, mock_vertical, context, result)

        context.apply_enabled_tools.assert_called_once()
        assert result.tools_applied == {"read", "write"}

    def test_filter_airgapped_tools(self, handler):
        """Test airgapped filtering."""
        tools = ["read", "web_search", "write"]
        filtered = handler._filter_airgapped_tools(tools)

        assert "web_search" not in filtered
        assert "read" in filtered

    def test_no_tools_error(self, handler, mock_vertical):
        """Test error handling when no tools."""
        mock_vertical.get_tools.return_value = []

        orchestrator = MagicMock()
        context = MagicMock()
        result = MagicMock()

        handler._do_apply(orchestrator, mock_vertical, context, result)

        result.add_error.assert_called_once()
```

### Integration Testing Strategy

Test the handler within the full pipeline:

```python
# tests/integration/framework/test_custom_handler_integration.py

import pytest
from victor.framework.vertical_integration import VerticalIntegrationPipeline
from victor.framework.step_handlers import StepHandlerRegistry

class TestCustomHandlerIntegration:
    """Test custom handler in full pipeline."""

    @pytest.fixture
    def registry(self):
        """Create registry with custom handler."""
        registry = StepHandlerRegistry.default()
        registry.add_handler(CustomToolsHandler())
        return registry

    @pytest.fixture
    def pipeline(self, registry):
        """Create pipeline with custom registry."""
        return VerticalIntegrationPipeline(step_registry=registry)

    def test_custom_handler_executes(self, pipeline, orchestrator):
        """Test that custom handler executes in pipeline."""
        result = pipeline.apply(orchestrator, MyVertical)

        assert result.success
        assert any("custom_tools" in status for status in result.step_status)

    def test_custom_handler_order(self, registry):
        """Test that custom handler executes in correct order."""
        handlers = registry.get_ordered_handlers()
        custom_idx = next(
            i for i, h in enumerate(handlers) if h.name == "custom_tools"
        )

        # Should be after capability_config (5)
        # Should be before prompt (20)
        assert handlers[custom_idx - 1].order < 10
        assert handlers[custom_idx + 1].order > 10
```

---

## Common Migration Mistakes

### Mistake 1: Not Using Context Methods

**Wrong:**
```python
def _do_apply(self, orchestrator, vertical, context, result):
    # Direct assignment - doesn't work!
    context.enabled_tools = {"read", "write"}
```

**Right:**
```python
def _do_apply(self, orchestrator, vertical, context, result):
    # Use context methods
    context.apply_enabled_tools({"read", "write"})
```

### Mistake 2: Private Attribute Access

**Wrong:**
```python
def _do_apply(self, orchestrator, vertical, context, result):
    # Private access - violates SOLID
    orchestrator._enabled_tools = {"read", "write"}
```

**Right:**
```python
def _do_apply(self, orchestrator, vertical, context, result):
    # Use capability-based approach
    if _check_capability(orchestrator, "enabled_tools"):
        _invoke_capability(orchestrator, "enabled_tools", {"read", "write"})
```

### Mistake 3: No Error Handling

**Wrong:**
```python
def _do_apply(self, orchestrator, vertical, context, result):
    # No error handling
    value = vertical.get_value()
    context.apply_value(value)
```

**Right:**
```python
def _do_apply(self, orchestrator, vertical, context, result):
    try:
        value = vertical.get_value()
        context.apply_value(value)
        result.add_info(f"Applied: {value}")
    except Exception as e:
        result.add_warning(f"Could not apply value: {e}")
```

### Mistake 4: Wrong Order

**Wrong:**
```python
class DependsOnToolsHandler(BaseStepHandler):
    order = 5  # Runs before tools - can't access tools!

    def _do_apply(self, orchestrator, vertical, context, result):
        # Tools not applied yet - will fail
        tools = context.enabled_tools
```

**Right:**
```python
class DependsOnToolsHandler(BaseStepHandler):
    order = 15  # Runs after tools (10)

    def _do_apply(self, orchestrator, vertical, context, result):
        # Tools already applied
        tools = context.enabled_tools
```

---

## Migration Checklist

Use this checklist to ensure complete migration:

### Phase 1: Analysis
- [ ] Identify all custom extension logic in vertical
- [ ] Group related functionality into logical handlers
- [ ] Determine appropriate order for each handler
- [ ] Identify dependencies between handlers

### Phase 2: Implementation
- [ ] Create handler classes inheriting from `BaseStepHandler`
- [ ] Implement `_do_apply()` methods
- [ ] Add error handling and logging
- [ ] Implement `_get_step_details()` for observability
- [ ] Remove old `apply_to_orchestrator()` methods

### Phase 3: Registration
- [ ] Create `StepHandlerRegistry` instance
- [ ] Register custom handlers
- [ ] Pass registry to `VerticalIntegrationPipeline`
- [ ] Verify handlers execute in correct order

### Phase 4: Testing
- [ ] Write unit tests for each handler
- [ ] Write integration tests for full pipeline
- [ ] Test error conditions
- [ ] Verify observability (step status, details)

### Phase 5: Documentation
- [ ] Document handler purpose and order
- [ ] Add examples to vertical documentation
- [ ] Update migration notes
- [ ] Document any breaking changes

---

## Advanced Migration Scenarios

### Scenario 1: Migrating Complex Multi-Step Logic

**Before:**
```python
class ComplexVertical(VerticalBase):
    def apply_to_orchestrator(self, orchestrator):
        # Step 1: Setup
        self._setup_tools(orchestrator)
        # Step 2: Validate
        self._validate_config(orchestrator)
        # Step 3: Apply
        self._apply_config(orchestrator)
        # Step 4: Cleanup
        self._cleanup(orchestrator)
```

**After:**
```python
# Split into multiple handlers
class SetupToolsHandler(BaseStepHandler):
    order = 10
    def _do_apply(self, orchestrator, vertical, context, result):
        # Setup logic
        pass

class ValidateConfigHandler(BaseStepHandler):
    order = 20
    def _do_apply(self, orchestrator, vertical, context, result):
        # Validation logic
        pass

class ApplyConfigHandler(BaseStepHandler):
    order = 30
    def _do_apply(self, orchestrator, vertical, context, result):
        # Application logic
        pass

class CleanupHandler(BaseStepHandler):
    order = 100
    def _do_apply(self, orchestrator, vertical, context, result):
        # Cleanup logic
        pass
```

### Scenario 2: Migrating Conditional Logic

**Before:**
```python
def apply_to_orchestrator(self, orchestrator):
    if self.name == "coding":
        self._apply_coding_extensions(orchestrator)
    elif self.name == "research":
        self._apply_research_extensions(orchestrator)
    # ...
```

**After:**
```python
class CodingExtensionsHandler(BaseStepHandler):
    order = 25
    def _do_apply(self, orchestrator, vertical, context, result):
        if vertical.name != "coding":
            return
        # Coding-specific logic

class ResearchExtensionsHandler(BaseStepHandler):
    order = 25
    def _do_apply(self, orchestrator, vertical, context, result):
        if vertical.name != "research":
            return
        # Research-specific logic
```

---

## Summary

Migrating to step handlers provides significant benefits:

**Key Improvements:**
- **Testability**: Unit test each handler independently
- **Maintainability**: Clear separation of concerns
- **Reusability**: Share handlers across verticals
- **Observability**: Per-step status tracking
- **SOLID Compliance**: Protocol-based communication

**Migration Steps:**
1. Identify extension logic
2. Create focused handler classes
3. Register handlers with StepHandlerRegistry
4. Write comprehensive tests
5. Update documentation

**Next Steps:**
- Review [Step Handler Examples](step_handler_examples.md) for more patterns
- See [Quick Reference](step_handler_quick_reference.md) for API details
- Check [Vertical Development Guide](../reference/internals/VERTICAL_DEVELOPMENT_GUIDE.md) for best practices

**Questions?** See [Troubleshooting](step_handler_guide.md#troubleshooting) in the main guide
