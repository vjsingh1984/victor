# Step Handler Examples

This document provides practical, working examples of step handlers for common vertical extension scenarios.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Tool Management](#tool-management)
3. [Middleware Integration](#middleware-integration)
4. [Workflow Registration](#workflow-registration)
5. [Safety & Security](#safety--security)
6. [Configuration Management](#configuration-management)
7. [Advanced Patterns](#advanced-patterns)
8. [Testing Examples](#testing-examples)

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

---

## Safety & Security

### Example 11: Safety Pattern Validation

Validate safety patterns before application:

```python
from typing import List, Any

class SafetyValidationHandler(BaseStepHandler):
    """Validate safety patterns."""

    @property
    def name(self) -> str:
        return "safety_validation"

    @property
    def order(self) -> int:
        return 28  # Before safety application (30)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate safety patterns."""
        # Get safety extensions
        safety_extensions = vertical.get_safety_extensions()

        if not safety_extensions:
            return

        # Collect and validate patterns
        all_patterns = []
        invalid_count = 0

        for ext in safety_extensions:
            patterns = self._get_extension_patterns(ext)
            for pattern in patterns:
                if self._is_valid_pattern(pattern):
                    all_patterns.append(pattern)
                else:
                    invalid_count += 1

        # Report validation results
        if invalid_count > 0:
            result.add_warning(f"Found {invalid_count} invalid patterns")

        # Store validated patterns in context
        context.apply_validated_safety_patterns(all_patterns)

        result.add_info(f"Validated {len(all_patterns)} safety patterns")

    def _get_extension_patterns(self, extension: Any) -> List[Any]:
        """Get patterns from extension."""
        patterns = []
        if hasattr(extension, "get_bash_patterns"):
            patterns.extend(extension.get_bash_patterns())
        if hasattr(extension, "get_file_patterns"):
            patterns.extend(extension.get_file_patterns())
        return patterns

    def _is_valid_pattern(self, pattern: Any) -> bool:
        """Check if pattern is valid."""
        if not hasattr(pattern, "pattern"):
            return False
        if not hasattr(pattern, "type"):
            return False
        return True
```

### Example 12: Strict Mode Enforcement

Enable strict validation based on settings:

```python
class StrictModeHandler(BaseStepHandler):
    """Enable strict validation mode."""

    @property
    def name(self) -> str:
        return "strict_mode"

    @property
    def order(self) -> int:
        return 32  # After safety (30)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Enable strict mode if configured."""
        # Check if strict mode should be enabled
        if not self._should_enable_strict(orchestrator):
            result.add_info("Strict mode not enabled")
            return

        # Enable via capability
        if _check_capability(orchestrator, "strict_safety"):
            _invoke_capability(orchestrator, "strict_safety", True)
            result.add_info("Enabled strict safety mode")
        else:
            result.add_warning("Strict safety capability not available")

    def _should_enable_strict(self, orchestrator: Any) -> bool:
        """Check if strict mode should be enabled."""
        if hasattr(orchestrator, "settings"):
            return getattr(orchestrator.settings, "strict_mode", False)
        return False
```

---

## Configuration Management

### Example 13: Mode Configuration Application

Apply mode configurations with validation:

```python
from typing import Dict, Any

class ModeConfigurationHandler(BaseStepHandler):
    """Apply mode configurations."""

    @property
    def name(self) -> str:
        return "mode_configuration"

    @property
    def order(self) -> int:
        return 42  # Part of config step (40)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply mode configurations."""
        # Get mode provider from vertical
        provider = self._get_mode_provider(vertical)

        if provider is None:
            result.add_info("No mode configuration provider")
            return

        # Get mode configs
        mode_configs = provider.get_mode_configs()
        default_mode = provider.get_default_mode()
        default_budget = provider.get_default_tool_budget()

        # Validate mode configs
        if not self._validate_mode_configs(mode_configs):
            result.add_error("Invalid mode configurations")
            return

        # Apply to context
        context.apply_mode_configs(mode_configs, default_mode, default_budget)

        # Apply to orchestrator via capability
        if _check_capability(orchestrator, "mode_configs"):
            _invoke_capability(orchestrator, "mode_configs", mode_configs)

        result.mode_configs_count = len(mode_configs)
        result.add_info(f"Applied {len(mode_configs)} mode configs")

    def _get_mode_provider(self, vertical: Type[VerticalBase]) -> Optional[Any]:
        """Get mode provider from vertical."""
        if hasattr(vertical, "get_mode_config_provider"):
            return vertical.get_mode_config_provider()
        return None

    def _validate_mode_configs(self, configs: Dict[str, Any]) -> bool:
        """Validate mode configurations."""
        # Check for required modes
        if not configs:
            return False

        # Validate each mode config
        for name, config in configs.items():
            if not hasattr(config, "exploration"):
                return False
            if not hasattr(config, "tool_budget_multiplier"):
                return False

        return True
```

### Example 14: Stage Configuration

Configure conversation stages:

```python
class StageConfigurationHandler(BaseStepHandler):
    """Configure conversation stages."""

    @property
    def name(self) -> str:
        return "stage_configuration"

    @property
    def order(self) -> int:
        return 41  # Part of config step (40)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply stage configuration."""
        # Get stages from vertical
        stages = vertical.get_stages()

        if not stages:
            result.add_info("No custom stages")
            return

        # Validate stages
        if not self._validate_stages(stages):
            result.add_error("Invalid stage configuration")
            return

        # Apply to context
        context.apply_stages(stages)

        # Log stage names
        stage_names = list(stages.keys())
        result.add_info(f"Applied {len(stages)} stages: {stage_names}")

    def _validate_stages(self, stages: Dict[str, Any]) -> bool:
        """Validate stage configuration."""
        # Check for required stages
        required = {"INITIAL", "EXECUTING", "COMPLETION"}
        return required.issubset(set(stages.keys()))
```

---

## Advanced Patterns

### Example 15: Handler Composition

Compose multiple validation steps:

```python
class CompositeValidationHandler(BaseStepHandler):
    """Compose multiple validation steps."""

    @property
    def name(self) -> str:
        return "composite_validation"

    @property
    def order(self) -> int:
        return 8  # Early, after capability_config (5)

    def __init__(self):
        super().__init__()
        self._validators = [
            self._validate_tools,
            self._validate_prompts,
            self._validate_config,
        ]

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Run all validation steps."""
        passed = 0
        failed = 0

        for validator in self._validators:
            try:
                validator(orchestrator, vertical, context, result)
                passed += 1
            except ValidationError as e:
                result.add_error(f"Validation failed: {e}")
                failed += 1

        result.add_info(f"Validation: {passed} passed, {failed} failed")

    def _validate_tools(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate tools."""
        tools = vertical.get_tools()
        if not tools:
            raise ValidationError("No tools configured")

    def _validate_prompts(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate system prompt."""
        prompt = vertical.get_system_prompt()
        if len(prompt) < 10:
            raise ValidationError("System prompt too short")

    def _validate_config(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate configuration."""
        config = vertical.get_config()
        if not config:
            raise ValidationError("No configuration")


class ValidationError(Exception):
    """Validation error."""
    pass
```

### Example 16: Async Handler

Load resources asynchronously:

```python
class AsyncResourceLoaderHandler(BaseStepHandler):
    """Load resources asynchronously."""

    @property
    def name(self) -> str:
        return "async_resource_loader"

    @property
    def order(self) -> int:
        return 9  # Early, before tools (10)

    async def apply_async(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
        strict_mode: bool = False,
    ) -> None:
        """Apply resources asynchronously."""
        # Load resources asynchronously
        resources = await self._load_resources_async(vertical)

        # Apply to context
        context.apply_resources(resources)

        result.add_info(f"Loaded {len(resources)} resources asynchronously")

    async def _load_resources_async(
        self,
        vertical: Type[VerticalBase],
    ) -> List[Any]:
        """Load resources asynchronously."""
        # Simulate async loading
        import asyncio

        await asyncio.sleep(0.1)  # Simulate I/O

        # Return loaded resources
        return [{"name": "resource1"}, {"name": "resource2"}]
```

### Example 17: Conditional Handler with Retry

Retry operations on failure:

```python
class RetryableOperationHandler(BaseStepHandler):
    """Retry operations on failure."""

    @property
    def name(self) -> str:
        return "retryable_operation"

    @property
    def order(self) -> int:
        return 70  # After framework (60)

    def __init__(self, max_retries: int = 3):
        super().__init__()
        self._max_retries = max_retries

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Perform operation with retry."""
        for attempt in range(self._max_retries):
            try:
                # Attempt operation
                self._perform_operation(orchestrator, vertical, context)
                result.add_info(f"Operation succeeded on attempt {attempt + 1}")
                return
            except Exception as e:
                if attempt < self._max_retries - 1:
                    result.add_warning(f"Attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    result.add_error(f"Operation failed after {self._max_retries} attempts")
                    raise

    def _perform_operation(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
    ) -> None:
        """Perform the actual operation."""
        # Simulate flaky operation
        import random

        if random.random() < 0.7:  # 70% failure rate
            raise RuntimeError("Operation failed")

        # Success
        context.apply_operation_result({"status": "success"})
```

---

## Testing Examples

### Example 18: Testable Handler

Write a handler designed for easy testing:

```python
class TestableHandler(BaseStepHandler):
    """Handler designed for easy testing."""

    @property
    def name(self) -> str:
        return "testable"

    @property
    def order(self) -> int:
        return 25

    def __init__(self, validator=None, processor=None):
        """Initialize with injectable dependencies."""
        super().__init__()
        self._validator = validator or self._default_validator
        self._processor = processor or self._default_processor

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply with testable components."""
        # Validate using injected validator
        if not self._validator(vertical):
            result.add_error("Validation failed")
            return

        # Process using injected processor
        processed = self._processor(vertical.get_data())

        # Apply result
        context.apply_data(processed)
        result.add_info("Processing successful")

    def _default_validator(self, vertical: Type[VerticalBase]) -> bool:
        """Default validation logic."""
        return len(vertical.get_tools()) > 0

    def _default_processor(self, data: Any) -> Any:
        """Default processing logic."""
        return data


# Test with mocks
def test_with_mocks():
    """Test handler with mocked dependencies."""
    # Create mock validator
    mock_validator = MagicMock(return_value=True)

    # Create mock processor
    mock_processor = MagicMock(return_value="processed")

    # Create handler with mocks
    handler = TestableHandler(
        validator=mock_validator,
        processor=mock_processor,
    )

    # Test
    handler._do_apply(orchestrator, vertical, context, result)

    # Verify
    mock_validator.assert_called_once_with(vertical)
    mock_processor.assert_called_once()
```

### Example 19: Handler with Test Hooks

Include hooks for test-specific behavior:

```python
class HandlerWithTestHooks(BaseStepHandler):
    """Handler with test-specific hooks."""

    @property
    def name(self) -> str:
        return "test_hooks"

    @property
    def order(self) -> int:
        return 25

    def __init__(self, test_mode: bool = False):
        """Initialize with optional test mode."""
        super().__init__()
        self._test_mode = test_mode

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply with test-specific behavior."""
        if self._test_mode:
            # Test mode behavior
            self._apply_test_behavior(orchestrator, context, result)
        else:
            # Production behavior
            self._apply_production_behavior(orchestrator, vertical, context, result)

    def _apply_test_behavior(
        self,
        orchestrator: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Test-specific behavior."""
        # Use mock data in tests
        context.apply_data({"test": True})
        result.add_info("Test mode applied")

    def _apply_production_behavior(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Production behavior."""
        # Real logic in production
        data = vertical.get_data()
        context.apply_data(data)
        result.add_info("Production mode applied")


# Test with test mode enabled
def test_handler_with_test_mode():
    """Test handler in test mode."""
    handler = HandlerWithTestHooks(test_mode=True)

    handler._do_apply(orchestrator, vertical, context, result)

    assert context.data == {"test": True}
    assert "Test mode" in result.info[0]
```

---

## Summary

These examples demonstrate common step handler patterns:

**Key Patterns:**
1. **Validation**: Validate before applying (Example 4, 11, 15)
2. **Filtering**: Filter based on conditions (Example 5, 6)
3. **Composition**: Compose multiple operations (Example 15)
4. **Async**: Load resources asynchronously (Example 16)
5. **Retry**: Retry on failure (Example 17)
6. **Testability**: Design for testing (Example 18, 19)

**Best Practices:**
- Use clear, descriptive names
- Choose appropriate order values
- Handle errors gracefully
- Provide step details for observability
- Design for testability

**Next Steps:**
- Review [Step Handler Guide](step_handler_guide.md) for concepts
- See [Migration Guide](step_handler_migration.md) for migration patterns
- Check [Quick Reference](step_handler_quick_reference.md) for API details

**Questions?** See main guide or troubleshooting section
