"""Custom Step Handlers Example - Demonstrating StepHandlerRegistry as Primary Extension Surface

This file provides practical examples of custom step handlers that extend
vertical integration capabilities in a SOLID-compliant, testable way.

**StepHandlerRegistry is the PRIMARY EXTENSION MECHANISM for verticals.**

See docs/extensions/ for complete documentation:
- step_handler_guide.md - Comprehensive guide
- step_handler_migration.md - Migrating from direct extension
- step_handler_examples.md - More examples
- step_handler_quick_reference.md - Quick reference card
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type

from victor.framework.step_handlers import BaseStepHandler, StepHandlerRegistry

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase
    from victor.core.verticals import VerticalContext
    from victor.framework.vertical_integration import IntegrationResult

logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Custom Tool Registration with Validation
# =============================================================================


class ValidatedToolsHandler(BaseStepHandler):
    """Handle tool registration with validation and filtering.

    Replaces direct tool registration in apply_to_orchestrator().
    """

    @property
    def name(self) -> str:
        return "validated_tools"

    @property
    def order(self) -> int:
        return 10  # Same as ToolStepHandler (can replace or complement)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply tools with validation."""
        tools = vertical.get_tools()

        if not tools:
            result.add_error("No tools configured")
            return

        # Validate tools
        validated_tools = self._validate_tools(tools)

        # Filter based on settings
        if hasattr(orchestrator, "settings"):
            if getattr(orchestrator.settings, "airgapped_mode", False):
                validated_tools = self._filter_airgapped_tools(validated_tools)

        # Apply via context (SOLID compliant)
        context.apply_enabled_tools(set(validated_tools))
        result.tools_applied = set(validated_tools)
        result.add_info(f"Applied {len(validated_tools)} validated tools")

    def _validate_tools(self, tools: List[str]) -> List[str]:
        """Validate tool names and requirements."""
        # Add custom validation logic here
        return tools

    def _filter_airgapped_tools(self, tools: List[str]) -> List[str]:
        """Filter out web tools in airgapped mode."""
        return [t for t in tools if not t.startswith("web_")]


# =============================================================================
# Example 2: Custom Middleware Injection with Ordering
# =============================================================================


class OrderedMiddlewareHandler(BaseStepHandler):
    """Handle middleware injection with custom ordering logic.

    Replaces direct middleware manipulation in apply_to_orchestrator().
    """

    @property
    def name(self) -> str:
        return "ordered_middleware"

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
        """Apply middleware with custom ordering."""
        middleware_list = vertical.get_middleware()

        # Add caching if enabled
        if self._should_enable_caching(orchestrator):
            from victor.framework.middleware import CachingMiddleware

            middleware_list.append(CachingMiddleware())

        # Order by priority
        middleware_list = self._order_middleware(middleware_list)

        # Apply via context
        context.apply_middleware(middleware_list)
        result.middleware_count = len(middleware_list)
        result.add_info(f"Applied {len(middleware_list)} ordered middleware")

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


# =============================================================================
# Example 3: Custom Workflow Registration with Validation
# =============================================================================


class ValidatedWorkflowHandler(BaseStepHandler):
    """Handle workflow registration with validation.

    Replaces direct workflow registration in apply_to_orchestrator().
    """

    @property
    def name(self) -> str:
        return "validated_workflows"

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

            # Register triggers
            self._register_workflow_triggers(orchestrator, vertical, valid_workflows, result)

        except ImportError:
            result.add_warning("Workflow registry not available")

    def _validate_workflow(self, workflow: Any) -> bool:
        """Validate workflow structure."""
        return hasattr(workflow, "nodes") and hasattr(workflow, "edges")

    def _register_workflow_triggers(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        workflows: Dict[str, Any],
        result: IntegrationResult,
    ) -> None:
        """Register auto-workflow triggers."""
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


# =============================================================================
# Example 4: Custom Safety Pattern Validation
# =============================================================================


class ValidatedSafetyHandler(BaseStepHandler):
    """Handle safety pattern application with validation.

    Replaces direct safety pattern application in apply_to_orchestrator().
    """

    @property
    def name(self) -> str:
        return "validated_safety"

    @property
    def order(self) -> int:
        return 30  # Same as SafetyStepHandler

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply safety patterns with validation."""
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
        result.add_info(f"Applied {len(all_patterns)} validated safety patterns")

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
        return hasattr(pattern, "pattern") and hasattr(pattern, "type")


# =============================================================================
# Example 5: Custom Event Handler Integration
# =============================================================================


class EventHandlerIntegrationHandler(BaseStepHandler):
    """Integrate custom event handlers with the event bus.

    Demonstrates extending vertical integration with custom event handling.
    """

    @property
    def name(self) -> str:
        return "event_handler_integration"

    @property
    def order(self) -> int:
        return 55  # Between extensions (45) and framework (60)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Integrate event handlers."""
        if not hasattr(vertical, "get_event_handlers"):
            return

        handlers = vertical.get_event_handlers()

        if not handlers:
            return

        # Register event handlers
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            if bus:
                for topic, handler in handlers.items():
                    # Subscribe handler to topic
                    # Implementation depends on event bus API
                    result.add_info(f"Registered event handler for topic: {topic}")
                result.add_info(f"Registered {len(handlers)} event handlers")
            else:
                result.add_warning("Event bus not available")

        except ImportError:
            result.add_warning("Event system not available")


# =============================================================================
# Example 6: Custom Validation Logic
# =============================================================================


class CustomValidationHandler(BaseStepHandler):
    """Apply custom validation logic to vertical configuration.

    Demonstrates adding validation as a cross-cutting concern.
    """

    @property
    def name(self) -> str:
        return "custom_validation"

    @property
    def order(self) -> int:
        return 5  # Very early, before capability_config

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply custom validation."""
        # Validate vertical configuration
        errors = []

        # Example: Check for required tools
        required_tools = getattr(vertical, "REQUIRED_TOOLS", [])
        available_tools = vertical.get_tools()
        missing_tools = set(required_tools) - set(available_tools)

        if missing_tools:
            errors.append(f"Missing required tools: {missing_tools}")

        # Example: Check system prompt length
        prompt = vertical.get_system_prompt()
        if len(prompt) < 50:
            errors.append("System prompt too short (min 50 chars)")

        # Report validation results
        if errors:
            for error in errors:
                result.add_error(f"Validation failed: {error}")
        else:
            result.add_info("Validation passed")


# =============================================================================
# Example 7: Vertical-Specific Configuration
# =============================================================================


class VerticalConfigHandler(BaseStepHandler):
    """Handle vertical-specific configuration setup.

    Demonstrates vertical-specific customization through step handlers.
    """

    @property
    def name(self) -> str:
        return "vertical_config"

    @property
    def order(self) -> int:
        return 25  # After prompt (20), before safety (30)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply vertical-specific configuration."""
        # Only apply to specific verticals
        if vertical.name not in ["coding", "research"]:
            result.add_info(f"Skipped {vertical.name} (coding/research only)")
            return

        # Apply vertical-specific settings
        if vertical.name == "coding":
            self._apply_coding_config(orchestrator, context, result)
        elif vertical.name == "research":
            self._apply_research_config(orchestrator, context, result)

    def _apply_coding_config(
        self,
        orchestrator: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply coding-specific configuration."""
        # Set coding-specific parameters
        config = {
            "max_edit_size": 1000,
            "enable_ast_analysis": True,
            "code_review_threshold": 0.8,
        }

        context.apply_capability_configs({"coding": config})
        result.add_info("Applied coding-specific configuration")

    def _apply_research_config(
        self,
        orchestrator: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply research-specific configuration."""
        # Set research-specific parameters
        config = {
            "max_sources": 20,
            "enable_citation_validation": True,
            "min_source_reliability": 0.7,
        }

        context.apply_capability_configs({"research": config})
        result.add_info("Applied research-specific configuration")


# =============================================================================
# Example 8: Composite Handler (Multiple Steps)
# =============================================================================


class CompositeValidationHandler(BaseStepHandler):
    """Compose multiple validation steps into one handler.

    Demonstrates composing multiple concerns in a single handler.
    """

    def __init__(self):
        """Initialize with multiple validators."""
        self._validators = [
            self._validate_tools,
            self._validate_prompts,
            self._validate_config,
        ]
        self._strict_mode = True

    @property
    def name(self) -> str:
        return "composite_validation"

    @property
    def order(self) -> int:
        return 6  # After capability_config (5), before tools (10)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Run all validators."""
        for validator in self._validators:
            try:
                validator(orchestrator, vertical, context, result)
            except Exception as e:
                result.add_error(f"Validation failed: {e}")
                if self._strict_mode:
                    raise

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
            result.add_warning("No tools configured")

    def _validate_prompts(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate prompts."""
        prompt = vertical.get_system_prompt()
        if not prompt:
            result.add_error("No system prompt configured")

    def _validate_config(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate config."""
        config = vertical.get_config()
        if not config:
            result.add_warning("No configuration provided")


# =============================================================================
# Usage Examples
# =============================================================================


def example_basic_usage():
    """Demonstrate basic usage of custom step handlers."""
    # Create registry with default handlers
    registry = StepHandlerRegistry.default()

    # Add custom handler
    registry.add_handler(ValidatedToolsHandler())

    # Create pipeline with custom registry
    from victor.framework.vertical_integration import VerticalIntegrationPipeline

    pipeline = VerticalIntegrationPipeline(step_registry=registry)

    # Apply vertical
    # result = pipeline.apply(orchestrator, MyVertical)


def example_multiple_custom_handlers():
    """Demonstrate using multiple custom handlers."""
    registry = StepHandlerRegistry.default()

    # Add multiple custom handlers
    registry.add_handler(ValidatedToolsHandler())
    registry.add_handler(OrderedMiddlewareHandler())
    registry.add_handler(ValidatedWorkflowHandler())
    registry.add_handler(ValidatedSafetyHandler())

    # Handlers are automatically sorted by order
    pipeline = VerticalIntegrationPipeline(step_registry=registry)

    # Apply vertical
    # result = pipeline.apply(orchestrator, MyVertical)


def example_replace_built_in_handler():
    """Demonstrate replacing a built-in handler."""
    registry = StepHandlerRegistry.default()

    # Remove default tools handler
    registry.remove_handler("tools")

    # Add custom tools handler
    registry.add_handler(ValidatedToolsHandler())

    # Custom handler will replace default
    pipeline = VerticalIntegrationPipeline(step_registry=registry)


def example_conditional_handler():
    """Demonstrate conditional handler execution."""
    # Create handler that only applies to specific verticals
    registry = StepHandlerRegistry.default()
    registry.add_handler(VerticalConfigHandler())

    # Handler will check vertical type and only apply if needed
    pipeline = VerticalIntegrationPipeline(step_registry=registry)


# =============================================================================
# Integration Test Example
# =============================================================================


def test_custom_handlers():
    """Test custom handlers in isolation."""
    import unittest
    from unittest.mock import MagicMock

    class TestValidatedToolsHandler(unittest.TestCase):
        """Tests for ValidatedToolsHandler."""

        def test_apply_tools_success(self):
            """Test successful tool application."""
            handler = ValidatedToolsHandler()

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
            context.apply_enabled_tools.assert_called_once()
            assert result.tools_applied == {"read", "write"}

        def test_filter_airgapped_tools(self):
            """Test airgapped mode filtering."""
            handler = ValidatedToolsHandler()

            tools = ["read", "web_search", "write", "web_scrape"]
            filtered = handler._filter_airgapped_tools(tools)

            assert "web_search" not in filtered
            assert "read" in filtered


if __name__ == "__main__":
    # Run basic example
    example_basic_usage()

    print("Custom Step Handlers Examples")
    print("=" * 60)
    print("See docs/extensions/ for complete documentation:")
    print("  - step_handler_guide.md (comprehensive guide)")
    print("  - step_handler_migration.md (migration guide)")
    print("  - step_handler_examples.md (more examples)")
    print("  - step_handler_quick_reference.md (quick reference)")
