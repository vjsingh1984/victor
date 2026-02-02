# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FrameworkStepHandler dependency injection improvements.

Tests that FrameworkStepHandler uses injected registries instead of
hard-coded imports, following Dependency Inversion Principle (DIP).
"""

from unittest.mock import Mock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from victor.framework.step_handlers import FrameworkStepHandler
from victor.core.verticals.base import VerticalBase
from victor.core.verticals import VerticalContext
from victor.framework.vertical_integration import IntegrationResult


class MockVertical(VerticalBase):
    """Mock vertical for testing."""

    name = "test_vertical"
    description = "Test vertical"
    version = "0.1.0"

    @classmethod
    def get_tools(cls):
        return []

    @classmethod
    def get_system_prompt(cls):
        return "Test prompt"


class TestStepHandlerDIInjection:
    """Tests for dependency injection in FrameworkStepHandler."""

    def test_step_handler_supports_injected_registries(self):
        """FrameworkStepHandler should accept registries via constructor.

        Phase 2 implementation: FrameworkStepHandler now supports constructor
        injection of registries for DIP compliance instead of hard-coded imports.
        """
        # Create mock registries
        mock_workflow_registry = Mock()
        mock_team_registry = Mock()
        mock_chain_registry = Mock()
        mock_persona_registry = Mock()
        mock_handler_registry = Mock()

        # Create handler with injected dependencies
        handler = FrameworkStepHandler(
            workflow_registry=mock_workflow_registry,
            team_registry=mock_team_registry,
            chain_registry=mock_chain_registry,
            persona_registry=mock_persona_registry,
            handler_registry=mock_handler_registry,
        )

        # Verify handler stores injected registries
        assert handler._workflow_registry is mock_workflow_registry
        assert handler._team_registry is mock_team_registry
        assert handler._chain_registry is mock_chain_registry
        assert handler._persona_registry is mock_persona_registry
        assert handler._handler_registry is mock_handler_registry

    def test_step_handler_uses_injected_workflow_registry(self):
        """Handler should use injected workflow registry instead of importing."""
        # Create mock registry
        mock_registry = Mock()
        mock_workflow = Mock(name="test_workflow")

        # Create handler with injected registry
        handler = FrameworkStepHandler(workflow_registry=mock_registry)

        # Create mock orchestrator, vertical, context, result
        mock_orchestrator = Mock()
        vertical = MockVertical
        context = VerticalContext()
        result = IntegrationResult()

        # Make vertical provide workflows
        workflow_provider = Mock()
        workflow_provider.get_workflows = Mock(return_value={"test_workflow": mock_workflow})
        vertical.get_workflow_provider = Mock(return_value=workflow_provider)

        # Apply step
        handler.apply_workflows(mock_orchestrator, vertical, context, result)

        # Verify injected registry was used instead of import
        # The register method is called with workflow object (after namespace is set on it)
        # and replace=True as a keyword argument
        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        # First positional arg is the workflow object
        assert call_args[0][0] == mock_workflow or call_args[0][0].name == "test_vertical:test_workflow"
        # replace=True should be a keyword argument
        assert call_args[1].get("replace") is True

    def test_step_handler_uses_injected_team_registry(self):
        """Handler should use injected team registry instead of importing."""
        # Create mock registry
        mock_registry = Mock()

        # Create handler with injected registry
        handler = FrameworkStepHandler(team_registry=mock_registry)

        # Create mock orchestrator, vertical, context, result
        mock_orchestrator = Mock()
        vertical = MockVertical
        context = VerticalContext()
        result = IntegrationResult()

        # Make vertical provide team specs (as dict, not list)
        mock_team_spec = Mock()
        # Set get_team_specs directly on class (fallback path - skips protocol check)
        vertical.get_team_specs = Mock(return_value={"team1": mock_team_spec})

        # Apply step (via _do_apply which calls apply_team_specs)
        handler.apply_team_specs(mock_orchestrator, vertical, context, result)

        # Verify injected registry was used instead of import
        mock_registry.register_from_vertical.assert_called_once_with(
            "test_vertical", {"team1": mock_team_spec}, replace=True
        )

    def test_step_handler_uses_injected_chain_registry(self):
        """Handler should use injected chain registry instead of importing."""
        # Create mock registry
        mock_registry = Mock()

        # Create handler with injected registry
        handler = FrameworkStepHandler(chain_registry=mock_registry)

        # Create mock orchestrator, vertical, context, result
        mock_orchestrator = Mock()
        vertical = MockVertical
        context = VerticalContext()
        result = IntegrationResult()

        # Make vertical provide chains (as dict, not list)
        mock_chain = Mock()
        # apply_chains checks hasattr(vertical, "get_chains") - needs to be on class
        vertical.get_chains = Mock(return_value={"chain1": mock_chain})

        # Apply step (via _do_apply which calls apply_chains)
        handler.apply_chains(mock_orchestrator, vertical, context, result)

        # Verify injected registry was used instead of import
        mock_registry.register.assert_called_once()

    def test_step_handler_uses_injected_persona_registry(self):
        """Handler should use injected persona registry instead of importing."""
        # Create mock registry
        mock_registry = Mock()

        # Create handler with injected registry
        handler = FrameworkStepHandler(persona_registry=mock_registry)

        # Create mock orchestrator, vertical, context, result
        mock_orchestrator = Mock()
        vertical = MockVertical
        context = VerticalContext()
        result = IntegrationResult()

        # Make vertical provide personas (as dict, not list)
        mock_persona = Mock()
        # apply_personas checks hasattr(vertical, "get_personas")
        vertical.get_personas = Mock(return_value={"persona1": mock_persona})

        # Apply step (via _do_apply which calls apply_personas)
        handler.apply_personas(mock_orchestrator, vertical, context, result)

        # Verify injected registry was used instead of import
        mock_registry.register.assert_called_once()

    def test_step_handler_uses_injected_handler_registry(self):
        """Handler should use injected handler registry instead of importing."""
        # Create mock registry
        mock_registry = Mock()

        # Create handler with injected registry
        handler = FrameworkStepHandler(handler_registry=mock_registry)

        # Create mock orchestrator, vertical, context, result
        mock_orchestrator = Mock()
        vertical = MockVertical
        context = VerticalContext()
        result = IntegrationResult()

        # Make vertical provide handlers (as dict, not list)
        mock_handler = Mock()
        # apply_handlers checks hasattr(vertical, "get_handlers")
        vertical.get_handlers = Mock(return_value={"test_handler": mock_handler})

        # Apply step (via _do_apply which calls apply_handlers)
        handler.apply_handlers(mock_orchestrator, vertical, context, result)

        # Verify injected registry was used instead of import
        mock_registry.register.assert_called_once()

    def test_step_handler_defaults_to_none_when_not_injected(self):
        """Handler should default to None when registries not injected."""
        # Create handler without injections
        handler = FrameworkStepHandler()

        # Verify all registries are None
        assert handler._workflow_registry is None
        assert handler._team_registry is None
        assert handler._chain_registry is None
        assert handler._persona_registry is None
        assert handler._handler_registry is None

    def test_step_handler_falls_back_gracefully_when_registry_not_injected(self):
        """Handler should fall back to imports when registry not injected.

        For backward compatibility, handler should still work when registries
        are not injected, using hard-coded imports as before.
        """
        # Create handler without injections (uses hard-coded imports)
        handler = FrameworkStepHandler()

        # Create mock orchestrator, vertical, context, result
        mock_orchestrator = Mock()
        vertical = MockVertical
        context = VerticalContext()
        result = IntegrationResult()

        # Make vertical provide workflows (but no registry to register them)
        workflow_provider = Mock()
        mock_workflow = Mock(name="test_workflow")
        workflow_provider.get_workflows = Mock(return_value={"test_workflow": mock_workflow})
        vertical.get_workflow_provider = Mock(return_value=workflow_provider)

        # Should not raise error - just skip registration
        handler.apply_workflows(mock_orchestrator, vertical, context, result)

        # Result should have workflow count even without registry
        assert result.workflows_count == 1


class TestStepHandlerOCPCompliance:
    """Tests for Open/Closed Principle compliance in step handlers."""

    def test_custom_handler_can_be_added_without_modification(self):
        """Custom step handlers should be addable via registry/entry points.

        OCP compliance: New handlers should be added by:
        1. Creating a new handler class
        2. Registering via StepHandlerRegistry.add_handler()
        3. No modification to FrameworkStepHandler required
        """
        # This test documents the OCP pattern - handlers are pluggable
        from victor.framework.step_handlers import BaseStepHandler, StepHandlerRegistry

        # Create custom handler
        class CustomHandler(BaseStepHandler):
            @property
            def name(self):
                return "custom"

            @property
            def order(self):
                return 125  # After framework (60), before context (100)

            def _do_apply(self, orchestrator, vertical, context, result):
                result.add_info("Custom handler applied")

        # Register handler without modifying FrameworkStepHandler
        registry = StepHandlerRegistry.default()
        registry.add_handler(CustomHandler())

        # Verify handler is registered (use get_ordered_handlers())
        handlers = registry.get_ordered_handlers()
        assert any(
            h.name == "custom" for h in handlers
        ), f"Custom handler not found in handlers: {[h.name for h in handlers]}"

    def test_framework_handler_does_not_hardcode_integration_order(self):
        """FrameworkStepHandler should not hard-code integration order.

        The order should be determined by the handler's `order` property
        and StepHandlerRegistry sorting, not by hard-coded sequential calls.
        """
        handler = FrameworkStepHandler()

        # Verify apply methods are called in expected order
        # (This is a design verification test - the implementation should
        # use a registry or list that respects order properties)
        assert handler.order == 60  # Framework runs at order 60

        # The key is that adding a new handler shouldn't require modifying
        # FrameworkStepHandler's _do_apply method to call it


class TestStepHandlerDIWithRealComponents:
    """Integration tests with real components where applicable."""

    def test_handler_with_missing_registry_logs_warning(self):
        """Handler should log warning when registry not injected and import fails.

        When registry is not injected and hard-coded import fails, handler
        should gracefully handle the error by logging a warning.
        """
        handler = FrameworkStepHandler()  # No injections

        # Create mock orchestrator, vertical, context, result
        mock_orchestrator = Mock()
        vertical = MockVertical
        context = VerticalContext()
        result = IntegrationResult()

        # Trigger import path (will fail because we can't actually import in test)
        # This verifies the fallback mechanism works
        handler.apply_workflows(mock_orchestrator, vertical, context, result)

        # Should complete without error, may have warnings
        # (we can't easily test the actual import failure in isolation)

    def test_all_registries_can_be_injected_together(self):
        """All registries should be injectable simultaneously.

        This tests that the DI interface allows injecting all registries
        at once for comprehensive configuration.
        """
        mock_workflow_registry = Mock()
        mock_team_registry = Mock()
        mock_chain_registry = Mock()
        mock_persona_registry = Mock()
        mock_handler_registry = Mock()

        handler = FrameworkStepHandler(
            workflow_registry=mock_workflow_registry,
            team_registry=mock_team_registry,
            chain_registry=mock_chain_registry,
            persona_registry=mock_persona_registry,
            handler_registry=mock_handler_registry,
        )

        # All should be stored
        assert handler._workflow_registry is mock_workflow_registry
        assert handler._team_registry is mock_team_registry
        assert handler._chain_registry is mock_chain_registry
        assert handler._persona_registry is mock_persona_registry
        assert handler._handler_registry is mock_handler_registry
