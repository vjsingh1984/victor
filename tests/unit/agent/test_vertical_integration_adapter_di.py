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

"""Tests for VerticalIntegrationAdapter DI and capability registry enforcement.

Tests that VerticalIntegrationAdapter uses capability registry methods instead of
writing to private fields, following Dependency Inversion Principle (DIP).
"""

import pytest
from unittest.mock import MagicMock, Mock, patch

from victor.core.errors import CapabilityRegistryRequiredError
from victor.agent.vertical_integration_adapter import VerticalIntegrationAdapter
from victor.core.verticals.protocols.middleware import MiddlewareProtocol
from victor.security.safety.types import SafetyPattern


class MockMiddleware(MiddlewareProtocol):
    """Mock middleware for testing."""

    def __init__(self, name: str = "test_middleware"):
        self.name = name
        self.pre_process_calls = []
        self.post_process_calls = []

    def pre_process(self, context: dict, tool_name: str, args: dict) -> dict:
        self.pre_process_calls.append((tool_name, args))
        return context

    def post_process(self, context: dict, tool_name: str, result: any) -> any:
        self.post_process_calls.append((tool_name, result))
        return result

    def before_tool_call(self, tool_name: str, args: dict) -> dict:
        """Before tool call hook."""
        return args

    def after_tool_call(self, tool_name: str, result: any) -> any:
        """After tool call hook."""
        return result

    def on_error(self, tool_name: str, error: Exception) -> None:
        """Error handling hook."""
        pass


class TestAdapterCapabilityRegistryRequired:
    """Tests for enforcing capability registry requirement in adapter."""

    def test_adapter_raises_error_without_capability_registry(self):
        """Adapter should raise CapabilityRegistryRequiredError when orchestrator lacks capability support.

        This test ensures that the adapter fails fast and clearly when the orchestrator
        doesn't support capability operations, preventing silent failures and fallback to
        private field writes.
        """
        # Create orchestrator mock without capability support
        mock_orchestrator = Mock(spec=[])
        # Explicitly ensure no capability methods exist
        for method in ['has_capability', 'get_capability', 'set_capability',
                       '_has_capability', '_get_capability_value', '_set_capability_value']:
            if hasattr(mock_orchestrator, method):
                delattr(mock_orchestrator, method)

        adapter = VerticalIntegrationAdapter(mock_orchestrator)

        # Attempt to apply middleware should raise error
        middleware = [MockMiddleware()]

        with pytest.raises(CapabilityRegistryRequiredError) as exc_info:
            adapter.apply_middleware(middleware)

        # Verify error attributes
        error = exc_info.value
        assert error.component == "VerticalIntegrationAdapter"
        assert error.capability_name == "vertical_middleware"
        assert len(error.required_methods) > 0

    def test_adapter_uses_capability_registry_for_middleware(self):
        """Adapter should use internal storage setters when available.

        The adapter uses internal storage setters (_set_*_storage) as the
        DIP-compliant approach for storing capabilities. These are controlled
        methods provided by the orchestrator, not direct attribute access.
        """
        # Create orchestrator mock with internal storage setter
        mock_orchestrator = Mock()
        mock_orchestrator._set_vertical_middleware_storage = Mock()

        adapter = VerticalIntegrationAdapter(mock_orchestrator)

        # Apply middleware
        middleware = [MockMiddleware()]
        adapter.apply_middleware(middleware)

        # Verify internal storage setter was called
        mock_orchestrator._set_vertical_middleware_storage.assert_called_once_with(middleware)

    def test_adapter_no_private_field_write_for_middleware(self):
        """Adapter should NOT write to _vertical_middleware private field.

        This is the critical LSP compliance test - ensure adapter doesn't
        break encapsulation by writing private fields.
        """
        mock_orchestrator = Mock()
        mock_orchestrator.has_capability = Mock(return_value=True)
        mock_orchestrator.set_capability = Mock()

        # Track if _vertical_middleware is accessed
        original_setattr = object.__setattr__
        private_writes = []

        def track_setattr(obj, name, value):
            if name == '_vertical_middleware':
                private_writes.append((name, value))
            return original_setattr(obj, name, value)

        with patch.object(type(mock_orchestrator), '__setattr__', track_setattr):
            adapter = VerticalIntegrationAdapter(mock_orchestrator)
            adapter.apply_middleware([MockMiddleware()])

        # Should have NO private field writes
        assert len(private_writes) == 0, \
            f"Adapter should not write _vertical_middleware, but got: {private_writes}"

    def test_adapter_no_private_field_write_for_middleware_chain(self):
        """Adapter should NOT write to _middleware_chain private field."""
        mock_orchestrator = Mock()
        mock_orchestrator.has_capability = Mock(return_value=True)
        mock_orchestrator.get_capability = Mock(return_value=None)
        mock_orchestrator.set_capability = Mock()

        # Track private field writes
        original_setattr = object.__setattr__
        private_writes = []

        def track_setattr(obj, name, value):
            if name == '_middleware_chain':
                private_writes.append((name, value))
            return original_setattr(obj, name, value)

        with patch.object(type(mock_orchestrator), '__setattr__', track_setattr):
            adapter = VerticalIntegrationAdapter(mock_orchestrator)
            adapter.apply_middleware([MockMiddleware()])

        # Should have NO private field writes to _middleware_chain
        assert len(private_writes) == 0, \
            f"Adapter should not write _middleware_chain, but got: {private_writes}"

    def test_adapter_no_private_field_write_for_safety_patterns(self):
        """Adapter should NOT write to _vertical_safety_patterns private field."""
        mock_orchestrator = Mock()
        mock_orchestrator.has_capability = Mock(return_value=True)
        mock_orchestrator.get_capability = Mock(return_value=None)
        mock_orchestrator.set_capability = Mock()

        # Track private field writes
        original_setattr = object.__setattr__
        private_writes = []

        def track_setattr(obj, name, value):
            if name == '_vertical_safety_patterns':
                private_writes.append((name, value))
            return original_setattr(obj, name, value)

        with patch.object(type(mock_orchestrator), '__setattr__', track_setattr):
            adapter = VerticalIntegrationAdapter(mock_orchestrator)
            patterns = [SafetyPattern(pattern="test", description="Test pattern")]
            adapter.apply_safety_patterns(patterns)

        # Should have NO private field writes to _vertical_safety_patterns
        assert len(private_writes) == 0, \
            f"Adapter should not write _vertical_safety_patterns, but got: {private_writes}"


class TestAdapterCapabilityRegistryIntegration:
    """Tests for adapter integration with capability registry."""

    def test_adapter_stores_via_capability_not_private(self):
        """Adapter should store via internal storage setters, not private fields.

        This test verifies the DIP compliance - adapter uses controlled methods
        (_set_*_storage) provided by orchestrator instead of writing to private fields.
        """
        # Create orchestrator with internal storage setters and capability support
        mock_orchestrator = Mock()
        mock_orchestrator._set_vertical_middleware_storage = Mock()

        # Mock vertical context
        mock_context = Mock()
        mock_context.apply_middleware = Mock()

        # Make get_capability_value return the context (adapter uses this method)
        mock_orchestrator.get_capability_value = Mock(return_value=mock_context)

        adapter = VerticalIntegrationAdapter(mock_orchestrator)
        middleware = [MockMiddleware("test1"), MockMiddleware("test2")]

        adapter.apply_middleware(middleware)

        # Verify internal storage setter was used
        mock_orchestrator._set_vertical_middleware_storage.assert_called_once_with(middleware)
        # Verify vertical context was used
        mock_context.apply_middleware.assert_called_once_with(middleware)

    def test_adapter_with_vertical_context_capability(self):
        """Adapter should use vertical_context capability when available."""
        mock_orchestrator = Mock()
        mock_orchestrator._set_vertical_middleware_storage = Mock()

        # Mock vertical context
        mock_context = Mock()
        mock_context.apply_middleware = Mock()

        # Make get_capability_value return the context (adapter uses this method)
        mock_orchestrator.get_capability_value = Mock(return_value=mock_context)

        adapter = VerticalIntegrationAdapter(mock_orchestrator)
        middleware = [MockMiddleware()]

        adapter.apply_middleware(middleware)

        # Verify vertical context was used
        mock_context.apply_middleware.assert_called_once_with(middleware)

    def test_adapter_handles_missing_capability_gracefully(self):
        """When capability is missing, adapter should use internal setters if available."""
        mock_orchestrator = Mock()
        mock_orchestrator.has_capability = Mock(return_value=False)
        mock_orchestrator.get_capability = Mock(return_value=None)

        # Add internal storage setter
        mock_orchestrator._set_vertical_middleware_storage = Mock()

        adapter = VerticalIntegrationAdapter(mock_orchestrator)
        middleware = [MockMiddleware()]

        # Should not raise error - uses internal setter
        adapter.apply_middleware(middleware)

        # Verify internal setter was called
        mock_orchestrator._set_vertical_middleware_storage.assert_called_once_with(middleware)


class TestAdapterBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_adapter_works_with_capability_less_orchestrator_strict_mode(self):
        """In strict mode, adapter should require capability registry and raise error.

        Phase 2 implementation: Adapter now enforces capability registry requirement
        and raises CapabilityRegistryRequiredError when orchestrator doesn't support it.
        """
        mock_orchestrator = Mock(spec=[])  # No methods at all

        adapter = VerticalIntegrationAdapter(mock_orchestrator)
        middleware = [MockMiddleware()]

        # Should raise CapabilityRegistryRequiredError (strict mode)
        with pytest.raises(CapabilityRegistryRequiredError):
            adapter.apply_middleware(middleware)
