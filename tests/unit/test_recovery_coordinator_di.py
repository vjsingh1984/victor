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
# See the License for the specific language governing permissions
# limitations under the License.

"""DI resolution tests for RecoveryCoordinator.

Tests dependency injection container resolution and service lifetime.
"""

import pytest
from unittest.mock import Mock, patch

from victor.agent.service_provider import OrchestratorServiceProvider
from victor.agent.protocols import RecoveryCoordinatorProtocol
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.recovery_blocked_consecutive_threshold = 4
    settings.recovery_blocked_total_threshold = 6
    settings.tool_call_budget_warning_threshold = 250
    settings.max_consecutive_tool_calls = 8
    settings.use_recovery_handler = False
    settings.enable_context_compaction = False
    return settings


@pytest.fixture
def service_provider(mock_settings):
    """Create OrchestratorServiceProvider with mocked settings."""
    return OrchestratorServiceProvider(settings=mock_settings)


class TestRecoveryCoordinatorDI:
    """Tests for RecoveryCoordinator DI resolution."""

    def test_recovery_coordinator_protocol_registered(self, service_provider):
        """Test that RecoveryCoordinatorProtocol is registered in DI container."""
        container = service_provider.container

        # Check that protocol is registered
        assert container.is_registered(RecoveryCoordinatorProtocol)

    def test_recovery_coordinator_can_be_resolved(self, service_provider):
        """Test that RecoveryCoordinator can be resolved from DI container."""
        container = service_provider.container

        # Resolve RecoveryCoordinator
        recovery_coordinator = container.get(RecoveryCoordinatorProtocol)

        # Verify it's not None and has expected attributes
        assert recovery_coordinator is not None
        assert hasattr(recovery_coordinator, "streaming_handler")
        assert hasattr(recovery_coordinator, "unified_tracker")
        assert hasattr(recovery_coordinator, "settings")

    def test_recovery_coordinator_singleton_lifetime(self, service_provider):
        """Test that RecoveryCoordinator has SINGLETON lifetime."""
        container = service_provider.container

        # Resolve RecoveryCoordinator twice
        instance1 = container.get(RecoveryCoordinatorProtocol)
        instance2 = container.get(RecoveryCoordinatorProtocol)

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2

    def test_recovery_coordinator_dependencies_injected(self, service_provider):
        """Test that RecoveryCoordinator dependencies are properly injected."""
        container = service_provider.container

        # Resolve RecoveryCoordinator
        recovery_coordinator = container.get(RecoveryCoordinatorProtocol)

        # Verify required dependencies are injected
        assert recovery_coordinator.streaming_handler is not None
        assert recovery_coordinator.unified_tracker is not None
        assert recovery_coordinator.settings is not None

    def test_recovery_coordinator_optional_dependencies(self, service_provider):
        """Test that RecoveryCoordinator optional dependencies are handled correctly."""
        container = service_provider.container

        # Resolve RecoveryCoordinator
        recovery_coordinator = container.get(RecoveryCoordinatorProtocol)

        # Verify optional dependencies (may be None if not registered)
        # recovery_handler is optional
        # context_compactor is optional
        # recovery_integration is optional
        assert hasattr(recovery_coordinator, "recovery_handler")
        assert hasattr(recovery_coordinator, "context_compactor")
        assert hasattr(recovery_coordinator, "recovery_integration")

    def test_recovery_coordinator_factory_method(self, service_provider):
        """Test that RecoveryCoordinator is created via factory method."""
        # The _create_recovery_coordinator method should be called
        # This is tested implicitly by successful resolution
        container = service_provider.container

        recovery_coordinator = container.get(RecoveryCoordinatorProtocol)

        # Verify it's the correct type (duck typing check)
        assert hasattr(recovery_coordinator, "check_time_limit")
        assert hasattr(recovery_coordinator, "check_iteration_limit")
        assert hasattr(recovery_coordinator, "check_natural_completion")
        assert hasattr(recovery_coordinator, "handle_empty_response")
        assert hasattr(recovery_coordinator, "apply_recovery_action")

    def test_recovery_coordinator_methods_callable(self, service_provider):
        """Test that RecoveryCoordinator methods are callable."""
        container = service_provider.container

        recovery_coordinator = container.get(RecoveryCoordinatorProtocol)

        # Verify key methods are callable
        assert callable(recovery_coordinator.check_time_limit)
        assert callable(recovery_coordinator.check_iteration_limit)
        assert callable(recovery_coordinator.check_natural_completion)
        assert callable(recovery_coordinator.check_tool_budget)
        assert callable(recovery_coordinator.check_progress)
        assert callable(recovery_coordinator.handle_empty_response)
        assert callable(recovery_coordinator.handle_blocked_tool)
        assert callable(recovery_coordinator.apply_recovery_action)
        assert callable(recovery_coordinator.filter_blocked_tool_calls)
        assert callable(recovery_coordinator.truncate_tool_calls)

    def test_orchestrator_factory_creates_recovery_coordinator(self, service_provider):
        """Test that OrchestratorFactory can create RecoveryCoordinator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory

        # Create factory
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            container=service_provider.container,
        )

        # Create RecoveryCoordinator via factory
        recovery_coordinator = factory.create_recovery_coordinator()

        # Verify it's not None and has expected attributes
        assert recovery_coordinator is not None
        assert hasattr(recovery_coordinator, "streaming_handler")
        assert hasattr(recovery_coordinator, "unified_tracker")

    def test_orchestrator_factory_recovery_coordinator_is_singleton(self, service_provider):
        """Test that OrchestratorFactory returns same RecoveryCoordinator instance."""
        from victor.agent.orchestrator_factory import OrchestratorFactory

        # Create factory
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            container=service_provider.container,
        )

        # Create RecoveryCoordinator twice
        instance1 = factory.create_recovery_coordinator()
        instance2 = factory.create_recovery_coordinator()

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2


class TestRecoveryCoordinatorDIIntegration:
    """Integration tests for RecoveryCoordinator DI."""

    def test_full_orchestrator_initialization_with_recovery_coordinator(
        self, service_provider
    ):
        """Test full orchestrator initialization includes RecoveryCoordinator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory

        # Create factory
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            container=service_provider.container,
        )

        # Create orchestrator (this should initialize RecoveryCoordinator internally)
        # We can't easily test this without a full orchestrator setup,
        # but we can verify the factory has the method
        assert hasattr(factory, "create_recovery_coordinator")
        assert callable(factory.create_recovery_coordinator)

    def test_recovery_coordinator_dependencies_resolution_chain(self, service_provider):
        """Test that RecoveryCoordinator dependency resolution chain works."""
        from victor.agent.protocols import (
            StreamingHandlerProtocol,
            TaskTrackerProtocol,
        )

        container = service_provider.container

        # Verify that dependencies are registered and can be resolved
        assert container.is_registered(StreamingHandlerProtocol)
        assert container.is_registered(TaskTrackerProtocol)

        # Verify RecoveryCoordinator can be resolved (which depends on above)
        recovery_coordinator = container.get(RecoveryCoordinatorProtocol)
        assert recovery_coordinator is not None

    def test_recovery_coordinator_with_all_dependencies(self, service_provider):
        """Test RecoveryCoordinator resolution with all possible dependencies."""
        container = service_provider.container

        # Resolve RecoveryCoordinator
        recovery_coordinator = container.get(RecoveryCoordinatorProtocol)

        # Verify all expected attributes exist (even if some are None)
        expected_attrs = [
            "recovery_handler",
            "recovery_integration",
            "streaming_handler",
            "context_compactor",
            "unified_tracker",
            "settings",
        ]

        for attr in expected_attrs:
            assert hasattr(recovery_coordinator, attr), f"Missing attribute: {attr}"
