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

"""DI resolution tests for TaskCoordinator.

Tests dependency injection container resolution and service lifetime.
"""

import pytest
from unittest.mock import Mock, MagicMock

from victor.agent.service_provider import OrchestratorServiceProvider
from victor.agent.protocols import TaskCoordinatorProtocol
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.temperature = 0.7
    settings.tool_budget = 15
    return settings


@pytest.fixture
def service_provider(mock_settings):
    """Create OrchestratorServiceProvider with mocked settings."""
    from victor.core.container import ServiceContainer

    provider = OrchestratorServiceProvider(settings=mock_settings)
    container = ServiceContainer()
    provider.container = container  # Store container reference for tests

    # Register all services via provider (no manual mocking needed for DI tests)
    provider.register_services(container)
    return provider


class TestTaskCoordinatorDI:
    """Tests for TaskCoordinator DI resolution."""

    def test_task_coordinator_protocol_registered(self, service_provider):
        """Test that TaskCoordinatorProtocol is registered in DI container."""
        container = service_provider.container

        # Check that protocol is registered
        assert container.is_registered(TaskCoordinatorProtocol)

    def test_task_coordinator_can_be_resolved(self, service_provider):
        """Test that TaskCoordinator can be resolved from DI container."""
        container = service_provider.container

        # Resolve TaskCoordinator
        task_coordinator = container.get(TaskCoordinatorProtocol)

        # Verify it's not None and has expected attributes
        assert task_coordinator is not None
        assert hasattr(task_coordinator, "task_analyzer")
        assert hasattr(task_coordinator, "unified_tracker")
        assert hasattr(task_coordinator, "prompt_builder")
        assert hasattr(task_coordinator, "settings")

    def test_task_coordinator_singleton_lifetime(self, service_provider):
        """Test that TaskCoordinator has SINGLETON lifetime."""
        container = service_provider.container

        # Resolve TaskCoordinator twice
        instance1 = container.get(TaskCoordinatorProtocol)
        instance2 = container.get(TaskCoordinatorProtocol)

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2

    def test_task_coordinator_dependencies_injected(self, service_provider):
        """Test that TaskCoordinator dependencies are properly injected."""
        container = service_provider.container

        # Resolve TaskCoordinator
        task_coordinator = container.get(TaskCoordinatorProtocol)

        # Verify required dependencies are injected
        assert task_coordinator.task_analyzer is not None
        assert task_coordinator.unified_tracker is not None
        assert task_coordinator.prompt_builder is not None
        assert task_coordinator.settings is not None

    def test_task_coordinator_methods_callable(self, service_provider):
        """Test that TaskCoordinator methods are callable."""
        container = service_provider.container

        task_coordinator = container.get(TaskCoordinatorProtocol)

        # Verify key methods are callable
        assert callable(task_coordinator.prepare_task)
        assert callable(task_coordinator.apply_intent_guard)
        assert callable(task_coordinator.apply_task_guidance)

    def test_orchestrator_factory_creates_task_coordinator(self, service_provider):
        """Test that OrchestratorFactory can create TaskCoordinator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from unittest.mock import Mock

        # Create factory (it will create its own container internally)
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=Mock(),
            model="test-model",
        )

        # Override factory's container with our test container
        factory._container = service_provider.container

        # Create TaskCoordinator via factory
        task_coordinator = factory.create_task_coordinator()

        # Verify it's not None and has expected attributes
        assert task_coordinator is not None
        assert hasattr(task_coordinator, "task_analyzer")
        assert hasattr(task_coordinator, "unified_tracker")

    def test_orchestrator_factory_task_coordinator_is_singleton(self, service_provider):
        """Test that OrchestratorFactory returns same TaskCoordinator instance."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from unittest.mock import Mock

        # Create factory (it will create its own container internally)
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=Mock(),
            model="test-model",
        )

        # Override factory's container with our test container
        factory._container = service_provider.container

        # Create TaskCoordinator twice
        instance1 = factory.create_task_coordinator()
        instance2 = factory.create_task_coordinator()

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2


class TestTaskCoordinatorDIIntegration:
    """Integration tests for TaskCoordinator DI."""

    def test_task_coordinator_dependencies_resolution_chain(self, service_provider):
        """Test that TaskCoordinator dependency resolution chain works."""
        from victor.agent.protocols import (
            TaskAnalyzerProtocol,
            TaskTrackerProtocol,
            SystemPromptBuilderProtocol,
        )

        container = service_provider.container

        # Verify that dependencies are registered and can be resolved
        assert container.is_registered(TaskAnalyzerProtocol)
        assert container.is_registered(TaskTrackerProtocol)
        assert container.is_registered(SystemPromptBuilderProtocol)

        # Verify TaskCoordinator can be resolved (which depends on above)
        task_coordinator = container.get(TaskCoordinatorProtocol)
        assert task_coordinator is not None

    def test_task_coordinator_with_all_dependencies(self, service_provider):
        """Test TaskCoordinator resolution with all possible dependencies."""
        container = service_provider.container

        # Resolve TaskCoordinator
        task_coordinator = container.get(TaskCoordinatorProtocol)

        # Verify all expected attributes exist
        expected_attrs = [
            "task_analyzer",
            "unified_tracker",
            "prompt_builder",
            "settings",
        ]

        for attr in expected_attrs:
            assert hasattr(task_coordinator, attr), f"Missing attribute: {attr}"

    def test_full_orchestrator_initialization_with_task_coordinator(
        self, service_provider
    ):
        """Test full orchestrator initialization includes TaskCoordinator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from unittest.mock import Mock

        # Create factory (it will create its own container internally)
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=Mock(),
            model="test-model",
        )

        # Verify the factory has the method
        assert hasattr(factory, "create_task_coordinator")
        assert callable(factory.create_task_coordinator)
