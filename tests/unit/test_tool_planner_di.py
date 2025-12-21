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

"""DI resolution tests for ToolPlanner.

Tests dependency injection container resolution and service lifetime.
"""

import pytest
from unittest.mock import Mock, MagicMock

from victor.agent.service_provider import OrchestratorServiceProvider
from victor.agent.protocols import ToolPlannerProtocol
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    return settings


@pytest.fixture
def service_provider(mock_settings):
    """Create OrchestratorServiceProvider with mocked settings."""
    from victor.core.container import ServiceContainer
    from victor.agent.protocols import ToolRegistrarProtocol
    from victor.core.container import ServiceLifetime

    provider = OrchestratorServiceProvider(settings=mock_settings)
    container = ServiceContainer()
    provider.container = container  # Store container reference for tests

    # Mock ToolRegistrarProtocol before registering services
    mock_registrar = MagicMock()
    mock_registrar.plan_tools.return_value = []
    mock_registrar.infer_goals_from_message.return_value = []

    container.register(
        ToolRegistrarProtocol,
        lambda c: mock_registrar,
        ServiceLifetime.SINGLETON,
    )

    provider.register_services(container)
    return provider


class TestToolPlannerDI:
    """Tests for ToolPlanner DI resolution."""

    def test_tool_planner_protocol_registered(self, service_provider):
        """Test that ToolPlannerProtocol is registered in DI container."""
        container = service_provider.container

        # Check that protocol is registered
        assert container.is_registered(ToolPlannerProtocol)

    def test_tool_planner_can_be_resolved(self, service_provider):
        """Test that ToolPlanner can be resolved from DI container."""
        container = service_provider.container

        # Resolve ToolPlanner
        tool_planner = container.get(ToolPlannerProtocol)

        # Verify it's not None and has expected attributes
        assert tool_planner is not None
        assert hasattr(tool_planner, "tool_registrar")
        assert hasattr(tool_planner, "settings")

    def test_tool_planner_singleton_lifetime(self, service_provider):
        """Test that ToolPlanner has SINGLETON lifetime."""
        container = service_provider.container

        # Resolve ToolPlanner twice
        instance1 = container.get(ToolPlannerProtocol)
        instance2 = container.get(ToolPlannerProtocol)

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2

    def test_tool_planner_dependencies_injected(self, service_provider):
        """Test that ToolPlanner dependencies are properly injected."""
        container = service_provider.container

        # Resolve ToolPlanner
        tool_planner = container.get(ToolPlannerProtocol)

        # Verify required dependencies are injected
        assert tool_planner.tool_registrar is not None
        assert tool_planner.settings is not None

    def test_tool_planner_methods_callable(self, service_provider):
        """Test that ToolPlanner methods are callable."""
        container = service_provider.container

        tool_planner = container.get(ToolPlannerProtocol)

        # Verify key methods are callable
        assert callable(tool_planner.plan_tools)
        assert callable(tool_planner.infer_goals_from_message)
        assert callable(tool_planner.filter_tools_by_intent)

    def test_orchestrator_factory_creates_tool_planner(self, service_provider):
        """Test that OrchestratorFactory can create ToolPlanner."""
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

        # Create ToolPlanner via factory
        tool_planner = factory.create_tool_planner()

        # Verify it's not None and has expected attributes
        assert tool_planner is not None
        assert hasattr(tool_planner, "tool_registrar")
        assert hasattr(tool_planner, "settings")

    def test_orchestrator_factory_tool_planner_is_singleton(self, service_provider):
        """Test that OrchestratorFactory returns same ToolPlanner instance."""
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

        # Create ToolPlanner twice
        instance1 = factory.create_tool_planner()
        instance2 = factory.create_tool_planner()

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2


class TestToolPlannerDIIntegration:
    """Integration tests for ToolPlanner DI."""

    def test_tool_planner_dependencies_resolution_chain(self, service_provider):
        """Test that ToolPlanner dependency resolution chain works."""
        from victor.agent.protocols import ToolRegistrarProtocol

        container = service_provider.container

        # Verify that dependencies are registered and can be resolved
        assert container.is_registered(ToolRegistrarProtocol)

        # Verify ToolPlanner can be resolved (which depends on above)
        tool_planner = container.get(ToolPlannerProtocol)
        assert tool_planner is not None

    def test_tool_planner_with_all_dependencies(self, service_provider):
        """Test ToolPlanner resolution with all possible dependencies."""
        container = service_provider.container

        # Resolve ToolPlanner
        tool_planner = container.get(ToolPlannerProtocol)

        # Verify all expected attributes exist
        expected_attrs = [
            "tool_registrar",
            "settings",
        ]

        for attr in expected_attrs:
            assert hasattr(tool_planner, attr), f"Missing attribute: {attr}"

    def test_full_orchestrator_initialization_with_tool_planner(
        self, service_provider
    ):
        """Test full orchestrator initialization includes ToolPlanner."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from unittest.mock import Mock

        # Create factory (it will create its own container internally)
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=Mock(),
            model="test-model",
        )

        # Verify the factory has the method
        assert hasattr(factory, "create_tool_planner")
        assert callable(factory.create_tool_planner)
