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

"""Unit tests for ProviderLifecycleManager.

TDD-first tests for Phase 1.2: Extract ProviderLifecycleManager.
These tests verify:
1. Protocol compliance
2. Exploration settings application
3. Prompt contributor retrieval
4. Prompt builder reinitialization
5. Tool budget management
6. DI registration
"""

from typing import Any
from unittest.mock import Mock
from dataclasses import dataclass


@dataclass
class MockCapabilities:
    """Mock for tool calling capabilities."""

    exploration_multiplier: float = 5.0
    continuation_patience: int = 3
    recommended_tool_budget: int = 100


@dataclass
class MockPromptContributor:
    """Mock prompt contributor."""

    name: str = "test_contributor"

    def contribute(self, context: Any) -> str:
        return f"Contribution from {self.name}"


class TestProviderLifecycleProtocol:
    """Tests for ProviderLifecycleProtocol interface."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        from victor.protocols.provider_lifecycle import ProviderLifecycleProtocol

        assert hasattr(ProviderLifecycleProtocol, "__protocol_attrs__") or hasattr(
            ProviderLifecycleProtocol, "_is_protocol"
        )

    def test_protocol_has_required_methods(self):
        """Protocol should define required methods."""
        from victor.protocols.provider_lifecycle import ProviderLifecycleProtocol

        # Verify protocol defines expected methods
        assert hasattr(ProviderLifecycleProtocol, "apply_exploration_settings")
        assert hasattr(ProviderLifecycleProtocol, "get_prompt_contributors")
        assert hasattr(ProviderLifecycleProtocol, "create_prompt_builder")
        assert hasattr(ProviderLifecycleProtocol, "calculate_tool_budget")


class TestProviderLifecycleManager:
    """Tests for ProviderLifecycleManager implementation."""

    def test_manager_implements_protocol(self):
        """Manager should implement ProviderLifecycleProtocol."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        manager = ProviderLifecycleManager(container)

        # Verify required methods exist
        assert hasattr(manager, "apply_exploration_settings")
        assert hasattr(manager, "get_prompt_contributors")
        assert hasattr(manager, "create_prompt_builder")
        assert hasattr(manager, "calculate_tool_budget")

    def test_apply_exploration_settings(self):
        """Should apply exploration settings to tracker."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        manager = ProviderLifecycleManager(container)

        # Create mock tracker
        tracker = Mock()
        tracker.set_model_exploration_settings = Mock()

        # Create mock capabilities
        caps = MockCapabilities(
            exploration_multiplier=10.0,
            continuation_patience=5,
        )

        # Apply settings
        manager.apply_exploration_settings(tracker, caps)

        # Verify tracker was called with correct args
        tracker.set_model_exploration_settings.assert_called_once_with(
            exploration_multiplier=10.0,
            continuation_patience=5,
        )

    def test_get_prompt_contributors_with_extensions(self):
        """Should retrieve prompt contributors from vertical extensions."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        # Create mock container with extensions
        mock_extensions = Mock()
        mock_extensions.prompt_contributors = [
            MockPromptContributor("contributor1"),
            MockPromptContributor("contributor2"),
        ]

        container = Mock()
        container.get_optional = Mock(return_value=mock_extensions)

        manager = ProviderLifecycleManager(container)

        # Get contributors
        contributors = manager.get_prompt_contributors()

        # Verify we got the contributors
        assert len(contributors) == 2
        assert contributors[0].name == "contributor1"
        assert contributors[1].name == "contributor2"

    def test_get_prompt_contributors_without_extensions(self):
        """Should return empty list when no extensions available."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        container.get_optional = Mock(return_value=None)

        manager = ProviderLifecycleManager(container)

        # Get contributors
        contributors = manager.get_prompt_contributors()

        # Should return empty list
        assert contributors == []

    def test_get_prompt_contributors_handles_import_error(self):
        """Should handle ImportError gracefully."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        container.get_optional = Mock(side_effect=ImportError("Module not found"))

        manager = ProviderLifecycleManager(container)

        # Should not raise, returns empty list
        contributors = manager.get_prompt_contributors()
        assert contributors == []

    def test_create_prompt_builder(self):
        """Should create a new SystemPromptBuilder with given parameters."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        manager = ProviderLifecycleManager(container)

        # Create mock inputs
        mock_adapter = Mock()
        mock_caps = MockCapabilities()
        mock_contributors = [MockPromptContributor()]

        # Create prompt builder
        builder = manager.create_prompt_builder(
            provider_name="anthropic",
            model="claude-3-opus",
            tool_adapter=mock_adapter,
            capabilities=mock_caps,
            prompt_contributors=mock_contributors,
        )

        # Verify builder was created (should be SystemPromptBuilder)
        assert builder is not None
        assert hasattr(builder, "build")

    def test_calculate_tool_budget_default(self):
        """Should calculate tool budget from capabilities."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        manager = ProviderLifecycleManager(container)

        # Mock capabilities with recommended budget
        caps = MockCapabilities(recommended_tool_budget=150)

        # Mock settings without override
        settings = Mock(spec=[])  # Empty spec = no attributes

        budget = manager.calculate_tool_budget(caps, settings)

        # Should use recommended budget (max with 50)
        assert budget == 150

    def test_calculate_tool_budget_minimum_50(self):
        """Should enforce minimum budget of 50."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        manager = ProviderLifecycleManager(container)

        # Mock capabilities with low recommended budget
        caps = MockCapabilities(recommended_tool_budget=20)

        # Mock settings without override
        settings = Mock(spec=[])

        budget = manager.calculate_tool_budget(caps, settings)

        # Should use minimum of 50
        assert budget == 50

    def test_calculate_tool_budget_settings_override(self):
        """Should use settings override when available."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        manager = ProviderLifecycleManager(container)

        # Mock capabilities
        caps = MockCapabilities(recommended_tool_budget=100)

        # Mock settings with override
        settings = Mock()
        settings.tool_call_budget = 200

        budget = manager.calculate_tool_budget(caps, settings)

        # Should use settings override
        assert budget == 200

    def test_should_respect_sticky_budget(self):
        """Should check if sticky budget should be respected."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        manager = ProviderLifecycleManager(container)

        # Tracker with sticky budget
        tracker_sticky = Mock()
        tracker_sticky._sticky_user_budget = True

        # Tracker without sticky budget
        tracker_not_sticky = Mock()
        tracker_not_sticky._sticky_user_budget = False

        # Tracker without attribute
        tracker_no_attr = Mock(spec=[])

        assert manager.should_respect_sticky_budget(tracker_sticky) is True
        assert manager.should_respect_sticky_budget(tracker_not_sticky) is False
        assert manager.should_respect_sticky_budget(tracker_no_attr) is False


class TestProviderLifecycleDIRegistration:
    """Tests for DI container registration."""

    def test_protocol_registered_in_container(self):
        """ProviderLifecycleProtocol should be registered in DI container."""
        from victor.protocols.provider_lifecycle import ProviderLifecycleProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        assert container.is_registered(ProviderLifecycleProtocol)

    def test_resolve_manager_from_container(self):
        """Should be able to resolve ProviderLifecycleProtocol from container."""
        from victor.protocols.provider_lifecycle import ProviderLifecycleProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        manager = container.get(ProviderLifecycleProtocol)

        assert manager is not None
        assert hasattr(manager, "apply_exploration_settings")
        assert hasattr(manager, "get_prompt_contributors")

    def test_manager_singleton_scope(self):
        """Manager should be registered as singleton."""
        from victor.protocols.provider_lifecycle import ProviderLifecycleProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        manager1 = container.get(ProviderLifecycleProtocol)
        manager2 = container.get(ProviderLifecycleProtocol)

        assert manager1 is manager2


class TestProviderLifecycleIntegration:
    """Integration tests for provider lifecycle management."""

    def test_full_post_switch_flow(self):
        """Test complete post-switch hook workflow."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        # Create mock container
        mock_extensions = Mock()
        mock_extensions.prompt_contributors = []

        container = Mock()
        container.get_optional = Mock(return_value=mock_extensions)

        manager = ProviderLifecycleManager(container)

        # Create mock tracker
        tracker = Mock()

        # Create mock capabilities
        caps = MockCapabilities(
            exploration_multiplier=5.0,
            continuation_patience=3,
            recommended_tool_budget=100,
        )

        # Create mock settings
        settings = Mock(spec=[])

        # Apply exploration settings
        manager.apply_exploration_settings(tracker, caps)
        tracker.set_model_exploration_settings.assert_called_once()

        # Get contributors
        contributors = manager.get_prompt_contributors()
        assert isinstance(contributors, list)

        # Create prompt builder
        mock_adapter = Mock()
        builder = manager.create_prompt_builder(
            provider_name="openai",
            model="gpt-4",
            tool_adapter=mock_adapter,
            capabilities=caps,
            prompt_contributors=contributors,
        )
        assert builder is not None

        # Calculate budget
        budget = manager.calculate_tool_budget(caps, settings)
        assert budget >= 50

    def test_handles_missing_vertical_extensions_gracefully(self):
        """Manager should handle missing vertical extensions gracefully."""
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        container = Mock()
        container.get_optional = Mock(side_effect=AttributeError("No extensions"))

        manager = ProviderLifecycleManager(container)

        # Should not raise
        contributors = manager.get_prompt_contributors()
        assert contributors == []
