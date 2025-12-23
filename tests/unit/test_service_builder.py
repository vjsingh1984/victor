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

"""Tests for ServiceBuilder.

Part of HIGH-005: Initialization Complexity reduction.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from victor.agent.builders.service_builder import ServiceBuilder
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.conversation_embeddings_enabled = False
    settings.intelligent_pipeline_enabled = False
    return settings


@pytest.fixture
def mock_factory():
    """Create mock factory."""
    factory = MagicMock()
    factory._container = MagicMock()
    return factory


def test_service_builder_initialization(mock_settings):
    """Test ServiceBuilder initialization."""
    builder = ServiceBuilder(mock_settings)

    assert builder.settings is mock_settings
    assert builder._factory is None
    assert builder._container is None


def test_service_builder_initialization_with_factory(mock_settings, mock_factory):
    """Test ServiceBuilder initialization with factory."""
    builder = ServiceBuilder(mock_settings, factory=mock_factory)

    assert builder.settings is mock_settings
    assert builder._factory is mock_factory


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_bootstraps_container(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test that build() bootstraps the DI container."""
    mock_container = MagicMock()
    mock_bootstrap.return_value = mock_container

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    builder.build()

    mock_bootstrap.assert_called_once_with(mock_settings)
    assert builder._container is mock_container


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_creates_core_services(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test that build() creates core services."""
    mock_bootstrap.return_value = MagicMock()

    # Setup factory mocks
    mock_factory.create_sanitizer.return_value = Mock(name="sanitizer")
    mock_factory.create_project_context.return_value = Mock(name="project_context")
    mock_factory.create_complexity_classifier.return_value = Mock(name="classifier")
    mock_factory.create_action_authorizer.return_value = Mock(name="authorizer")
    mock_factory.create_search_router.return_value = Mock(name="router")

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    services = builder.build()

    assert "sanitizer" in services
    assert "project_context" in services
    assert "complexity_classifier" in services
    assert "action_authorizer" in services
    assert "search_router" in services


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_creates_conversation_services(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test that build() creates conversation services."""
    mock_bootstrap.return_value = MagicMock()

    mock_factory.create_conversation_state_machine.return_value = Mock(name="state_machine")
    mock_factory.create_intent_classifier.return_value = Mock(name="intent_classifier")

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    services = builder.build()

    assert "conversation_state" in services
    assert "intent_classifier" in services


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_creates_analytics_services(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test that build() creates analytics services."""
    mock_bootstrap.return_value = MagicMock()

    mock_factory.create_streaming_metrics_collector.return_value = Mock(name="metrics")
    mock_factory.create_usage_analytics.return_value = Mock(name="analytics")
    mock_factory.create_sequence_tracker.return_value = Mock(name="tracker")

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    services = builder.build()

    assert "streaming_metrics_collector" in services
    assert "usage_analytics" in services
    assert "sequence_tracker" in services


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_creates_recovery_components(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test that build() creates recovery components."""
    mock_bootstrap.return_value = MagicMock()

    mock_recovery_handler = Mock(name="recovery_handler")
    mock_factory.create_recovery_handler.return_value = mock_recovery_handler
    mock_factory.create_recovery_integration.return_value = Mock(name="integration")
    mock_factory.create_recovery_coordinator.return_value = Mock(name="coordinator")

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    services = builder.build()

    assert "recovery_handler" in services
    assert "recovery_integration" in services
    assert "recovery_coordinator" in services
    mock_factory.create_recovery_integration.assert_called_once_with(mock_recovery_handler)


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_creates_coordination_components(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test that build() creates coordination components."""
    mock_bootstrap.return_value = MagicMock()

    mock_factory.create_chunk_generator.return_value = Mock(name="chunk_gen")
    mock_factory.create_tool_planner.return_value = Mock(name="planner")
    mock_factory.create_task_coordinator.return_value = Mock(name="coordinator")

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    services = builder.build()

    assert "chunk_generator" in services
    assert "tool_planner" in services
    assert "task_coordinator" in services


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
@patch("victor.agent.task_analyzer.get_task_analyzer")
def test_service_builder_build_creates_task_analyzer(
    mock_get_analyzer, mock_bootstrap, mock_settings, mock_factory
):
    """Test that build() creates task analyzer."""
    mock_bootstrap.return_value = MagicMock()
    mock_analyzer = Mock(name="task_analyzer")
    mock_get_analyzer.return_value = mock_analyzer

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    services = builder.build()

    assert "task_analyzer" in services
    assert services["task_analyzer"] is mock_analyzer


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_registers_components(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test that build() registers all built components."""
    mock_bootstrap.return_value = MagicMock()

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    services = builder.build()

    # Check that components are registered
    for name, service in services.items():
        if service is not None:
            assert builder.has_component(name)
            assert builder.get_component(name) is service


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_conversation_controller(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test building conversation controller."""
    mock_bootstrap.return_value = MagicMock()
    mock_controller = Mock(name="conversation_controller")
    mock_factory.create_conversation_controller.return_value = mock_controller

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    builder.build()  # Build base services first

    mock_provider = Mock()
    mock_conversation = Mock()
    mock_state = Mock()
    mock_memory = Mock()

    controller = builder.build_conversation_controller(
        provider=mock_provider,
        model="test-model",
        conversation=mock_conversation,
        conversation_state=mock_state,
        memory_manager=mock_memory,
        memory_session_id="session-123",
        system_prompt="Test prompt",
    )

    assert controller is mock_controller
    assert builder.has_component("conversation_controller")
    mock_factory.create_conversation_controller.assert_called_once()


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_streaming_controller(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test building streaming controller."""
    mock_bootstrap.return_value = MagicMock()
    mock_controller = Mock(name="streaming_controller")
    mock_factory.create_streaming_controller.return_value = mock_controller

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    builder.build()

    mock_metrics = Mock()
    mock_callback = Mock()

    controller = builder.build_streaming_controller(
        streaming_metrics_collector=mock_metrics,
        on_session_complete=mock_callback,
    )

    assert controller is mock_controller
    assert builder.has_component("streaming_controller")


@patch("victor.agent.builders.service_builder.ensure_bootstrapped")
def test_service_builder_build_context_compactor(
    mock_bootstrap, mock_settings, mock_factory
):
    """Test building context compactor."""
    mock_bootstrap.return_value = MagicMock()
    mock_compactor = Mock(name="context_compactor")
    mock_factory.create_context_compactor.return_value = mock_compactor

    builder = ServiceBuilder(mock_settings, factory=mock_factory)
    builder.build()

    mock_conv_controller = Mock()

    compactor = builder.build_context_compactor(
        conversation_controller=mock_conv_controller
    )

    assert compactor is mock_compactor
    assert builder.has_component("context_compactor")
