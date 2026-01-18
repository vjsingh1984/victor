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

"""Tests for ConfigWorkflowBuilder."""

from unittest.mock import MagicMock, patch

from victor.agent.builders.config_workflow_builder import ConfigWorkflowBuilder
from victor.agent.coordinators.conversation_coordinator import ConversationCoordinator
from victor.agent.coordinators.search_coordinator import SearchCoordinator


def test_config_workflow_builder_wires_components():
    """ConfigWorkflowBuilder assigns configuration and workflow components."""
    settings = MagicMock()
    settings.execution_timeout = 120
    settings.session_idle_timeout = 180

    factory = MagicMock()
    factory.create_workflow_optimization_components.return_value = "workflow-optimization"

    orchestrator = MagicMock()
    orchestrator.conversation = MagicMock()
    orchestrator.memory_manager = "memory-store"
    orchestrator._memory_session_id = "session-id"
    orchestrator._lifecycle_manager = MagicMock()
    orchestrator.usage_logger = "usage-logger"
    orchestrator.search_router = "search-router"
    orchestrator._streaming_handler = MagicMock()
    orchestrator._streaming_handler.session_idle_timeout = 10
    orchestrator.provider_name = "provider"
    orchestrator.model = "model"

    with patch(
        "victor.agent.builders.config_workflow_builder.create_configuration_manager",
        return_value="config-manager",
    ), patch(
        "victor.agent.builders.config_workflow_builder.create_memory_manager",
        return_value="memory-wrapper",
    ), patch(
        "victor.agent.builders.config_workflow_builder.create_session_recovery_manager",
        return_value="session-recovery",
    ), patch(
        "victor.agent.builders.config_workflow_builder.ProgressMetrics"
    ) as progress_cls, patch(
        "victor.config.config_loaders.get_provider_limits"
    ) as get_provider_limits:
        progress_instance = MagicMock()
        progress_cls.return_value = progress_instance
        limits = MagicMock()
        limits.context_window = 8192
        limits.session_idle_timeout = 300
        get_provider_limits.return_value = limits

        builder = ConfigWorkflowBuilder(settings=settings, factory=factory)
        components = builder.build(orchestrator)

    assert orchestrator._configuration_manager == "config-manager"
    assert orchestrator._memory_manager_wrapper == "memory-wrapper"
    assert isinstance(orchestrator._conversation_coordinator, ConversationCoordinator)
    assert isinstance(orchestrator._search_coordinator, SearchCoordinator)
    assert orchestrator._team_coordinator is None
    assert orchestrator._workflow_optimization == "workflow-optimization"
    assert orchestrator._session_recovery_manager == "session-recovery"
    assert orchestrator._progress_metrics == progress_instance
    assert orchestrator._session_idle_timeout == 300
    assert orchestrator._streaming_handler.session_idle_timeout == 300
    progress_instance.initialize_token_budget.assert_called_once_with(8192)
    assert components["workflow_optimization"] == "workflow-optimization"
