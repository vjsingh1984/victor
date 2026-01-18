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

"""Tests for FinalizationBuilder."""

from unittest.mock import MagicMock, patch

from victor.agent.builders.finalization_builder import FinalizationBuilder


def test_finalization_builder_wires_dependencies():
    """FinalizationBuilder wires lifecycle and vertical integration."""
    settings = MagicMock()
    factory = MagicMock()

    orchestrator = MagicMock()
    orchestrator._recovery_handler = "recovery-handler"
    orchestrator._context_compactor = "context-compactor"
    orchestrator._observability = "observability"
    orchestrator.conversation_state = "conversation-state"
    orchestrator.provider = "provider"
    orchestrator.code_manager = "code-manager"
    orchestrator.usage_logger = "usage-logger"
    orchestrator._background_tasks = {"task"}
    orchestrator.flush_analytics = MagicMock()
    orchestrator.stop_health_monitoring = MagicMock()
    orchestrator.__init_capability_registry__ = MagicMock()
    orchestrator._lifecycle_manager = MagicMock()

    with (
        patch(
            "victor.agent.builders.finalization_builder.create_vertical_context",
            return_value="vertical-context",
        ),
        patch(
            "victor.agent.builders.finalization_builder.VerticalIntegrationAdapter",
            return_value="vertical-adapter",
        ),
    ):
        builder = FinalizationBuilder(settings=settings, factory=factory)
        components = builder.build(orchestrator)

    factory.wire_component_dependencies.assert_called_once_with(
        recovery_handler="recovery-handler",
        context_compactor="context-compactor",
        observability="observability",
        conversation_state="conversation-state",
    )
    assert orchestrator._vertical_context == "vertical-context"
    assert orchestrator._vertical_integration_adapter == "vertical-adapter"
    assert orchestrator._mode_workflow_team_coordinator is None
    orchestrator.__init_capability_registry__.assert_called_once()
    orchestrator._lifecycle_manager.set_provider.assert_called_once_with("provider")
    orchestrator._lifecycle_manager.set_code_manager.assert_called_once_with("code-manager")
    orchestrator._lifecycle_manager.set_usage_logger.assert_called_once_with("usage-logger")
    orchestrator._lifecycle_manager.set_background_tasks.assert_called_once_with(
        list(orchestrator._background_tasks)
    )
    orchestrator._lifecycle_manager.set_flush_analytics_callback.assert_called_once_with(
        orchestrator.flush_analytics
    )
    orchestrator._lifecycle_manager.set_stop_health_monitoring_callback.assert_called_once_with(
        orchestrator.stop_health_monitoring
    )
    assert components["vertical_context"] == "vertical-context"
