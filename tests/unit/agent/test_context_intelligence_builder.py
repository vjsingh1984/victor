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

"""Tests for ContextIntelligenceBuilder."""

from unittest.mock import MagicMock, patch

from victor.agent.builders.context_intelligence_builder import ContextIntelligenceBuilder


def test_context_intelligence_builder_wires_components():
    """ContextIntelligenceBuilder assigns context and intelligence components."""
    settings = MagicMock()
    factory = MagicMock()
    factory.create_rl_coordinator.return_value = MagicMock()
    factory.create_context_compactor.return_value = "context-compactor"
    factory.create_usage_analytics.return_value = "usage-analytics"
    factory.create_sequence_tracker.return_value = "sequence-tracker"
    factory.create_tool_output_formatter.return_value = "tool-formatter"

    orchestrator = MagicMock()
    orchestrator.settings = settings
    orchestrator.provider_name = "test-provider"
    orchestrator.model = "test-model"
    orchestrator.provider = "provider"
    orchestrator._conversation_controller = "conversation-controller"
    orchestrator.debug_logger = "debug-logger"

    with patch(
        "victor.agent.builders.context_intelligence_builder.get_task_analyzer",
        return_value="task-analyzer",
    ), patch(
        "victor.agent.builders.context_intelligence_builder.create_context_manager",
        return_value="context-manager",
    ):
        builder = ContextIntelligenceBuilder(settings=settings, factory=factory)
        components = builder.build(orchestrator)

    assert orchestrator._task_analyzer == "task-analyzer"
    assert orchestrator._context_compactor == "context-compactor"
    assert orchestrator._context_manager == "context-manager"
    assert orchestrator._usage_analytics == "usage-analytics"
    assert orchestrator._sequence_tracker == "sequence-tracker"
    assert orchestrator._tool_output_formatter == "tool-formatter"
    assert components["context_manager"] == "context-manager"
