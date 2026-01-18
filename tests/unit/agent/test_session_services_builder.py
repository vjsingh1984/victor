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

"""Tests for SessionServicesBuilder."""

from unittest.mock import MagicMock, patch

from victor.agent.builders.session_services_builder import SessionServicesBuilder


def test_session_services_builder_wires_components():
    """SessionServicesBuilder assigns session state and core services."""
    settings = MagicMock()
    factory = MagicMock()
    factory.initialize_tool_budget.return_value = 5
    factory.create_complexity_classifier.return_value = "classifier"
    factory.create_action_authorizer.return_value = "authorizer"
    factory.create_search_router.return_value = "router"
    factory.create_presentation_adapter.return_value = "presentation"
    factory.create_reminder_manager.return_value = "reminder"

    orchestrator = MagicMock()
    orchestrator._tool_calling_caps_internal = "caps"
    orchestrator.provider_name = "provider"

    with (
        patch(
            "victor.agent.builders.session_services_builder.create_session_state_manager",
            return_value="session-state",
        ) as create_session_state,
        patch(
            "victor.agent.builders.session_services_builder.TaskCompletionDetector"
        ) as completion_cls,
    ):
        completion_instance = MagicMock()
        completion_cls.return_value = completion_instance

        builder = SessionServicesBuilder(settings=settings, factory=factory)
        components = builder.build(orchestrator)

    factory.initialize_tool_budget.assert_called_once_with("caps")
    create_session_state.assert_called_once_with(tool_budget=5)
    factory.create_reminder_manager.assert_called_once_with(
        provider="provider",
        task_complexity="medium",
        tool_budget=5,
    )

    assert orchestrator.tool_budget == 5
    assert orchestrator._session_state == "session-state"
    assert orchestrator.task_classifier == "classifier"
    assert orchestrator.intent_detector == "authorizer"
    assert orchestrator.search_router == "router"
    assert orchestrator._presentation == "presentation"
    assert orchestrator._task_completion_detector == completion_instance
    assert orchestrator.reminder_manager == "reminder"
    assert components["tool_budget"] == 5
