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

"""Tests for PromptingBuilder."""

from unittest.mock import MagicMock, patch

from victor.agent.builders.prompting_builder import PromptingBuilder


def test_prompting_builder_wires_components():
    """PromptingBuilder assigns prompt-related components."""
    settings = MagicMock()
    factory = MagicMock()
    factory.create_sanitizer.return_value = "sanitizer"
    factory.create_system_prompt_builder.return_value = "prompt-builder"
    project_context = MagicMock()
    project_context.content = True
    project_context.get_system_prompt_addition.return_value = "context-add"
    project_context.context_file = "context-file"
    factory.create_project_context.return_value = project_context

    orchestrator = MagicMock()
    orchestrator.provider_name = "provider"
    orchestrator.tool_adapter = "tool-adapter"
    orchestrator._tool_calling_caps_internal = "tool-caps"
    orchestrator.mode_controller = "mode-controller"
    orchestrator._build_system_prompt_with_adapter.return_value = "base-prompt"

    with (
        patch(
            "victor.agent.builders.prompting_builder.ResponseCoordinator",
            return_value="response-coordinator",
        ),
        patch(
            "victor.agent.coordinators.prompt_coordinator.PromptBuilderCoordinator",
            return_value="prompt-coordinator",
        ),
        patch(
            "victor.agent.coordinators.mode_coordinator.ModeCoordinator",
            return_value="mode-coordinator",
        ),
    ):
        builder = PromptingBuilder(settings=settings, factory=factory)
        components = builder.build(orchestrator, model="model")

    assert orchestrator.sanitizer == "sanitizer"
    assert orchestrator._response_coordinator == "response-coordinator"
    assert orchestrator.prompt_builder == "prompt-builder"
    assert orchestrator.project_context == project_context
    assert orchestrator._prompt_coordinator == "prompt-coordinator"
    assert orchestrator._mode_coordinator == "mode-coordinator"
    assert orchestrator._system_prompt == "base-prompt\n\ncontext-add"
    assert orchestrator._system_added is False
    assert components["system_prompt"] == "base-prompt\n\ncontext-add"
