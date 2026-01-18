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

"""Tests for WorkflowMemoryBuilder."""

from unittest.mock import MagicMock

from victor.agent.builders.workflow_memory_builder import WorkflowMemoryBuilder


def test_workflow_memory_builder_wires_core_components():
    """WorkflowMemoryBuilder assigns workflow/memory components to orchestrator."""
    settings = MagicMock()
    settings.conversation_embeddings_enabled = True

    factory = MagicMock()
    factory.create_tool_cache.return_value = "tool-cache"
    factory.create_tool_dependency_graph.return_value = "tool-graph"
    factory.create_code_execution_manager.return_value = "code-manager"
    factory.create_workflow_registry.return_value = "workflow-registry"
    factory.create_message_history.return_value = "message-history"
    factory.create_memory_components.return_value = (None, None)
    factory.create_conversation_state_machine.return_value = "conversation-state"
    factory.create_state_coordinator.return_value = "state-coordinator"
    factory.create_intent_classifier.return_value = "intent-classifier"

    orchestrator = MagicMock()
    orchestrator.provider_name = "test-provider"
    orchestrator._system_prompt = "test-prompt"
    orchestrator._session_state = "session-state"
    orchestrator._register_default_workflows = MagicMock()
    tool_caps = MagicMock()
    tool_caps.native_tool_calls = True
    orchestrator._tool_calling_caps_internal = tool_caps

    builder = WorkflowMemoryBuilder(settings=settings, factory=factory)
    components = builder.build(orchestrator)

    assert orchestrator.tool_cache == "tool-cache"
    assert orchestrator.tool_graph == "tool-graph"
    assert orchestrator.code_manager == "code-manager"
    assert orchestrator.workflow_registry == "workflow-registry"
    assert orchestrator.conversation == "message-history"
    assert orchestrator.memory_manager is None
    assert orchestrator._memory_session_id is None
    assert orchestrator.conversation_state == "conversation-state"
    assert orchestrator._state_coordinator == "state-coordinator"
    assert orchestrator.intent_classifier == "intent-classifier"
    assert components["tool_cache"] == "tool-cache"
    assert components["tool_graph"] == "tool-graph"
