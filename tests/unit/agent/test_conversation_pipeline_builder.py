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

"""Tests for ConversationPipelineBuilder."""

from unittest.mock import MagicMock

from victor.agent.builders.conversation_pipeline_builder import ConversationPipelineBuilder


def test_conversation_pipeline_builder_wires_components():
    """ConversationPipelineBuilder assigns conversation and streaming components."""
    settings = MagicMock()
    factory = MagicMock()
    factory.create_conversation_controller.return_value = "conversation-controller"
    factory.create_lifecycle_manager.return_value = "lifecycle-manager"
    factory.create_tool_deduplication_tracker.return_value = "dedup-tracker"
    pipeline = MagicMock()
    factory.create_tool_pipeline.return_value = pipeline
    factory.create_streaming_controller.return_value = "streaming-controller"
    factory.create_streaming_coordinator.return_value = "streaming-coordinator"
    factory.create_streaming_chat_handler.return_value = "streaming-handler"

    orchestrator = MagicMock()
    orchestrator.conversation = "conversation"
    orchestrator.conversation_state = "conversation-state"
    orchestrator.memory_manager = "memory"
    orchestrator._memory_session_id = "session-id"
    orchestrator._system_prompt = "system"
    orchestrator._metrics_collector = "metrics"
    orchestrator._context_compactor = None
    orchestrator._sequence_tracker = None
    orchestrator._usage_analytics = None
    orchestrator._reminder_manager = None
    orchestrator.tools = "tools"
    orchestrator.tool_executor = "tool-executor"
    orchestrator.tool_budget = 3
    orchestrator.tool_cache = "tool-cache"
    orchestrator.argument_normalizer = "normalizer"
    orchestrator._on_tool_start_callback = MagicMock()
    orchestrator._on_tool_complete_callback = MagicMock()
    orchestrator._middleware_chain = "middleware"
    orchestrator.streaming_metrics_collector = "stream-metrics"
    orchestrator._on_streaming_session_complete = MagicMock()
    orchestrator._pending_semantic_cache = None

    builder = ConversationPipelineBuilder(settings=settings, factory=factory)
    components = builder.build(orchestrator, provider="provider", model="model")

    assert orchestrator._conversation_controller == "conversation-controller"
    assert orchestrator._lifecycle_manager == "lifecycle-manager"
    assert orchestrator._deduplication_tracker == "dedup-tracker"
    assert orchestrator._tool_pipeline == pipeline
    assert orchestrator._streaming_controller == "streaming-controller"
    assert orchestrator._streaming_coordinator == "streaming-coordinator"
    assert orchestrator._streaming_handler == "streaming-handler"
    assert orchestrator._iteration_coordinator is None
    assert components["tool_pipeline"] == pipeline
