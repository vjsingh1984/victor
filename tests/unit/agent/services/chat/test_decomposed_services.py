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

"""Tests for chat helper services and the canonical chat surface."""

import importlib
import pytest

from victor.agent.services.chat import (
    ResponseAggregationService,
    ResponseAggregationServiceConfig,
    ContinuationService,
    ContinuationServiceConfig,
)
from victor.agent.services.chat_service import ChatService, ChatServiceConfig
from victor.providers.base import CompletionResponse, StreamChunk

# =============================================================================
# Mock Dependencies
# =============================================================================


class MockProviderService:
    """Mock provider service."""

    def __init__(self):
        self.healthy = True

    def is_healthy(self):
        return self.healthy

    async def chat_completion(self, messages, **kwargs):
        return CompletionResponse(
            content="Mock response",
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

    async def stream_completion(self, messages, **kwargs):
        yield StreamChunk(content="Mock chunk")


class MockToolService:
    """Mock tool service."""

    def __init__(self):
        self.healthy = True

    def is_healthy(self):
        return self.healthy

    async def execute_tool(self, tool_name, arguments):
        from victor.tools.base import ToolResult

        return ToolResult(success=True, output=f"Executed {tool_name}")


class MockContextService:
    """Mock context service."""

    def __init__(self):
        self.healthy = True
        self.messages = []

    def is_healthy(self):
        return self.healthy

    async def check_context_overflow(self):
        return False

    async def compact_context(self):
        self.messages = self.messages[-6:]

    def get_messages(self):
        return self.messages

    @property
    def message_count(self):
        return len(self.messages)

    def add_message(self, message):
        self.messages.append(message)

    def clear_messages(self, retain_system=True):
        self.messages.clear()


class MockRecoveryService:
    """Mock recovery service."""

    def __init__(self):
        self.healthy = True

    def is_healthy(self):
        return True


class MockConversationController:
    """Mock conversation controller."""

    def reset(self):
        pass


class MockStreamingCoordinator:
    """Mock streaming coordinator."""

    pass


# =============================================================================
# ResponseAggregationService Tests
# =============================================================================


class TestResponseAggregationService:
    """Tests for ResponseAggregationService."""

    def test_aggregate_chunks_basic(self):
        """Test basic chunk aggregation."""
        config = ResponseAggregationServiceConfig()
        service = ResponseAggregationService(config)

        chunks = [
            StreamChunk(content="Hello "),
            StreamChunk(content="world"),
            StreamChunk(content="!"),
        ]

        response = service.aggregate_chunks(chunks)

        assert response.content == "Hello world!"
        assert response.stop_reason == "stop"

    def test_aggregate_empty_chunks(self):
        """Test aggregating empty chunk list raises error."""
        config = ResponseAggregationServiceConfig()
        service = ResponseAggregationService(config)

        with pytest.raises(ValueError, match="Cannot aggregate empty"):
            service.aggregate_chunks([])

    def test_format_response(self):
        """Test response formatting."""
        config = ResponseAggregationServiceConfig()
        service = ResponseAggregationService(config)

        response = CompletionResponse(
            content="  Test response  ",
            stop_reason="stop",
            usage=None,
        )

        formatted = service.format_response(response)

        assert formatted == "Test response"

    def test_normalize_response(self):
        """Test response normalization."""
        config = ResponseAggregationServiceConfig()
        service = ResponseAggregationService(config)

        # Response with missing usage (Pydantic doesn't allow None for content)
        response = CompletionResponse(
            content="Test",
            stop_reason="stop",
            usage=None,
        )

        normalized = service.normalize_response(response)

        assert normalized.content == "Test"
        assert normalized.stop_reason == "stop"
        assert normalized.usage == {"prompt_tokens": 0, "completion_tokens": 0}

    def test_get_aggregation_metrics(self):
        """Test aggregation metrics."""
        config = ResponseAggregationServiceConfig()
        service = ResponseAggregationService(config)

        chunks = [StreamChunk(content=f"Chunk {i}") for i in range(5)]
        service.aggregate_chunks(chunks)

        metrics = service.get_aggregation_metrics()

        assert metrics["aggregation_count"] == 1
        assert metrics["total_chunks_processed"] == 5
        assert metrics["average_chunks_per_aggregation"] == 5.0


# =============================================================================
# ContinuationService Tests
# =============================================================================


class TestContinuationService:
    """Tests for ContinuationService."""

    def test_needs_continuation_max_tokens(self):
        """Test continuation needed for max_tokens stop reason."""
        config = ContinuationServiceConfig()
        service = ContinuationService(config)

        response = CompletionResponse(
            content="Some content",
            stop_reason="max_tokens",
            usage=None,
        )

        assert service.needs_continuation(response) is True

    def test_needs_continuation_short_content(self):
        """Test continuation needed for short content."""
        config = ContinuationServiceConfig(continuation_threshold=100)
        service = ContinuationService(config)

        response = CompletionResponse(
            content="Short",
            stop_reason="stop",
            usage=None,
        )

        assert service.needs_continuation(response) is True

    def test_needs_continuation_complete(self):
        """Test no continuation needed for complete response."""
        config = ContinuationServiceConfig(continuation_threshold=50)
        service = ContinuationService(config)

        response = CompletionResponse(
            content="This is a long enough response that exceeds the threshold.",
            stop_reason="stop",
            usage=None,
        )

        assert service.needs_continuation(response) is False

    def test_should_stop_continuation_max_prompts(self):
        """Test stopping at max continuation prompts."""
        config = ContinuationServiceConfig(max_continuation_prompts=3)
        service = ContinuationService(config)

        service._continuation_count = 3

        assert service.should_stop_continuation(iteration=1, max_iterations=10) is True

    def test_should_stop_continuation_max_iterations(self):
        """Test stopping at max iterations."""
        config = ContinuationServiceConfig()
        service = ContinuationService(config)

        assert service.should_stop_continuation(iteration=200, max_iterations=200) is True

    def test_create_continuation_prompt(self):
        """Test continuation prompt creation."""
        config = ContinuationServiceConfig()
        service = ContinuationService(config)

        response = CompletionResponse(
            content="This is the previous content",
            stop_reason="max_tokens",
            usage=None,
        )

        import asyncio

        prompt = asyncio.run(service.create_continuation_prompt(response))

        assert "Please continue" in prompt
        assert "previous content" in prompt

    def test_get_continuation_metrics(self):
        """Test continuation metrics."""
        config = ContinuationServiceConfig()
        service = ContinuationService(config)

        service._continuation_count = 2
        service._total_continuations = 5

        metrics = service.get_continuation_metrics()

        assert metrics["current_continuation_count"] == 2
        assert metrics["total_continuations"] == 5
        assert metrics["max_continuation_prompts"] == 3


# =============================================================================
# Canonical ChatService Surface Tests
# =============================================================================


class TestCanonicalChatServiceSurface:
    """Tests for the canonical ChatService surface."""

    def test_helper_package_does_not_export_chat_service(self):
        """Helper package should not export a duplicate ChatService."""
        chat_pkg = importlib.import_module("victor.agent.services.chat")

        assert not hasattr(chat_pkg, "ChatService")
        assert not hasattr(chat_pkg, "ChatServiceConfig")

    def test_legacy_facade_module_removed(self):
        """Legacy facade module should point callers to the canonical service."""
        with pytest.raises(ImportError, match="chat_service\\.ChatService"):
            importlib.import_module("victor.agent.services.chat.chat_service_facade")

    def test_initialization(self):
        """Test canonical ChatService initialization."""
        config = ChatServiceConfig()

        provider_service = MockProviderService()
        tool_service = MockToolService()
        context_service = MockContextService()
        recovery_service = MockRecoveryService()
        conversation_controller = MockConversationController()

        service = ChatService(
            config=config,
            provider_service=provider_service,
            tool_service=tool_service,
            context_service=context_service,
            recovery_service=recovery_service,
            conversation_controller=conversation_controller,
            streaming_coordinator=MockStreamingCoordinator(),
        )

        assert service is not None

    def test_is_healthy(self):
        """Test health check."""
        config = ChatServiceConfig()

        service = ChatService(
            config=config,
            provider_service=MockProviderService(),
            tool_service=MockToolService(),
            context_service=MockContextService(),
            recovery_service=MockRecoveryService(),
            conversation_controller=MockConversationController(),
            streaming_coordinator=MockStreamingCoordinator(),
        )

        assert service.is_healthy() is True

    def test_conversation_empty(self):
        """Test empty conversation check."""
        config = ChatServiceConfig()

        service = ChatService(
            config=config,
            provider_service=MockProviderService(),
            tool_service=MockToolService(),
            context_service=MockContextService(),
            recovery_service=MockRecoveryService(),
            conversation_controller=MockConversationController(),
            streaming_coordinator=MockStreamingCoordinator(),
        )

        assert service.is_conversation_empty() is True

    def test_get_message_count(self):
        """Test message count."""
        config = ChatServiceConfig()
        context_service = MockContextService()
        context_service.messages = [{"role": "user"}, {"role": "assistant"}]

        service = ChatService(
            config=config,
            provider_service=MockProviderService(),
            tool_service=MockToolService(),
            context_service=context_service,
            recovery_service=MockRecoveryService(),
            conversation_controller=MockConversationController(),
            streaming_coordinator=MockStreamingCoordinator(),
        )

        assert service.get_message_count() == 2

    def test_get_conversation_stats(self):
        """Test canonical conversation stats."""
        config = ChatServiceConfig()
        context_service = MockContextService()
        context_service.messages = [
            {"role": "system"},
            {"role": "user"},
            {"role": "assistant"},
            {"role": "tool"},
        ]

        service = ChatService(
            config=config,
            provider_service=MockProviderService(),
            tool_service=MockToolService(),
            context_service=context_service,
            recovery_service=MockRecoveryService(),
            conversation_controller=MockConversationController(),
            streaming_coordinator=MockStreamingCoordinator(),
        )

        stats = service.get_conversation_stats()

        assert stats["message_count"] == 4
        assert stats["is_empty"] is False

    def test_bind_runtime_components(self):
        """Test binding canonical runtime collaborators."""
        config = ChatServiceConfig()

        service = ChatService(
            config=config,
            provider_service=MockProviderService(),
            tool_service=MockToolService(),
            context_service=MockContextService(),
            recovery_service=MockRecoveryService(),
            conversation_controller=MockConversationController(),
            streaming_coordinator=MockStreamingCoordinator(),
        )

        async def _stream_handler(user_message: str, **kwargs):
            _ = user_message, kwargs
            yield StreamChunk(content="chunk")

        service.bind_runtime_components(
            turn_executor=object(),
            planning_handler=lambda user_message: user_message,
            stream_chat_handler=_stream_handler,
        )

        assert service._turn_executor is not None
        assert service._planning_handler is not None
        assert service._stream_chat_handler is _stream_handler
