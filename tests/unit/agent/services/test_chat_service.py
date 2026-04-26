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

"""Tests for ChatService implementation."""

import asyncio
import inspect
from unittest import mock

import pytest

from victor.agent.services.chat_service import ChatService, ChatServiceConfig
from victor.providers.base import CompletionResponse

# =============================================================================
# Mock Dependencies
# =============================================================================


class MockProviderService:
    """Mock provider service."""

    def __init__(self):
        self.healthy = True
        self.current_provider = self

    def is_healthy(self):
        return self.healthy

    def get_current_provider(self):
        return self

    async def chat_completion(self, messages, **kwargs):
        from victor.providers.base import CompletionResponse

        return CompletionResponse(
            content="Mock response",
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

    async def stream_completion(self, messages, **kwargs):
        from victor.providers.base import StreamChunk

        yield StreamChunk(content="Mock chunk")

    async def check_provider_health(self, provider=None):
        return True


class MockToolService:
    """Mock tool service."""

    def __init__(self):
        self.healthy = True
        self.tools_executed = []

    def is_healthy(self):
        return self.healthy

    async def execute_tool(self, tool_name, arguments):
        from victor.tools.base import ToolResult

        self.tools_executed.append((tool_name, arguments))
        return ToolResult(success=True, output=f"Executed {tool_name}")


class MockContextService:
    """Mock context service."""

    def __init__(self):
        self.healthy = True
        self.messages = []
        self.overflow = False

    def is_healthy(self):
        return self.healthy

    async def check_context_overflow(self):
        return self.overflow

    async def compact_context(self):
        self.messages = self.messages[-6:]

    def get_messages(self):
        return self.messages

    def add_message(self, message):
        self.messages.append(message)

    def clear_messages(self, retain_system=True):
        self.messages.clear()


class MockRecoveryService:
    """Mock recovery service."""

    def __init__(self):
        self.recovery_attempts = 0

    def is_healthy(self):
        return True

    async def execute_recovery(self, context):
        self.recovery_attempts += 1
        return False  # Recovery fails by default


class MockConversationController:
    """Mock conversation controller."""

    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class MockStreamingCoordinator:
    """Mock streaming coordinator."""

    pass


# =============================================================================
# Base Test Class with Helper Methods
# =============================================================================


class BaseChatServiceTest:
    """Base class for ChatService tests with helper methods."""

    def _create_test_service(self, config=None):
        """Create a test ChatService with mocked dependencies."""
        if config is None:
            config = ChatServiceConfig()

        provider = MockProviderService()
        tools = MockToolService()
        context = MockContextService()
        recovery = MockRecoveryService()
        conversation = MockConversationController()
        streaming = MockStreamingCoordinator()

        return ChatService(
            config=config,
            provider_service=provider,
            tool_service=tools,
            context_service=context,
            recovery_service=recovery,
            conversation_controller=conversation,
            streaming_coordinator=streaming,
        )


# =============================================================================
# Tests
# =============================================================================


class TestChatServiceConfig(BaseChatServiceTest):
    """Tests for ChatServiceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChatServiceConfig()

        assert config.max_iterations == 200
        assert config.max_continuation_prompts == 3
        assert config.stream_chunk_size == 100
        assert config.enable_response_caching is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChatServiceConfig(
            max_iterations=100,
            max_continuation_prompts=5,
            stream_chunk_size=200,
            enable_response_caching=False,
        )

        assert config.max_iterations == 100
        assert config.max_continuation_prompts == 5
        assert config.stream_chunk_size == 200
        assert config.enable_response_caching is False


class TestChatServiceInit(BaseChatServiceTest):
    """Tests for ChatService initialization."""

    def test_initialization(self):
        """Test service initialization with all dependencies."""
        config = ChatServiceConfig()
        provider = MockProviderService()
        tools = MockToolService()
        context = MockContextService()
        recovery = MockRecoveryService()
        conversation = MockConversationController()
        streaming = MockStreamingCoordinator()

        service = ChatService(
            config=config,
            provider_service=provider,
            tool_service=tools,
            context_service=context,
            recovery_service=recovery,
            conversation_controller=conversation,
            streaming_coordinator=streaming,
        )

        assert service._config is config
        assert service._provider is provider
        assert service._tools is tools
        assert service._context is context
        assert service._recovery is recovery
        assert service._conversation is conversation
        assert service._streaming is streaming


class TestChatServiceHealth(BaseChatServiceTest):
    """Tests for ChatService health checks."""

    def test_is_healthy_all_services_healthy(self):
        """Test health check when all services are healthy."""
        service = self._create_test_service()
        assert service.is_healthy() is True

    def test_is_healthy_provider_unhealthy(self):
        """Test health check when provider is unhealthy."""
        service = self._create_test_service()
        service._provider.healthy = False
        assert service.is_healthy() is False

    def test_is_healthy_tools_unhealthy(self):
        """Test health check when tools are unhealthy."""
        service = self._create_test_service()
        service._tools.healthy = False
        assert service.is_healthy() is False

    def test_is_healthy_context_unhealthy(self):
        """Test health check when context is unhealthy."""
        service = self._create_test_service()
        service._context.healthy = False
        assert service.is_healthy() is False


class TestChatServiceReset(BaseChatServiceTest):
    """Tests for ChatService.reset_conversation method."""

    def test_reset_conversation(self):
        """Test conversation reset."""
        service = self._create_test_service()

        # Add some context
        service._context.messages = [{"role": "user", "content": "test"}]

        service.reset_conversation()

        assert service._context.messages == []
        assert service._conversation.reset_count == 1


class TestChatServiceControllerBackedContext(BaseChatServiceTest):
    """Direct ChatService paths should share conversation state with controller-backed context."""

    def _create_controller_backed_service(self):
        from victor.agent.conversation.controller import ConversationController
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        config = ChatServiceConfig()
        provider = MockProviderService()
        tools = MockToolService()
        controller = ConversationController()
        context = ContextServiceAdapter(controller)
        recovery = MockRecoveryService()
        streaming = MockStreamingCoordinator()

        service = ChatService(
            config=config,
            provider_service=provider,
            tool_service=tools,
            context_service=context,
            recovery_service=recovery,
            conversation_controller=controller,
            streaming_coordinator=streaming,
        )
        return service, controller

    @pytest.mark.asyncio
    async def test_chat_records_user_and_assistant_messages_in_controller_backed_context(self):
        service, controller = self._create_controller_backed_service()
        response = CompletionResponse(content="done", role="assistant", stop_reason="stop")
        service._run_agentic_loop = mock.AsyncMock(return_value=response)

        result = await service.chat("hello")

        assert result is response
        assert [message.role for message in controller.messages] == ["user", "assistant"]
        assert [message.content for message in controller.messages] == ["hello", "done"]

    def test_add_tool_result_supports_controller_backed_context_adapter(self):
        service, controller = self._create_controller_backed_service()
        result = mock.MagicMock(output="tool output", error=None)

        service._add_tool_result_to_context(
            "read",
            result,
            tool_call_id="call-1",
            formatted_content="formatted output",
        )

        tool_message = controller.messages[-1]
        assert tool_message.role == "tool"
        assert tool_message.content == "formatted output"
        assert tool_message.tool_call_id == "call-1"
        assert tool_message.name == "read"

    def test_chat_service_declares_single_keyword_based_context_helpers(self):
        source = inspect.getsource(ChatService)

        assert source.count("def _add_user_message_to_context(") == 1
        assert source.count("def _add_assistant_message_to_context(") == 1
        assert "self._context.add_message(msg)" not in source


class TestStreamingPipelineIntegration:
    """Tests verifying streaming pipeline has AgenticLoop components."""

    def test_pipeline_accepts_perception(self):
        """StreamingChatPipeline constructor accepts perception parameter."""
        from unittest.mock import MagicMock
        from victor.agent.streaming.pipeline import StreamingChatPipeline

        mock_coord = MagicMock()
        mock_perception = MagicMock()
        pipeline = StreamingChatPipeline(mock_coord, perception=mock_perception)
        assert pipeline._perception is mock_perception

    def test_pipeline_accepts_fulfillment(self):
        """StreamingChatPipeline constructor accepts fulfillment parameter."""
        from unittest.mock import MagicMock
        from victor.agent.streaming.pipeline import StreamingChatPipeline

        mock_coord = MagicMock()
        mock_fulfillment = MagicMock()
        pipeline = StreamingChatPipeline(mock_coord, fulfillment=mock_fulfillment)
        assert pipeline._fulfillment is mock_fulfillment

    def test_pipeline_has_progress_tracking(self):
        """StreamingChatPipeline tracks progress scores."""
        from unittest.mock import MagicMock
        from victor.agent.streaming.pipeline import StreamingChatPipeline

        pipeline = StreamingChatPipeline(MagicMock())
        assert hasattr(pipeline, "_progress_scores")
        assert pipeline._progress_scores == []

    def test_factory_passes_components(self):
        """create_streaming_chat_pipeline passes perception and fulfillment."""
        from unittest.mock import MagicMock
        from victor.agent.streaming.pipeline import create_streaming_chat_pipeline

        mock_perception = MagicMock()
        mock_fulfillment = MagicMock()
        pipeline = create_streaming_chat_pipeline(
            MagicMock(), perception=mock_perception, fulfillment=mock_fulfillment
        )
        assert pipeline._perception is mock_perception
        assert pipeline._fulfillment is mock_fulfillment


# =============================================================================
# Test Helpers
# =============================================================================


def test_chat_service_integration_with_feature_flags():
    """Test that ChatService works with feature flags."""
    from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

    # Enable new chat service flag
    manager = get_feature_flag_manager()
    manager.enable(FeatureFlag.USE_NEW_CHAT_SERVICE)

    # Flag should be enabled
    assert manager.is_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE)


class TestToolService:
    """Tests for ToolService implementation."""

    def test_initialization(self):
        """Test service initialization."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        config = ToolServiceConfig()
        selector = mock.Mock()
        executor = mock.Mock()
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        assert service._config is config
        assert service._selector is selector
        assert service._executor is executor
        assert service._registrar is registrar

    @pytest.mark.asyncio
    async def test_select_tools(self):
        """Test tool selection."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        config = ToolServiceConfig()
        selector = mock.Mock()
        selector.select = mock.AsyncMock(return_value=["tool1", "tool2"])
        executor = mock.Mock()
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        context = mock.Mock()
        tools = await service.select_tools(context, max_tools=10)

        assert tools == ["tool1", "tool2"]
        selector.select.assert_called_once_with(context, 10)

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig
        from victor.tools.base import ToolResult

        config = ToolServiceConfig()
        selector = mock.Mock()
        executor = mock.Mock()
        executor.execute = mock.AsyncMock(return_value=ToolResult(success=True, output="Success"))
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        result = await service.execute_tool("test_tool", {"arg": "value"})

        assert result.success is True
        assert result.output == "Success"
        executor.execute.assert_called_once_with("test_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_tool_budget(self):
        """Test tool budget management."""
        from victor.agent.services.tool_service import (
            ToolService,
            ToolServiceConfig,
            ToolBudgetExceededError,
        )
        from victor.tools.base import ToolResult

        config = ToolServiceConfig(default_tool_budget=2)
        selector = mock.Mock()
        executor = mock.Mock()
        executor.execute = mock.AsyncMock(return_value=ToolResult(success=True, output="Success"))
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        # Initial budget
        assert service.get_tool_budget() == 2

        # Execute tool
        await service.execute_tool("test", {})

        # Budget decreased
        assert service.get_tool_budget() == 1

        # Execute another tool
        await service.execute_tool("test", {})

        # Budget exhausted
        assert service.get_tool_budget() == 0

        # Next execution should fail
        with pytest.raises(ToolBudgetExceededError):
            await service.execute_tool("test", {})

    def test_get_tool_usage_stats(self):
        """Test getting tool usage statistics."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        config = ToolServiceConfig()
        selector = mock.Mock()
        executor = mock.Mock()
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        # Track some usage
        service._track_tool_usage("read", True)
        service._track_tool_usage("read", True)
        service._track_tool_usage("write", True)
        service._track_tool_usage("error:grep", False)

        stats = service.get_tool_usage_stats()

        assert stats["total_calls"] == 4
        assert stats["successful_calls"] == 3
        assert stats["failed_calls"] == 1
        assert stats["success_rate"] == 0.75
