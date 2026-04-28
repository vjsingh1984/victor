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

"""Unit tests for ChatCoordinator.

Tests for chat and streaming chat operations extracted from orchestrator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any

from victor.agent.services.chat_compat import ChatCoordinator
from victor.core.errors import ProviderRateLimitError


def _make_deprecated_chat_coordinator(mock_orchestrator):
    """Construct the deprecated ChatCoordinator shim with an explicit warning assertion."""
    with pytest.warns(DeprecationWarning, match="ChatCoordinator is deprecated"):
        return ChatCoordinator(mock_orchestrator)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator satisfying ChatOrchestratorProtocol."""
    orch = MagicMock()

    # Conversation
    orch.messages = []
    orch.add_message = MagicMock()
    orch.conversation = MagicMock()
    orch.conversation.ensure_system_prompt = MagicMock()
    orch.conversation.message_count = MagicMock(return_value=0)

    # Settings
    orch.settings = MagicMock()
    orch.settings.chat_max_iterations = 10

    # Task classification
    orch.task_classifier = MagicMock()
    task_class = MagicMock()
    task_class.complexity = MagicMock(value="moderate")
    task_class.tool_budget = 5
    orch.task_classifier.classify.return_value = task_class

    # Tool selector
    orch.tool_selector = MagicMock()
    orch.tool_selector.select_tools = AsyncMock(return_value=[])
    orch.tool_selector.prioritize_by_stage = MagicMock(return_value=[])

    # Provider
    orch.provider = MagicMock()
    orch.provider.supports_tools = MagicMock(return_value=True)
    orch.provider.chat = AsyncMock()
    orch._provider_service = MagicMock()
    orch._provider_service.get_rate_limit_wait_time = MagicMock(return_value=1.0)

    # Budget settings
    orch.tool_budget = 15
    orch.tool_calls_used = 0

    # Other attributes
    orch.temperature = 0.7
    orch.max_tokens = 4096
    orch.model = "claude-3-sonnet"
    orch.use_semantic_selection = False
    orch.thinking = None

    # Context compaction
    orch._context_compactor = None

    # Token tracking
    orch._cumulative_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    # Response completer
    orch.response_completer = MagicMock()
    orch.response_completer.ensure_response = AsyncMock(
        return_value=MagicMock(content="Fallback response")
    )

    # Tool calls handler
    orch.execute_tool_calls = AsyncMock(return_value=[])

    # Task coordinator
    orch.task_coordinator = MagicMock()
    orch.task_coordinator.apply_intent_guard = MagicMock()
    orch.task_coordinator.current_intent = None
    orch.task_coordinator.temperature = 0.7
    orch.task_coordinator.tool_budget = 15
    orch.task_coordinator.apply_task_guidance = MagicMock()

    # Task analyzer
    orch._task_analyzer = MagicMock()
    orch._task_analyzer.classify_task_keywords = MagicMock(
        return_value={
            "is_analysis_task": False,
            "is_action_task": True,
            "needs_execution": True,
            "coarse_task_type": "coding",
        }
    )

    # Task coordinator
    orch.task_coordinator.prepare_task = MagicMock(return_value=MagicMock())

    # Conversation controller
    orch.conversation_controller = MagicMock()

    # Context
    orch._current_intent = None
    orch._get_max_context_chars = MagicMock(return_value=200000)

    # _tool_planner
    orch._tool_planner = MagicMock()
    orch._tool_planner.filter_tools_by_intent = MagicMock(return_value=[])

    # Capability protocol — return False so stream_chat finally block is skipped
    orch.has_capability = MagicMock(return_value=False)
    orch.get_capability_value = MagicMock(return_value=None)

    return orch


@pytest.fixture
def chat_coordinator(mock_orchestrator):
    """Create ChatCoordinator with mock orchestrator."""
    return _make_deprecated_chat_coordinator(mock_orchestrator)


# =============================================================================
# Test Helper Methods
# =============================================================================


class TestExtractRequiredFilesFromPrompt:
    """Tests for _extract_required_files_from_prompt (moved to streaming.pipeline)."""

    def test_extract_absolute_paths(self):
        """Extract absolute file paths from message."""
        from victor.agent.streaming.pipeline import _extract_required_files_from_prompt

        result = _extract_required_files_from_prompt(
            "Please fix /home/user/project/main.py and /home/user/project/utils.py"
        )
        assert set(result) == {
            "/home/user/project/main.py",
            "/home/user/project/utils.py",
        }

    def test_extract_relative_paths(self):
        """Extract relative file paths from message."""
        from victor.agent.streaming.pipeline import _extract_required_files_from_prompt

        result = _extract_required_files_from_prompt("Check ./src/main.py and ../utils/helper.py")
        assert set(result) == {"./src/main.py", "../utils/helper.py"}

    def test_extract_mixed_paths(self):
        """Extract mixed path types."""
        from victor.agent.streaming.pipeline import _extract_required_files_from_prompt

        result = _extract_required_files_from_prompt(
            "Review /absolute/path.py and ./relative/path.py"
        )
        assert set(result) == {"/absolute/path.py", "./relative/path.py"}

    def test_deduplicate_paths(self):
        """Remove duplicate paths."""
        from victor.agent.streaming.pipeline import _extract_required_files_from_prompt

        result = _extract_required_files_from_prompt(
            "Fix ./file.py and also fix ./file.py and /abs/file.py"
        )
        assert len(result) == 2  # Deduplicated: ./file.py and /abs/file.py

    def test_no_paths_returns_empty(self):
        """Return empty list when no paths found."""
        from victor.agent.streaming.pipeline import _extract_required_files_from_prompt

        result = _extract_required_files_from_prompt("Hello, how are you today?")
        assert result == []


class TestExtractRequiredOutputsFromPrompt:
    """Tests for _extract_required_outputs_from_prompt (moved to streaming.pipeline)."""

    def test_returns_empty_list(self):
        """Always returns empty list as outputs are advisory."""
        from victor.agent.streaming.pipeline import (
            _extract_required_outputs_from_prompt,
        )

        result = _extract_required_outputs_from_prompt("Create output.txt and results.json")
        assert result == []


class TestPrepareTask:
    """Tests for _prepare_task."""

    def test_prepare_task(self, chat_coordinator, mock_orchestrator):
        """Prepare task through task coordinator."""
        from victor.agent.unified_task_tracker import TrackerTaskType

        mock_orchestrator.task_coordinator.prepare_task.return_value = MagicMock(task_type="edit")
        result = chat_coordinator._prepare_task(
            "Write a function",
            TrackerTaskType.EDIT,
        )
        mock_orchestrator.task_coordinator.prepare_task.assert_called_once()


class TestChatServiceShim:
    """Tests for ChatCoordinator delegation to canonical ChatService."""

    @pytest.mark.asyncio
    async def test_chat_delegates_to_bound_chat_service(self, chat_coordinator):
        chat_service = MagicMock()
        response = MagicMock(content="service response")
        chat_service.chat = AsyncMock(return_value=response)

        chat_coordinator.bind_chat_service(chat_service)

        with pytest.warns(
            DeprecationWarning,
            match="ChatCoordinator.chat\\(\\) is deprecated compatibility surface",
        ):
            result = await chat_coordinator.chat("hello", use_planning=True)

        assert result is response
        chat_service.chat.assert_awaited_once_with("hello", use_planning=True)

    def test_turn_executor_requires_bound_runtime(self, mock_orchestrator):
        coordinator = _make_deprecated_chat_coordinator(mock_orchestrator)

        with pytest.warns(DeprecationWarning, match="ChatCoordinator.turn_executor is deprecated"):
            with pytest.raises(RuntimeError, match="requires a bound ChatService runtime"):
                _ = coordinator.turn_executor


class TestGetRateLimitWaitTime:
    """Tests for _get_rate_limit_wait_time."""

    def test_calculates_wait_time_with_backoff(self, chat_coordinator, mock_orchestrator):
        """Calculate wait time with exponential backoff."""
        mock_orchestrator._provider_service.get_rate_limit_wait_time.return_value = 2.0
        exc = Exception("Rate limited")

        # First attempt
        wait = chat_coordinator._get_rate_limit_wait_time(exc, 0)
        assert wait == 2.0  # 2.0 * 2^0 = 2.0

        # Second attempt
        wait = chat_coordinator._get_rate_limit_wait_time(exc, 1)
        assert wait == 4.0  # 2.0 * 2^1 = 4.0

        # Third attempt
        wait = chat_coordinator._get_rate_limit_wait_time(exc, 2)
        assert wait == 8.0  # 2.0 * 2^2 = 8.0

    def test_caps_wait_at_max(self, chat_coordinator, mock_orchestrator):
        """Cap wait time at 300 seconds."""
        mock_orchestrator._provider_service.get_rate_limit_wait_time.return_value = 100.0
        exc = Exception("Rate limited")

        # Would be 100 * 2^4 = 1600, but capped at 300
        wait = chat_coordinator._get_rate_limit_wait_time(exc, 4)
        assert wait == 300.0

    # NOTE: TestSelectToolsForTurn, TestParseAndValidateToolCalls, TestCreateRecoveryContext,
    # TestHandleRecoveryWithIntegration, and TestApplyRecoveryAction delegation tests removed.
    # These methods were moved to the streaming pipeline (direct orch calls).
    # The real logic is tested in test_orchestrator_core.py.


class TestHandleEmptyResponseRecovery:
    """Tests for _handle_empty_response_recovery."""

    @pytest.mark.asyncio
    async def test_empty_response_recovery_logic(self, chat_coordinator, mock_orchestrator):
        """Test empty response recovery handles provider stream."""
        from victor.providers.base import StreamChunk

        # Mock streaming provider
        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content="recovered content", role="assistant")

        mock_orchestrator.provider.stream = mock_stream
        mock_orchestrator.add_message = MagicMock()
        mock_orchestrator.sanitizer.sanitize = MagicMock(return_value="sanitized content")
        mock_orchestrator._chunk_generator.generate_content_chunk = MagicMock(
            return_value=MagicMock()
        )

        success, tool_calls, chunk = await chat_coordinator._handle_empty_response_recovery(
            MagicMock(), []
        )

        # Should recover content
        assert success is True
        assert tool_calls is None
        assert chunk is not None

    # NOTE: TestValidateIntelligentResponse delegation test removed.
    # This method was moved to the streaming pipeline (direct orch call).
    # The real logic is tested in test_orchestrator_core.py.


class TestStreamingShimBehavior:
    """Tests for ChatCoordinator.stream_chat shim delegation."""

    @pytest.mark.asyncio
    async def test_stream_chat_prefers_bound_chat_service(self, mock_orchestrator):
        """Shim delegates to bound ChatService before orchestrator helpers."""
        from victor.providers.base import StreamChunk

        chunk = StreamChunk(content="streamed", role="assistant")
        chat_service = MagicMock()

        async def _stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            yield chunk

        coordinator = _make_deprecated_chat_coordinator(mock_orchestrator)
        chat_service.stream_chat = _stream_chat
        coordinator.bind_chat_service(chat_service)

        with pytest.warns(
            DeprecationWarning,
            match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            results = [c async for c in coordinator.stream_chat("hello")]
        assert results == [chunk]

    @pytest.mark.asyncio
    async def test_stream_chat_prefers_orchestrator_public_runtime(self, mock_orchestrator):
        """Shim delegates to orchestrator.stream_chat when no ChatService is bound."""
        from victor.providers.base import StreamChunk

        chunk = StreamChunk(content="runtime", role="assistant")

        async def _stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"mode": "test"}
            yield chunk

        mock_orchestrator.stream_chat = _stream_chat
        coordinator = _make_deprecated_chat_coordinator(mock_orchestrator)

        with pytest.warns(
            DeprecationWarning,
            match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            results = [c async for c in coordinator.stream_chat("hello", mode="test")]

        assert results == [chunk]

    @pytest.mark.asyncio
    async def test_stream_chat_prefers_public_runtime_over_internal_compatibility_hooks(
        self, mock_orchestrator
    ):
        """Shim uses orchestrator.stream_chat before deprecated internal hooks."""
        from victor.providers.base import StreamChunk

        chunk = StreamChunk(content="runtime", role="assistant")

        async def _runtime_stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"mode": "test"}
            yield chunk

        async def _legacy_stream_chat(user_message: str, **kwargs):
            raise AssertionError("legacy compatibility hook should not run")

        mock_orchestrator.stream_chat = _runtime_stream_chat
        mock_orchestrator._get_service_streaming_runtime = MagicMock(
            side_effect=AssertionError("internal streaming runtime getter should not run")
        )
        mock_orchestrator._stream_chat_runtime = _legacy_stream_chat
        coordinator = _make_deprecated_chat_coordinator(mock_orchestrator)

        with pytest.warns(
            DeprecationWarning,
            match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            results = [c async for c in coordinator.stream_chat("hello", mode="test")]

        assert results == [chunk]
        mock_orchestrator._get_service_streaming_runtime.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_chat_prefers_bound_chat_service_over_other_paths(self, mock_orchestrator):
        """Shim uses bound ChatService before orchestrator runtime and internal hooks."""
        from victor.providers.base import StreamChunk

        chunk = StreamChunk(content="service", role="assistant")
        chat_service = MagicMock()

        async def _service_stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"mode": "test"}
            yield chunk

        async def _runtime_stream_chat(user_message: str, **kwargs):
            raise AssertionError("service runtime should not run when ChatService is bound")

        async def _legacy_stream_chat(user_message: str, **kwargs):
            raise AssertionError("legacy compatibility hook should not run")

        chat_service.stream_chat = _service_stream_chat
        mock_orchestrator.stream_chat = _runtime_stream_chat
        mock_orchestrator._get_service_streaming_runtime = MagicMock(
            side_effect=AssertionError("internal streaming runtime getter should not run")
        )
        mock_orchestrator._stream_chat_runtime = _legacy_stream_chat
        coordinator = _make_deprecated_chat_coordinator(mock_orchestrator)
        coordinator.bind_chat_service(chat_service)

        with pytest.warns(
            DeprecationWarning,
            match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            results = [c async for c in coordinator.stream_chat("hello", mode="test")]

        assert results == [chunk]
        mock_orchestrator._get_service_streaming_runtime.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_chat_requires_service_runtime_not_legacy_hook(self, mock_orchestrator):
        """Shim no longer preserves the legacy _stream_chat_runtime fallback hook."""

        async def _runtime(user_message: str, **kwargs):
            yield MagicMock()

        mock_orchestrator._stream_chat_runtime = _runtime
        coordinator = _make_deprecated_chat_coordinator(mock_orchestrator)

        with pytest.warns(
            DeprecationWarning,
            match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            with pytest.raises(RuntimeError, match="no bound ChatService or streaming runtime"):
                _ = [c async for c in coordinator.stream_chat("hello")]

    @pytest.mark.asyncio
    async def test_stream_chat_raises_when_no_runtime_path(self, mock_orchestrator):
        """Shim fails fast when no ChatService or service runtime is wired."""
        coordinator = _make_deprecated_chat_coordinator(mock_orchestrator)

        with pytest.warns(
            DeprecationWarning,
            match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            with pytest.raises(RuntimeError, match="no bound ChatService or streaming runtime"):
                _ = [c async for c in coordinator.stream_chat("hello")]
