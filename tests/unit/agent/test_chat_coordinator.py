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

from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.core.errors import ProviderRateLimitError

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
    orch._provider_coordinator = MagicMock()
    orch._provider_coordinator.get_rate_limit_wait_time = MagicMock(return_value=1.0)

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
    orch._handle_tool_calls = AsyncMock(return_value=[])

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

    return orch


@pytest.fixture
def chat_coordinator(mock_orchestrator):
    """Create ChatCoordinator with mock orchestrator."""
    return ChatCoordinator(mock_orchestrator)


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


class TestGetRateLimitWaitTime:
    """Tests for _get_rate_limit_wait_time."""

    def test_calculates_wait_time_with_backoff(self, chat_coordinator, mock_orchestrator):
        """Calculate wait time with exponential backoff."""
        mock_orchestrator._provider_coordinator.get_rate_limit_wait_time.return_value = 2.0
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
        mock_orchestrator._provider_coordinator.get_rate_limit_wait_time.return_value = 100.0
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


class StubStreamingPipeline:
    """Test double for StreamingChatPipeline."""

    def __init__(self):
        self.calls = []

    async def run(self, user_message: str):
        from victor.providers.base import StreamChunk

        self.calls.append(user_message)
        yield StreamChunk(content=f"resp:{user_message}")


class TestStreamingPipelineIntegration:
    """Tests for ChatCoordinator.stream_chat pipeline delegation."""

    @pytest.mark.asyncio
    async def test_stream_chat_uses_pipeline(self, chat_coordinator, monkeypatch):
        """ChatCoordinator should delegate streaming to StreamingChatPipeline."""
        pipeline = StubStreamingPipeline()
        created = []

        def fake_factory(coordinator, **kwargs):
            created.append(coordinator)
            return pipeline

        monkeypatch.setattr(
            "victor.agent.streaming.create_streaming_chat_pipeline",
            fake_factory,
        )

        chunks = []
        async for chunk in chat_coordinator.stream_chat("hello world"):
            chunks.append(chunk.content)

        assert chunks == ["resp:hello world"]
        assert pipeline.calls == ["hello world"]
        assert created == [chat_coordinator]

    @pytest.mark.asyncio
    async def test_pipeline_is_cached(self, chat_coordinator, monkeypatch):
        """Pipeline is instantiated once and reused across turns."""
        pipeline = StubStreamingPipeline()
        factory_calls = 0

        def fake_factory(_, **kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return pipeline

        monkeypatch.setattr(
            "victor.agent.streaming.create_streaming_chat_pipeline",
            fake_factory,
        )

        async for _ in chat_coordinator.stream_chat("first"):
            pass
        async for _ in chat_coordinator.stream_chat("second"):
            pass

        assert factory_calls == 1
        assert pipeline.calls == ["first", "second"]
