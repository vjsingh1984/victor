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
    """Tests for _extract_required_files_from_prompt."""

    def test_extract_absolute_paths(self, chat_coordinator):
        """Extract absolute file paths from message."""
        result = chat_coordinator._extract_required_files_from_prompt(
            "Please fix /home/user/project/main.py and /home/user/project/utils.py"
        )
        assert set(result) == {"/home/user/project/main.py", "/home/user/project/utils.py"}

    def test_extract_relative_paths(self, chat_coordinator):
        """Extract relative file paths from message."""
        result = chat_coordinator._extract_required_files_from_prompt(
            "Check ./src/main.py and ../utils/helper.py"
        )
        assert set(result) == {"./src/main.py", "../utils/helper.py"}

    def test_extract_mixed_paths(self, chat_coordinator):
        """Extract mixed path types."""
        result = chat_coordinator._extract_required_files_from_prompt(
            "Review /absolute/path.py and ./relative/path.py"
        )
        assert set(result) == {"/absolute/path.py", "./relative/path.py"}

    def test_deduplicate_paths(self, chat_coordinator):
        """Remove duplicate paths."""
        result = chat_coordinator._extract_required_files_from_prompt(
            "Fix ./file.py and also fix ./file.py and /abs/file.py"
        )
        assert len(result) == 2  # Deduplicated: ./file.py and /abs/file.py

    def test_no_paths_returns_empty(self, chat_coordinator):
        """Return empty list when no paths found."""
        result = chat_coordinator._extract_required_files_from_prompt(
            "Hello, how are you today?"
        )
        assert result == []


class TestExtractRequiredOutputsFromPrompt:
    """Tests for _extract_required_outputs_from_prompt."""

    def test_returns_empty_list(self, chat_coordinator):
        """Always returns empty list as outputs are advisory."""
        result = chat_coordinator._extract_required_outputs_from_prompt(
            "Create output.txt and results.json"
        )
        assert result == []


class TestGetMaxContextChars:
    """Tests for _get_max_context_chars."""

    def test_delegates_to_orchestrator(self, chat_coordinator, mock_orchestrator):
        """Delegates to orchestrator's _get_max_context_chars."""
        mock_orchestrator._get_max_context_chars.return_value = 150000
        result = chat_coordinator._get_max_context_chars()
        assert result == 150000
        mock_orchestrator._get_max_context_chars.assert_called_once()


class TestClassifyTaskKeywords:
    """Tests for _classify_task_keywords."""

    def test_delegates_to_task_analyzer(self, chat_coordinator, mock_orchestrator):
        """Delegates to orchestrator's task analyzer."""
        mock_orchestrator._task_analyzer.classify_task_keywords.return_value = {
            "is_analysis_task": True,
            "is_action_task": False,
            "needs_execution": False,
            "coarse_task_type": "analysis",
        }
        result = chat_coordinator._classify_task_keywords("Explain this code")
        assert result["is_analysis_task"] is True
        assert result["coarse_task_type"] == "analysis"
        mock_orchestrator._task_analyzer.classify_task_keywords.assert_called_once_with(
            "Explain this code"
        )


class TestApplyIntentGuard:
    """Tests for _apply_intent_guard."""

    def test_applies_intent_guard(self, chat_coordinator, mock_orchestrator):
        """Apply intent guard through task coordinator."""
        chat_coordinator._apply_intent_guard("What files exist?")
        mock_orchestrator.task_coordinator.apply_intent_guard.assert_called_once()
        assert mock_orchestrator._current_intent == mock_orchestrator.task_coordinator.current_intent


class TestApplyTaskGuidance:
    """Tests for _apply_task_guidance."""

    def test_applies_task_guidance(self, chat_coordinator, mock_orchestrator):
        """Apply task guidance through task coordinator."""
        from victor.agent.unified_task_tracker import TrackerTaskType

        chat_coordinator._apply_task_guidance(
            user_message="Fix the bug",
            unified_task_type=TrackerTaskType.EDIT,
            is_analysis_task=False,
            is_action_task=True,
            needs_execution=True,
            max_exploration_iterations=3,
        )

        # Verify temperature and tool_budget are set
        assert mock_orchestrator.task_coordinator.temperature == 0.7
        assert mock_orchestrator.task_coordinator.tool_budget == 15
        mock_orchestrator.task_coordinator.apply_task_guidance.assert_called_once()


class TestPrepareTask:
    """Tests for _prepare_task."""

    def test_prepare_task(self, chat_coordinator, mock_orchestrator):
        """Prepare task through task coordinator."""
        from victor.agent.unified_task_tracker import TrackerTaskType

        mock_orchestrator.task_coordinator.prepare_task.return_value = MagicMock(
            task_type="edit"
        )
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


class TestSelectToolsForTurn:
    """Tests for _select_tools_for_turn."""

    @pytest.mark.asyncio
    async def test_select_tools_with_intent_filtering(
        self, chat_coordinator, mock_orchestrator
    ):
        """Select tools with intent-based filtering."""
        mock_orchestrator._tool_planner.filter_tools_by_intent.return_value = [
            MagicMock(name="read_file"),
            MagicMock(name="write_file"),
        ]

        result = await chat_coordinator._select_tools_for_turn(
            "Read the file", MagicMock(goals=["analysis"])
        )

        mock_orchestrator._tool_planner.filter_tools_by_intent.assert_called()
