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

"""Unit tests for ContinuationStrategy.

Tests continuation decision logic extracted from orchestrator.
"""

import pytest
from unittest.mock import Mock

from victor.agent.continuation_strategy import ContinuationStrategy


class TestDetectMentionedTools:
    """Tests for detect_mentioned_tools method."""

    def test_detect_tool_with_call_pattern(self):
        """Test detecting tool mentioned with 'call' pattern."""
        text = "Let me call read_file to check the content."
        all_tools = ["read_file", "write_file", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, all_tools, aliases)

        assert "read_file" in result

    def test_detect_tool_with_function_call_pattern(self):
        """Test detecting tool mentioned with function call pattern."""
        text = "I'll use search() to find the files."
        all_tools = ["read_file", "write_file", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, all_tools, aliases)

        assert "search" in result

    def test_detect_multiple_tools(self):
        """Test detecting multiple tools mentioned."""
        text = "I'll call read_file first, then use write_file() to save changes."
        all_tools = ["read_file", "write_file", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, all_tools, aliases)

        assert "read_file" in result
        assert "write_file" in result
        assert len(result) == 2

    def test_no_tools_mentioned(self):
        """Test when no tools are mentioned."""
        text = "Here is my analysis of the codebase structure."
        all_tools = ["read_file", "write_file", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, all_tools, aliases)

        assert len(result) == 0

    def test_resolve_tool_alias(self):
        """Test resolving tool aliases to canonical names."""
        text = "Let me call ls to list files."
        all_tools = ["ls", "list_directory"]
        aliases = {"ls": "list_directory"}

        result = ContinuationStrategy.detect_mentioned_tools(text, all_tools, aliases)

        assert "list_directory" in result
        assert "ls" not in result


class TestDetermineContinuationAction:
    """Tests for determine_continuation_action method."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.max_continuation_prompts_analysis = 6
        settings.max_continuation_prompts_action = 5
        settings.max_continuation_prompts_default = 3
        settings.continuation_prompt_overrides = {}
        return settings

    @pytest.fixture
    def mock_intent_result(self):
        """Create mock intent result."""
        from victor.embeddings.intent_classifier import IntentType
        result = Mock()
        result.intent = IntentType.CONTINUATION
        result.confidence = 0.8
        return result

    @pytest.fixture
    def tracker_config(self):
        """Create mock tracker config."""
        return {"max_total_iterations": 50}

    def test_finish_when_summary_already_requested(self, mock_settings, mock_intent_result, tracker_config):
        """Test finishing when summary was already requested."""
        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=mock_intent_result,
            is_analysis_task=False,
            is_action_task=False,
            content_length=100,
            full_content="Some content",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=True,  # Already requested
            settings=mock_settings,
            rl_coordinator=None,
            provider_name="test",
            model="test-model",
            tool_budget=15,
            unified_tracker_config=tracker_config,
        )

        assert result["action"] == "finish"
        assert result["reason"] == "Summary already requested - final response received"

    def test_handle_stuck_loop_intent(self, mock_settings, tracker_config):
        """Test handling STUCK_LOOP intent."""
        from victor.embeddings.intent_classifier import IntentType

        stuck_intent = Mock()
        stuck_intent.intent = IntentType.STUCK_LOOP

        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=stuck_intent,
            is_analysis_task=False,
            is_action_task=False,
            content_length=100,
            full_content="I will call read_file... I plan to use search...",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=mock_settings,
            rl_coordinator=None,
            provider_name="test",
            model="test-model",
            tool_budget=15,
            unified_tracker_config=tracker_config,
        )

        assert result["action"] == "request_summary"
        assert "stuck in a planning loop" in result["message"]

    def test_force_tool_execution_for_hallucinated_calls(self, mock_settings, mock_intent_result, tracker_config):
        """Test forcing tool execution when tools are mentioned but not called."""
        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=mock_intent_result,
            is_analysis_task=False,
            is_action_task=False,
            content_length=100,
            full_content="Let me call read_file",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=["read_file"],  # Tool mentioned but not called
            max_prompts_summary_requested=False,
            settings=mock_settings,
            rl_coordinator=None,
            provider_name="test",
            model="test-model",
            tool_budget=15,
            unified_tracker_config=tracker_config,
        )

        assert result["action"] == "force_tool_execution"
        assert "didn't actually make the tool call" in result["message"]

    def test_completion_intent_finishes(self, mock_settings, tracker_config):
        """Test that COMPLETION intent causes finish."""
        from victor.embeddings.intent_classifier import IntentType

        completion_intent = Mock()
        completion_intent.intent = IntentType.COMPLETION

        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=completion_intent,
            is_analysis_task=False,
            is_action_task=False,
            content_length=100,
            full_content="Task completed",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=mock_settings,
            rl_coordinator=None,
            provider_name="test",
            model="test-model",
            tool_budget=15,
            unified_tracker_config=tracker_config,
        )

        assert result["action"] == "finish"
        assert result["reason"] == "Model indicated task completion"

    def test_prompt_tool_call_for_analysis_task(self, mock_settings, mock_intent_result, tracker_config):
        """Test prompting for tool calls in analysis tasks."""
        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=mock_intent_result,
            is_analysis_task=True,  # Analysis task
            is_action_task=False,
            content_length=100,
            full_content="Here's what I found",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=mock_settings,
            rl_coordinator=None,
            provider_name="test",
            model="test-model",
            tool_budget=15,
            unified_tracker_config=tracker_config,
        )

        assert result["action"] == "prompt_tool_call"
        assert "read_file" in result["message"] or "analysis" in result["message"].lower()
        assert result["updates"]["continuation_prompts"] == 1
