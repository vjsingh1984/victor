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

"""Tests for continuation strategy."""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.continuation_strategy import ContinuationStrategy
from victor.storage.embeddings.intent_classifier import IntentType
from victor.storage.embeddings.question_classifier import QuestionType, QuestionClassificationResult


# =============================================================================
# DETECT MENTIONED TOOLS TESTS
# =============================================================================


class TestDetectMentionedTools:
    """Tests for detect_mentioned_tools static method."""

    def test_detects_call_pattern(self):
        """Test detects 'call tool' pattern."""
        text = "Let me call read to see the file"
        tools = ["read", "search", "write"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_detects_use_pattern(self):
        """Test detects 'use tool' pattern."""
        text = "I'll use search to find it"
        tools = ["read", "search", "write"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "search" in result

    def test_detects_execute_pattern(self):
        """Test detects 'execute tool' pattern."""
        text = "Let me execute grep"
        tools = ["grep", "read", "write"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "grep" in result

    def test_detects_parenthesis_pattern(self):
        """Test detects 'tool()' pattern."""
        text = "I'll call read() to get the content"
        tools = ["read", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_detects_the_tool_pattern(self):
        """Test detects 'the X tool' pattern."""
        text = "I'll use the read tool to check"
        tools = ["read", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_resolves_aliases(self):
        """Test resolves tool aliases to canonical names."""
        text = "Let me call read_file"
        tools = ["read_file"]
        aliases = {"read_file": "read"}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_no_duplicates(self):
        """Test no duplicate tool names returned."""
        text = "I'll call read and use read to check"
        tools = ["read"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert len(result) == 1
        assert "read" in result

    def test_returns_empty_for_no_matches(self):
        """Test returns empty list when no tools mentioned."""
        text = "Just a simple response without any tools"
        tools = ["read", "search", "write"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert result == []

    def test_case_insensitive(self):
        """Test detection is case insensitive."""
        text = "Let me CALL READ to check"
        tools = ["read"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_run_pattern(self):
        """Test detects 'run tool' pattern."""
        text = "Let me run bash to execute the command"
        tools = ["bash", "read"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "bash" in result

    def test_invoke_pattern(self):
        """Test detects 'invoke tool' pattern."""
        text = "I need to invoke search"
        tools = ["search", "read"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "search" in result

    def test_perform_pattern(self):
        """Test detects 'perform tool' pattern."""
        text = "Let me perform write to save"
        tools = ["write", "read"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "write" in result

    def test_multiple_tools(self):
        """Test detects multiple tools mentioned."""
        text = "I'll call read and then use write"
        tools = ["read", "write", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result
        assert "write" in result


# =============================================================================
# CONTINUATION STRATEGY INITIALIZATION
# =============================================================================


class TestContinuationStrategyInit:
    """Tests for ContinuationStrategy initialization."""

    def test_init_with_event_bus(self):
        """Test initialization with custom event bus."""
        mock_bus = MagicMock()
        strategy = ContinuationStrategy(event_bus=mock_bus)

        assert strategy._event_bus is mock_bus

    def test_init_without_event_bus(self):
        """Test initialization without event bus uses singleton."""
        strategy = ContinuationStrategy()

        assert strategy._event_bus is not None


# =============================================================================
# DETERMINE CONTINUATION ACTION TESTS
# =============================================================================


class TestDetermineContinuationAction:
    """Tests for determine_continuation_action method."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy with mocked event bus."""
        mock_bus = MagicMock()
        return ContinuationStrategy(event_bus=mock_bus)

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.max_continuation_prompts_analysis = 6
        settings.max_continuation_prompts_action = 5
        settings.max_continuation_prompts_default = 3
        settings.continuation_prompt_overrides = {}
        return settings

    @pytest.fixture
    def base_kwargs(self, mock_settings):
        """Create base kwargs for determine_continuation_action."""
        return {
            "is_analysis_task": False,
            "is_action_task": False,
            "content_length": 100,
            "full_content": "Some content",
            "continuation_prompts": 0,
            "asking_input_prompts": 0,
            "one_shot_mode": False,
            "mentioned_tools": None,
            "max_prompts_summary_requested": False,
            "settings": mock_settings,
            "rl_coordinator": None,
            "provider_name": "openai",
            "model": "gpt-4",
            "tool_budget": 50,
            "unified_tracker_config": {"max_total_iterations": 50},
        }

    def test_finish_when_summary_already_requested(self, strategy, base_kwargs):
        """Test finishes when summary was already requested."""
        base_kwargs["max_prompts_summary_requested"] = True

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "finish"
        assert "already requested" in result["reason"]

    def test_request_summary_on_stuck_loop(self, strategy, base_kwargs):
        """Test requests summary on stuck loop intent."""
        mock_intent = MagicMock()
        mock_intent.intent = IntentType.STUCK_LOOP

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "request_summary"
        assert "STUCK_LOOP" in result["reason"]

    def test_force_tool_execution_on_hallucinated_calls(self, strategy, base_kwargs):
        """Test forces tool execution when tools mentioned but not called."""
        base_kwargs["mentioned_tools"] = ["read", "search"]

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "force_tool_execution"
        assert "Hallucinated" in result["reason"]

    def test_return_to_user_for_asking_input_one_shot(self, strategy, base_kwargs):
        """Test returns to user when asking input in one-shot mode."""
        base_kwargs["one_shot_mode"] = True

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.ASKING_INPUT

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "return_to_user"

    def test_continue_asking_input_auto_respond(self, strategy, base_kwargs):
        """Test auto-responds to asking input intent when question is rhetorical."""
        mock_intent = MagicMock()
        mock_intent.intent = IntentType.ASKING_INPUT

        # Mock classify_question to return a rhetorical question that should auto-continue
        # Note: should_auto_continue is a property computed from question_type and confidence
        mock_result = QuestionClassificationResult(
            question_type=QuestionType.RHETORICAL,
            confidence=0.9,  # >= 0.6 so should_auto_continue will be True
            matched_pattern="Should I proceed?",
        )
        with patch(
            "victor.agent.continuation_strategy.classify_question", return_value=mock_result
        ):
            result = strategy.determine_continuation_action(
                intent_result=mock_intent, **base_kwargs
            )

        assert result["action"] == "continue_asking_input"
        assert "asking_input_prompts" in result["updates"]

    def test_return_to_user_max_asking_input(self, strategy, base_kwargs):
        """Test returns to user when max asking input reached."""
        base_kwargs["asking_input_prompts"] = 3

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.ASKING_INPUT

        # Mock classify_question to return a rhetorical question (which would normally auto-continue)
        # But since we hit max prompts, it should return to user instead
        # Note: should_auto_continue is a property computed from question_type and confidence
        mock_result = QuestionClassificationResult(
            question_type=QuestionType.RHETORICAL,
            confidence=0.9,  # >= 0.6 so should_auto_continue will be True
            matched_pattern="Should I proceed?",
        )
        with patch(
            "victor.agent.continuation_strategy.classify_question", return_value=mock_result
        ):
            result = strategy.determine_continuation_action(
                intent_result=mock_intent, **base_kwargs
            )

        assert result["action"] == "return_to_user"

    def test_finish_on_completion_intent(self, strategy, base_kwargs):
        """Test finishes on completion intent."""
        mock_intent = MagicMock()
        mock_intent.intent = IntentType.COMPLETION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "finish"
        assert "completion" in result["reason"].lower()

    def test_prompt_tool_call_for_analysis_task(self, strategy, base_kwargs):
        """Test prompts for tool calls on analysis task."""
        base_kwargs["is_analysis_task"] = True

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "prompt_tool_call"
        assert "analysis" in result["message"].lower()

    def test_prompt_tool_call_for_action_task(self, strategy, base_kwargs):
        """Test prompts for tool calls on action task."""
        base_kwargs["is_action_task"] = True

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "prompt_tool_call"
        assert "implementation" in result["message"].lower()

    def test_request_summary_max_continuation_prompts(self, strategy, base_kwargs):
        """Test requests summary when max continuation prompts reached."""
        base_kwargs["is_analysis_task"] = True
        base_kwargs["continuation_prompts"] = 6

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "request_summary"
        assert result["updates"].get("max_prompts_summary_requested") is True

    def test_finish_default(self, strategy, base_kwargs):
        """Test finishes by default when no conditions met."""
        mock_intent = MagicMock()
        # Use a neutral intent that doesn't match specific handlers
        mock_intent.intent = IntentType.NEUTRAL

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] == "finish"


class TestRLCoordinator:
    """Tests for RL coordinator integration."""

    @pytest.fixture
    def strategy(self):
        mock_bus = MagicMock()
        return ContinuationStrategy(event_bus=mock_bus)

    def test_uses_rl_recommendations(self, strategy):
        """Test uses RL coordinator recommendations."""
        mock_settings = MagicMock()
        mock_settings.max_continuation_prompts_analysis = 6
        mock_settings.max_continuation_prompts_action = 5
        mock_settings.max_continuation_prompts_default = 3
        mock_settings.continuation_prompt_overrides = {}

        mock_rl = MagicMock()
        mock_recommendation = MagicMock()
        mock_recommendation.value = 10
        mock_recommendation.confidence = 0.9
        mock_rl.get_recommendation.return_value = mock_recommendation

        base_kwargs = {
            "is_analysis_task": True,
            "is_action_task": False,
            "content_length": 100,
            "full_content": "Content",
            "continuation_prompts": 0,
            "asking_input_prompts": 0,
            "one_shot_mode": False,
            "mentioned_tools": None,
            "max_prompts_summary_requested": False,
            "settings": mock_settings,
            "rl_coordinator": mock_rl,
            "provider_name": "openai",
            "model": "gpt-4",
            "tool_budget": 50,
            "unified_tracker_config": {"max_total_iterations": 50},
        }

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        # RL coordinator should have been called
        mock_rl.get_recommendation.assert_called()
