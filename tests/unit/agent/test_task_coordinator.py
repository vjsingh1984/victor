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

"""Unit tests for TaskCoordinator.

Tests task preparation, intent detection, and task-specific guidance.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Any

from victor.agent.task_coordinator import TaskCoordinator
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.temperature = 0.7
    settings.tool_budget = 15
    return settings


@pytest.fixture
def mock_task_analyzer():
    """Create mock TaskAnalyzer."""
    analyzer = MagicMock()

    # Configure complexity classification
    complexity_result = Mock()
    complexity_result.complexity = Mock(value="moderate")
    complexity_result.confidence = 0.8
    complexity_result.prompt_hint = "Test hint"
    analyzer.classify_complexity.return_value = complexity_result

    # Configure intent detection
    intent_result = Mock()
    intent_result.intent = Mock(value="write_allowed")
    intent_result.prompt_guard = None
    analyzer.detect_intent.return_value = intent_result

    return analyzer


@pytest.fixture
def mock_unified_tracker():
    """Create mock UnifiedTaskTracker."""
    tracker = MagicMock()
    tracker.config = {"max_total_iterations": 50}
    tracker.set_tool_budget = Mock()
    return tracker


@pytest.fixture
def mock_conversation_controller():
    """Create mock ConversationController."""
    controller = MagicMock()
    controller.add_message = Mock()
    return controller


@pytest.fixture
def mock_prompt_builder():
    """Create mock PromptBuilder."""
    builder = MagicMock()
    builder.prompt_contributors = []
    return builder


@pytest.fixture
def task_coordinator(
    mock_task_analyzer,
    mock_unified_tracker,
    mock_prompt_builder,
    mock_settings,
):
    """Create TaskCoordinator with mocked dependencies."""
    return TaskCoordinator(
        task_analyzer=mock_task_analyzer,
        unified_tracker=mock_unified_tracker,
        prompt_builder=mock_prompt_builder,
        settings=mock_settings,
    )


class TestTaskPreparation:
    """Tests for task preparation operations."""

    def test_prepare_task_basic(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test basic task preparation."""
        unified_type = Mock(value="edit")
        message = "Fix the bug in test.py"

        classification, budget = task_coordinator.prepare_task(
            message, unified_type, mock_conversation_controller
        )

        # Verify task analyzer was called
        mock_task_analyzer.classify_complexity.assert_called_once_with(message)

        # Verify classification result returned
        assert classification is not None
        assert isinstance(budget, int)

    def test_prepare_task_with_task_hint(self, task_coordinator, mock_conversation_controller):
        """Test task preparation with task hint injection."""
        from unittest.mock import patch

        unified_type = Mock(value="edit")
        message = "Refactor the code"

        with patch(
            "victor.agent.prompt_builder.get_task_type_hint",
            return_value="Test task hint",
        ):
            task_coordinator.prepare_task(message, unified_type, mock_conversation_controller)

            # Verify hint was injected
            mock_conversation_controller.add_message.assert_called()
            calls = [
                call
                for call in mock_conversation_controller.add_message.call_args_list
                if len(call[0]) >= 2 and call[0][0] == "system"
            ]
            assert len(calls) > 0


class TestIntentDetection:
    """Tests for intent detection operations."""

    def test_apply_intent_guard_write_allowed(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test intent guard with WRITE_ALLOWED intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Configure mock for WRITE_ALLOWED
        intent_result = Mock()
        intent_result.intent = ActionIntent.WRITE_ALLOWED
        intent_result.prompt_guard = None
        mock_task_analyzer.detect_intent.return_value = intent_result

        message = "Create a new feature"
        task_coordinator.apply_intent_guard(message, mock_conversation_controller)

        # Verify intent detection was called
        mock_task_analyzer.detect_intent.assert_called_once_with(message)

        # Verify current intent was set
        assert task_coordinator.current_intent == ActionIntent.WRITE_ALLOWED

        # No prompt guard should be added for WRITE_ALLOWED
        # (can't assert this directly without checking call args)

    def test_apply_intent_guard_display_only(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test intent guard with DISPLAY_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Configure mock for DISPLAY_ONLY
        intent_result = Mock()
        intent_result.intent = ActionIntent.DISPLAY_ONLY
        intent_result.prompt_guard = "READ ONLY MODE: Do not modify files."
        mock_task_analyzer.detect_intent.return_value = intent_result

        message = "Show me the code"
        task_coordinator.apply_intent_guard(message, mock_conversation_controller)

        # Verify prompt guard was injected
        mock_conversation_controller.add_message.assert_called_with(
            "system", "READ ONLY MODE: Do not modify files."
        )

        # Verify current intent was set
        assert task_coordinator.current_intent == ActionIntent.DISPLAY_ONLY

    def test_apply_intent_guard_read_only(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test intent guard with READ_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Configure mock for READ_ONLY
        intent_result = Mock()
        intent_result.intent = ActionIntent.READ_ONLY
        intent_result.prompt_guard = "This is read-only analysis."
        mock_task_analyzer.detect_intent.return_value = intent_result

        message = "Analyze the codebase"
        task_coordinator.apply_intent_guard(message, mock_conversation_controller)

        # Verify prompt guard was injected
        mock_conversation_controller.add_message.assert_called_with(
            "system", "This is read-only analysis."
        )


class TestTaskGuidance:
    """Tests for task-specific guidance."""

    def test_apply_task_guidance_analysis_task(
        self, task_coordinator, mock_conversation_controller
    ):
        """Test task guidance for analysis tasks."""
        unified_type = Mock(value="analyze")
        message = "Analyze the project structure"

        # Set initial temperature
        task_coordinator.temperature = 0.7

        task_coordinator.apply_task_guidance(
            user_message=message,
            unified_task_type=unified_type,
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=50,
            conversation_controller=mock_conversation_controller,
        )

        # Verify temperature was increased for analysis
        assert task_coordinator.temperature > 0.7

        # Verify system messages were added
        assert mock_conversation_controller.add_message.call_count >= 2

    def test_apply_task_guidance_action_task(self, task_coordinator, mock_conversation_controller):
        """Test task guidance for action tasks."""
        unified_type = Mock(value="create")
        message = "Create a new script"

        task_coordinator.apply_task_guidance(
            user_message=message,
            unified_task_type=unified_type,
            is_analysis_task=False,
            is_action_task=True,
            needs_execution=True,
            max_exploration_iterations=10,
            conversation_controller=mock_conversation_controller,
        )

        # Verify system message was added for action task
        mock_conversation_controller.add_message.assert_called()
        calls = mock_conversation_controller.add_message.call_args_list
        assert any("action-oriented" in str(call).lower() for call in calls)

    def test_apply_task_guidance_analysis_tool_budget(
        self, task_coordinator, mock_conversation_controller
    ):
        """Test that analysis tasks increase tool budget."""
        unified_type = Mock(value="analyze")

        # Set initial tool budget
        task_coordinator.tool_budget = 15

        task_coordinator.apply_task_guidance(
            user_message="Analyze all files",
            unified_task_type=unified_type,
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=100,
            conversation_controller=mock_conversation_controller,
        )

        # Verify tool budget was increased for analysis
        assert task_coordinator.tool_budget >= 200


class TestTaskCoordinatorProperties:
    """Tests for TaskCoordinator properties."""

    def test_temperature_property(self, task_coordinator):
        """Test temperature property getter/setter."""
        task_coordinator.temperature = 0.9
        assert task_coordinator.temperature == 0.9

    def test_tool_budget_property(self, task_coordinator):
        """Test tool_budget property getter/setter."""
        task_coordinator.tool_budget = 25
        assert task_coordinator.tool_budget == 25

    def test_observed_files_property(self, task_coordinator):
        """Test observed_files property getter/setter."""
        files = ["test.py", "main.py"]
        task_coordinator.observed_files = files
        assert task_coordinator.observed_files == files

    def test_current_intent_property(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test current_intent property."""
        from victor.agent.action_authorizer import ActionIntent

        # Set intent via apply_intent_guard
        intent_result = Mock()
        intent_result.intent = ActionIntent.READ_ONLY
        intent_result.prompt_guard = "Guard"
        mock_task_analyzer.detect_intent.return_value = intent_result

        task_coordinator.apply_intent_guard("test message", mock_conversation_controller)

        assert task_coordinator.current_intent == ActionIntent.READ_ONLY


class TestTaskCoordinatorInitialization:
    """Tests for TaskCoordinator initialization."""

    def test_initialization_with_valid_dependencies(
        self,
        mock_task_analyzer,
        mock_unified_tracker,
        mock_prompt_builder,
        mock_settings,
    ):
        """Test successful initialization with valid dependencies."""
        coordinator = TaskCoordinator(
            task_analyzer=mock_task_analyzer,
            unified_tracker=mock_unified_tracker,
            prompt_builder=mock_prompt_builder,
            settings=mock_settings,
        )

        assert coordinator.task_analyzer is mock_task_analyzer
        assert coordinator.unified_tracker is mock_unified_tracker
        assert coordinator.prompt_builder is mock_prompt_builder
        assert coordinator.settings is mock_settings

    def test_set_reminder_manager(self, task_coordinator):
        """Test setting reminder manager."""
        reminder_manager = Mock()
        task_coordinator.set_reminder_manager(reminder_manager)

        assert task_coordinator._reminder_manager is reminder_manager
