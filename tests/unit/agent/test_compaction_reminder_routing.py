"""Tests for compaction reminder routing through ContextReminderManager."""

from unittest.mock import MagicMock

import pytest

from victor.agent.conversation_controller import ConversationController
from victor.providers.base import Message


@pytest.fixture
def mock_reminder_manager():
    manager = MagicMock()
    manager.state = MagicMock()
    manager.state.compaction_summary = ""
    return manager


@pytest.fixture
def controller_with_manager(mock_reminder_manager):
    controller = ConversationController(
        context_reminder_manager=mock_reminder_manager
    )
    controller.set_system_prompt("System prompt")
    controller.add_message("user", "Hello")
    return controller


@pytest.fixture
def controller_without_manager():
    controller = ConversationController()
    controller.set_system_prompt("System prompt")
    controller.add_message("user", "Hello")
    return controller


class TestCompactionReminderRouting:
    def test_compaction_routes_through_reminder_manager(
        self, controller_with_manager, mock_reminder_manager
    ):
        controller_with_manager._compaction_summaries = ["Summary 1", "Summary 2"]
        result = controller_with_manager.inject_compaction_context()

        assert result is True
        # Should have updated the reminder manager's state
        assert mock_reminder_manager.state.compaction_summary != ""
        # Should NOT have inserted a direct message (reminder manager handles it)
        msgs = controller_with_manager.messages
        assistant_reminders = [
            m for m in msgs if m.role == "assistant" and "Context reminder" in m.content
        ]
        assert len(assistant_reminders) == 0

    def test_compaction_fallback_without_manager(self, controller_without_manager):
        controller_without_manager._compaction_summaries = ["Summary A"]
        result = controller_without_manager.inject_compaction_context()

        assert result is True
        # Should have inserted a direct message
        msgs = controller_without_manager.messages
        reminders = [
            m for m in msgs if m.role == "assistant" and "Context reminder" in m.content
        ]
        assert len(reminders) == 1

    def test_compaction_summary_updates_state(
        self, controller_with_manager, mock_reminder_manager
    ):
        controller_with_manager._compaction_summaries = ["S1", "S2", "S3"]
        controller_with_manager.inject_compaction_context()

        summary = mock_reminder_manager.state.compaction_summary
        assert "S1" in summary or "S2" in summary or "S3" in summary

    def test_no_summaries_returns_false(self, controller_with_manager):
        controller_with_manager._compaction_summaries = []
        result = controller_with_manager.inject_compaction_context()
        assert result is False

    def test_reminder_manager_none_uses_direct_injection(self, controller_without_manager):
        controller_without_manager._compaction_summaries = ["Test summary"]
        result = controller_without_manager.inject_compaction_context()

        assert result is True
        msgs = controller_without_manager.messages
        # Direct injection: assistant message with [Context reminder: ...]
        assert any("Context reminder" in m.content for m in msgs if m.role == "assistant")
