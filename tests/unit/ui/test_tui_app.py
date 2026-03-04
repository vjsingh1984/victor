"""Unit tests for Victor TUI app behavior."""

from unittest.mock import MagicMock

from textual.messages import UpdateScroll

from victor.ui.tui.app import VictorTUI


def test_ctrl_f_binding_maps_to_toggle_follow_mode() -> None:
    """Ctrl+F should stay wired to follow-mode toggle."""
    assert any(
        binding.key == "ctrl+f" and binding.action == "toggle_follow_mode"
        for binding in VictorTUI.BINDINGS
    )


def test_update_jump_button_label_with_unread_count() -> None:
    """Jump button should show unread count when auto-follow is disabled."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = False
    app._conversation_log.follow_paused = False
    app._conversation_log.unread_count = 3
    app._jump_button = MagicMock()
    app._jump_button.label = "Jump to bottom"

    app._update_jump_to_bottom()

    app._jump_button.add_class.assert_called_once_with("visible")
    assert app._jump_button.label == "Jump to bottom (3 new)"


def test_update_jump_button_hides_when_at_bottom() -> None:
    """Jump button should hide and reset label when auto-follow is enabled."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = True
    app._conversation_log.follow_paused = False
    app._conversation_log.unread_count = 0
    app._jump_button = MagicMock()
    app._jump_button.label = "Jump to bottom (4 new)"

    app._update_jump_to_bottom()

    app._jump_button.remove_class.assert_called_once_with("visible")
    assert app._jump_button.label == "Jump to bottom"


def test_action_page_up_disables_auto_scroll_and_pages() -> None:
    """Page-up action should disable auto-follow and page up the conversation."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._update_jump_to_bottom = MagicMock()

    app.action_page_up()

    app._conversation_log.disable_auto_scroll.assert_called_once()
    app._conversation_log.scroll_page_up.assert_called_once_with(animate=False)
    app._conversation_log.update_auto_scroll_state.assert_called_once()
    app._update_jump_to_bottom.assert_called_once()


def test_action_page_down_pages_and_updates_state() -> None:
    """Page-down action should page down and refresh auto-follow state."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._update_jump_to_bottom = MagicMock()

    app.action_page_down()

    app._conversation_log.scroll_page_down.assert_called_once_with(animate=False)
    app._conversation_log.update_auto_scroll_state.assert_called_once()
    app._update_jump_to_bottom.assert_called_once()


def test_on_update_scroll_updates_jump_button_for_conversation_log() -> None:
    """Scroll updates from the conversation log should refresh jump button state."""
    app = VictorTUI()
    conversation_log = MagicMock()
    app._conversation_log = conversation_log
    app._update_jump_to_bottom = MagicMock()

    event = UpdateScroll().set_sender(conversation_log)
    app.on_update_scroll(event)

    app._update_jump_to_bottom.assert_called_once()


def test_on_update_scroll_ignores_other_senders() -> None:
    """Scroll updates from other widgets should not affect jump button state."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._update_jump_to_bottom = MagicMock()
    other_widget = MagicMock()

    event = UpdateScroll().set_sender(other_widget)
    app.on_update_scroll(event)

    app._update_jump_to_bottom.assert_not_called()


def test_action_toggle_unread_marker_flips_state_and_announces() -> None:
    """Unread marker toggle should update log config and emit status message."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.unread_separator_enabled = True
    app._add_system_message = MagicMock()
    app._update_jump_to_bottom = MagicMock()

    app.action_toggle_unread_marker()

    app._conversation_log.set_unread_separator_enabled.assert_called_once_with(False)
    app._add_system_message.assert_called_once_with("Unread marker hidden")
    app._update_jump_to_bottom.assert_called_once()


def test_action_jump_unread_uses_separator_when_present() -> None:
    """Ctrl+N should jump to unread marker when available."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.jump_to_unread_separator.return_value = True
    app._update_jump_to_bottom = MagicMock()

    app.action_jump_unread()

    app._conversation_log.jump_to_unread_separator.assert_called_once()
    app._conversation_log.disable_auto_scroll.assert_called_once()
    app._conversation_log.update_auto_scroll_state.assert_called_once()
    app._conversation_log.scroll_to_bottom.assert_not_called()
    app._update_jump_to_bottom.assert_called_once()


def test_action_jump_unread_falls_back_to_bottom_when_no_marker() -> None:
    """Ctrl+N should jump to bottom when no unread marker exists."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.jump_to_unread_separator.return_value = False
    app._conversation_log.follow_paused = False
    app._update_jump_to_bottom = MagicMock()

    app.action_jump_unread()

    app._conversation_log.scroll_to_bottom.assert_called_once_with(animate=False)
    app._conversation_log.set_follow_paused.assert_not_called()
    app._update_jump_to_bottom.assert_called_once()


def test_action_jump_unread_resumes_follow_when_paused_and_no_marker() -> None:
    """Ctrl+N should resume follow if paused and no unread marker is available."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.jump_to_unread_separator.return_value = False
    app._conversation_log.follow_paused = True
    app._update_jump_to_bottom = MagicMock()

    app.action_jump_unread()

    app._conversation_log.set_follow_paused.assert_called_once_with(False, jump_to_bottom=True)
    app._conversation_log.scroll_to_bottom.assert_not_called()
    app._update_jump_to_bottom.assert_called_once()


def test_update_jump_button_shows_resume_follow_when_paused() -> None:
    """Paused follow mode should show a resume label."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = False
    app._conversation_log.follow_paused = True
    app._conversation_log.unread_count = 2
    app._jump_button = MagicMock()
    app._jump_button.label = "Jump to bottom"
    app._status_bar = MagicMock()

    app._update_jump_to_bottom()

    app._jump_button.add_class.assert_called_once_with("visible")
    assert app._jump_button.label == "Resume follow (2 new)"
    app._status_bar.update_follow.assert_called_once_with(True)
    app._status_bar.update_unread.assert_called_once_with(2)


def test_update_jump_button_marks_paused_when_user_scrolled() -> None:
    """Indicator should show paused when follow is off due manual scrolling."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = False
    app._conversation_log.follow_paused = False
    app._conversation_log.unread_count = 0
    app._jump_button = MagicMock()
    app._jump_button.label = "Jump to bottom"
    app._status_bar = MagicMock()

    app._update_jump_to_bottom()

    app._status_bar.update_follow.assert_called_once_with(True)
    app._status_bar.update_unread.assert_called_once_with(0)
    assert app._jump_button.label == "Jump to bottom"


def test_update_jump_button_marks_following_when_auto_follow_active() -> None:
    """Indicator should show following when auto-follow is active."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = True
    app._conversation_log.follow_paused = False
    app._conversation_log.unread_count = 0
    app._jump_button = MagicMock()
    app._jump_button.label = "Jump to bottom"
    app._status_bar = MagicMock()

    app._update_jump_to_bottom()

    app._status_bar.update_follow.assert_called_once_with(False)
    app._status_bar.update_unread.assert_called_once_with(0)


def test_action_toggle_follow_mode_pauses_follow() -> None:
    """Ctrl+F action should pause sticky follow mode."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.follow_paused = False
    app._update_jump_to_bottom = MagicMock()

    app.action_toggle_follow_mode()

    app._conversation_log.set_follow_paused.assert_called_once_with(True, jump_to_bottom=False)
    app._update_jump_to_bottom.assert_called_once()


def test_action_toggle_follow_mode_resumes_follow() -> None:
    """Ctrl+F action should resume follow and jump to latest."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.follow_paused = True
    app._update_jump_to_bottom = MagicMock()

    app.action_toggle_follow_mode()

    app._conversation_log.set_follow_paused.assert_called_once_with(False, jump_to_bottom=True)
    app._update_jump_to_bottom.assert_called_once()


def test_jump_button_resumes_follow_when_paused() -> None:
    """Jump button should resume follow mode when follow is paused."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.follow_paused = True
    app._update_jump_to_bottom = MagicMock()
    button = MagicMock()
    button.id = "jump-to-bottom"
    event = MagicMock()
    event.button = button

    app.on_button_pressed(event)

    app._conversation_log.set_follow_paused.assert_called_once_with(False, jump_to_bottom=True)
    app._conversation_log.scroll_to_bottom.assert_not_called()
    app._update_jump_to_bottom.assert_called_once()
