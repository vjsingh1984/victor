"""Unit tests for Victor TUI widget behavior."""

from unittest.mock import MagicMock, patch

from rich.markdown import Markdown
from textual.messages import UpdateScroll

from victor.ui.tui.widgets import EnhancedConversationLog, StatusBar, StreamingMessageBlock


def test_streaming_does_not_force_scroll_when_auto_scroll_disabled() -> None:
    """Streaming updates should not jump viewport when user scrolled away from bottom."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True),
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True) as scroll_end,
    ):
        log = EnhancedConversationLog()

        log.start_streaming()
        assert scroll_end.call_count == 1

        log.disable_auto_scroll()
        log.append_streaming_chunk("hello")
        log.update_streaming("hello world")
        log.finish_streaming()

        # No additional forced scrolls after auto-scroll is disabled.
        assert scroll_end.call_count == 1


def test_streaming_scrolls_when_auto_scroll_enabled() -> None:
    """Streaming should follow output with guaranteed start/end anchoring."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True),
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True) as scroll_end,
    ):
        log = EnhancedConversationLog()

        log.start_streaming()
        log.append_streaming_chunk("a")
        log.update_streaming("ab")
        log.finish_streaming()

        # Start and finish are always forced. Mid-stream follow updates are throttled.
        assert scroll_end.call_count >= 2
        assert scroll_end.call_count <= 4


def test_user_message_reenables_auto_scroll() -> None:
    """Submitting a user message should bring focus back to latest context."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True),
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True) as scroll_end,
    ):
        log = EnhancedConversationLog()
        log.disable_auto_scroll()

        log.add_user_message("check this")

        assert log.auto_scroll_enabled is True
        assert scroll_end.call_count == 1


def test_streaming_message_block_renders_text_then_markdown() -> None:
    """Assistant streaming should use fast text updates, then finalize as markdown."""
    block = StreamingMessageBlock(role="assistant", initial_content="# Heading")
    body = MagicMock()

    with patch.object(StreamingMessageBlock, "query_one", return_value=body):
        block.is_streaming = True
        block._update_display()
        streaming_render = body.update.call_args.args[0]
        assert isinstance(streaming_render, str)
        assert streaming_render == "# Heading"

        body.reset_mock()
        block.finish_streaming()
        final_render = body.update.call_args.args[0]
        assert isinstance(final_render, Markdown)


def test_streaming_message_block_flushes_buffer_on_finish() -> None:
    """Buffered chunks should flush when streaming completes."""
    block = StreamingMessageBlock(role="assistant", initial_content="")
    body = MagicMock()

    with (
        patch.object(StreamingMessageBlock, "query_one", return_value=body),
        patch("victor.ui.tui.widgets.time.monotonic", side_effect=[1.0, 1.01, 2.0]),
    ):
        block.is_streaming = True
        block.append_chunk("A")
        block.append_chunk("B")

        # Second chunk was buffered due throttle window.
        assert block.content == "A"

        block.finish_streaming()
        assert block.content == "AB"
        final_render = body.update.call_args.args[0]
        assert isinstance(final_render, Markdown)


def test_unread_count_tracks_new_messages_when_auto_scroll_disabled() -> None:
    """Unread count should increase while user is off-bottom and clear when returning."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True),
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True),
    ):
        log = EnhancedConversationLog()
        assert log.unread_count == 0

        log.disable_auto_scroll()
        log.add_assistant_message("new message")
        assert log.unread_count == 1

        # Streaming chunks should not inflate unread count after stream starts.
        log.start_streaming()
        assert log.unread_count == 2
        log.append_streaming_chunk("a")
        log.append_streaming_chunk("b")
        log.finish_streaming()
        assert log.unread_count == 2

        log.scroll_to_bottom()
        assert log.unread_count == 0
        assert log.auto_scroll_enabled is True


def test_update_scroll_event_updates_auto_scroll_state() -> None:
    """UpdateScroll should re-evaluate whether auto-follow stays enabled."""
    log = EnhancedConversationLog()
    event = UpdateScroll()

    with patch.object(log, "_is_at_bottom", side_effect=[False, True]):
        log.on_update_scroll(event)
        assert log.auto_scroll_enabled is False

        log.on_update_scroll(event)
        assert log.auto_scroll_enabled is True


def test_update_scroll_ignores_programmatic_guard_window() -> None:
    """Programmatic scroll updates should not immediately flip follow state."""
    log = EnhancedConversationLog()
    event = UpdateScroll()
    log._ignore_scroll_update_until = 10.0
    log._auto_scroll = True

    with (
        patch("victor.ui.tui.widgets.time.monotonic", return_value=9.95),
        patch.object(log, "_is_at_bottom", return_value=True),
        patch.object(log, "update_auto_scroll_state") as update_auto_scroll_state,
    ):
        log.on_update_scroll(event)

    update_auto_scroll_state.assert_not_called()


def test_update_scroll_processes_during_guard_when_user_scrolled_away() -> None:
    """Manual scroll-away should bypass guard and disable follow immediately."""
    log = EnhancedConversationLog()
    event = UpdateScroll()
    log._ignore_scroll_update_until = 10.0
    log._auto_scroll = True

    with (
        patch("victor.ui.tui.widgets.time.monotonic", return_value=9.95),
        patch.object(log, "_is_at_bottom", return_value=False),
        patch.object(log, "update_auto_scroll_state") as update_auto_scroll_state,
    ):
        log.on_update_scroll(event)

    update_auto_scroll_state.assert_called_once()


def test_update_scroll_processes_after_programmatic_guard_window() -> None:
    """Scroll updates should resume once guard window expires."""
    log = EnhancedConversationLog()
    event = UpdateScroll()
    log._ignore_scroll_update_until = 10.0

    with (
        patch("victor.ui.tui.widgets.time.monotonic", return_value=10.05),
        patch.object(log, "update_auto_scroll_state") as update_auto_scroll_state,
    ):
        log.on_update_scroll(event)

    update_auto_scroll_state.assert_called_once()


def test_unread_separator_inserted_once_before_first_unread_message() -> None:
    """Unread separator should be created once before the first unread message."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True) as mount,
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True),
    ):
        log = EnhancedConversationLog(show_unread_separator=True)
        log.disable_auto_scroll()

        log.add_assistant_message("first unread")
        log.add_assistant_message("second unread")

        assert log.unread_count == 2
        assert log._unread_separator is not None
        assert mount.call_count == 3
        separator_widget = mount.call_args_list[0].args[1]
        assert "unread-separator" in separator_widget.classes


def test_unread_separator_disabled_does_not_insert_marker() -> None:
    """Disabling unread separator should keep unread counting without marker widget."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True) as mount,
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True),
    ):
        log = EnhancedConversationLog(show_unread_separator=False)
        log.disable_auto_scroll()

        log.add_assistant_message("new message")

        assert log.unread_count == 1
        assert log._unread_separator is None
        assert mount.call_count == 1


def test_unread_separator_removed_when_reaching_bottom() -> None:
    """Unread separator should clear when user returns to bottom."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True),
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True),
    ):
        log = EnhancedConversationLog(show_unread_separator=True)
        log.disable_auto_scroll()
        log.add_assistant_message("new message")
        assert log._unread_separator is not None

        log.scroll_to_bottom()

        assert log._unread_separator is None
        assert log._unread_boundary_id is None
        assert log.unread_count == 0


def test_enabling_separator_with_existing_unread_inserts_marker() -> None:
    """Re-enabling marker should place it at the original unread boundary."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True) as mount,
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True),
    ):
        log = EnhancedConversationLog(show_unread_separator=False)
        log.disable_auto_scroll()
        log.add_assistant_message("new message")
        assert log.unread_count == 1
        assert log._unread_separator is None
        assert log._unread_boundary_id == "msg-0"

        boundary_target = MagicMock()
        with patch.object(log, "query_one", return_value=boundary_target):
            log.set_unread_separator_enabled(True)

        assert log._unread_separator is not None
        assert mount.call_count == 2
        separator_call = mount.call_args_list[1]
        assert separator_call.kwargs["before"] is boundary_target


def test_jump_to_unread_separator_scrolls_when_present() -> None:
    """Unread jump should scroll to separator when marker exists."""
    log = EnhancedConversationLog(show_unread_separator=True)
    fake_separator = MagicMock()
    log._unread_separator = fake_separator

    with patch.object(log, "scroll_to_widget", return_value=True) as scroll_to_widget:
        jumped = log.jump_to_unread_separator()

    assert jumped is True
    scroll_to_widget.assert_called_once()


def test_jump_to_unread_separator_uses_boundary_target_when_marker_hidden() -> None:
    """Unread jump should still work when marker is disabled but boundary is known."""
    log = EnhancedConversationLog(show_unread_separator=False)
    log._unread_boundary_id = "msg-4"
    boundary_target = MagicMock()

    with (
        patch.object(log, "query_one", return_value=boundary_target),
        patch.object(log, "scroll_to_widget", return_value=True) as scroll_to_widget,
    ):
        jumped = log.jump_to_unread_separator()

    assert jumped is True
    scroll_to_widget.assert_called_once()


def test_jump_to_unread_separator_returns_false_without_marker() -> None:
    """Unread jump should report false when no unread marker exists."""
    log = EnhancedConversationLog(show_unread_separator=True)
    log._unread_separator = None
    log._unread_boundary_id = None

    assert log.jump_to_unread_separator() is False


def test_follow_pause_is_sticky_across_messages() -> None:
    """Sticky follow pause should prevent auto-follow for new messages."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True),
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True) as scroll_end,
    ):
        log = EnhancedConversationLog()
        log.set_follow_paused(True)

        log.add_user_message("user message")
        log.add_assistant_message("assistant message")

        assert log.follow_paused is True
        assert log.auto_scroll_enabled is False
        assert log.unread_count == 1
        assert scroll_end.call_count == 0


def test_resuming_follow_clears_unread_and_jumps_to_bottom() -> None:
    """Resuming follow should clear unread state and jump once when requested."""
    with (
        patch.object(EnhancedConversationLog, "mount", autospec=True),
        patch.object(EnhancedConversationLog, "scroll_end", autospec=True) as scroll_end,
    ):
        log = EnhancedConversationLog()
        log.set_follow_paused(True)
        log.add_assistant_message("pending")
        assert log.unread_count == 1

        log.set_follow_paused(False, jump_to_bottom=True)

        assert log.follow_paused is False
        assert log.auto_scroll_enabled is True
        assert log.unread_count == 0
        assert scroll_end.call_count == 1


def test_update_auto_scroll_state_respects_sticky_pause() -> None:
    """Sticky pause should keep auto-follow off even at bottom."""
    log = EnhancedConversationLog()
    log.set_follow_paused(True)
    log._unread_count = 2

    with patch.object(log, "_is_at_bottom", return_value=True):
        log.update_auto_scroll_state()

    assert log.follow_paused is True
    assert log.auto_scroll_enabled is False
    assert log.unread_count == 0


def test_maybe_scroll_end_sets_programmatic_guard_when_following() -> None:
    """Programmatic follow scrolls should set a short guard against transient events."""
    with patch.object(EnhancedConversationLog, "scroll_end", autospec=True):
        log = EnhancedConversationLog()

        with patch("victor.ui.tui.widgets.time.monotonic", return_value=20.0):
            log._maybe_scroll_end(force=True)

        assert log._ignore_scroll_update_until > 20.0


def test_status_bar_updates_follow_indicator() -> None:
    """Status bar should show explicit Following/Paused indicator text."""
    bar = StatusBar()
    label = MagicMock()

    with (
        patch.object(StatusBar, "query_one", return_value=label),
        patch.object(bar, "update_shortcuts"),
    ):
        bar.update_follow(paused=True)
        label.update.assert_called_with("Paused")

        bar.update_follow(paused=False)
        label.update.assert_called_with("Following")


def test_status_bar_updates_unread_indicator() -> None:
    """Status bar should show/hide unread badge based on unread count."""
    bar = StatusBar()
    label = MagicMock()

    with patch.object(StatusBar, "query_one", return_value=label):
        bar.update_unread(4)
        label.update.assert_called_with("4 new")
        label.add_class.assert_called_with("visible")

        bar.update_unread(0)
        label.update.assert_called_with("")
        label.remove_class.assert_called_with("visible")


def test_status_bar_shortcuts_reflect_follow_action_state() -> None:
    """Shortcut hints should show pause/resume follow based on current state."""
    bar = StatusBar()
    hints = MagicMock()

    with patch.object(StatusBar, "query_one", return_value=hints):
        bar._follow_paused = False
        bar._update_shortcuts_idle()
        idle_following_text = hints.update.call_args.args[0]
        assert "pause follow" in idle_following_text.plain

        bar._follow_paused = True
        bar._update_shortcuts_idle()
        idle_paused_text = hints.update.call_args.args[0]
        assert "resume follow" in idle_paused_text.plain
