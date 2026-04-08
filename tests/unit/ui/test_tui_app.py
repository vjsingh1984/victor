"""Unit tests for Victor TUI app behavior."""

import asyncio
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, call, patch

from textual.messages import UpdateScroll

from victor.ui.tui.app import TUIConsoleAdapter, VictorTUI
from victor.ui.tui.session import Message
from victor.ui.tui.widgets import ToolCallWidget
from victor.providers.base import StreamChunk


def test_ctrl_f_binding_maps_to_toggle_follow_mode() -> None:
    """Ctrl+F should stay wired to follow-mode toggle."""
    assert any(
        binding.key == "ctrl+f" and binding.action == "toggle_follow_mode"
        for binding in VictorTUI.BINDINGS
    )


def test_handle_console_line_records_and_refreshes_ui() -> None:
    """Console output lines should be recorded and refresh unread/jump affordances."""
    app = VictorTUI()
    app._record_message = MagicMock()
    app._update_jump_to_bottom = MagicMock()

    app._handle_console_line("system output")

    app._record_message.assert_called_once_with("system", "system output")
    app._update_jump_to_bottom.assert_called_once()


def test_handle_console_line_ignores_empty_lines() -> None:
    """Empty console output lines should be ignored."""
    app = VictorTUI()
    app._record_message = MagicMock()
    app._update_jump_to_bottom = MagicMock()

    app._handle_console_line("")

    app._record_message.assert_not_called()
    app._update_jump_to_bottom.assert_not_called()


def test_console_adapter_sends_each_line_to_callback_and_log() -> None:
    """Console adapter should emit each rendered line to callback and conversation log."""
    log = MagicMock()
    on_line = MagicMock()
    adapter = TUIConsoleAdapter(log, on_line=on_line)

    adapter.print("first\nsecond")

    on_line.assert_any_call("first")
    on_line.assert_any_call("second")
    assert on_line.call_count == 2
    log.add_system_message.assert_any_call("first")
    log.add_system_message.assert_any_call("second")
    assert log.add_system_message.call_count == 2


def test_console_adapter_invokes_callback_after_log_update() -> None:
    """Console callback should run after unread state is updated in the log."""

    class FakeLog:
        def __init__(self) -> None:
            self.unread_count = 0

        def add_system_message(self, _line: str) -> None:
            self.unread_count += 1

    log = FakeLog()
    observed_counts: list[int] = []
    adapter = TUIConsoleAdapter(
        log,  # type: ignore[arg-type]
        on_line=lambda _line: observed_counts.append(log.unread_count),
    )

    adapter.print("line")

    assert observed_counts == [1]


def test_action_clear_clears_session_messages_and_agent_state() -> None:
    """Clear action should reset transcript state and session history consistently."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._session_messages = [MagicMock(), MagicMock()]
    app.agent = MagicMock()
    app._add_system_message = MagicMock()

    app.action_clear()

    app._conversation_log.clear.assert_called_once()
    app.agent.reset_conversation.assert_called_once()
    app._add_system_message.assert_called_once_with("Conversation cleared")
    assert app._session_messages == []


def test_handle_command_clear_routes_through_clear_action() -> None:
    """Slash clear should use the unified clear action and restore input focus."""
    app = VictorTUI()
    app.action_clear = MagicMock()
    app._input_widget = MagicMock()

    asyncio.run(app._handle_command("/clear"))

    app.action_clear.assert_called_once()
    app._input_widget.focus_input.assert_called_once()


def test_render_message_replay_uses_history_path() -> None:
    """Replay rendering should bypass normal follow/unread behavior paths."""
    app = VictorTUI()
    app._conversation_log = MagicMock()

    app._render_message("assistant", "from history", replay=True)

    app._conversation_log.add_history_message.assert_called_once_with(
        "assistant", "from history"
    )
    app._conversation_log.add_assistant_message.assert_not_called()
    app._conversation_log.add_user_message.assert_not_called()
    app._conversation_log.add_system_message.assert_not_called()
    app._conversation_log.add_error_message.assert_not_called()


def test_replay_transcript_batches_and_reanchors_once() -> None:
    """Transcript replay should clear, batch-mount history, and jump to latest once."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app.batch_update = MagicMock(return_value=nullcontext())

    app._replay_transcript([("assistant", "one"), ("user", "two")])

    app.batch_update.assert_called_once_with()
    app._conversation_log.clear.assert_called_once_with()
    assert app._conversation_log.add_history_message.call_args_list == [
        call("assistant", "one"),
        call("user", "two"),
    ]
    app._conversation_log.set_follow_paused.assert_called_once_with(
        False, jump_to_bottom=True
    )


def test_replay_transcript_async_uses_sync_path_for_small_histories() -> None:
    """Small histories should keep using single-pass replay."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._replay_transcript = MagicMock()
    messages = [("assistant", "short")]

    asyncio.run(app._replay_transcript_async(messages, status_label="Loading session"))

    app._replay_transcript.assert_called_once_with(messages)


def test_replay_transcript_async_chunks_large_histories() -> None:
    """Large histories should replay in async chunks with progress updates."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app.batch_update = MagicMock(return_value=nullcontext())
    app._set_status = MagicMock()
    messages = [
        ("assistant", f"m{i}")
        for i in range(app._ASYNC_REPLAY_THRESHOLD + app._ASYNC_REPLAY_CHUNK_SIZE)
    ]

    sleep_mock = AsyncMock()
    with patch("victor.ui.tui.app.asyncio.sleep", sleep_mock):
        asyncio.run(
            app._replay_transcript_async(messages, status_label="Loading session")
        )

    app._conversation_log.clear.assert_called_once_with()
    assert app._conversation_log.add_history_message.call_count == len(messages)
    app._conversation_log.set_follow_paused.assert_called_once_with(
        False, jump_to_bottom=True
    )
    assert app.batch_update.call_count >= 4
    assert sleep_mock.await_count >= 1
    assert any(
        call_args.args
        and call_args.args[0].startswith("Loading session (")
        and "/" in call_args.args[0]
        for call_args in app._set_status.call_args_list
    )


def test_start_session_restore_without_running_loop_uses_fallback() -> None:
    """Restore starter should run sync fallback when no event loop is active."""
    app = VictorTUI()
    fallback = MagicMock()

    async def _noop() -> None:
        return None

    app._start_session_restore(_noop(), fallback=fallback)

    fallback.assert_called_once_with()


def test_action_cancel_stream_cancels_active_session_restore() -> None:
    """Ctrl+X should cancel restore tasks before checking stream state."""
    app = VictorTUI()
    restore_task = MagicMock()
    restore_task.done.return_value = False
    app._session_restore_task = restore_task
    app._add_system_message = MagicMock()
    app._set_status = MagicMock()
    app.agent = MagicMock()

    app.action_cancel_stream()

    restore_task.cancel.assert_called_once_with()
    app._add_system_message.assert_called_once_with("Session restore canceled")
    app._set_status.assert_called_once_with("Idle", "idle")
    app.agent.request_cancellation.assert_not_called()


def test_load_session_async_uses_async_replay_path() -> None:
    """Async session loader should use chunk-capable replay helper."""
    app = VictorTUI()
    app._replay_transcript_async = AsyncMock()
    app._restore_agent_conversation = MagicMock()
    app._add_system_message = MagicMock()
    app._set_status = MagicMock()

    session = MagicMock()
    session.id = "session-12345678"
    session.name = "Replay Test"
    session.messages = [Message(role="assistant", content=f"m{i}") for i in range(3)]
    manager = MagicMock()
    with (
        patch("victor.ui.tui.session.SessionManager", return_value=manager),
        patch(
            "victor.ui.tui.app.asyncio.to_thread", AsyncMock(return_value=session)
        ) as to_thread,
    ):
        asyncio.run(app._load_session_async("session-12345678"))

    to_thread.assert_awaited_once_with(manager.load, "session-12345678")
    app._replay_transcript_async.assert_awaited_once_with(
        [(msg.role, msg.content) for msg in session.messages],
        status_label="Loading session",
    )
    app._restore_agent_conversation.assert_called_once_with(session.messages)
    app._add_system_message.assert_called_once_with(
        "Session loaded: Replay Test (3 messages)"
    )


def test_load_project_session_async_uses_to_thread() -> None:
    """Project async loader should fetch session data via background thread."""
    app = VictorTUI()
    app._replay_transcript_async = AsyncMock()
    app._add_system_message = MagicMock()
    app._set_status = MagicMock()
    app.agent = None

    history = MagicMock()
    history.messages = [MagicMock(role="assistant", content="a1")]
    persistence = MagicMock()
    persistence.load_session = MagicMock()

    with (
        patch(
            "victor.agent.sqlite_session_persistence.get_sqlite_session_persistence",
            return_value=persistence,
        ),
        patch(
            "victor.agent.message_history.MessageHistory.from_dict",
            return_value=history,
        ),
        patch(
            "victor.ui.tui.app.asyncio.to_thread",
            AsyncMock(
                return_value={
                    "conversation": {"messages": []},
                    "metadata": {"title": "P"},
                }
            ),
        ) as to_thread,
    ):
        asyncio.run(app._load_project_session_async("project-session-1"))

    to_thread.assert_awaited_once_with(persistence.load_session, "project-session-1")
    app._replay_transcript_async.assert_awaited_once_with(
        [("assistant", "a1")],
        status_label="Loading project session",
    )


def test_load_session_uses_status_and_single_completion_message() -> None:
    """Large restore should report progress in status bar, not transcript spam."""
    app = VictorTUI()
    app._replay_transcript = MagicMock()
    app._restore_agent_conversation = MagicMock()
    app._add_system_message = MagicMock()
    app._set_status = MagicMock()

    session = MagicMock()
    session.id = "session-12345678"
    session.name = "Replay Test"
    session.messages = [Message(role="assistant", content=f"m{i}") for i in range(60)]
    manager = MagicMock()
    manager.load.return_value = session

    with patch("victor.ui.tui.session.SessionManager", return_value=manager):
        app._load_session("session-12345678")

    app._set_status.assert_has_calls(
        [
            call("Loading session (60 messages)...", "busy"),
            call("Idle", "idle"),
        ]
    )
    assert app._set_status.call_count == 2
    app._restore_agent_conversation.assert_called_once_with(session.messages)
    app._replay_transcript.assert_called_once_with(
        [(msg.role, msg.content) for msg in session.messages]
    )
    app._add_system_message.assert_called_once_with(
        "Session loaded: Replay Test (60 messages)"
    )


def test_input_submit_resumes_follow_when_paused() -> None:
    """Submitting a prompt should resume live follow if sticky pause was active."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.follow_paused = True
    app._input_widget = MagicMock()
    app._add_user_message = MagicMock()
    app._process_message_async = AsyncMock()
    event = MagicMock()
    event.value = "hello"

    asyncio.run(app.on_input_widget_submitted(event))

    app._conversation_log.set_follow_paused.assert_called_once_with(
        False, jump_to_bottom=True
    )
    app._add_user_message.assert_called_once_with("hello")
    app._process_message_async.assert_awaited_once_with("hello")


def test_finish_tool_call_keeps_follow_up_widgets_visible_longer() -> None:
    """Tool widgets with follow-up actions should stay visible longer."""
    app = VictorTUI()
    widget = MagicMock()
    app._current_tool_widget = widget
    app._schedule_tool_widget_cleanup = MagicMock()
    app._prune_tool_widgets = MagicMock()
    follow_ups = [
        {
            "command": 'graph(mode="trace", node="main", depth=3)',
            "description": "Trace execution starting from main.",
        }
    ]

    app._finish_tool_call(success=True, elapsed=0.5, follow_up_suggestions=follow_ups)

    widget.update_status.assert_called_once_with(
        "success",
        0.5,
        follow_up_suggestions=follow_ups,
    )
    app._schedule_tool_widget_cleanup.assert_called_once_with(
        widget,
        timeout=20.0,
    )


def test_follow_up_selection_prefills_input() -> None:
    """Selecting a tool follow-up should prefill the prompt input."""
    app = VictorTUI()
    app._input_widget = MagicMock()
    app._add_system_message = MagicMock()
    event = ToolCallWidget.FollowUpSelected(
        'graph(mode="callers", node="parse_json", depth=1)'
    )

    app.on_tool_call_widget_follow_up_selected(event)

    app._input_widget.set_value.assert_called_once_with(
        'graph(mode="callers", node="parse_json", depth=1)'
    )
    app._input_widget.focus_input.assert_called_once_with()
    app._add_system_message.assert_called_once()


def test_stream_response_handles_metadata_tool_results_with_follow_ups() -> None:
    """Streaming path should handle metadata-based tool results and follow-ups."""
    app = VictorTUI()
    app.agent = MagicMock()
    app._conversation_log = MagicMock()
    app._start_streaming_ui = AsyncMock()
    app._set_status = MagicMock()
    app._finish_tool_call = MagicMock()
    app._hide_thinking = MagicMock()
    app._record_message = MagicMock()

    follow_ups = [
        {
            "command": 'graph(mode="trace", node="main", depth=3)',
            "description": "Trace execution starting from main.",
        }
    ]

    async def _stream():
        yield StreamChunk(
            content="",
            metadata={
                "tool_result": {
                    "name": "code_search",
                    "success": True,
                    "elapsed": 0.5,
                    "arguments": {"query": "main entry point"},
                    "follow_up_suggestions": follow_ups,
                }
            },
        )

    app.agent.stream_chat = MagicMock(return_value=_stream())

    asyncio.run(app._stream_response("trace main"))

    app._finish_tool_call.assert_called_once_with(
        success=True,
        elapsed=0.5,
        follow_up_suggestions=follow_ups,
    )


def test_input_submit_ignored_while_processing_keeps_draft() -> None:
    """Submitting while busy should not clear input or enqueue a second send."""
    app = VictorTUI()
    app._is_processing = True
    app._input_widget = MagicMock()
    app._add_user_message = MagicMock()
    app._process_message_async = AsyncMock()
    app._set_status = MagicMock()
    event = MagicMock()
    event.value = "hello"

    asyncio.run(app.on_input_widget_submitted(event))

    app._input_widget.add_to_history.assert_not_called()
    app._input_widget.clear.assert_not_called()
    app._add_user_message.assert_not_called()
    app._process_message_async.assert_not_awaited()
    app._set_status.assert_called_once_with("Working", "busy")


def test_process_message_async_toggles_input_busy_state() -> None:
    """Processing lifecycle should set input busy true then false."""
    app = VictorTUI()
    app._input_widget = MagicMock()

    asyncio.run(app._process_message_async("hello"))

    app._input_widget.set_busy.assert_any_call(True)
    app._input_widget.set_busy.assert_any_call(False)
    app._input_widget.focus_input.assert_called_once()


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


def test_update_jump_button_avoids_redundant_visible_updates() -> None:
    """Repeated refreshes with same state should not thrash jump button visibility."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = False
    app._conversation_log.follow_paused = False
    app._conversation_log.unread_count = 2
    app._jump_button = MagicMock()
    app._jump_button.label = "Jump to bottom"

    app._update_jump_to_bottom()
    app._update_jump_to_bottom()

    app._jump_button.add_class.assert_called_once_with("visible")
    app._jump_button.remove_class.assert_not_called()


def test_update_jump_button_avoids_redundant_hidden_updates() -> None:
    """Repeated refreshes at bottom should not repeatedly remove visibility class."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = True
    app._conversation_log.follow_paused = False
    app._conversation_log.unread_count = 0
    app._jump_button = MagicMock()
    app._jump_button.label = "Jump to bottom"

    app._update_jump_to_bottom()
    app._update_jump_to_bottom()

    app._jump_button.remove_class.assert_called_once_with("visible")
    app._jump_button.add_class.assert_not_called()


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


def test_action_scroll_bottom_resumes_follow_when_paused() -> None:
    """Ctrl+End should resume sticky follow mode when paused."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.follow_paused = True
    app._update_jump_to_bottom = MagicMock()

    app.action_scroll_bottom()

    app._conversation_log.set_follow_paused.assert_called_once_with(
        False, jump_to_bottom=True
    )
    app._conversation_log.scroll_to_bottom.assert_not_called()
    app._update_jump_to_bottom.assert_called_once()


def test_action_scroll_bottom_scrolls_when_follow_not_paused() -> None:
    """Ctrl+End should jump to bottom when sticky follow is not paused."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.follow_paused = False
    app._update_jump_to_bottom = MagicMock()

    app.action_scroll_bottom()

    app._conversation_log.scroll_to_bottom.assert_called_once_with(animate=False)
    app._conversation_log.set_follow_paused.assert_not_called()
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

    app._conversation_log.set_follow_paused.assert_called_once_with(
        False, jump_to_bottom=True
    )
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
    app._conversation_log.auto_scroll_enabled = True
    app._conversation_log.follow_paused = False
    app._update_jump_to_bottom = MagicMock()

    app.action_toggle_follow_mode()

    app._conversation_log.set_follow_paused.assert_called_once_with(
        True, jump_to_bottom=False
    )
    app._update_jump_to_bottom.assert_called_once()


def test_action_toggle_follow_mode_resumes_follow() -> None:
    """Ctrl+F action should resume follow and jump to latest."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = False
    app._conversation_log.follow_paused = True
    app._update_jump_to_bottom = MagicMock()

    app.action_toggle_follow_mode()

    app._conversation_log.set_follow_paused.assert_called_once_with(
        False, jump_to_bottom=True
    )
    app._update_jump_to_bottom.assert_called_once()


def test_action_toggle_follow_mode_resumes_when_user_scrolled() -> None:
    """Ctrl+F should resume following when paused by manual scroll-away."""
    app = VictorTUI()
    app._conversation_log = MagicMock()
    app._conversation_log.auto_scroll_enabled = False
    app._conversation_log.follow_paused = False
    app._update_jump_to_bottom = MagicMock()

    app.action_toggle_follow_mode()

    app._conversation_log.set_follow_paused.assert_called_once_with(
        False, jump_to_bottom=True
    )
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

    app._conversation_log.set_follow_paused.assert_called_once_with(
        False, jump_to_bottom=True
    )
    app._conversation_log.scroll_to_bottom.assert_not_called()
    app._update_jump_to_bottom.assert_called_once()
