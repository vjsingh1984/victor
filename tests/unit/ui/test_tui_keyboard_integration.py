"""Integration-style keyboard tests for Victor TUI."""

from __future__ import annotations

import asyncio

from victor.ui.tui.app import VictorTUI
from victor.ui.tui.widgets import InputWidget, StatusBar


def _static_content_text(widget: object) -> str:
    """Extract textual content from Textual Static/Label widget."""
    content = getattr(widget, "_Static__content", "")
    return str(content)


def test_enter_submits_to_transcript_and_shows_busy_feedback() -> None:
    """Pressing Enter in the real TUI should submit and provide immediate keyboard feedback."""

    async def _run() -> None:
        async def on_message(message: str) -> str:
            await asyncio.sleep(0)
            return f"echo:{message}"

        app = VictorTUI(stream=False, on_message=on_message)

        async with app.run_test() as pilot:
            input_widget = app.query_one("#input-widget", InputWidget)

            await pilot.press("h", "i", "enter")
            for _ in range(10):
                await pilot.pause()
                if any(
                    msg.role == "assistant" and msg.content == "echo:hi"
                    for msg in app._session_messages
                ):
                    break

            # Enter should send the prompt and clear the input field.
            assert input_widget.value == ""
            assert any(msg.role == "user" and msg.content == "hi" for msg in app._session_messages)
            assert any(
                msg.role == "assistant" and msg.content == "echo:hi"
                for msg in app._session_messages
            )
            assert input_widget._input is not None
            assert input_widget._input.disabled is False
            assert input_widget._prompt_label is not None
            assert _static_content_text(input_widget._prompt_label) == "❯"
            assert input_widget._hint_label is not None
            hint_text = _static_content_text(input_widget._hint_label)
            assert "Enter" in hint_text
            assert "Shift+Enter" in hint_text
            assert "commands" in hint_text

    asyncio.run(_run())


def test_ctrl_f_toggles_sticky_follow_mode_and_status_state() -> None:
    """Ctrl+F should toggle sticky follow mode and keep app status state in sync."""

    async def _run() -> None:
        app = VictorTUI(stream=False)
        async with app.run_test() as pilot:
            app.query_one(StatusBar)
            assert app._conversation_log is not None
            assert app._jump_button is not None
            assert app._status_bar is not None
            follow_label = app.query_one("#follow-indicator")
            assert app._conversation_log.follow_paused is False
            assert app._status_bar._follow_paused is False
            assert _static_content_text(follow_label) == "Following"

            await pilot.press("ctrl+f")
            await pilot.pause()
            assert app._conversation_log.follow_paused is True
            assert app._status_bar._follow_paused is True
            assert _static_content_text(follow_label) == "Paused"
            assert str(app._jump_button.label).startswith("Resume follow")

            await pilot.press("ctrl+f")
            await pilot.pause()
            assert app._conversation_log.follow_paused is False
            assert app._status_bar._follow_paused is False
            assert _static_content_text(follow_label) == "Following"
            assert str(app._jump_button.label) == "Jump to bottom"

    asyncio.run(_run())
