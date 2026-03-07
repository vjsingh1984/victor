"""Unit tests for TUI input enter/submit behavior."""

from __future__ import annotations

import asyncio
import warnings
from unittest.mock import MagicMock

from textual.app import App, ComposeResult

from victor.ui.tui.widgets import InputWidget


class _InputHarness(App[None]):
    """Minimal app harness to capture InputWidget submit messages."""

    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputWidget(id="input")

    async def on_input_widget_submitted(self, event: InputWidget.Submitted) -> None:
        self.submissions.append(event.value)


def test_enter_submits_prompt_without_unawaited_key_warnings() -> None:
    """Pressing Enter should dispatch submission without coroutine warnings."""

    async def _run() -> None:
        app = _InputHarness()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            async with app.run_test() as pilot:
                await pilot.press("h", "e", "l", "l", "o")
                await pilot.press("enter")
                await pilot.pause()
        assert app.submissions == ["hello"]
        assert not any("was never awaited" in str(w.message) for w in caught)

    asyncio.run(_run())


def test_enter_mid_line_inserts_newline_then_ctrl_enter_submits() -> None:
    """Enter in the middle of text should insert newline instead of submitting."""

    async def _run() -> None:
        app = _InputHarness()
        async with app.run_test() as pilot:
            await pilot.press("h", "i")
            await pilot.press("left")
            await pilot.press("enter")
            await pilot.pause()
            assert app.submissions == []
            input_widget = app.query_one("#input", InputWidget)
            assert input_widget.value == "h\ni"

            await pilot.press("ctrl+enter")
            await pilot.pause()

        assert app.submissions == ["h\ni"]

    asyncio.run(_run())


def test_set_busy_updates_prompt_hint_and_input_disable_state() -> None:
    """Busy state should visually acknowledge Enter and disable editing while running."""
    widget = InputWidget()
    widget._input = MagicMock()
    widget._prompt_label = MagicMock()
    widget._hint_label = MagicMock()

    widget.set_busy(True)

    assert widget._input.disabled is True
    widget._prompt_label.update.assert_called_with("⋯")
    widget._prompt_label.add_class.assert_called_with("busy")
    widget._hint_label.update.assert_called_with(widget._BUSY_HINT)

    widget._prompt_label.reset_mock()
    widget._hint_label.reset_mock()

    widget.set_busy(False)

    assert widget._input.disabled is False
    widget._prompt_label.update.assert_called_with("❯")
    widget._prompt_label.remove_class.assert_called_with("busy")
    widget._hint_label.update.assert_called_with(widget._IDLE_HINT)
