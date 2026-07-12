"""Unit tests for LiveManager.

Tests verify the Live display lifecycle: start/stop, pause/resume with
nested depth counting, content buffer management, incremental rendering
(HEAD/TAIL split), and section separators.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from victor.ui.theme import victor_theme

from victor.ui.rendering.live_manager import LiveManager


@pytest.fixture
def console() -> Console:
    return Console(theme=victor_theme)


@pytest.fixture
def mock_live() -> MagicMock:
    return MagicMock()


@pytest.fixture
def manager(console: Console, mock_live: MagicMock) -> LiveManager:
    mgr = LiveManager(console)
    mgr._live = mock_live
    return mgr


class TestStart:
    """LiveManager.start() creates and starts a Rich Live display."""

    @patch("victor.ui.rendering.live_manager.Live")
    def test_start_creates_live(self, mock_live_class: MagicMock, console: Console) -> None:
        mock_live_instance = MagicMock()
        mock_live_class.return_value = mock_live_instance
        mgr = LiveManager(console)
        mgr.start()
        mock_live_class.assert_called_once()
        mock_live_instance.start.assert_called_once()
        assert mgr._is_paused is False
        assert mgr._pause_count == 0

    @patch("victor.ui.rendering.live_manager.Live")
    def test_start_invalidates_head_cache(
        self, mock_live_class: MagicMock, console: Console
    ) -> None:
        mock_live_class.return_value = MagicMock()
        mgr = LiveManager(console)
        mgr._rendered_head = "stale"
        mgr._rendered_head_source = "stale_source"
        mgr.start()
        assert mgr._rendered_head is None
        assert mgr._rendered_head_source == ""


class TestPause:
    """LiveManager.pause() stops the Live display with depth counting."""

    def test_pause_stops_live(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager.pause()
        mock_live.stop.assert_called_once()
        assert manager._is_paused is True
        assert manager._pause_count == 1

    def test_pause_nested_does_not_stop_again(
        self, manager: LiveManager, mock_live: MagicMock
    ) -> None:
        manager.pause()
        manager.pause()
        assert mock_live.stop.call_count == 1
        assert manager._pause_count == 2

    def test_pause_when_no_live(self, console: Console) -> None:
        mgr = LiveManager(console)
        mgr.pause()
        assert mgr._pause_count == 1

    def test_pause_records_shown_content(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager._content_buffer = "AAA"
        manager.pause()
        assert manager._content_shown_before_pause == "AAA"

    def test_pause_already_paused(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager._is_paused = True
        manager.pause()
        assert manager._pause_count == 1
        mock_live.stop.assert_not_called()


class TestResume:
    """LiveManager.resume() restarts the Live display."""

    def test_resume_restarts_live(self, manager: LiveManager, mock_live: MagicMock) -> None:
        with patch("victor.ui.rendering.live_manager.Live") as mock_live_class:
            new_live = MagicMock()
            mock_live_class.return_value = new_live
            manager._content_buffer = "AAA"
            manager.pause()
            manager.resume()
            assert manager._is_paused is False
            assert manager._pause_count == 0
            new_live.start.assert_called_once()

    def test_resume_nested_does_not_restart(
        self, manager: LiveManager, mock_live: MagicMock
    ) -> None:
        manager.pause()
        manager.pause()
        manager.resume()
        assert manager._is_paused is True
        assert manager._pause_count == 1

    def test_resume_nested_full_unwind(self, manager: LiveManager, mock_live: MagicMock) -> None:
        with patch("victor.ui.rendering.live_manager.Live") as mock_live_class:
            mock_live_class.return_value = MagicMock()
            manager.pause()
            manager.pause()
            manager.resume()
            manager.resume()
            assert manager._is_paused is False
            assert manager._pause_count == 0

    def test_resume_without_pause_is_noop(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager.resume()
        assert manager._pause_count == 0
        assert mock_live.start.call_count == 0

    def test_resume_creates_new_live_with_content(
        self, manager: LiveManager, mock_live: MagicMock
    ) -> None:
        with patch("victor.ui.rendering.live_manager.Live") as mock_live_class:
            new_live = MagicMock()
            mock_live_class.return_value = new_live
            manager._content_buffer = "AAA"
            manager.pause()
            manager._content_buffer += "BBB"
            manager.resume()
            mock_live_class.assert_called_once()
            new_live.start.assert_called_once()

    def test_resume_updates_metrics(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager.pause()
        with patch("victor.ui.rendering.live_manager.Live") as mock_live_class:
            mock_live_class.return_value = MagicMock()
            manager.resume()
        assert manager._metrics.pause_count == 1
        assert manager._metrics.total_pause_ms >= 0


class TestContentBuffer:
    """Content buffer management."""

    def test_append_adds_text(self, manager: LiveManager) -> None:
        manager.append_content("Hello")
        assert manager._content_buffer == "Hello"

    def test_append_accumulates(self, manager: LiveManager) -> None:
        manager.append_content("A")
        manager.append_content("B")
        manager.append_content("C")
        assert manager._content_buffer == "ABC"

    def test_content_buffer_property(self, manager: LiveManager) -> None:
        manager._content_buffer = "test"
        assert manager.content_buffer == "test"

    def test_visible_content_no_pause(self, manager: LiveManager) -> None:
        manager._content_buffer = "full"
        assert manager.visible_content == "full"

    def test_visible_content_after_pause(self, manager: LiveManager) -> None:
        manager._content_buffer = "AAA"
        manager.pause()
        manager._content_buffer += "BBB"
        assert manager.visible_content == "BBB"

    def test_advance_shown_content(self, manager: LiveManager) -> None:
        manager._content_buffer = "AAA"
        manager.advance_shown_content()
        assert manager._content_shown_before_pause == "AAA"
        assert manager.visible_content == ""

    def test_buffer_cap_enforced(self, manager: LiveManager) -> None:
        manager.MAX_CONTENT_BUFFER_SIZE = 10
        manager._content_buffer = "A" * 9
        manager.append_content("B" * 5)
        assert len(manager._content_buffer) == 10
        assert (
            manager._content_buffer == "A" * 5 + "B" * 5
        )  # excess=4 trimmed from front, leaving 5 As

    def test_buffer_cap_exact_fit(self, manager: LiveManager) -> None:
        manager.MAX_CONTENT_BUFFER_SIZE = 10
        manager._content_buffer = "A" * 5
        manager.append_content("B" * 5)
        assert manager._content_buffer == "A" * 5 + "B" * 5

    def test_is_paused_property(self, manager: LiveManager) -> None:
        assert manager.is_paused is False
        manager.pause()
        assert manager.is_paused is True
        with patch("victor.ui.rendering.live_manager.Live") as mock_live_class:
            mock_live_class.return_value = MagicMock()
            manager.resume()
        assert manager.is_paused is False

    def test_live_property(self, manager: LiveManager, mock_live: MagicMock) -> None:
        assert manager.live is mock_live


class TestIncrementalRender:
    """Incremental HEAD/TAIL rendering."""

    @patch("victor.ui.rendering.live_manager.render_markdown_with_hooks")
    def test_update_live_no_live(self, mock_render: MagicMock, console: Console) -> None:
        mgr = LiveManager(console)
        mgr.update_live()
        mock_render.assert_not_called()

    @patch("victor.ui.rendering.live_manager.render_markdown_with_hooks")
    def test_update_live_full_render_fallback(
        self, mock_render: MagicMock, manager: LiveManager, mock_live: MagicMock
    ) -> None:
        mock_render.side_effect = lambda c: f"R({c})"
        manager._content_buffer = "AAA"
        with patch.object(LiveManager, "_incremental_render_enabled", return_value=False):
            manager.update_live()
        mock_live.update.assert_called_once_with("R(AAA)")

    @patch("victor.ui.rendering.live_manager.render_markdown_with_hooks")
    def test_update_live_empty_buffer(
        self, mock_render: MagicMock, manager: LiveManager, mock_live: MagicMock
    ) -> None:
        manager._content_buffer = ""
        manager.update_live()
        mock_live.update.assert_called_once()

    @patch("victor.ui.rendering.live_manager.render_markdown_with_hooks")
    def test_update_live_with_renderable(
        self, mock_render: MagicMock, manager: LiveManager, mock_live: MagicMock
    ) -> None:
        renderable = MagicMock()
        manager.update_live_with_renderable(renderable)
        mock_live.update.assert_called_once_with(renderable)

    @patch("victor.ui.rendering.live_manager.render_markdown_with_hooks")
    def test_update_live_with_renderable_no_live(
        self, mock_render: MagicMock, console: Console
    ) -> None:
        mgr = LiveManager(console)
        mgr.update_live_with_renderable("anything")

    @patch("victor.ui.rendering.live_manager.render_markdown_with_hooks")
    def test_invalidate_head_cache(self, mock_render: MagicMock, manager: LiveManager) -> None:
        manager._rendered_head = "stale"
        manager._rendered_head_source = "stale"
        manager._invalidate_head_cache()
        assert manager._rendered_head is None
        assert manager._rendered_head_source == ""


class TestIncrementalRenderEnabled:
    """_incremental_render_enabled() env-var gating."""

    def test_default_enabled(self) -> None:
        assert LiveManager._incremental_render_enabled() is True

    def test_env_0_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VICTOR_INCREMENTAL_RENDER", "0")
        assert LiveManager._incremental_render_enabled() is False

    def test_env_false_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VICTOR_INCREMENTAL_RENDER", "false")
        assert LiveManager._incremental_render_enabled() is False

    def test_env_off_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VICTOR_INCREMENTAL_RENDER", "off")
        assert LiveManager._incremental_render_enabled() is False

    def test_env_1_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VICTOR_INCREMENTAL_RENDER", "1")
        assert LiveManager._incremental_render_enabled() is True

    def test_env_arbitrary_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VICTOR_INCREMENTAL_RENDER", "yes")
        assert LiveManager._incremental_render_enabled() is True


class TestSectionSeparator:
    """print_section_separator() renders a Rich Rule."""

    def test_separator_with_title(self, manager: LiveManager, console: Console) -> None:
        with patch.object(console, "print") as mock_print:
            manager.print_section_separator("Response")
        mock_print.assert_called_once()

    def test_separator_without_title(self, manager: LiveManager, console: Console) -> None:
        with patch.object(console, "print") as mock_print:
            manager.print_section_separator()
        mock_print.assert_called_once()


class TestStop:
    """LiveManager.stop() cleans up the Live display."""

    def test_stop_stops_live(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager.stop()
        mock_live.stop.assert_called_once()

    def test_stop_no_live(self, console: Console) -> None:
        mgr = LiveManager(console)
        mgr.stop()

    def test_stop_resets_state(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager.stop()
        assert manager._is_paused is False
        assert manager._pause_count == 0
        assert manager._live is None


class TestEdgeCases:
    """Edge cases and defensive scenarios."""

    def test_double_start(self, console: Console) -> None:
        with patch("victor.ui.rendering.live_manager.Live") as mock_live_class:
            first_instance = MagicMock()
            second_instance = MagicMock()
            mock_live_class.side_effect = [first_instance, second_instance]
            mgr = LiveManager(console)
            mgr.start()
            assert mgr._live is first_instance
            mgr.start()
            assert mgr._live is second_instance

    def test_double_stop(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager.stop()
        # Second stop is safe: _live is now None, so stop() is a no-op
        manager.stop()
        assert mock_live.stop.call_count == 1

    def test_pause_resume_no_content(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager.pause()
        with patch("victor.ui.rendering.live_manager.Live") as mock_live_class:
            mock_live_class.return_value = MagicMock()
            manager.resume()
        assert manager._is_paused is False

    def test_append_empty_string(self, manager: LiveManager) -> None:
        manager.append_content("")
        assert manager._content_buffer == ""

    def test_append_none(self, manager: LiveManager) -> None:
        with pytest.raises(TypeError):
            manager.append_content(None)  # type: ignore[arg-type]

    def test_visible_content_advance_after_pause(self, manager: LiveManager) -> None:
        manager._content_buffer = "AAA"
        manager.pause()
        manager._content_buffer += "BBB"
        manager.advance_shown_content()
        assert manager.visible_content == ""
        manager._content_buffer += "CCC"
        assert manager.visible_content == "CCC"

    def test_stop_clears_content_shown(self, manager: LiveManager, mock_live: MagicMock) -> None:
        manager._content_shown_before_pause = "old"
        manager.stop()
        assert manager._content_shown_before_pause == ""
