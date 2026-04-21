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

"""Unit tests for LiveDisplayRenderer content display fixes.

These tests verify the fixes for console rendering issues where content
was being buffered but not displayed to users.
"""

import pytest
from unittest.mock import MagicMock, patch
from rich.console import Console
from victor.ui.rendering import LiveDisplayRenderer


class TestLiveDisplayRendererContentDisplay:
    """Tests for content display fixes in Phase 1."""

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_content_visible_in_thinking_mode(self, mock_live_class):
        """Content should be visible even when in thinking mode.

        This is a regression test for the bug where early return in on_content()
        prevented Live display updates during thinking mode.
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        console = Console()
        renderer = LiveDisplayRenderer(console)

        renderer.start()
        renderer.on_thinking_start()

        # Add content during thinking mode
        renderer.on_content("This should be visible during thinking")

        # CRITICAL: Live display should be updated even in thinking mode
        assert mock_live.update.called, "Live display should be updated in thinking mode"
        assert "This should be visible during thinking" in renderer._content_buffer
        assert renderer._content_buffer == renderer._thinking_buffer

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_finalize_shows_buffered_content(self, mock_live_class):
        """finalize() should ensure all buffered content is displayed.

        This tests the fail-safe mechanism that displays content if Live display
        was never started or was stopped early.
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        console = Console()
        renderer = LiveDisplayRenderer(console)

        # Simulate stream ending in thinking mode without Live update
        renderer.on_thinking_start()
        renderer.on_content("Buffered content in thinking mode")
        renderer._live = None  # Simulate Live being stopped

        result = renderer.finalize()

        # Content should be returned even if Live wasn't updated
        assert result == "Buffered content in thinking mode"

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_thinking_end_updates_live_display(self, mock_live_class):
        """on_thinking_end() should update Live display with buffered content.

        This verifies the fix that forces a Live display update when thinking
        mode ends, ensuring content buffered during thinking is visible.
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        console = Console()
        renderer = LiveDisplayRenderer(console)

        renderer.start()
        renderer.on_thinking_start()
        renderer.on_content("Content during thinking")
        renderer.on_thinking_end()

        # Live should be updated after thinking ends
        assert mock_live.update.called
        assert "Content during thinking" in renderer._content_buffer

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_content_always_added_to_buffer(self, mock_live_class):
        """Content should always be added to buffer, regardless of mode.

        This ensures content is preserved for final display even if stream
        ends in thinking mode.
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        console = Console()
        renderer = LiveDisplayRenderer(console)

        renderer.start()

        # Add content in normal mode
        renderer.on_content("Normal content")
        assert "Normal content" in renderer._content_buffer

        # Add content in thinking mode
        renderer.on_thinking_start()
        renderer.on_content("Thinking content")
        assert "Thinking content" in renderer._content_buffer

        # Both should be in buffer
        assert "Normal content" in renderer._content_buffer
        assert "Thinking content" in renderer._content_buffer

    def test_finalize_with_thinking_buffer(self):
        """finalize() should flush remaining thinking buffer before cleanup."""
        console = Console()
        renderer = LiveDisplayRenderer(console)

        # Simulate thinking buffer not flushed
        renderer.on_thinking_start()
        renderer.on_content("Unflushed thinking content")
        renderer._in_thinking_mode = True  # Simulate still in thinking mode

        result = renderer.finalize()

        # Thinking buffer should be flushed
        assert renderer._thinking_buffer == ""
        # Content should be in result
        assert "Unflushed thinking content" in result

    def test_finalize_with_empty_buffers(self):
        """finalize() should handle empty buffers gracefully."""
        console = Console()
        renderer = LiveDisplayRenderer(console)

        # Don't add any content
        result = renderer.finalize()

        # Should return empty string
        assert result == ""

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_multiple_thinking_cycles(self, mock_live_class):
        """Multiple thinking cycles should handle content correctly."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        console = Console()
        renderer = LiveDisplayRenderer(console)

        renderer.start()

        # First thinking cycle
        renderer.on_thinking_start()
        renderer.on_content("First thinking")
        renderer.on_thinking_end()

        # Normal content
        renderer.on_content("Normal between")

        # Second thinking cycle
        renderer.on_thinking_start()
        renderer.on_content("Second thinking")
        renderer.on_thinking_end()

        # All content should be in buffer
        assert "First thinking" in renderer._content_buffer
        assert "Normal between" in renderer._content_buffer
        assert "Second thinking" in renderer._content_buffer

        # finalize should work
        result = renderer.finalize()
        assert len(result) > 0


class TestLiveDisplayRendererFailSafe:
    """Tests for fail-safe mechanisms in LiveDisplayRenderer."""

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_fail_safe_display_when_live_never_started(self, mock_live_class):
        """Fail-safe should display content if Live was never started."""
        mock_live_class.return_value = None  # Live never starts
        console = Console()
        renderer = LiveDisplayRenderer(console)

        # Add content without starting Live
        renderer.on_content("Content without Live")

        result = renderer.finalize()

        # Content should be in result
        assert "Content without Live" in result

    def test_fail_safe_preserves_content_on_error(self):
        """Content should be preserved even if Live update fails."""
        console = Console()
        renderer = LiveDisplayRenderer(console)

        # Start Live
        renderer.start()

        # Add content
        content = "Important content"
        renderer.on_content(content)

        # Stop Live (simulating early termination)
        renderer._live = None

        # finalize should still return content
        result = renderer.finalize()

        assert result == content
        assert content in result


class TestLiveDisplayRendererDebugLogging:
    """Tests for debug logging in LiveDisplayRenderer."""

    def test_finalize_logs_buffer_lengths(self, caplog):
        """finalize() should log buffer lengths for debugging."""
        console = Console()
        renderer = LiveDisplayRenderer(console)

        # Add content
        renderer.on_content("Test content")

        with caplog.at_level("DEBUG"):
            result = renderer.finalize()

        # Should log debug info
        assert any("content_buffer_len" in record.message for record in caplog.records)
        assert any("thinking_buffer_len" in record.message for record in caplog.records)
        assert any("in_thinking_mode" in record.message for record in caplog.records)

    def test_finalize_logs_empty_content_warning(self, caplog):
        """finalize() should warn if content is empty."""
        console = Console()
        renderer = LiveDisplayRenderer(console)

        # Don't add any content
        with caplog.at_level("DEBUG"):
            result = renderer.finalize()

        # Should not warn for empty content (this is valid)
        # Warning only if content was expected but not received
        assert not any("empty content" in record.message for record in caplog.records)
