"""Tests for rendering module.

Tests cover:
- FormatterRenderer class methods
- LiveDisplayRenderer class methods
- stream_response() unified handler
- Protocol compliance
- Edge cases and error handling
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from rich.console import Console

from victor.ui.rendering import (
    StreamRenderer,
    FormatterRenderer,
    LiveDisplayRenderer,
    stream_response,
    format_tool_args,
)
from victor.providers.base import StreamChunk


class TestFormatterRenderer:
    """Tests for FormatterRenderer class."""

    @pytest.fixture
    def mock_formatter(self):
        """Create a mock OutputFormatter."""
        formatter = MagicMock()
        return formatter

    @pytest.fixture
    def mock_console(self):
        """Create a mock Console."""
        return MagicMock(spec=Console)

    @pytest.fixture
    def renderer(self, mock_formatter, mock_console):
        """Create a FormatterRenderer instance."""
        return FormatterRenderer(mock_formatter, mock_console)

    def test_start_calls_formatter_start_streaming(self, renderer, mock_formatter):
        """Test start() delegates to formatter.start_streaming()."""
        renderer.start()
        mock_formatter.start_streaming.assert_called_once()

    def test_pause_calls_formatter_end_streaming(self, renderer, mock_formatter):
        """Test pause() delegates to formatter.end_streaming()."""
        renderer.pause()
        mock_formatter.end_streaming.assert_called_once()

    def test_resume_calls_formatter_start_streaming(self, renderer, mock_formatter):
        """Test resume() delegates to formatter.start_streaming()."""
        renderer.resume()
        mock_formatter.start_streaming.assert_called_once()

    def test_on_tool_start_pauses_and_resumes(self, renderer, mock_formatter):
        """Test on_tool_start() pauses, shows status, and resumes."""
        renderer.on_tool_start("read", {"path": "/test.py"})

        # Should pause, call tool_start, status, then resume
        mock_formatter.end_streaming.assert_called_once()
        mock_formatter.tool_start.assert_called_once_with("read", {"path": "/test.py"})
        mock_formatter.status.assert_called_once()
        assert "read" in mock_formatter.status.call_args[0][0]
        mock_formatter.start_streaming.assert_called_once()

    def test_on_tool_result_success(self, renderer, mock_formatter):
        """Test on_tool_result() for successful tool execution."""
        renderer.on_tool_result(
            name="write",
            success=True,
            elapsed=0.5,
            arguments={"path": "/out.py"},
        )

        mock_formatter.end_streaming.assert_called_once()
        mock_formatter.tool_result.assert_called_once_with(
            tool_name="write",
            success=True,
            error=None,
        )
        mock_formatter.start_streaming.assert_called_once()

    def test_on_tool_result_failure(self, renderer, mock_formatter):
        """Test on_tool_result() for failed tool execution."""
        renderer.on_tool_result(
            name="bash",
            success=False,
            elapsed=1.2,
            arguments={"command": "ls"},
            error="Permission denied",
        )

        mock_formatter.tool_result.assert_called_once_with(
            tool_name="bash",
            success=False,
            error="Permission denied",
        )

    def test_on_status(self, renderer, mock_formatter):
        """Test on_status() shows status message."""
        renderer.on_status("Thinking...")

        mock_formatter.end_streaming.assert_called_once()
        mock_formatter.status.assert_called_once_with("Thinking...")
        mock_formatter.start_streaming.assert_called_once()

    def test_on_file_preview(self, renderer, mock_formatter, mock_console):
        """Test on_file_preview() displays syntax-highlighted content."""
        renderer.on_file_preview("/test.py", "print('hello')")

        mock_formatter.end_streaming.assert_called_once()
        mock_console.print.assert_called_once()
        # Check Panel was printed
        call_args = mock_console.print.call_args[0][0]
        assert hasattr(call_args, "title")  # Panel has title attribute
        mock_formatter.start_streaming.assert_called_once()

    def test_on_file_preview_no_extension(self, renderer, mock_formatter, mock_console):
        """Test on_file_preview() handles files without extension."""
        renderer.on_file_preview("Makefile", "all: build")

        mock_console.print.assert_called_once()

    def test_on_edit_preview(self, renderer, mock_formatter, mock_console):
        """Test on_edit_preview() displays colored diff."""
        diff = "-old line\n+new line\n context"
        renderer.on_edit_preview("/test.py", diff)

        mock_formatter.end_streaming.assert_called_once()
        # Should print path header and 3 diff lines
        assert mock_console.print.call_count == 4
        mock_formatter.start_streaming.assert_called_once()

    def test_on_content_accumulates(self, renderer, mock_formatter):
        """Test on_content() accumulates content and streams chunks."""
        renderer.on_content("Hello ")
        renderer.on_content("World")

        assert mock_formatter.stream_chunk.call_count == 2
        mock_formatter.stream_chunk.assert_any_call("Hello ")
        mock_formatter.stream_chunk.assert_any_call("World")

    def test_finalize_returns_content(self, renderer, mock_formatter):
        """Test finalize() returns accumulated content."""
        renderer.on_content("Test content")
        result = renderer.finalize()

        assert result == "Test content"
        mock_formatter.response.assert_called_once_with(content="Test content")

    def test_cleanup_is_noop(self, renderer):
        """Test cleanup() is a no-op for FormatterRenderer."""
        renderer.cleanup()  # Should not raise

    def test_on_thinking_content_prints_dimmed(self, renderer, mock_console):
        """Test on_thinking_content() prints dimmed/italic text."""
        renderer.on_thinking_content("Analyzing the problem...")

        # Should print styled text (not added to buffer)
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args
        assert call_args[1].get("end") == ""  # No newline

    def test_on_thinking_start_shows_indicator(self, renderer, mock_formatter, mock_console):
        """Test on_thinking_start() shows thinking indicator."""
        renderer.on_thinking_start()

        # Should pause streaming and show indicator
        mock_formatter.end_streaming.assert_called_once()
        mock_console.print.assert_called_once()
        call_text = str(mock_console.print.call_args[0][0])
        assert "Thinking" in call_text

    def test_on_thinking_end_resumes_streaming(self, renderer, mock_formatter, mock_console):
        """Test on_thinking_end() resumes streaming."""
        renderer.on_thinking_end()

        # Should print newline and resume streaming
        mock_console.print.assert_called_once()
        mock_formatter.start_streaming.assert_called_once()

    def test_thinking_content_not_in_buffer(self, renderer, mock_formatter, mock_console):
        """Test thinking content is not accumulated in buffer."""
        renderer.on_content("Normal content")
        renderer.on_thinking_content("Thinking content")
        result = renderer.finalize()

        # Only normal content should be in result
        assert result == "Normal content"
        assert "Thinking" not in result


class TestLiveDisplayRenderer:
    """Tests for LiveDisplayRenderer class."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock Console."""
        return MagicMock(spec=Console)

    @pytest.fixture
    def renderer(self, mock_console):
        """Create a LiveDisplayRenderer instance."""
        return LiveDisplayRenderer(mock_console)

    def test_format_args_empty(self):
        """Test format_tool_args() with empty dict."""
        result = format_tool_args({})
        assert result == ""

    def test_format_args_single(self):
        """Test format_tool_args() with single argument."""
        result = format_tool_args({"path": "/test.py"})
        assert "path=" in result
        assert "/test.py" in result

    def test_format_args_multiple(self):
        """Test format_tool_args() with multiple arguments."""
        result = format_tool_args({"a": 1, "b": "test"})
        assert "a=" in result
        assert "b=" in result

    def test_format_args_truncates_long_values(self):
        """Test format_tool_args() truncates values longer than 60 chars."""
        long_value = "x" * 100
        result = format_tool_args({"long": long_value})
        assert len(result) < len(long_value) + 20  # Much shorter than original

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_start_creates_live_display(self, mock_live_class, renderer, mock_console):
        """Test start() creates and starts a Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()

        mock_live_class.assert_called_once()
        mock_live.start.assert_called_once()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_pause_stops_live_display(self, mock_live_class, renderer, mock_console):
        """Test pause() stops the Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.pause()

        mock_live.stop.assert_called_once()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_resume_creates_new_live_display(self, mock_live_class, renderer, mock_console):
        """Test resume() creates a new Live display with current content."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer._content_buffer = "Previous content"
        renderer.resume()

        assert mock_live_class.call_count == 1
        mock_live.start.assert_called_once()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_start_prints_status(self, mock_live_class, renderer, mock_console):
        """Test on_tool_start() prints tool invocation status."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_tool_start("read", {"path": "/test.py"})

        # Should print status
        mock_console.print.assert_called()
        call_str = str(mock_console.print.call_args)
        assert "read" in call_str

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_result_success_prints_checkmark(self, mock_live_class, renderer, mock_console):
        """Test on_tool_result() prints green checkmark for success."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_tool_result(
            name="read",
            success=True,
            elapsed=0.1,
            arguments={"path": "/test.py"},
        )

        mock_console.print.assert_called()
        call_str = str(mock_console.print.call_args)
        assert "green" in call_str
        assert "✓" in call_str

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_result_failure_prints_x(self, mock_live_class, renderer, mock_console):
        """Test on_tool_result() prints red X for failure."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_tool_result(
            name="write",
            success=False,
            elapsed=0.5,
            arguments={},
            error="Failed",
        )

        mock_console.print.assert_called()
        call_str = str(mock_console.print.call_args)
        assert "red" in call_str
        assert "✗" in call_str

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_status_prints_dim(self, mock_live_class, renderer, mock_console):
        """Test on_status() prints message with dim styling."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_status("Thinking deeply...")

        mock_console.print.assert_called()
        call_str = str(mock_console.print.call_args)
        assert "dim" in call_str
        assert "Thinking" in call_str

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_content_updates_live_display(self, mock_live_class, renderer, mock_console):
        """Test on_content() updates the Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_content("Hello ")
        renderer.on_content("World")

        assert mock_live.update.call_count == 2
        assert renderer._content_buffer == "Hello World"

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_finalize_returns_content_and_cleans_up(self, mock_live_class, renderer, mock_console):
        """Test finalize() returns content and stops Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_content("Final content")
        result = renderer.finalize()

        assert result == "Final content"
        mock_live.stop.assert_called()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_cleanup_stops_live_if_running(self, mock_live_class, renderer, mock_console):
        """Test cleanup() stops Live display if it exists."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.cleanup()

        mock_live.stop.assert_called()
        assert renderer._live is None

    def test_cleanup_handles_no_live(self, renderer):
        """Test cleanup() handles case when Live was never started."""
        renderer.cleanup()  # Should not raise

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_thinking_content_prints_styled(self, mock_live_class, renderer, mock_console):
        """Test on_thinking_content() prints dimmed/italic styled text."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_thinking_content("Reasoning about the problem...")

        # Should print styled text with end="" (no newline)
        assert mock_console.print.call_count >= 1
        last_call = mock_console.print.call_args
        assert last_call[1].get("end") == ""

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_thinking_start_shows_indicator(self, mock_live_class, renderer, mock_console):
        """Test on_thinking_start() shows thinking indicator."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_thinking_start()

        # Should stop live display and print indicator
        mock_live.stop.assert_called()
        call_text = str(mock_console.print.call_args[0][0])
        assert "Thinking" in call_text

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_thinking_end_resumes(self, mock_live_class, renderer, mock_console):
        """Test on_thinking_end() resumes live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_thinking_end()

        # Should print newline
        mock_console.print.assert_called()


class TestStreamResponse:
    """Tests for stream_response() unified handler."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock AgentOrchestrator."""
        agent = MagicMock()
        return agent

    @pytest.fixture
    def mock_renderer(self):
        """Create a mock StreamRenderer."""
        renderer = MagicMock()
        renderer.finalize.return_value = "Response content"
        return renderer

    @pytest.mark.asyncio
    async def test_processes_tool_start_event(self, mock_agent, mock_renderer):
        """Test stream_response handles tool_start metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "tool_start": {
                        "name": "read",
                        "arguments": {"path": "/test.py"},
                    }
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_tool_start.assert_called_once_with(
            name="read",
            arguments={"path": "/test.py"},
        )

    @pytest.mark.asyncio
    async def test_processes_tool_result_event(self, mock_agent, mock_renderer):
        """Test stream_response handles tool_result metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "tool_result": {
                        "name": "write",
                        "success": True,
                        "elapsed": 0.5,
                        "arguments": {"path": "/out.py"},
                    }
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_tool_result.assert_called_once_with(
            name="write",
            success=True,
            elapsed=0.5,
            arguments={"path": "/out.py"},
            error=None,
        )

    @pytest.mark.asyncio
    async def test_processes_tool_result_with_error(self, mock_agent, mock_renderer):
        """Test stream_response handles tool_result with error."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "tool_result": {
                        "name": "bash",
                        "success": False,
                        "elapsed": 1.0,
                        "arguments": {"cmd": "rm -rf /"},
                        "error": "Permission denied",
                    }
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_tool_result.assert_called_once_with(
            name="bash",
            success=False,
            elapsed=1.0,
            arguments={"cmd": "rm -rf /"},
            error="Permission denied",
        )

    @pytest.mark.asyncio
    async def test_processes_status_event(self, mock_agent, mock_renderer):
        """Test stream_response handles status metadata."""

        async def mock_stream():
            yield StreamChunk(content="", metadata={"status": "Thinking..."})

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_status.assert_called_once_with("Thinking...")

    @pytest.mark.asyncio
    async def test_processes_file_preview_event(self, mock_agent, mock_renderer):
        """Test stream_response handles file_preview metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "path": "/test.py",
                    "file_preview": "print('hello')",
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_file_preview.assert_called_once_with(
            path="/test.py",
            content="print('hello')",
        )

    @pytest.mark.asyncio
    async def test_processes_edit_preview_event(self, mock_agent, mock_renderer):
        """Test stream_response handles edit_preview metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "path": "/file.py",
                    "edit_preview": "-old\n+new",
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_edit_preview.assert_called_once_with(
            path="/file.py",
            diff="-old\n+new",
        )

    @pytest.mark.asyncio
    async def test_processes_content_chunks(self, mock_agent, mock_renderer):
        """Test stream_response handles content chunks."""

        async def mock_stream():
            yield StreamChunk(content="Hello ", metadata=None)
            yield StreamChunk(content="World", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        assert mock_renderer.on_content.call_count == 2
        mock_renderer.on_content.assert_any_call("Hello ")
        mock_renderer.on_content.assert_any_call("World")

    @pytest.mark.asyncio
    async def test_calls_lifecycle_methods(self, mock_agent, mock_renderer):
        """Test stream_response calls start, finalize, cleanup."""

        async def mock_stream():
            yield StreamChunk(content="test", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        result = await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.start.assert_called_once()
        mock_renderer.finalize.assert_called_once()
        mock_renderer.cleanup.assert_called_once()
        assert result == "Response content"

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, mock_agent, mock_renderer):
        """Test stream_response cleans up even on exception."""

        async def mock_stream():
            yield StreamChunk(content="test", metadata=None)
            raise RuntimeError("Simulated error")

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        with pytest.raises(RuntimeError, match="Simulated error"):
            await stream_response(mock_agent, "test message", mock_renderer)

        # Cleanup should still be called
        mock_renderer.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_generator_on_success(self, mock_agent, mock_renderer):
        """Test stream_response closes async generator on success."""
        mock_gen = AsyncMock()
        mock_gen.__aiter__ = MagicMock(return_value=iter([]))
        mock_gen.aclose = AsyncMock()

        async def mock_stream():
            return
            yield  # Never reached

        gen = mock_stream()
        mock_agent.stream_chat = MagicMock(return_value=gen)

        await stream_response(mock_agent, "test message", mock_renderer)

        # Generator should be closed (aclose called in finally)

    @pytest.mark.asyncio
    async def test_handles_mixed_events(self, mock_agent, mock_renderer):
        """Test stream_response handles mixed event types correctly."""

        async def mock_stream():
            yield StreamChunk(content="", metadata={"status": "Starting..."})
            yield StreamChunk(
                content="",
                metadata={"tool_start": {"name": "read", "arguments": {}}},
            )
            yield StreamChunk(
                content="",
                metadata={
                    "tool_result": {
                        "name": "read",
                        "success": True,
                        "elapsed": 0.1,
                        "arguments": {},
                    }
                },
            )
            yield StreamChunk(content="Response ", metadata=None)
            yield StreamChunk(content="text", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_status.assert_called_once_with("Starting...")
        mock_renderer.on_tool_start.assert_called_once()
        mock_renderer.on_tool_result.assert_called_once()
        assert mock_renderer.on_content.call_count == 2

    @pytest.mark.asyncio
    async def test_filters_deepseek_thinking_markers(self, mock_agent, mock_renderer):
        """Test stream_response filters DeepSeek thinking markers."""

        async def mock_stream():
            # DeepSeek thinking output with markers
            yield StreamChunk(
                content="<｜begin▁of▁thinking｜>Let me analyze this...",
                metadata=None,
            )
            yield StreamChunk(
                content="<｜end▁of▁thinking｜>Here is the answer.",
                metadata=None,
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        # Should call on_thinking_start when entering thinking
        mock_renderer.on_thinking_start.assert_called()
        # Should call on_thinking_end when exiting thinking
        mock_renderer.on_thinking_end.assert_called()

    @pytest.mark.asyncio
    async def test_filters_qwen_thinking_markers(self, mock_agent, mock_renderer):
        """Test stream_response filters Qwen3 thinking markers."""

        async def mock_stream():
            yield StreamChunk(content="<think>Analyzing...", metadata=None)
            yield StreamChunk(content="</think>The answer is 42.", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        # Should handle thinking transitions
        mock_renderer.on_thinking_start.assert_called()
        mock_renderer.on_thinking_end.assert_called()

    @pytest.mark.asyncio
    async def test_suppress_thinking_parameter(self, mock_agent, mock_renderer):
        """Test suppress_thinking parameter hides thinking content."""

        async def mock_stream():
            yield StreamChunk(content="<think>Hidden thinking</think>Visible", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer, suppress_thinking=True)

        # Should still call lifecycle methods
        mock_renderer.start.assert_called()
        mock_renderer.finalize.assert_called()


class TestProtocolCompliance:
    """Tests for StreamRenderer protocol compliance."""

    def test_formatter_renderer_is_stream_renderer(self):
        """Test FormatterRenderer is a StreamRenderer."""
        mock_formatter = MagicMock()
        mock_console = MagicMock(spec=Console)
        renderer = FormatterRenderer(mock_formatter, mock_console)
        assert isinstance(renderer, StreamRenderer)

    def test_live_display_renderer_is_stream_renderer(self):
        """Test LiveDisplayRenderer is a StreamRenderer."""
        mock_console = MagicMock(spec=Console)
        renderer = LiveDisplayRenderer(mock_console)
        assert isinstance(renderer, StreamRenderer)

    def test_protocol_has_required_methods(self):
        """Test StreamRenderer protocol defines required methods."""
        required_methods = [
            "start",
            "pause",
            "resume",
            "on_tool_start",
            "on_tool_result",
            "on_status",
            "on_file_preview",
            "on_edit_preview",
            "on_content",
            "on_thinking_content",
            "on_thinking_start",
            "on_thinking_end",
            "finalize",
            "cleanup",
        ]
        for method in required_methods:
            assert hasattr(StreamRenderer, method)
