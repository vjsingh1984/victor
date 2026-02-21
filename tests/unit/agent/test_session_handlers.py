"""Unit tests for session management handlers.

Tests the unified session management across CLI, TUI, and one-shot modes.

Test Coverage:
- SessionConfig dataclass
- SessionMetrics dataclass
- BaseSessionHandler
- OneshotSessionHandler
- InteractiveSessionHandler
- TUISessionHandler
- Error handling and edge cases
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from victor.agent.session_manager_base import (
    SessionMode,
    SessionConfig,
    SessionMetrics,
    ISessionHandler,
    BaseSessionHandler,
    OneshotSessionHandler,
    InteractiveSessionHandler,
    TUISessionHandler,
)


class TestSessionMode:
    """Test SessionMode enum."""

    def test_interactive_value(self):
        """Test INTERACTIVE mode value."""
        assert SessionMode.INTERACTIVE.value == "interactive"

    def test_oneshot_value(self):
        """Test ONESHOT mode value."""
        assert SessionMode.ONESHOT.value == "oneshot"

    def test_tui_value(self):
        """Test TUI mode value."""
        assert SessionMode.TUI.value == "tui"


class TestSessionConfig:
    """Test SessionConfig dataclass."""

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters."""
        config = SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

        assert config.mode is SessionMode.INTERACTIVE
        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet"
        assert config.profile == "default"
        assert config.thinking is False
        assert config.vertical is None
        assert config.tool_budget is None
        assert config.max_iterations is None

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        config = SessionConfig(
            mode=SessionMode.ONESHOT,
            provider="openai",
            model="gpt-4",
            profile="custom",
            thinking=True,
            vertical="research",
            tool_budget=50,
            max_iterations=25,
            mode_name="explore",
        )

        assert config.mode is SessionMode.ONESHOT
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.thinking is True
        assert config.vertical == "research"
        assert config.tool_budget == 50
        assert config.max_iterations == 25
        assert config.mode_name == "explore"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SessionConfig(
            mode=SessionMode.TUI,
            provider="test-provider",
            model="test-model",
            profile="test-profile",
            thinking=True,
            vertical="coding",
        )

        result = config.to_dict()

        assert result["mode"] == "tui"
        assert result["provider"] == "test-provider"
        assert result["model"] == "test-model"
        assert result["thinking"] is True
        assert result["vertical"] == "coding"


class TestSessionMetrics:
    """Test SessionMetrics dataclass."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        metrics = SessionMetrics()

        assert isinstance(metrics.start_time, float)
        assert metrics.end_time is None
        assert metrics.tool_calls == 0
        assert metrics.tokens_used == 0
        assert metrics.iterations == 0
        assert metrics.success is True

    def test_duration_before_end(self):
        """Test duration calculation before session ends."""
        metrics = SessionMetrics()
        time.sleep(0.1)

        duration = metrics.duration_seconds
        assert duration >= 0.1

    def test_duration_after_end(self):
        """Test duration calculation after session ends."""
        metrics = SessionMetrics()
        time.sleep(0.1)
        metrics.end_time = time.time()

        duration = metrics.duration_seconds
        assert duration >= 0.1
        # Duration should not change after end_time is set
        time.sleep(0.05)
        assert metrics.duration_seconds == duration

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = SessionMetrics(
            tool_calls=5,
            tokens_used=100,
            iterations=3,
        )

        result = metrics.to_dict()

        assert "start_time" in result
        assert "end_time" in result
        assert result["tool_calls"] == 5
        assert result["tokens_used"] == 100
        assert result["iterations"] == 3
        assert result["success"] is True
        assert "duration_seconds" in result


class TestBaseSessionHandler:
    """Test BaseSessionHandler implementation."""

    @pytest.fixture
    def handler(self):
        """Return BaseSessionHandler instance."""
        return BaseSessionHandler()

    @pytest.fixture
    def mock_config(self):
        """Return mock SessionConfig."""
        return SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

    @pytest.mark.asyncio
    async def test_initialize_creates_agent(self, handler, mock_config):
        """Test that initialize creates an agent."""
        mock_agent = MagicMock()
        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            agent = await handler.initialize(mock_config)

        assert agent is mock_agent

    @pytest.mark.asyncio
    async def test_process_message_with_streaming(self, handler):
        """Test process_message with streaming enabled."""
        mock_agent = MagicMock()
        mock_agent.provider.supports_streaming.return_value = True

        # Mock stream_chat to return chunks
        async def mock_stream(message):
            chunks = [
                MagicMock(content="Hello ", type="content"),
                MagicMock(content="world!", type="content"),
            ]
            for chunk in chunks:
                yield chunk

        mock_agent.stream_chat = mock_stream

        result = await handler.process_message(mock_agent, "Test message", stream=True)

        assert result == "Hello world!"

    @pytest.mark.asyncio
    async def test_process_message_without_streaming(self, handler):
        """Test process_message with streaming disabled."""
        mock_agent = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False

        mock_response = MagicMock()
        mock_response.content = "Direct response"
        mock_agent.chat = AsyncMock(return_value=mock_response)

        result = await handler.process_message(mock_agent, "Test message", stream=False)

        assert result == "Direct response"
        mock_agent.chat.assert_called_once_with("Test message")

    @pytest.mark.asyncio
    async def test_cleanup_finalizes_metrics(self, handler):
        """Test that cleanup finalizes session metrics."""
        mock_agent = MagicMock()
        mock_agent.get_session_metrics.return_value = {
            "tool_calls": 10,
            "tokens": 500,
            "iterations": 5,
        }

        metrics = SessionMetrics()

        await handler.cleanup(mock_agent, metrics)

        assert metrics.end_time is not None
        assert metrics.tool_calls == 10
        assert metrics.tokens_used == 500
        assert metrics.iterations == 5

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_metrics(self, handler):
        """Test cleanup when agent doesn't have session metrics."""
        mock_agent = MagicMock()
        mock_agent.get_session_metrics.return_value = None

        metrics = SessionMetrics()

        await handler.cleanup(mock_agent, metrics)

        # Should not crash, metrics should remain at defaults
        assert metrics.end_time is not None
        assert metrics.tool_calls == 0
        assert metrics.tokens_used == 0

    @pytest.mark.asyncio
    async def test_initialize_applies_overrides(self, handler):
        """Test that initialize applies budget and iteration overrides."""
        mock_config = SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
            tool_budget=100,
            max_iterations=50,
            mode_name="plan",
        )

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            await handler.initialize(mock_config)

        # Verify overrides were applied
        mock_agent.unified_tracker.set_tool_budget.assert_called_once_with(100, user_override=True)
        mock_agent.unified_tracker.set_max_iterations.assert_called_once_with(
            50, user_override=True
        )


class TestOneshotSessionHandler:
    """Test OneshotSessionHandler."""

    @pytest.fixture
    def handler(self):
        """Return OneshotSessionHandler instance."""
        return OneshotSessionHandler()

    @pytest.fixture
    def mock_config(self):
        """Return mock SessionConfig for oneshot."""
        return SessionConfig(
            mode=SessionMode.ONESHOT,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

    @pytest.mark.asyncio
    async def test_execute_successful_session(self, handler, mock_config):
        """Test successful one-shot session execution."""
        mock_agent = MagicMock()
        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        # Mock process_message
        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch.object(handler, "process_message", return_value="Response"):
                metrics = await handler.execute(mock_config, "Test message")

        assert metrics.success is True
        assert metrics.end_time is not None

    @pytest.mark.asyncio
    async def test_execute_handles_errors(self, handler, mock_config):
        """Test that execute handles errors gracefully."""
        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(side_effect=Exception("Test error"))

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with pytest.raises(Exception, match="Test error"):
                await handler.execute(mock_config, "Test message")

    @pytest.mark.asyncio
    async def test_execute_cleans_up_on_error(self, handler, mock_config):
        """Test that execute cleans up even when error occurs."""
        mock_agent = MagicMock()
        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch.object(handler, "process_message", side_effect=Exception("Process error")):
                with patch.object(handler, "cleanup") as mock_cleanup:
                    with pytest.raises(Exception):
                        await handler.execute(mock_config, "Test message")

                    # Cleanup should still be called
                    mock_cleanup.assert_called_once()


class TestInteractiveSessionHandler:
    """Test InteractiveSessionHandler."""

    @pytest.fixture
    def handler(self):
        """Return InteractiveSessionHandler with callback."""
        callback = AsyncMock()
        return InteractiveSessionHandler(on_message_callback=callback)

    @pytest.fixture
    def mock_config(self):
        """Return mock SessionConfig for interactive."""
        return SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

    def test_init_with_callback(self):
        """Test initialization with callback."""
        callback = MagicMock()
        handler = InteractiveSessionHandler(on_message_callback=callback)

        assert handler._on_message is callback

    def test_init_without_callback(self):
        """Test initialization without callback."""
        handler = InteractiveSessionHandler()

        assert handler._on_message is None

    @pytest.mark.asyncio
    async def test_get_user_input_raises_not_implemented(self, handler):
        """Test that _get_user_input raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await handler._get_user_input()


class TestTUISessionHandler:
    """Test TUISessionHandler."""

    @pytest.fixture
    def handler(self):
        """Return TUISessionHandler instance."""
        return TUISessionHandler()

    @pytest.fixture
    def mock_config(self):
        """Return mock SessionConfig for TUI."""
        return SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_start_tui_with_pre_configured_agent(self, handler, mock_config):
        """Test starting TUI with pre-configured agent."""
        mock_agent = MagicMock()
        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
            await handler.start_tui(mock_config, agent=mock_agent)

        mock_tui.run_async.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_start_tui_creates_agent(self, handler, mock_config):
        """Test starting TUI creates agent if not provided."""
        mock_agent = MagicMock()
        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(mock_config, agent=None)

        # Verify agent was created
        mock_shim.create_orchestrator.assert_called_once()
        # Verify TUI was started
        mock_tui.run_async.assert_called_once()


class TestISessionHandlerInterface:
    """Test ISessionHandler as an abstract interface."""

    def test_cannot_instantiate_abstract_handler(self):
        """Test that abstract handler cannot be instantiated."""
        with pytest.raises(TypeError):
            ISessionHandler()

    def test_handler_requires_initialize(self):
        """Test that handler requires initialize method."""
        assert hasattr(ISessionHandler, "initialize")
        assert hasattr(ISessionHandler, "process_message")
        assert hasattr(ISessionHandler, "cleanup")


class TestSessionHandlerErrorHandling:
    """Test error handling across all handler types."""

    @pytest.mark.asyncio
    async def test_base_handler_cleanup_without_agent(self):
        """Test cleanup without agent doesn't crash."""
        handler = BaseSessionHandler()
        metrics = SessionMetrics()

        # Should not crash
        await handler.cleanup(None, metrics)

        assert metrics.end_time is not None

    @pytest.mark.asyncio
    async def test_process_message_with_exception_in_stream(self):
        """Test process_message handles exception in streaming."""
        handler = BaseSessionHandler()
        mock_agent = MagicMock()
        mock_agent.provider.supports_streaming.return_value = True

        # Mock stream_chat that raises exception
        async def mock_stream_error(message):
            yield MagicMock(content="Start", type="content")
            raise Exception("Stream error")

        mock_agent.stream_chat = mock_stream_error

        with pytest.raises(Exception, match="Stream error"):
            await handler.process_message(mock_agent, "Test", stream=True)

    @pytest.mark.asyncio
    async def test_initialize_with_invalid_mode_switch(self):
        """Test initialize handles invalid mode switch gracefully."""
        from victor.config.settings import Settings

        settings = MagicMock(spec=Settings)

        config = SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
            mode_name="invalid_mode",
        )

        handler = BaseSessionHandler()
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.agent.mode_controller.get_mode_controller") as mock_get_controller:
                mock_controller = MagicMock()
                mock_controller.switch_mode.side_effect = Exception("Invalid mode")
                mock_get_controller.return_value = mock_controller

                # Should not crash despite mode switch failure
                agent = await handler.initialize(config)

                assert agent is mock_agent
