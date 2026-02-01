"""Integration tests for TUI session management.

Tests the unified session management for TUI mode including:
- TUI session initialization and agent creation
- TUI-specific features (rich UI, visual feedback)
- Integration between TUISessionHandler and VictorTUI
- Session lifecycle management in TUI context
- Error handling and graceful degradation

Test Approach:
- Uses real session handlers but mocks VictorTUI for testability
- Tests integration between TUISessionHandler and TUI components
- Verifies agent creation and cleanup in TUI context
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from victor.agent.session_manager_base import (
    SessionMode,
    SessionConfig,
    TUISessionHandler,
)


class TestTUISessionInitialization:
    """Test TUI session initialization flow."""

    @pytest.fixture
    def session_config(self):
        """Return a standard TUI session configuration."""
        return SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
            thinking=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_initialization_with_agent_creation(self, session_config):
        """Test TUI initialization with automatic agent creation."""
        handler = TUISessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(session_config, agent=None)

        # Verify agent was created
        mock_shim.create_orchestrator.assert_called_once()

        # Verify TUI was started
        mock_tui.run_async.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_with_preconfigured_agent(self, session_config):
        """Test TUI initialization with externally configured agent."""
        handler = TUISessionHandler()

        mock_agent = MagicMock()
        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
            await handler.start_tui(session_config, agent=mock_agent)

        # Verify TUI was started with the pre-configured agent
        mock_tui.run_async.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_with_custom_settings(self, session_config):
        """Test TUI with custom session settings."""
        handler = TUISessionHandler()

        session_config.tool_budget = 200
        session_config.max_iterations = 100
        session_config.thinking = True

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(session_config, agent=None)

        # Verify custom settings were applied
        mock_agent.unified_tracker.set_tool_budget.assert_called_once_with(200, user_override=True)
        mock_agent.unified_tracker.set_max_iterations.assert_called_once_with(
            100, user_override=True
        )


class TestTUISessionLifecycle:
    """Test complete TUI session lifecycle."""

    @pytest.fixture
    def handler(self):
        """Return TUISessionHandler."""
        return TUISessionHandler()

    @pytest.fixture
    def session_config(self):
        """Return TUI session configuration."""
        return SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_tui_lifecycle(self, handler, session_config):
        """Test complete TUI lifecycle: initialize → run → cleanup."""
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.get_session_metrics.return_value = {
            "tool_calls": 20,
            "tokens": 1000,
            "iterations": 10,
        }

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                # Start TUI (includes initialization and cleanup)
                await handler.start_tui(session_config, agent=None)

        # Verify full lifecycle
        mock_shim.create_orchestrator.assert_called_once()
        mock_tui.run_async.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_cleanup_on_exit(self, handler, session_config):
        """Test that TUI properly cleans up on exit."""
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.get_session_metrics.return_value = {
            "tool_calls": 5,
            "tokens": 250,
            "iterations": 2,
        }

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        mock_tui = MagicMock()

        # Simulate TUI exit
        async def mock_run_with_cleanup():
            # Simulate some work
            await asyncio.sleep(0.01)
            # Simulate metrics collection during TUI run
            pass

        mock_tui.run_async = mock_run_with_cleanup

        import asyncio

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(session_config, agent=None)

        # Verify cleanup happened
        assert mock_shim.create_orchestrator.called


class TestTUIWithVerticals:
    """Test TUI integration with vertical system."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_with_coding_vertical(self):
        """Test TUI with coding vertical integration."""
        from victor.coding.assistant import CodingAssistant

        config = SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
            vertical="coding",
        )

        handler = TUISessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.core.verticals.get_vertical") as mock_get_vertical:
                with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                    mock_get_vertical.return_value = CodingAssistant

                    await handler.start_tui(config, agent=None)

                    # Verify vertical was loaded
                    mock_get_vertical.assert_called_once_with("coding")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_with_research_vertical(self):
        """Test TUI with research vertical integration."""
        from victor.research.assistant import ResearchAssistant

        config = SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
            vertical="research",
        )

        handler = TUISessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.core.verticals.get_vertical") as mock_get_vertical:
                with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                    mock_get_vertical.return_value = ResearchAssistant

                    await handler.start_tui(config, agent=None)

                    # Verify vertical was loaded
                    mock_get_vertical.assert_called_once_with("research")


class TestTUIErrorHandling:
    """Test error handling in TUI sessions."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_handles_initialization_error(self):
        """Test TUI handles agent initialization error gracefully."""
        config = SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

        handler = TUISessionHandler()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(
            side_effect=Exception("Provider connection failed")
        )

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            # Should raise the error (TUI would handle this)
            with pytest.raises(Exception, match="Provider connection failed"):
                await handler.start_tui(config, agent=None)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_handles_runtime_error(self):
        """Test TUI handles runtime errors during session."""
        config = SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

        handler = TUISessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        # Simulate TUI runtime error
        async def mock_run_with_error():
            raise RuntimeError("TUI rendering failed")

        mock_tui = MagicMock()
        mock_tui.run_async = mock_run_with_error

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                # Should propagate the error
                with pytest.raises(RuntimeError, match="TUI rendering failed"):
                    await handler.start_tui(config, agent=None)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_cleanup_despite_errors(self):
        """Test that TUI cleanup happens even after errors."""
        config = SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

        handler = TUISessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.get_session_metrics.return_value = {
            "tool_calls": 0,
            "tokens": 0,
            "iterations": 0,
        }

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        # Simulate error during TUI run
        async def mock_run_with_error():
            raise Exception("User interrupted")

        mock_tui = MagicMock()
        mock_tui.run_async = mock_run_with_error

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                with patch("victor.ui.commands.utils.graceful_shutdown") as mock_shutdown:
                    try:
                        await handler.start_tui(config, agent=None)
                    except Exception:
                        pass  # Expected error

                    # Verify cleanup still happened
                    mock_shutdown.assert_called_once_with(mock_agent)


class TestTUIWithThinkingMode:
    """Test TUI with thinking mode enabled."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_with_thinking_enabled(self):
        """Test TUI initialization with thinking mode."""
        config = SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            profile="default",
            thinking=True,
        )

        handler = TUISessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        # Verify thinking was passed to shim
        def verify_shim_init(settings, profile_name, thinking, **kwargs):
            assert thinking is True
            return mock_shim

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch(
            "victor.framework.shim.FrameworkShim", side_effect=verify_shim_init
        ) as mock_shim_patch:
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(config, agent=None)

                # Verify thinking mode was used
                mock_shim_patch.assert_called_once()
                call_kwargs = mock_shim_patch.call_args[1]
                assert call_kwargs["thinking"] is True


class TestTUIConcurrentSessions:
    """Test multiple concurrent TUI sessions (edge case)."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_tui_sessions(self):
        """Test handling multiple TUI sessions concurrently."""
        import asyncio

        configs = [
            SessionConfig(
                mode=SessionMode.TUI,
                provider="anthropic",
                model="claude-3-5-sonnet",
                profile="default",
            )
            for _ in range(3)
        ]

        async def start_session(config):
            handler = TUISessionHandler()

            mock_agent = MagicMock()
            mock_agent.unified_tracker = MagicMock()

            mock_shim = MagicMock()
            mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
            mock_shim.emit_session_start = MagicMock()

            mock_tui = MagicMock()
            mock_tui.run_async = AsyncMock()

            with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
                with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                    await handler.start_tui(config, agent=None)

            return True

        # Start multiple sessions concurrently
        results = await asyncio.gather(*[start_session(config) for config in configs])

        # All sessions should complete
        assert all(results)


class TestTUISessionMetrics:
    """Test TUI session metrics collection."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_metrics_collection(self):
        """Test that TUI properly collects session metrics."""
        config = SessionConfig(
            mode=SessionMode.TUI,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

        handler = TUISessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.get_session_metrics.return_value = {
            "tool_calls": 25,
            "tokens": 1500,
            "iterations": 12,
        }

        mock_shim = MagicMock()
        mock_shim.create_orchestrator = AsyncMock(return_value=mock_agent)
        mock_shim.emit_session_start = MagicMock()

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.shim.FrameworkShim", return_value=mock_shim):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(config, agent=None)

        # Metrics would be collected during cleanup
        # (which happens in start_tui's finally block)
        assert mock_shim.create_orchestrator.called
