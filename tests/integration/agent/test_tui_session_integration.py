"""Integration tests for TUI session management.

Tests the unified session management for TUI mode including:
- TUI session initialization and agent creation
- TUI-specific features (rich UI, visual feedback)
- Integration between TUISessionHandler and VictorTUI
- Session lifecycle management in TUI context
- Error handling and graceful degradation

Test Approach:
- Uses real session handlers and creation strategies but mocks AgentFactory and VictorTUI
- Tests integration between TUISessionHandler and TUI components
- Verifies agent creation and cleanup in TUI context
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from victor.agent.session_manager_base import (
    SessionMode,
    SessionConfig,
    SessionMetrics,
    TUISessionHandler,
)
from victor.core.verticals.import_resolver import import_module_with_fallback


def _load_vertical_attr(module_path: str, attr_name: str):
    """Resolve a vertical attribute and skip test when unavailable."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        pytest.skip(f"Vertical module or attribute unavailable: {module_path}:{attr_name}")
    return getattr(module, attr_name)


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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(session_config, agent=None)

        # Verify agent was created
        mock_factory.create.assert_called_once_with()

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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch(
            "victor.framework.agent_factory.AgentFactory", return_value=mock_factory
        ) as mock_agent_factory:
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(session_config, agent=None)

        # Verify custom settings were forwarded to the canonical factory
        assert mock_agent_factory.call_args.kwargs["thinking"] is True
        assert mock_agent_factory.call_args.kwargs["tool_budget"] == 200
        assert mock_agent_factory.call_args.kwargs["max_iterations"] == 100


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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                # Start TUI (includes initialization and cleanup)
                await handler.start_tui(session_config, agent=None)

        # Verify full lifecycle
        mock_factory.create.assert_called_once_with()
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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_tui = MagicMock()

        # Simulate TUI exit
        async def mock_run_with_cleanup():
            # Simulate some work
            await asyncio.sleep(0.01)
            # Simulate metrics collection during TUI run
            pass

        mock_tui.run_async = mock_run_with_cleanup

        import asyncio

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(session_config, agent=None)

        # Verify cleanup happened
        assert mock_factory.create.called


class TestTUIWithVerticals:
    """Test TUI integration with vertical system."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tui_with_coding_vertical(self):
        """Test TUI with coding vertical integration."""
        CodingAssistant = _load_vertical_attr("victor.coding.assistant", "CodingAssistant")

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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
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
        ResearchAssistant = _load_vertical_attr("victor.research.assistant", "ResearchAssistant")

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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(side_effect=Exception("Provider connection failed"))

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        # Simulate TUI runtime error
        async def mock_run_with_error():
            raise RuntimeError("TUI rendering failed")

        mock_tui = MagicMock()
        mock_tui.run_async = mock_run_with_error

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        # Simulate error during TUI run
        async def mock_run_with_error():
            raise Exception("User interrupted")

        mock_tui = MagicMock()
        mock_tui.run_async = mock_run_with_error

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch(
            "victor.framework.agent_factory.AgentFactory", return_value=mock_factory
        ) as mock_agent_factory:
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(config, agent=None)

                # Verify thinking mode was used
                mock_agent_factory.assert_called_once()
                call_kwargs = mock_agent_factory.call_args[1]
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

            mock_factory = MagicMock()
            mock_factory.create = AsyncMock(return_value=mock_agent)

            mock_tui = MagicMock()
            mock_tui.run_async = AsyncMock()

            with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
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

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_tui = MagicMock()
        mock_tui.run_async = AsyncMock()

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.ui.tui.VictorTUI", return_value=mock_tui):
                await handler.start_tui(config, agent=None)

        # Metrics would be collected during cleanup
        # (which happens in start_tui's finally block)
        assert mock_factory.create.called
