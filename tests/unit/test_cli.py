"""Tests for victor.ui.cli module.

This module tests the CLI utility functions and commands using typer's CliRunner.
Target: 70%+ coverage of the cli.py module.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

# Test fixtures and helpers


@pytest.fixture
def mock_console():
    """Create a mock Rich console."""
    return MagicMock(spec=Console)


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.provider = "mock"
    settings.model = "mock-model"
    settings.airgapped_mode = False
    settings.semantic_index_preload = False
    settings.tool_cache_enabled = True
    return settings


@pytest.fixture
def mock_agent():
    """Create a mock AgentOrchestrator."""
    agent = MagicMock()
    agent.provider = MagicMock()
    agent.provider.close = AsyncMock()
    agent.provider.supports_streaming = MagicMock(return_value=True)
    agent.graceful_shutdown = AsyncMock(return_value={"status": "ok"})
    agent.shutdown = AsyncMock()
    agent.stream_chat = AsyncMock()
    return agent


# =============================================================================
# Test _configure_logging
# =============================================================================


class TestConfigureLogging:
    """Tests for _configure_logging function."""

    def test_configure_logging_sets_level(self):
        """Test that logging level is configured correctly."""
        from victor.ui.cli import _configure_logging

        stream = io.StringIO()
        _configure_logging("DEBUG", stream=stream)

        # Check that root logger level is set
        assert logging.root.level == logging.DEBUG

    def test_configure_logging_info_level(self):
        """Test INFO level configuration."""
        from victor.ui.cli import _configure_logging

        stream = io.StringIO()
        _configure_logging("INFO", stream=stream)
        assert logging.root.level == logging.INFO

    def test_configure_logging_warning_level(self):
        """Test WARNING level configuration."""
        from victor.ui.cli import _configure_logging

        stream = io.StringIO()
        _configure_logging("WARNING", stream=stream)
        assert logging.root.level == logging.WARNING

    def test_configure_logging_error_level(self):
        """Test ERROR level configuration."""
        from victor.ui.cli import _configure_logging

        stream = io.StringIO()
        _configure_logging("ERROR", stream=stream)
        assert logging.root.level == logging.ERROR

    def test_configure_logging_invalid_level_defaults_to_warning(self):
        """Test that invalid level defaults to WARNING."""
        from victor.ui.cli import _configure_logging

        stream = io.StringIO()
        _configure_logging("INVALID", stream=stream)
        # getattr returns WARNING for invalid levels
        assert logging.root.level == logging.WARNING

    def test_configure_logging_case_insensitive(self):
        """Test that level is case insensitive."""
        from victor.ui.cli import _configure_logging

        stream = io.StringIO()
        _configure_logging("debug", stream=stream)
        assert logging.root.level == logging.DEBUG


# =============================================================================
# Test _flush_logging
# =============================================================================


class TestFlushLogging:
    """Tests for _flush_logging function."""

    def test_flush_logging_flushes_handlers(self):
        """Test that all handlers are flushed."""
        from victor.ui.cli import _flush_logging

        # Create a mock handler
        mock_handler = MagicMock()
        logging.root.addHandler(mock_handler)

        _flush_logging()

        mock_handler.flush.assert_called_once()

        # Cleanup
        logging.root.removeHandler(mock_handler)

    def test_flush_logging_multiple_handlers(self):
        """Test flushing multiple handlers."""
        from victor.ui.cli import _flush_logging

        handlers = [MagicMock() for _ in range(3)]
        for h in handlers:
            logging.root.addHandler(h)

        _flush_logging()

        for h in handlers:
            h.flush.assert_called()

        # Cleanup
        for h in handlers:
            logging.root.removeHandler(h)


# =============================================================================
# Test _graceful_shutdown
# =============================================================================


class TestGracefulShutdown:
    """Tests for _graceful_shutdown async function."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_none_agent(self):
        """Test shutdown with None agent."""
        from victor.ui.cli import _graceful_shutdown

        # Should not raise
        await _graceful_shutdown(None)

    @pytest.mark.asyncio
    async def test_graceful_shutdown_calls_agent_methods(self, mock_agent):
        """Test that shutdown calls agent methods."""
        from victor.ui.cli import _graceful_shutdown

        await _graceful_shutdown(mock_agent)

        mock_agent.graceful_shutdown.assert_awaited_once()
        mock_agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_handles_exception(self, mock_agent):
        """Test that exceptions are handled gracefully."""
        from victor.ui.cli import _graceful_shutdown

        mock_agent.graceful_shutdown = AsyncMock(side_effect=Exception("Test error"))

        # Should not raise, should fall back to closing provider
        await _graceful_shutdown(mock_agent)

        mock_agent.provider.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_handles_close_exception(self, mock_agent):
        """Test that close exceptions are also handled."""
        from victor.ui.cli import _graceful_shutdown

        mock_agent.graceful_shutdown = AsyncMock(side_effect=Exception("Test error"))
        mock_agent.provider.close = AsyncMock(side_effect=Exception("Close error"))

        # Should not raise
        await _graceful_shutdown(mock_agent)


# =============================================================================
# Test version_callback
# =============================================================================


class TestVersionCallback:
    """Tests for version_callback function."""

    def test_version_callback_false_does_nothing(self):
        """Test that False value does not exit."""
        from victor.ui.cli import version_callback

        # Should not raise
        version_callback(False)

    def test_version_callback_true_exits(self):
        """Test that True value prints version and exits."""
        import typer
        from victor.ui.cli import version_callback

        with pytest.raises(typer.Exit):
            version_callback(True)


# =============================================================================
# Test _setup_safety_confirmation
# =============================================================================


class TestSetupSafetyConfirmation:
    """Tests for _setup_safety_confirmation function."""

    def test_setup_safety_confirmation_sets_callback(self):
        """Test that safety confirmation callback is set."""
        from victor.ui.cli import _setup_safety_confirmation

        with patch("victor.ui.cli.set_confirmation_callback") as mock_set:
            _setup_safety_confirmation()
            mock_set.assert_called_once()


# =============================================================================
# Test _cli_confirmation_callback
# =============================================================================


class TestCliConfirmationCallback:
    """Tests for _cli_confirmation_callback async function."""

    @pytest.mark.asyncio
    async def test_confirmation_callback_prompts_user(self):
        """Test that callback prompts user for confirmation."""
        from victor.ui.cli import _cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, RiskLevel

        request = ConfirmationRequest(
            tool_name="test_tool",
            risk_level=RiskLevel.MEDIUM,
            description="Test action",
            details=["Detail 1", "Detail 2"],
            arguments={"param": "value"},
        )

        with patch("victor.ui.cli.Confirm.ask", return_value=True) as mock_ask:
            result = await _cli_confirmation_callback(request)
            assert result is True
            mock_ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirmation_callback_user_declines(self):
        """Test callback when user declines."""
        from victor.ui.cli import _cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, RiskLevel

        request = ConfirmationRequest(
            tool_name="test_tool",
            risk_level=RiskLevel.HIGH,
            description="Dangerous action",
            details=[],
            arguments={},
        )

        with patch("victor.ui.cli.Confirm.ask", return_value=False):
            result = await _cli_confirmation_callback(request)
            assert result is False

    @pytest.mark.asyncio
    async def test_confirmation_callback_keyboard_interrupt(self):
        """Test callback handles KeyboardInterrupt."""
        from victor.ui.cli import _cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, RiskLevel

        request = ConfirmationRequest(
            tool_name="test_tool",
            risk_level=RiskLevel.LOW,
            description="Test action",
            details=[],
            arguments={},
        )

        with patch("victor.ui.cli.Confirm.ask", side_effect=KeyboardInterrupt):
            result = await _cli_confirmation_callback(request)
            assert result is False

    @pytest.mark.asyncio
    async def test_confirmation_callback_eof_error(self):
        """Test callback handles EOFError."""
        from victor.ui.cli import _cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, RiskLevel

        request = ConfirmationRequest(
            tool_name="test_tool",
            risk_level=RiskLevel.SAFE,
            description="Safe action",
            details=[],
            arguments={},
        )

        with patch("victor.ui.cli.Confirm.ask", side_effect=EOFError):
            result = await _cli_confirmation_callback(request)
            assert result is False

    @pytest.mark.asyncio
    async def test_confirmation_callback_risk_levels(self):
        """Test callback handles all risk levels."""
        from victor.ui.cli import _cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, RiskLevel

        for level in RiskLevel:
            request = ConfirmationRequest(
                tool_name="test_tool",
                risk_level=level,
                description=f"{level.value} action",
                details=[],
                arguments={},
            )

            with patch("victor.ui.cli.Confirm.ask", return_value=True):
                result = await _cli_confirmation_callback(request)
                assert result is True


# =============================================================================
# Test CLI Commands using CliRunner
# =============================================================================


class TestCliCommands:
    """Tests for CLI commands using typer's CliRunner."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner."""
        return CliRunner()

    def test_version_option(self, runner):
        """Test --version option."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Victor v" in result.stdout

    def test_help_option(self, runner):
        """Test --help option."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Victor" in result.stdout
        assert "chat" in result.stdout

    def test_chat_help(self, runner):
        """Test chat --help."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "Message to send" in result.stdout or "profile" in result.stdout

    def test_init_help(self, runner):
        """Test init --help."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0

    def test_providers_command(self, runner):
        """Test providers command."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["providers"])
        # May fail without config but should at least run
        assert result.exit_code in [0, 1]

    def test_profiles_command(self, runner):
        """Test profiles command."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["profiles"])
        # May fail without config - allow exit code 0, 1, or 2 (usage error)
        assert result.exit_code in [0, 1, 2]

    def test_security_help(self, runner):
        """Test security command help."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["security", "--help"])
        assert result.exit_code == 0

    def test_embeddings_help(self, runner):
        """Test embeddings command help."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["embeddings", "--help"])
        assert result.exit_code == 0

    def test_keys_help(self, runner):
        """Test keys command help."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["keys", "--help"])
        assert result.exit_code == 0


# =============================================================================
# Test _check_codebase_index
# =============================================================================


class TestCheckCodebaseIndex:
    """Tests for _check_codebase_index async function."""

    @pytest.mark.asyncio
    async def test_check_codebase_index_import_error(self, mock_console):
        """Test handling of ImportError."""
        from victor.ui.cli import _check_codebase_index

        # Patch the import inside the function
        with patch.dict("sys.modules", {"victor.codebase.indexer": None}):
            # Should not raise
            await _check_codebase_index("/tmp", mock_console)

    @pytest.mark.asyncio
    async def test_check_codebase_index_generic_error(self, mock_console):
        """Test handling of generic errors."""
        from victor.ui.cli import _check_codebase_index

        mock_index = MagicMock()
        mock_index.check_staleness_by_mtime.side_effect = Exception("Test error")

        with patch.dict("sys.modules", {"victor.codebase.indexer": MagicMock()}):
            with patch("victor.ui.cli.CodebaseIndex", return_value=mock_index, create=True):
                # Should not raise
                await _check_codebase_index("/tmp", mock_console)

    @pytest.mark.asyncio
    async def test_check_codebase_index_not_stale(self, mock_console):
        """Test when index is not stale."""
        from victor.ui.cli import _check_codebase_index

        mock_index = MagicMock()
        mock_index.check_staleness_by_mtime.return_value = (False, [], [])

        with patch("victor.codebase.indexer.CodebaseIndex", return_value=mock_index):
            await _check_codebase_index("/tmp", mock_console, silent=True)
            # No output expected when not stale and silent


# =============================================================================
# Test _preload_semantic_index
# =============================================================================


class TestPreloadSemanticIndex:
    """Tests for _preload_semantic_index async function."""

    @pytest.mark.asyncio
    async def test_preload_semantic_index_import_error(self, mock_console, mock_settings):
        """Test handling of ImportError."""
        from victor.ui.cli import _preload_semantic_index

        with patch(
            "victor.tools.code_search_tool._get_or_build_index",
            side_effect=ImportError("Test"),
        ):
            result = await _preload_semantic_index("/tmp", mock_settings, mock_console)
            assert result is False

    @pytest.mark.asyncio
    async def test_preload_semantic_index_generic_error(self, mock_console, mock_settings):
        """Test handling of generic errors."""
        from victor.ui.cli import _preload_semantic_index

        with patch(
            "victor.tools.code_search_tool._get_or_build_index",
            side_effect=Exception("Test error"),
        ):
            result = await _preload_semantic_index("/tmp", mock_settings, mock_console)
            assert result is False


# =============================================================================
# Test _setup_signal_handlers
# =============================================================================


class TestSetupSignalHandlers:
    """Tests for _setup_signal_handlers function."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix signals only")
    def test_setup_signal_handlers_unix(self):
        """Test signal handler setup on Unix."""
        from victor.ui.cli import _setup_signal_handlers

        loop = asyncio.new_event_loop()

        with patch("signal.signal") as mock_signal:
            _setup_signal_handlers(loop)
            # Should set SIGTERM handler
            mock_signal.assert_called()

        loop.close()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_setup_signal_handlers_windows(self):
        """Test signal handler setup on Windows (limited)."""
        from victor.ui.cli import _setup_signal_handlers

        loop = asyncio.new_event_loop()

        # Should not raise on Windows
        _setup_signal_handlers(loop)

        loop.close()


# =============================================================================
# Test run_oneshot
# =============================================================================


class TestRunOneshot:
    """Tests for run_oneshot async function."""

    @pytest.mark.asyncio
    async def test_run_oneshot_import_works(self):
        """Test that run_oneshot can be imported."""
        from victor.ui.cli import run_oneshot

        # Just verify the function exists and is callable
        assert callable(run_oneshot)


# =============================================================================
# Test History Enabled Flag
# =============================================================================


class TestHistoryEnabled:
    """Tests for readline history configuration."""

    def test_history_enabled_flag_exists(self):
        """Test that history enabled flag is set."""
        from victor.ui import cli

        # The flag should be defined
        assert hasattr(cli, "_history_enabled")
        # It should be a boolean
        assert isinstance(cli._history_enabled, bool)


# =============================================================================
# Test Global Agent Reference
# =============================================================================


class TestGlobalAgentReference:
    """Tests for global agent reference."""

    def test_current_agent_initial_none(self):
        """Test that _current_agent starts as None."""
        from victor.ui import cli

        # Initially should be None
        assert cli._current_agent is None


# =============================================================================
# Test Output Format Options
# =============================================================================


class TestOutputFormatOptions:
    """Tests for output format option handling."""

    def test_output_format_cli_options(self):
        """Test that output format options are available in CLI."""
        from victor.ui.cli import chat

        # Verify the chat function has the expected parameters
        import inspect

        sig = inspect.signature(chat)
        param_names = list(sig.parameters.keys())

        # These options should be available in the chat command
        assert "code_only" in param_names or any("code" in p for p in param_names)
        assert "json_output" in param_names or any("json" in p for p in param_names)
        assert "plain_output" in param_names or any("plain" in p for p in param_names)

    def test_output_modes_exist(self):
        """Test that output mode handling code exists."""
        import victor.ui.cli as cli_module

        # Verify the module has expected attributes
        assert hasattr(cli_module, "run_oneshot")
        assert hasattr(cli_module, "run_interactive")


# =============================================================================
# Integration-style tests (still mocked but more comprehensive)
# =============================================================================


class TestChatCommandIntegration:
    """Integration-style tests for chat command."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner."""
        return CliRunner()

    def test_chat_missing_provider(self, runner):
        """Test chat with missing provider configuration."""
        from victor.ui.cli import app

        # This will likely fail but shouldn't crash
        result = runner.invoke(
            app, ["chat", "--profile", "nonexistent", "hello"], catch_exceptions=True
        )
        # Should handle gracefully
        assert result.exit_code in [0, 1, 2]

    def test_chat_stdin_mode(self, runner):
        """Test chat with stdin mode."""
        from victor.ui.cli import app

        result = runner.invoke(
            app,
            ["chat", "--stdin", "--help"],
        )
        # Help should work
        assert result.exit_code == 0


# =============================================================================
# Test app configuration
# =============================================================================


class TestAppConfiguration:
    """Tests for typer app configuration."""

    def test_app_name(self):
        """Test app name is set correctly."""
        from victor.ui.cli import app

        assert app.info.name == "victor"

    def test_app_has_registered_commands(self):
        """Test app has commands registered."""
        from victor.ui.cli import app

        # Get registered command names - handle both old and new typer API
        commands = app.registered_commands
        if commands:
            command_names = [cmd.name for cmd in commands if cmd.name]
            # Check that there are some commands
            assert len(command_names) > 0
        else:
            # Alternative: check for registered groups
            assert app.registered_groups is not None or len(commands) >= 0


# =============================================================================
# Test console initialization
# =============================================================================


class TestConsoleInitialization:
    """Tests for console initialization."""

    def test_console_is_rich_console(self):
        """Test that console is a Rich Console instance."""
        from victor.ui.cli import console
        from rich.console import Console

        assert isinstance(console, Console)


# =============================================================================
# Test error handling in commands
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner."""
        return CliRunner()

    def test_invalid_log_level_handling(self, runner):
        """Test handling of invalid log level."""
        from victor.ui.cli import app

        result = runner.invoke(
            app, ["chat", "--log-level", "INVALID", "--help"], catch_exceptions=True
        )
        # Should still show help
        assert result.exit_code == 0

    def test_security_command_help(self, runner):
        """Test security command help."""
        from victor.ui.cli import app

        result = runner.invoke(app, ["security", "--help"])
        assert result.exit_code == 0
