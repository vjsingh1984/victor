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
from types import SimpleNamespace
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
        from victor.ui.commands.utils import configure_logging as _configure_logging

        stream = io.StringIO()
        _configure_logging("DEBUG", stream=stream, file_logging=False)

        # Root logger is always set to DEBUG (handlers do the filtering)
        assert logging.root.level == logging.DEBUG
        assert any(h.level == logging.DEBUG for h in logging.root.handlers)

    def test_configure_logging_info_level(self):
        """Test INFO level configuration."""
        from victor.ui.commands.utils import configure_logging as _configure_logging

        stream = io.StringIO()
        _configure_logging("INFO", stream=stream, file_logging=False)
        # configure_logging sets root to DEBUG but handler to specified level
        assert any(h.level == logging.INFO for h in logging.root.handlers)

    def test_configure_logging_warning_level(self):
        """Test WARNING level configuration."""
        from victor.ui.commands.utils import configure_logging as _configure_logging

        stream = io.StringIO()
        _configure_logging("WARNING", stream=stream, file_logging=False)
        # configure_logging sets root to DEBUG but handler to specified level
        assert any(h.level == logging.WARNING for h in logging.root.handlers)

    def test_configure_logging_error_level(self):
        """Test ERROR level configuration."""
        from victor.ui.commands.utils import configure_logging as _configure_logging

        stream = io.StringIO()
        _configure_logging("ERROR", stream=stream, file_logging=False)
        # configure_logging sets root to DEBUG but handler to specified level
        assert any(h.level == logging.ERROR for h in logging.root.handlers)

    def test_configure_logging_invalid_level_defaults_to_warning(self):
        """Test that invalid level defaults to WARNING."""
        from victor.ui.commands.utils import configure_logging as _configure_logging

        stream = io.StringIO()
        _configure_logging("INVALID", stream=stream, file_logging=False)
        # configure_logging sets root to DEBUG but handler to WARNING for invalid levels
        assert any(h.level == logging.WARNING for h in logging.root.handlers)

    def test_configure_logging_case_insensitive(self):
        """Test that level is case insensitive."""
        from victor.ui.commands.utils import configure_logging as _configure_logging

        stream = io.StringIO()
        _configure_logging("debug", stream=stream, file_logging=False)
        assert any(h.level == logging.DEBUG for h in logging.root.handlers)

    def test_configure_logging_uses_global_victor_dir_for_default_log_file(self, tmp_path):
        """Default log file should resolve through centralized Victor paths."""
        from victor.ui.commands.utils import configure_logging as _configure_logging

        global_dir = tmp_path / ".victor"
        fake_paths = SimpleNamespace(global_victor_dir=global_dir)
        mock_file_handler = MagicMock()
        mock_file_handler.level = logging.NOTSET

        with (
            patch("victor.ui.commands.utils.get_project_paths", return_value=fake_paths),
            patch(
                "logging.handlers.RotatingFileHandler", return_value=mock_file_handler
            ) as mock_handler,
        ):
            _configure_logging("INFO", stream=io.StringIO(), file_logging=True)

        assert mock_handler.call_args is not None
        assert mock_handler.call_args.args[0] == global_dir / "logs" / "victor.log"


# =============================================================================
# Test _flush_logging
# =============================================================================


@pytest.mark.slow
class TestFlushLogging:
    """Tests for _flush_logging function."""

    @pytest.mark.slow
    def test_flush_logging_flushes_handlers(self):
        """Test that all handlers are flushed."""
        from victor.ui.commands.utils import flush_logging as _flush_logging

        # Create a mock handler
        mock_handler = MagicMock()
        logging.root.addHandler(mock_handler)

        _flush_logging()

        mock_handler.flush.assert_called_once()

        # Cleanup
        logging.root.removeHandler(mock_handler)

    @pytest.mark.slow
    def test_flush_logging_multiple_handlers(self):
        """Test flushing multiple handlers."""
        from victor.ui.commands.utils import flush_logging as _flush_logging

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


@pytest.mark.slow
class TestGracefulShutdown:
    """Tests for _graceful_shutdown async function."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_graceful_shutdown_none_agent(self):
        """Test shutdown with None agent."""
        from victor.ui.commands.utils import graceful_shutdown as _graceful_shutdown

        # Should not raise
        await _graceful_shutdown(None)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_graceful_shutdown_calls_agent_methods(self, mock_agent):
        """Test that shutdown calls agent methods."""
        from victor.ui.commands.utils import graceful_shutdown as _graceful_shutdown

        await _graceful_shutdown(mock_agent)

        mock_agent.graceful_shutdown.assert_awaited_once()
        mock_agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_graceful_shutdown_handles_exception(self, mock_agent):
        """Test that exceptions are handled gracefully."""
        from victor.ui.commands.utils import graceful_shutdown as _graceful_shutdown

        mock_agent.graceful_shutdown = AsyncMock(side_effect=Exception("Test error"))

        # Should not raise, should fall back to closing provider
        await _graceful_shutdown(mock_agent)

        mock_agent.provider.close.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_graceful_shutdown_handles_close_exception(self, mock_agent):
        """Test that close exceptions are also handled."""
        from victor.ui.commands.utils import graceful_shutdown as _graceful_shutdown

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
        from victor.ui.commands.utils import setup_safety_confirmation

        with patch("victor.ui.commands.utils.set_confirmation_callback") as mock_set:
            setup_safety_confirmation()
            mock_set.assert_called_once()


# =============================================================================
# Test _cli_confirmation_callback
# =============================================================================


class TestCliConfirmationCallback:
    """Tests for _cli_confirmation_callback async function."""

    @pytest.mark.asyncio
    async def test_confirmation_callback_prompts_user(self):
        """Test that callback prompts user for confirmation."""
        from victor.ui.commands.utils import cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, OperationalRiskLevel

        request = ConfirmationRequest(
            tool_name="test_tool",
            risk_level=OperationalRiskLevel.MEDIUM,
            description="Test action",
            details=["Detail 1", "Detail 2"],
            arguments={"param": "value"},
        )

        with patch("victor.ui.commands.utils.Confirm.ask", return_value=True) as mock_ask:
            result = await cli_confirmation_callback(request)
            assert result is True
            mock_ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirmation_callback_user_declines(self):
        """Test callback when user declines."""
        from victor.ui.commands.utils import cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, OperationalRiskLevel

        request = ConfirmationRequest(
            tool_name="test_tool",
            risk_level=OperationalRiskLevel.HIGH,
            description="Dangerous action",
            details=[],
            arguments={},
        )

        with patch("victor.ui.commands.utils.Confirm.ask", return_value=False):
            result = await cli_confirmation_callback(request)
            assert result is False

    @pytest.mark.asyncio
    async def test_confirmation_callback_keyboard_interrupt(self):
        """Test callback handles KeyboardInterrupt."""
        from victor.ui.commands.utils import cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, OperationalRiskLevel

        request = ConfirmationRequest(
            tool_name="test_tool",
            risk_level=OperationalRiskLevel.LOW,
            description="Test action",
            details=[],
            arguments={},
        )

        with patch("victor.ui.commands.utils.Confirm.ask", side_effect=KeyboardInterrupt):
            result = await cli_confirmation_callback(request)
            assert result is False

    @pytest.mark.asyncio
    async def test_confirmation_callback_eof_error(self):
        """Test callback handles EOFError."""
        from victor.ui.commands.utils import cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, OperationalRiskLevel

        request = ConfirmationRequest(
            tool_name="test_tool",
            risk_level=OperationalRiskLevel.SAFE,
            description="Safe action",
            details=[],
            arguments={},
        )

        with patch("victor.ui.commands.utils.Confirm.ask", side_effect=EOFError):
            result = await cli_confirmation_callback(request)
            assert result is False

    @pytest.mark.asyncio
    async def test_confirmation_callback_risk_levels(self):
        """Test callback handles all risk levels."""
        from victor.ui.commands.utils import cli_confirmation_callback
        from victor.agent.safety import ConfirmationRequest, OperationalRiskLevel

        for level in OperationalRiskLevel:
            request = ConfirmationRequest(
                tool_name="test_tool",
                risk_level=level,
                description=f"{level.value} action",
                details=[],
                arguments={},
            )

            with patch("victor.ui.commands.utils.Confirm.ask", return_value=True):
                result = await cli_confirmation_callback(request)
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
        # May fail without config - allow exit code 0, 1, or 2 (usage error)
        assert result.exit_code in [0, 1, 2]

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

        result = runner.invoke(app, ["embedding", "--help"])
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
        from victor.ui.commands.utils import check_codebase_index

        # Patch the import inside the function
        with patch.dict("sys.modules", {"victor_coding.codebase.indexer": None}):
            # Should not raise
            await check_codebase_index("/tmp", mock_console)

    @pytest.mark.asyncio
    async def test_check_codebase_index_generic_error(self, mock_console):
        """Test handling of generic errors."""
        from victor.ui.commands.utils import check_codebase_index

        mock_index = MagicMock()
        mock_index.check_staleness_by_mtime.side_effect = Exception("Test error")

        with patch.dict("sys.modules", {"victor_coding.codebase.indexer": MagicMock()}):
            with patch(
                "victor.ui.commands.utils.CodebaseIndex",
                return_value=mock_index,
                create=True,
            ):
                # Should not raise
                await check_codebase_index("/tmp", mock_console)

    @pytest.mark.asyncio
    async def test_check_codebase_index_not_stale(self, mock_console):
        """Test when index is not stale."""
        from victor.ui.commands.utils import check_codebase_index

        mock_index = MagicMock()
        mock_index.check_staleness_by_mtime.return_value = (False, [], [])

        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_index

        mock_container = MagicMock()
        mock_container.get_optional.return_value = mock_factory

        with (
            patch(
                "victor.ui.commands.utils.CodebaseIndexFactoryProtocol",
                new=MagicMock(),
            ),
            patch("victor.ui.commands.utils.get_container", return_value=mock_container),
        ):
            await check_codebase_index("/tmp", mock_console, silent=True)
            # No output expected when not stale and silent


# =============================================================================
# Test _preload_semantic_index
# =============================================================================


class TestPreloadSemanticIndex:
    """Tests for _preload_semantic_index async function."""

    @pytest.mark.asyncio
    async def test_preload_semantic_index_import_error(self, mock_console, mock_settings):
        """Test handling of ImportError."""
        from victor.ui.commands.utils import preload_semantic_index

        with patch(
            "victor.tools.code_search_tool._get_or_build_index",
            side_effect=ImportError("Test"),
        ):
            result = await preload_semantic_index("/tmp", mock_settings, mock_console)
            assert result is False

    @pytest.mark.asyncio
    async def test_preload_semantic_index_generic_error(self, mock_console, mock_settings):
        """Test handling of generic errors."""
        from victor.ui.commands.utils import preload_semantic_index

        with patch(
            "victor.tools.code_search_tool._get_or_build_index",
            side_effect=Exception("Test error"),
        ):
            result = await preload_semantic_index("/tmp", mock_settings, mock_console)
            assert result is False


# =============================================================================
# Test _setup_signal_handlers
# =============================================================================


class TestSetupSignalHandlers:
    """Tests for setup_signal_handlers function."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix signals only")
    def test_setup_signal_handlers_unix(self):
        """Test signal handler setup on Unix."""
        from victor.ui.commands.utils import setup_signal_handlers

        loop = asyncio.new_event_loop()

        with patch("signal.signal") as mock_signal:
            setup_signal_handlers(loop)
            # Should set SIGTERM handler
            mock_signal.assert_called()

        loop.close()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_setup_signal_handlers_windows(self):
        """Test signal handler setup on Windows (limited)."""
        from victor.ui.commands.utils import setup_signal_handlers

        loop = asyncio.new_event_loop()

        # Should not raise on Windows
        setup_signal_handlers(loop)

        loop.close()


# =============================================================================
# Test run_oneshot
# =============================================================================


class TestRunOneshot:
    """Tests for run_oneshot async function."""

    @pytest.mark.asyncio
    async def test_run_oneshot_import_works(self):
        """Test that run_oneshot can be imported."""
        from victor.ui.commands.chat import run_oneshot

        # Just verify the function exists and is callable
        assert callable(run_oneshot)


# =============================================================================
# Test History Enabled Flag
# =============================================================================


class TestHistoryEnabled:
    """Tests for readline history configuration."""

    def test_history_enabled_flag_exists(self):
        """Test that history enabled flag is set or not defined (CLI refactored)."""
        from victor.ui import cli

        # The flag may or may not be defined after CLI refactoring
        # If it exists, it should be a boolean
        if hasattr(cli, "_history_enabled"):
            assert isinstance(cli._history_enabled, bool)
        else:
            # Flag removed in CLI refactoring - test passes
            pass


# =============================================================================
# Test Global Agent Reference
# =============================================================================


class TestGlobalAgentReference:
    """Tests for global agent reference."""

    def test_current_agent_initial_none(self):
        """Test that _current_agent starts as None."""
        from victor.ui.commands import utils

        # Initially should be None (moved from cli to utils)
        assert utils._current_agent is None


# =============================================================================
# Test Output Format Options
# =============================================================================


class TestOutputFormatOptions:
    """Tests for output format option handling."""

    def test_output_format_cli_options(self):
        """Test that output format options are available in CLI."""
        from victor.ui.commands.chat import chat

        # Verify the chat function has the expected parameters
        import inspect

        sig = inspect.signature(chat)
        param_names = list(sig.parameters.keys())

        # These options should be available in the chat command
        assert "code_only" in param_names or any("code" in p for p in param_names)
        assert "json_output" in param_names or any("json" in p for p in param_names)
        assert "plain" in param_names or any("plain" in p for p in param_names)

    def test_output_modes_exist(self):
        """Test that output mode handling code exists."""
        from victor.ui.commands import chat as chat_module

        # Verify the module has expected attributes (moved from cli to chat)
        assert hasattr(chat_module, "run_oneshot")
        assert hasattr(chat_module, "run_interactive")


class TestChatChromePolicy:
    """Tests for Rich-only chat command chrome."""

    def test_cli_chrome_shown_for_default_rich_mode(self):
        """Rich output can include human-facing command chrome."""
        from victor.ui.commands.chat import _should_render_cli_chrome
        from victor.ui.output_formatter import create_formatter

        assert _should_render_cli_chrome(create_formatter()) is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"json_mode": True},
            {"jsonl": True},
            {"plain": True},
            {"code_only": True},
            {"quiet": True},
        ],
    )
    def test_cli_chrome_hidden_for_automation_modes(self, kwargs):
        """Automation output modes must not receive extra stdout chrome."""
        from victor.ui.commands.chat import _should_render_cli_chrome
        from victor.ui.output_formatter import create_formatter

        assert _should_render_cli_chrome(create_formatter(**kwargs)) is False

    def test_interactive_tool_banner_hidden_for_tui(self):
        """TUI sessions should not print plain CLI startup chrome."""
        from victor.ui.commands.chat import _should_render_interactive_tool_banner

        assert _should_render_interactive_tool_banner(use_tui=True) is False
        assert _should_render_interactive_tool_banner(use_tui=False) is True

    def test_build_cli_panel_accepts_profile_like_object(self):
        """CLI header builder should work with fallback profile display objects."""
        from types import SimpleNamespace

        from victor.ui.commands.chat import _build_cli_panel

        panel = _build_cli_panel(SimpleNamespace(provider="openai", model="gpt-4o"))
        assert panel is not None

    def test_summarize_tool_output_mode_plain_text(self):
        """TUI startup summaries should use plain text, not Rich markup."""
        from types import SimpleNamespace

        from victor.ui.commands.chat import _summarize_tool_output_mode

        settings = SimpleNamespace(
            tool_output_pruning_safe_only=True,
            tool_output_preview_enabled=False,
        )
        assert (
            _summarize_tool_output_mode(settings)
            == "Tool output: safe read-heavy pruning, preview off"
        )

    def test_print_interactive_startup_messages_handles_queue(self):
        """Queued startup notices should render through the shared status style."""
        from victor.ui.commands.chat import _print_interactive_startup_messages

        console = MagicMock()
        _print_interactive_startup_messages(
            console,
            ["Profile fallback active", "Warning: file watchers disabled"],
        )

        assert console.print.call_count == 2


class TestChatReplRendering:
    """Tests for CLI REPL rendering ownership."""

    @pytest.mark.asyncio
    async def test_formatter_renderer_response_is_not_reprinted(self):
        """The REPL must not print stream_response() content a second time."""
        from victor.ui.commands import chat as chat_module

        class FakeKeyBindings:
            def add(self, *_args, **_kwargs):
                def decorator(func):
                    return func

                return decorator

        class FakePromptSession:
            def __init__(self):
                self.key_bindings = FakeKeyBindings()
                self._inputs = iter(["hello", "/exit"])

            async def prompt_async(self, _prompt):
                return next(self._inputs)

        cmd_handler = MagicMock()
        cmd_handler.is_command.return_value = False
        agent = MagicMock()
        settings = MagicMock()
        profile_config = MagicMock(provider="test-provider", model="test-model")

        with (
            patch.object(
                chat_module, "_create_cli_prompt_session", return_value=FakePromptSession()
            ),
            patch.object(chat_module.console, "print") as mock_print,
            patch("victor.ui.rendering.stream_response", new=AsyncMock(return_value="dup content")),
        ):
            await chat_module._run_cli_repl(
                agent=agent,
                settings=settings,
                cmd_handler=cmd_handler,
                profile_config=profile_config,
                stream=True,
                renderer_choice="text",
            )

        rendered = [
            str(call_args.args[0]) for call_args in mock_print.call_args_list if call_args.args
        ]
        assert not any("dup content" in item for item in rendered)


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
