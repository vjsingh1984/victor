#!/usr/bin/env python3
"""Test CLI logging control functionality."""

import os
import logging
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, AsyncMock, MagicMock

from victor.ui.cli import app, main


runner = CliRunner()


def test_default_logging_level():
    """Test that default logging level is INFO when no arguments provided."""

    # Mock the async agent orchestrator
    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            result = runner.invoke(app, ["main", "test message"])

            # Verify logging was configured with INFO level
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs["level"] == logging.INFO


def test_log_level_debug():
    """Test --log-level DEBUG sets logging to DEBUG level."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            result = runner.invoke(app, ["main", "--log-level", "DEBUG", "test message"])

            # Verify logging was configured with DEBUG level
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs["level"] == logging.DEBUG


def test_log_level_info():
    """Test --log-level INFO sets logging to INFO level."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            result = runner.invoke(app, ["main", "--log-level", "INFO", "test message"])

            # Verify logging was configured with INFO level
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs["level"] == logging.INFO


def test_log_level_warn_maps_to_warning():
    """Test --log-level WARN correctly maps to WARNING level."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            result = runner.invoke(app, ["main", "--log-level", "WARN", "test message"])

            # Verify WARN was mapped to WARNING
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs["level"] == logging.WARNING


def test_log_level_error():
    """Test --log-level ERROR sets logging to ERROR level."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            result = runner.invoke(app, ["main", "--log-level", "ERROR", "test message"])

            # Verify logging was configured with ERROR level
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs["level"] == logging.ERROR


def test_log_level_critical():
    """Test --log-level CRITICAL sets logging to CRITICAL level."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            result = runner.invoke(app, ["main", "--log-level", "CRITICAL", "test message"])

            # Verify logging was configured with CRITICAL level
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs["level"] == logging.CRITICAL


def test_invalid_log_level_shows_error():
    """Test that invalid log level shows error and exits."""
    result = runner.invoke(app, ["main", "--log-level", "INVALID", "test message"])

    # Should exit with error code
    assert result.exit_code == 1

    # Should show error message with valid options
    output = result.stdout
    assert "Invalid log level" in output or "invalid" in output.lower()
    assert "INVALID" in output or "invalid" in output.lower()


def test_environment_variable_fallback():
    """Test that VICTOR_LOG_LEVEL environment variable is used as fallback."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch.dict(os.environ, {"VICTOR_LOG_LEVEL": "DEBUG"}):
        with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
            with patch("logging.basicConfig") as mock_logging:
                result = runner.invoke(app, ["main", "test message"])

                # Verify logging was configured with DEBUG from env var
                assert mock_logging.called
                call_kwargs = mock_logging.call_args[1]
                assert call_kwargs["level"] == logging.DEBUG


def test_cli_argument_overrides_environment_variable():
    """Test that CLI --log-level argument overrides environment variable."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch.dict(os.environ, {"VICTOR_LOG_LEVEL": "DEBUG"}):
        with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
            with patch("logging.basicConfig") as mock_logging:
                # Pass ERROR as CLI argument, should override DEBUG from env var
                result = runner.invoke(app, ["main", "--log-level", "ERROR", "test message"])

                # Verify logging was configured with ERROR (CLI arg) not DEBUG (env var)
                assert mock_logging.called
                call_kwargs = mock_logging.call_args[1]
                assert call_kwargs["level"] == logging.ERROR


def test_log_level_case_insensitive():
    """Test that log level is case-insensitive."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            # Test lowercase
            result = runner.invoke(app, ["main", "--log-level", "debug", "test message"])

            # Verify logging was configured with DEBUG level
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs["level"] == logging.DEBUG


def test_logging_format_is_configured():
    """Test that logging format is properly configured."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            result = runner.invoke(app, ["main", "--log-level", "INFO", "test message"])

            # Verify logging format is set
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert "format" in call_kwargs
            assert "asctime" in call_kwargs["format"]
            assert "name" in call_kwargs["format"]
            assert "levelname" in call_kwargs["format"]
            assert "message" in call_kwargs["format"]


def test_logging_force_override():
    """Test that logging.basicConfig is called with force=True to override existing config."""

    async def mock_from_settings(*args, **kwargs):
        mock_agent = MagicMock()
        mock_agent.start_embedding_preload = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False
        mock_agent.provider.close = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(content="Test response"))
        return mock_agent

    with patch("victor.ui.cli.AgentOrchestrator.from_settings", side_effect=mock_from_settings):
        with patch("logging.basicConfig") as mock_logging:
            result = runner.invoke(app, ["main", "--log-level", "DEBUG", "test message"])

            # Verify force=True is set
            assert mock_logging.called
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs["force"] is True


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
