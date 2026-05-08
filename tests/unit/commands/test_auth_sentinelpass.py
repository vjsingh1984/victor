"""Tests for SentinelPass-backed Victor auth configuration."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from victor.ui.commands.auth import auth_app


def test_auth_add_sentinelpass_stores_lookup_reference_without_prompting_for_key():
    """`--source sentinelpass` stores a lookup reference, not an API key value."""
    runner = CliRunner()
    manager = MagicMock()

    with patch("victor.ui.commands.auth.get_account_manager", return_value=manager):
        with patch("victor.ui.commands.auth.Prompt.ask") as prompt:
            result = runner.invoke(
                auth_app,
                [
                    "add",
                    "--provider",
                    "anthropic",
                    "--model",
                    "claude-sonnet-4-5",
                    "--name",
                    "claude-vault",
                    "--source",
                    "sentinelpass",
                    "--sentinelpass-domain",
                    "api.anthropic.com",
                ],
            )

    assert result.exit_code == 0
    prompt.assert_not_called()
    manager.save_account.assert_called_once()
    account = manager.save_account.call_args.args[0]
    assert account.name == "claude-vault"
    assert account.provider == "anthropic"
    assert account.auth.method == "api_key"
    assert account.auth.source == "sentinelpass"
    assert account.auth.value == "api.anthropic.com"
