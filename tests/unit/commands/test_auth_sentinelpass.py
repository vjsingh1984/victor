"""Tests for SentinelPass-backed Victor auth configuration."""

from types import SimpleNamespace
import yaml
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


def test_auth_add_openai_codex_oauth_source_stores_generation_defaults():
    """`--source codex` should configure OpenAI OAuth without prompting for an API key."""
    runner = CliRunner()
    manager = MagicMock()
    manager.load_config.return_value = SimpleNamespace(defaults=SimpleNamespace(account="default"))

    with patch("victor.ui.commands.auth.get_account_manager", return_value=manager):
        with patch("victor.ui.commands.auth.Prompt.ask") as prompt:
            result = runner.invoke(
                auth_app,
                [
                    "add",
                    "--provider",
                    "openai",
                    "--model",
                    "gpt-5-nano",
                    "--name",
                    "openai-cheap",
                    "--auth-method",
                    "oauth",
                    "--source",
                    "codex",
                    "--temperature",
                    "0.2",
                    "--max-tokens",
                    "2048",
                    "--default",
                ],
            )

    assert result.exit_code == 0
    prompt.assert_not_called()
    account = manager.save_account.call_args.args[0]
    assert account.provider == "openai"
    assert account.model == "gpt-5-nano"
    assert account.auth.method == "oauth"
    assert account.auth.source == "codex"
    assert account.temperature == 0.2
    assert account.max_tokens == 2048
    saved_config = manager.save_config.call_args.args[0]
    assert saved_config.defaults.account == "openai-cheap"


def test_auth_add_creates_matching_chat_profile(tmp_path):
    """Adding an auth account should make the same name usable as a chat profile."""
    runner = CliRunner()
    manager = MagicMock()

    with (
        patch("victor.ui.commands.auth.get_account_manager", return_value=manager),
        patch("victor.ui.commands.auth.get_project_paths") as mock_paths,
        patch("victor.ui.commands.auth.Prompt.ask") as prompt,
    ):
        mock_paths.return_value.global_victor_dir = tmp_path
        result = runner.invoke(
            auth_app,
            [
                "add",
                "--provider",
                "openai",
                "--model",
                "gpt-5-nano",
                "--name",
                "openai-cheap",
                "--auth-method",
                "oauth",
                "--source",
                "codex",
                "--temperature",
                "0.2",
                "--max-tokens",
                "2048",
            ],
        )

    assert result.exit_code == 0
    prompt.assert_not_called()
    profile_data = yaml.safe_load((tmp_path / "profiles.yaml").read_text())
    profile = profile_data["profiles"]["openai-cheap"]
    assert profile["provider"] == "openai"
    assert profile["model"] == "gpt-5-nano"
    assert profile["auth_mode"] == "oauth"
    assert profile["oauth_source"] == "codex"
    assert profile["temperature"] == 0.2
    assert profile["max_tokens"] == 2048
    assert profile["account"] == "openai-cheap"


def test_auth_add_anthropic_claude_code_oauth_source():
    """`--source claude-code` should configure Anthropic OAuth without an API key."""
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
                    "claude-haiku-4-5-20251001",
                    "--name",
                    "claude-cheap",
                    "--auth-method",
                    "oauth",
                    "--source",
                    "claude-code",
                ],
            )

    assert result.exit_code == 0
    prompt.assert_not_called()
    account = manager.save_account.call_args.args[0]
    assert account.provider == "anthropic"
    assert account.auth.method == "oauth"
    assert account.auth.source == "claude-code"


def test_auth_add_rejects_codex_source_for_non_openai_oauth():
    runner = CliRunner()

    result = runner.invoke(
        auth_app,
        [
            "add",
            "--provider",
            "anthropic",
            "--model",
            "claude-haiku-4-5-20251001",
            "--auth-method",
            "oauth",
            "--source",
            "codex",
        ],
    )

    assert result.exit_code == 1
    assert "only valid with OpenAI OAuth" in result.stdout
