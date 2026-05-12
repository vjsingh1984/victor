"""Tests for importing Codex OAuth credentials into Victor auth."""

import json
import stat
from pathlib import Path

import yaml
from typer.testing import CliRunner

from victor.ui.commands.auth import auth_app

runner = CliRunner()


def _write_codex_auth(path: Path, *, access_token: str = "codex_access") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "codex_refresh",
                    "id_token": "codex_id",
                    "token_type": "Bearer",
                    "scope": "openid profile email offline_access",
                },
            }
        )
    )


def test_import_codex_openai_writes_victor_oauth_store_without_leaking_tokens(tmp_path):
    """Codex import should persist tokens securely and keep stdout redacted."""
    codex_auth = tmp_path / ".codex" / "auth.json"
    storage_dir = tmp_path / ".victor"
    _write_codex_auth(codex_auth)

    result = runner.invoke(
        auth_app,
        [
            "import-codex",
            "openai",
            "--codex-auth",
            str(codex_auth),
            "--storage-dir",
            str(storage_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Imported OpenAI OAuth tokens from Codex" in result.stdout
    assert "codex_access" not in result.stdout
    assert "codex_refresh" not in result.stdout

    token_file = storage_dir / "oauth_tokens.yaml"
    token_data = yaml.safe_load(token_file.read_text())
    assert token_data["openai"]["access_token"] == "codex_access"
    assert token_data["openai"]["refresh_token"] == "codex_refresh"
    assert token_data["openai"]["id_token"] == "codex_id"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    profile_data = yaml.safe_load((storage_dir / "profiles.yaml").read_text())
    profile = profile_data["profiles"]["openai-oauth"]
    assert profile["provider"] == "openai"
    assert profile["model"] == "gpt-5.4"
    assert profile["auth_mode"] == "oauth"
    assert profile["account"] == "openai-oauth"


def test_import_codex_openai_dry_run_validates_without_writing(tmp_path):
    """Dry-run should validate Codex tokens without creating Victor token storage."""
    codex_auth = tmp_path / ".codex" / "auth.json"
    storage_dir = tmp_path / ".victor"
    _write_codex_auth(codex_auth)

    result = runner.invoke(
        auth_app,
        [
            "import-codex",
            "openai",
            "--codex-auth",
            str(codex_auth),
            "--storage-dir",
            str(storage_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "tokens are importable" in result.stdout
    assert not (storage_dir / "oauth_tokens.yaml").exists()


def test_import_codex_openai_preserves_existing_tokens_without_force(tmp_path):
    """Existing Victor OAuth tokens should not be overwritten unless forced."""
    codex_auth = tmp_path / ".codex" / "auth.json"
    storage_dir = tmp_path / ".victor"
    token_file = storage_dir / "oauth_tokens.yaml"
    _write_codex_auth(codex_auth, access_token="new_access")
    storage_dir.mkdir(parents=True, exist_ok=True)
    token_file.write_text(
        yaml.safe_dump(
            {
                "openai": {
                    "access_token": "existing_access",
                    "refresh_token": "existing_refresh",
                    "id_token": None,
                    "token_type": "Bearer",
                    "expires_at": None,
                    "scopes": [],
                }
            }
        )
    )

    result = runner.invoke(
        auth_app,
        [
            "import-codex",
            "openai",
            "--codex-auth",
            str(codex_auth),
            "--storage-dir",
            str(storage_dir),
        ],
    )

    assert result.exit_code == 1
    token_data = yaml.safe_load(token_file.read_text())
    assert token_data["openai"]["access_token"] == "existing_access"


def test_import_codex_openai_force_replaces_existing_tokens(tmp_path):
    """Force mode should intentionally replace existing Victor OAuth tokens."""
    codex_auth = tmp_path / ".codex" / "auth.json"
    storage_dir = tmp_path / ".victor"
    token_file = storage_dir / "oauth_tokens.yaml"
    _write_codex_auth(codex_auth, access_token="new_access")
    storage_dir.mkdir(parents=True, exist_ok=True)
    token_file.write_text(
        yaml.safe_dump(
            {
                "openai": {
                    "access_token": "existing_access",
                    "refresh_token": "existing_refresh",
                    "id_token": None,
                    "token_type": "Bearer",
                    "expires_at": None,
                    "scopes": [],
                }
            }
        )
    )

    result = runner.invoke(
        auth_app,
        [
            "import-codex",
            "openai",
            "--codex-auth",
            str(codex_auth),
            "--storage-dir",
            str(storage_dir),
            "--force",
        ],
    )

    assert result.exit_code == 0
    token_data = yaml.safe_load(token_file.read_text())
    assert token_data["openai"]["access_token"] == "new_access"
