# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for profiles CLI commands - achieving 70%+ coverage."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from typer.testing import CliRunner
import typer

from victor.ui.commands.profiles import (
    profiles_app,
    list_profiles,
    create_profile,
    edit_profile,
    delete_profile,
    show_profile,
    set_default_profile,
    _load_profiles_yaml,
    _save_profiles_yaml,
)


runner = CliRunner()


class TestLoadProfilesYaml:
    """Tests for _load_profiles_yaml helper function."""

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from a nonexistent file returns empty dict."""
        result = _load_profiles_yaml(tmp_path / "nonexistent.yaml")
        assert result == {"profiles": {}}

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid profiles.yaml file."""
        profiles_file = tmp_path / "profiles.yaml"
        test_data = {
            "profiles": {
                "default": {
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(test_data, f)

        result = _load_profiles_yaml(profiles_file)
        assert result == test_data

    def test_load_empty_yaml(self, tmp_path):
        """Test loading an empty yaml file."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.touch()

        result = _load_profiles_yaml(profiles_file)
        assert result == {}

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid yaml returns empty dict with error."""
        profiles_file = tmp_path / "profiles.yaml"
        with open(profiles_file, "w") as f:
            f.write("invalid: yaml: content: [")

        result = _load_profiles_yaml(profiles_file)
        assert result == {"profiles": {}}


class TestSaveProfilesYaml:
    """Tests for _save_profiles_yaml helper function."""

    def test_save_profiles_creates_directory(self, tmp_path):
        """Test saving creates parent directories if needed."""
        nested_dir = tmp_path / "nested" / "dir"
        profiles_file = nested_dir / "profiles.yaml"
        test_data = {"profiles": {"test": {"provider": "test"}}}

        _save_profiles_yaml(profiles_file, test_data)

        assert profiles_file.exists()
        with open(profiles_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded == test_data

    def test_save_profiles_overwrites_existing(self, tmp_path):
        """Test saving overwrites existing file."""
        profiles_file = tmp_path / "profiles.yaml"

        # Write initial data
        initial_data = {"profiles": {"old": {"provider": "old"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(initial_data, f)

        # Save new data
        new_data = {"profiles": {"new": {"provider": "new"}}}
        _save_profiles_yaml(profiles_file, new_data)

        with open(profiles_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded == new_data

    def test_save_profiles_error_handling(self, tmp_path):
        """Test saving handles permission errors."""
        from click.exceptions import Exit

        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        profiles_file = readonly_dir / "profiles.yaml"

        # Write initial file
        profiles_file.touch()

        # Make directory read-only - skip on Windows
        import sys

        if sys.platform != "win32":
            readonly_dir.chmod(0o444)
            try:
                with pytest.raises(Exit):
                    _save_profiles_yaml(profiles_file, {"test": "data"})
            finally:
                readonly_dir.chmod(0o755)


class TestListProfiles:
    """Tests for list_profiles command."""

    def test_list_no_profiles(self):
        """Test listing when no profiles configured."""
        mock_settings = MagicMock()
        mock_settings.load_profiles.return_value = {}

        with patch("victor.ui.commands.profiles.load_settings", return_value=mock_settings):
            result = runner.invoke(profiles_app, ["list"])

        assert result.exit_code == 0
        assert "No profiles configured" in result.stdout

    def test_list_with_profiles(self):
        """Test listing profiles displays them correctly."""
        mock_profile = MagicMock()
        mock_profile.provider = "ollama"
        mock_profile.model = "llama2"
        mock_profile.temperature = 0.7
        mock_profile.max_tokens = 4096
        mock_profile.description = "Test profile"

        mock_settings = MagicMock()
        mock_settings.load_profiles.return_value = {"default": mock_profile}
        mock_settings.get_config_dir.return_value = Path("/tmp")

        with patch("victor.ui.commands.profiles.load_settings", return_value=mock_settings):
            result = runner.invoke(profiles_app, ["list"])

        assert result.exit_code == 0
        assert "default" in result.stdout or "Configured Profiles" in result.stdout


class TestCreateProfile:
    """Tests for create_profile command."""

    def test_create_new_profile(self, tmp_path):
        """Test creating a new profile successfully."""
        profiles_file = tmp_path / "profiles.yaml"

        with patch.object(Path, "__truediv__", return_value=profiles_file):
            with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
                mock_settings_cls.get_config_dir.return_value = tmp_path

                result = runner.invoke(
                    profiles_app,
                    [
                        "create",
                        "test_profile",
                        "--provider",
                        "ollama",
                        "--model",
                        "llama2",
                        "--temperature",
                        "0.5",
                        "--max-tokens",
                        "8192",
                        "--description",
                        "Test description",
                    ],
                )

        assert result.exit_code == 0
        assert "Created profile" in result.stdout

    def test_create_profile_already_exists(self, tmp_path):
        """Test creating a profile that already exists shows error."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"existing": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(
                profiles_app,
                ["create", "existing", "--provider", "anthropic", "--model", "claude"],
            )

        assert "already exists" in result.stdout

    def test_create_profile_without_description(self, tmp_path):
        """Test creating a profile without description."""
        profiles_file = tmp_path / "profiles.yaml"

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(
                profiles_app,
                ["create", "simple", "--provider", "ollama", "--model", "llama2"],
            )

        assert result.exit_code == 0


class TestEditProfile:
    """Tests for edit_profile command."""

    def test_edit_existing_profile(self, tmp_path):
        """Test editing an existing profile."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "edit_me": {
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(
                profiles_app,
                ["edit", "edit_me", "--temperature", "0.3"],
            )

        assert result.exit_code == 0
        assert "Updated profile" in result.stdout

    def test_edit_nonexistent_profile(self, tmp_path):
        """Test editing a profile that doesn't exist."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.touch()

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(
                profiles_app,
                ["edit", "nonexistent", "--temperature", "0.5"],
            )

        assert "not found" in result.stdout

    def test_edit_no_changes(self, tmp_path):
        """Test editing with no changes specified."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"no_change": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["edit", "no_change"])

        assert "No changes specified" in result.stdout

    def test_edit_all_fields(self, tmp_path):
        """Test editing all profile fields."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "full_edit": {
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(
                profiles_app,
                [
                    "edit",
                    "full_edit",
                    "--provider",
                    "anthropic",
                    "--model",
                    "claude",
                    "--temperature",
                    "0.5",
                    "--max-tokens",
                    "8192",
                    "--description",
                    "Updated",
                ],
            )

        assert result.exit_code == 0
        assert "Updated profile" in result.stdout


class TestDeleteProfile:
    """Tests for delete_profile command."""

    def test_delete_with_force(self, tmp_path):
        """Test deleting a profile with --force flag."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"delete_me": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["delete", "delete_me", "--force"])

        assert result.exit_code == 0
        assert "Deleted profile" in result.stdout

    def test_delete_nonexistent_profile(self, tmp_path):
        """Test deleting a profile that doesn't exist."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.touch()

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["delete", "nonexistent", "--force"])

        assert "not found" in result.stdout

    def test_delete_with_confirmation_yes(self, tmp_path):
        """Test deleting a profile with confirmation."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"confirm_delete": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["delete", "confirm_delete"], input="y\n")

        assert "Deleted profile" in result.stdout

    def test_delete_with_confirmation_no(self, tmp_path):
        """Test cancelling profile deletion."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"keep_me": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["delete", "keep_me"], input="n\n")

        assert "Cancelled" in result.stdout


class TestShowProfile:
    """Tests for show_profile command."""

    def test_show_existing_profile(self):
        """Test showing an existing profile."""
        mock_profile = MagicMock()
        mock_profile.provider = "ollama"
        mock_profile.model = "llama2"
        mock_profile.temperature = 0.7
        mock_profile.max_tokens = 4096
        mock_profile.description = "Test profile"
        mock_profile.tool_selection = None

        mock_settings = MagicMock()
        mock_settings.load_profiles.return_value = {"default": mock_profile}

        with patch("victor.ui.commands.profiles.load_settings", return_value=mock_settings):
            result = runner.invoke(profiles_app, ["show", "default"])

        assert result.exit_code == 0
        assert "ollama" in result.stdout or "Provider" in result.stdout

    def test_show_nonexistent_profile(self):
        """Test showing a profile that doesn't exist."""
        mock_settings = MagicMock()
        mock_settings.load_profiles.return_value = {"other": MagicMock()}

        with patch("victor.ui.commands.profiles.load_settings", return_value=mock_settings):
            result = runner.invoke(profiles_app, ["show", "nonexistent"])

        assert "not found" in result.stdout

    def test_show_profile_with_tool_selection(self):
        """Test showing a profile with tool_selection configured."""
        mock_profile = MagicMock()
        mock_profile.provider = "anthropic"
        mock_profile.model = "claude"
        mock_profile.temperature = 0.5
        mock_profile.max_tokens = 8192
        mock_profile.description = None
        mock_profile.tool_selection = "semantic"

        mock_settings = MagicMock()
        mock_settings.load_profiles.return_value = {"with_tools": mock_profile}

        with patch("victor.ui.commands.profiles.load_settings", return_value=mock_settings):
            result = runner.invoke(profiles_app, ["show", "with_tools"])

        assert result.exit_code == 0


class TestSetDefaultProfile:
    """Tests for set_default_profile command."""

    def test_set_default_success(self, tmp_path):
        """Test setting a profile as default."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "default": {"provider": "ollama", "model": "llama2"},
                "anthropic": {"provider": "anthropic", "model": "claude"},
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["set-default", "anthropic"])

        assert result.exit_code == 0
        assert "Set" in result.stdout or "default" in result.stdout

    def test_set_default_nonexistent(self, tmp_path):
        """Test setting nonexistent profile as default."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.touch()

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["set-default", "nonexistent"])

        assert "not found" in result.stdout

    def test_set_default_already_default(self, tmp_path):
        """Test setting default profile as default (no-op)."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "default": {"provider": "ollama", "model": "llama2"},
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["set-default", "default"])

        assert "already the default" in result.stdout

    def test_set_default_without_existing_default(self, tmp_path):
        """Test setting default when no previous default exists."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "new_default": {"provider": "anthropic", "model": "claude"},
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            result = runner.invoke(profiles_app, ["set-default", "new_default"])

        assert result.exit_code == 0


class TestProfilesAppIntegration:
    """Integration tests for profiles CLI app."""

    def test_create_edit_delete_flow(self, tmp_path):
        """Test full create-edit-delete workflow."""
        with patch("victor.ui.commands.profiles.Settings") as mock_settings_cls:
            mock_settings_cls.get_config_dir.return_value = tmp_path

            # Create
            result = runner.invoke(
                profiles_app,
                ["create", "workflow_test", "--provider", "ollama", "--model", "test"],
            )
            assert "Created" in result.stdout

            # Edit
            result = runner.invoke(
                profiles_app,
                ["edit", "workflow_test", "--temperature", "0.8"],
            )
            assert "Updated" in result.stdout

            # Delete
            result = runner.invoke(profiles_app, ["delete", "workflow_test", "--force"])
            assert "Deleted" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
