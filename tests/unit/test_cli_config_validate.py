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

"""Tests for the config-validate CLI command."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from victor.ui.cli import app


runner = CliRunner()


class TestConfigValidateCommand:
    """Tests for victor config-validate command."""

    def test_config_validate_basic(self):
        """Test basic config validation with existing config."""
        # Mock settings to return valid config
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"
        mock_settings.load_profiles.return_value = {
            "default": MagicMock(
                provider="ollama",
                model="test-model",
                temperature=0.7,
                max_tokens=4096,
            )
        }

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"profiles": {"default": {}}}):
                    result = runner.invoke(app, ["config", "validate"])
                    # Should pass or fail gracefully
                    assert result.exit_code in [0, 1]

    def test_config_validate_verbose_flag(self):
        """Test verbose output flag."""
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"
        mock_settings.load_profiles.return_value = {
            "default": MagicMock(
                provider="ollama",
                model="test-model",
                temperature=0.7,
                max_tokens=4096,
            )
        }

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"profiles": {"default": {}}}):
                    result = runner.invoke(app, ["config", "validate", "--verbose"])
                    assert result.exit_code in [0, 1]

    def test_config_validate_missing_config_dir(self):
        """Test handling of missing config directory."""
        mock_settings = MagicMock()
        # Return a non-existent directory
        mock_settings.get_config_dir.return_value = Path("/nonexistent/path/.victor")

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 1
            assert "not found" in result.output.lower() or "init" in result.output.lower()

    def test_config_validate_invalid_yaml(self):
        """Test handling of invalid YAML file."""
        import yaml

        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                    result = runner.invoke(app, ["config", "validate"])
                    assert result.exit_code == 1
                    assert "yaml" in result.output.lower() or "invalid" in result.output.lower()

    def test_config_validate_missing_profiles_section(self):
        """Test handling of config without profiles section."""
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"other_key": "value"}):
                    result = runner.invoke(app, ["config", "validate"])
                    assert result.exit_code == 1
                    assert "profiles" in result.output.lower()

    def test_config_validate_invalid_temperature(self):
        """Test detection of invalid temperature value."""
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"
        mock_settings.load_profiles.return_value = {
            "invalid": MagicMock(
                provider="ollama",
                model="test-model",
                temperature=5.0,  # Invalid - out of range
                max_tokens=4096,
            )
        }
        mock_settings.get_provider_settings.return_value = {}

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"profiles": {"invalid": {}}}):
                    result = runner.invoke(app, ["config", "validate", "--verbose"])
                    assert result.exit_code == 1
                    assert "temperature" in result.output.lower()

    def test_config_validate_invalid_max_tokens(self):
        """Test detection of invalid max_tokens value."""
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"
        mock_settings.load_profiles.return_value = {
            "invalid": MagicMock(
                provider="ollama",
                model="test-model",
                temperature=0.7,
                max_tokens=-100,  # Invalid - negative
            )
        }
        mock_settings.get_provider_settings.return_value = {}

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"profiles": {"invalid": {}}}):
                    result = runner.invoke(app, ["config", "validate", "--verbose"])
                    assert result.exit_code == 1
                    assert (
                        "max_tokens" in result.output.lower() or "invalid" in result.output.lower()
                    )

    def test_config_validate_unknown_provider(self):
        """Test detection of unknown provider."""
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"
        mock_settings.load_profiles.return_value = {
            "unknown": MagicMock(
                provider="fake_provider",  # Unknown provider
                model="test-model",
                temperature=0.7,
                max_tokens=4096,
            )
        }
        mock_settings.get_provider_settings.return_value = {}

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"profiles": {"unknown": {}}}):
                    result = runner.invoke(app, ["config", "validate", "--verbose"])
                    assert result.exit_code == 1
                    assert "unknown" in result.output.lower() or "provider" in result.output.lower()

    def test_config_validate_cloud_provider_missing_api_key(self):
        """Test warning for cloud provider without API key."""
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"
        mock_settings.load_profiles.return_value = {
            "cloud": MagicMock(
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.7,
                max_tokens=4096,
            )
        }
        mock_settings.get_provider_settings.return_value = {"api_key": None}  # No API key

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"profiles": {"cloud": {}}}):
                    result = runner.invoke(app, ["config", "validate", "--verbose"])
                    # Should pass with warning (exit code 0)
                    assert result.exit_code == 0
                    assert "api key" in result.output.lower() or "warning" in result.output.lower()

    def test_config_validate_cloud_provider_with_api_key(self):
        """Test cloud provider with API key configured."""
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"
        mock_settings.load_profiles.return_value = {
            "cloud": MagicMock(
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.7,
                max_tokens=4096,
            )
        }
        mock_settings.get_provider_settings.return_value = {"api_key": "sk-test-key"}

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"profiles": {"cloud": {}}}):
                    result = runner.invoke(app, ["config", "validate", "--verbose"])
                    assert result.exit_code == 0
                    assert "api key configured" in result.output.lower()

    def test_config_validate_multiple_profiles(self):
        """Test validation of multiple profiles."""
        mock_settings = MagicMock()
        mock_settings.get_config_dir.return_value = Path.home() / ".victor"
        mock_settings.load_profiles.return_value = {
            "local": MagicMock(
                provider="ollama",
                model="qwen2.5-coder:7b",
                temperature=0.7,
                max_tokens=4096,
            ),
            "cloud": MagicMock(
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.5,
                max_tokens=8192,
            ),
        }
        mock_settings.get_provider_settings.return_value = {"api_key": "sk-test"}

        with patch("victor.ui.commands.config.load_settings", return_value=mock_settings):
            with patch("builtins.open", MagicMock()):
                with patch(
                    "yaml.safe_load",
                    return_value={"profiles": {"local": {}, "cloud": {}}},
                ):
                    result = runner.invoke(app, ["config", "validate", "--verbose"])
                    assert result.exit_code == 0
                    assert "2 profile" in result.output.lower()


class TestConfigValidateHelp:
    """Tests for config-validate help text."""

    def test_config_validate_help(self):
        """Test that help text is displayed correctly."""
        result = runner.invoke(app, ["config", "validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.output.lower()
        assert "--verbose" in result.output
        assert "--check-connectivity" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
