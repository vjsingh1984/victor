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

"""Tests for interactive onboarding wizard."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from victor.ui.commands.onboarding import OnboardingWizard


class TestOnboardingWizard:
    """Tests for OnboardingWizard class."""

    def test_init(self):
        """Wizard initialization works."""
        wizard = OnboardingWizard()
        assert wizard.config_dir == Path.home() / ".victor"
        assert wizard.state["step"] == 0

    def test_init_with_custom_console(self):
        """Wizard can accept custom console."""
        from rich.console import Console

        console = Console()
        wizard = OnboardingWizard(console=console)
        assert wizard.console == console


class TestEnvironmentDetection:
    """Tests for environment detection."""

    @patch("victor.ui.commands.onboarding.subprocess.run")
    def test_check_ollama_running(self, mock_run):
        """Ollama detection works when running."""
        mock_run.return_value = MagicMock(returncode=0)
        wizard = OnboardingWizard()
        result = wizard._check_ollama()
        assert result is True

    @patch("victor.ui.commands.onboarding.subprocess.run")
    def test_check_ollama_not_running(self, mock_run):
        """Ollama detection handles not running."""
        mock_run.return_value = MagicMock(returncode=1)
        wizard = OnboardingWizard()
        result = wizard._check_ollama()
        assert result is False

    @patch("victor.ui.commands.onboarding.subprocess.run")
    def test_check_ollama_not_found(self, mock_run):
        """Ollama detection handles not installed."""
        mock_run.side_effect = FileNotFoundError()
        wizard = OnboardingWizard()
        result = wizard._check_ollama()
        assert result is False

    def test_check_api_keys(self, monkeypatch):
        """API key detection works."""
        wizard = OnboardingWizard()

        # No keys
        keys = wizard._check_api_keys()
        assert keys == []

        # Add Anthropic key
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        keys = wizard._check_api_keys()
        assert "Anthropic" in keys

        # Add OpenAI key
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        keys = wizard._check_api_keys()
        assert "Anthropic" in keys
        assert "OpenAI" in keys


class TestProviderConfiguration:
    """Tests for provider configuration."""

    def test_get_models_for_provider(self):
        """Getting models for provider returns expected list."""
        wizard = OnboardingWizard()

        # Ollama
        models = wizard._get_models_for_provider("ollama")
        assert len(models) > 0
        assert "qwen2.5-coder:7b" in [m["id"] for m in models]

        # Anthropic
        models = wizard._get_models_for_provider("anthropic")
        assert len(models) > 0
        assert any("claude" in m["id"].lower() for m in models)

        # Unknown provider
        models = wizard._get_models_for_provider("unknown")
        assert models == []

    def test_get_default_model(self):
        """Getting default model works."""
        wizard = OnboardingWizard()

        assert "qwen2.5-coder" in wizard._get_default_model("ollama")
        assert "claude" in wizard._get_default_model("anthropic").lower()
        assert "gpt" in wizard._get_default_model("openai").lower()

    def test_auto_detect_provider(self, monkeypatch):
        """Auto-detect provider picks correctly."""
        wizard = OnboardingWizard()
        wizard.state["has_cloud_keys"] = False
        wizard.state["ollama_available"] = False

        # No keys, no Ollama -> defaults to ollama
        provider = wizard._auto_detect_provider()
        assert provider == "ollama"

        # Has Anthropic key
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        wizard.state["has_cloud_keys"] = True
        provider = wizard._auto_detect_provider()
        assert provider == "anthropic"


class TestProfileSelection:
    """Tests for interactive profile selection."""

    @patch("victor.ui.commands.onboarding.Prompt.ask", return_value="basic")
    def test_select_profile_sets_selected_profile(self, _mock_prompt):
        """Profile selection stores resolved profile in wizard state."""
        wizard = OnboardingWizard()
        result = wizard._select_profile()

        assert result is True
        assert wizard.state["selected_profile"] is not None
        assert wizard.state["selected_profile"].name == "basic"


class TestProfileYAMLGeneration:
    """Tests for profile YAML generation."""

    @patch("victor.ui.commands.onboarding.install_profile")
    def test_apply_configuration(self, mock_install):
        """Applying configuration calls install_profile."""
        wizard = OnboardingWizard()
        wizard.state["selected_profile"] = MagicMock(name="test", display_name="Test Profile")
        wizard.state["provider"] = "ollama"
        wizard.state["model"] = "qwen2.5-coder:7b"

        wizard._apply_configuration()

        # Should have called install_profile
        mock_install.assert_called_once()


class TestWizardFlow:
    """Tests for wizard flow."""

    @patch("victor.ui.commands.onboarding.OnboardingWizard._show_welcome")
    def test_welcome_screen(self, mock_welcome):
        """Welcome screen displays without errors."""
        wizard = OnboardingWizard()
        wizard._show_welcome()

        # Should have called _show_welcome
        mock_welcome.assert_called_once()

    @patch("victor.ui.commands.onboarding.OnboardingWizard._detect_environment")
    @patch("victor.ui.commands.onboarding.OnboardingWizard._show_welcome")
    @patch("victor.ui.commands.onboarding.Confirm.ask")
    def test_confirm_flow(self, mock_confirm, mock_welcome, mock_detect):
        """Test confirmation flow."""
        wizard = OnboardingWizard()
        wizard._show_welcome()

        # User cancels
        mock_confirm.return_value = False
        result = wizard._confirm_start()
        assert result is False

    @patch("victor.ui.commands.onboarding.OnboardingWizard._show_completion")
    @patch("victor.ui.commands.onboarding.OnboardingWizard._validate_configuration")
    @patch("victor.ui.commands.onboarding.OnboardingWizard._apply_configuration")
    @patch("victor.ui.commands.onboarding.OnboardingWizard._configure_provider")
    @patch("victor.ui.commands.onboarding.OnboardingWizard._select_profile")
    @patch("victor.ui.commands.onboarding.OnboardingWizard._detect_environment")
    @patch("victor.ui.commands.onboarding.OnboardingWizard._show_welcome")
    @patch("victor.ui.commands.onboarding.OnboardingWizard._confirm_start")
    @patch("victor.ui.commands.onboarding.Confirm.ask")
    def test_full_run_success(
        self,
        mock_confirm,
        mock_confirm_start,
        mock_welcome,
        mock_detect,
        mock_select,
        mock_configure,
        mock_apply,
        mock_validate,
        mock_complete,
    ):
        """Test full wizard run with all steps."""
        wizard = OnboardingWizard()

        # Mock all user inputs and steps
        mock_confirm.return_value = True
        mock_confirm_start.return_value = True
        mock_detect.return_value = None  # Environment detected
        mock_select.return_value = True  # Profile selected
        mock_configure.return_value = True  # Provider configured
        mock_apply.return_value = None  # Configuration applied
        mock_validate.return_value = True  # Validation passed
        mock_complete.return_value = None  # Completion shown

        exit_code = wizard.run()

        # Should complete successfully
        assert exit_code == 0

    @patch("victor.ui.commands.onboarding.OnboardingWizard._show_welcome")
    @patch("victor.ui.commands.onboarding.Confirm.ask")
    def test_run_user_cancels(self, mock_ask, mock_welcome):
        """Test wizard run when user cancels at start."""
        wizard = OnboardingWizard()
        mock_ask.return_value = False

        exit_code = wizard.run()

        # Should exit with 0 (not an error, just cancelled)
        assert exit_code == 0


class TestRunOnboarding:
    """Tests for run_onboarding entry point."""

    @patch("victor.ui.commands.onboarding.OnboardingWizard.run", return_value=0)
    def test_run_onboarding_entry_point(self, mock_run):
        """run_onboarding creates wizard and runs it."""
        from victor.ui.commands.onboarding import run_onboarding

        exit_code = run_onboarding()

        assert exit_code == 0
        mock_run.assert_called_once()

    @patch(
        "victor.ui.commands.onboarding.OnboardingWizard.run", side_effect=Exception("Test error")
    )
    def test_run_onboarding_handles_exception(self, mock_run):
        """run_onboarding handles exceptions."""
        from victor.ui.commands.onboarding import run_onboarding

        exit_code = run_onboarding()

        assert exit_code == 1
        mock_run.assert_called_once()

    @patch("victor.ui.commands.onboarding.OnboardingWizard", side_effect=Exception("Init error"))
    def test_run_onboarding_handles_wizard_init_exception(self, mock_wizard):
        """run_onboarding handles wizard initialization exceptions."""
        from victor.ui.commands.onboarding import run_onboarding

        exit_code = run_onboarding()

        assert exit_code == 1
        mock_wizard.assert_called_once()
