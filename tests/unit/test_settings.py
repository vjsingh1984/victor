"""Tests for config/settings.py module."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from victor.config.settings import (
    ProviderConfig,
    ProfileConfig,
    Settings,
    load_settings,
)


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_provider_config_defaults(self):
        """Test ProviderConfig with default values."""
        config = ProviderConfig()

        assert config.api_key is None
        assert config.base_url is None
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.organization is None

    def test_provider_config_with_values(self):
        """Test ProviderConfig with custom values."""
        config = ProviderConfig(
            api_key="test_key",
            base_url="https://api.example.com",
            timeout=120,
            max_retries=5,
            organization="test_org"
        )

        assert config.api_key == "test_key"
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 120
        assert config.max_retries == 5
        assert config.organization == "test_org"


class TestProfileConfig:
    """Tests for ProfileConfig model."""

    def test_profile_config_creation(self):
        """Test creating a ProfileConfig."""
        config = ProfileConfig(
            provider="ollama",
            model="qwen2.5-coder:7b"
        )

        assert config.provider == "ollama"
        assert config.model == "qwen2.5-coder:7b"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_profile_config_custom_values(self):
        """Test ProfileConfig with custom values."""
        config = ProfileConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.5,
            max_tokens=8192
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192


class TestSettings:
    """Tests for Settings class."""

    def test_settings_defaults(self):
        """Test Settings with default values."""
        settings = Settings()

        assert settings.default_provider == "ollama"
        assert settings.default_model == "qwen2.5-coder:7b"
        assert settings.default_temperature == 0.7
        assert settings.default_max_tokens == 4096
        assert settings.log_level == "INFO"
        assert settings.airgapped_mode is False
        assert settings.stream_responses is True

    def test_get_config_dir(self):
        """Test getting config directory."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/mock/home")

            with patch("pathlib.Path.mkdir") as mock_mkdir:
                config_dir = Settings.get_config_dir()

                assert config_dir == Path("/mock/home") / ".victor"
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_load_profiles_no_file(self):
        """Test loading profiles when file doesn't exist."""
        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            profiles = Settings.load_profiles()

            assert "default" in profiles
            assert profiles["default"].provider == "ollama"
            assert profiles["default"].model == "qwen2.5-coder:7b"

    def test_load_profiles_with_file(self):
        """Test loading profiles from YAML file."""
        yaml_content = """
profiles:
  dev:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.8
    max_tokens: 2048
  prod:
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    temperature: 0.5
    max_tokens: 8192
"""

        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", mock_open(read_data=yaml_content)):
                profiles = Settings.load_profiles()

                assert "dev" in profiles
                assert "prod" in profiles
                assert profiles["dev"].provider == "ollama"
                assert profiles["prod"].provider == "anthropic"
                assert profiles["prod"].temperature == 0.5

    def test_load_profiles_error_handling(self):
        """Test error handling when loading profiles fails."""
        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", side_effect=IOError("Read error")):
                profiles = Settings.load_profiles()

                # Should return empty dict on error
                assert profiles == {}

    def test_load_provider_config_no_file(self):
        """Test loading provider config when file doesn't exist."""
        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            config = Settings.load_provider_config("anthropic")

            assert config is None

    def test_load_provider_config_with_file(self):
        """Test loading provider config from YAML file."""
        yaml_content = """
providers:
  anthropic:
    api_key: test_key_123
    timeout: 90
    max_retries: 5
  openai:
    api_key: openai_key
    base_url: https://api.openai.com/v1
"""

        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", mock_open(read_data=yaml_content)):
                config = Settings.load_provider_config("anthropic")

                assert config is not None
                assert config.api_key == "test_key_123"
                assert config.timeout == 90
                assert config.max_retries == 5

    def test_load_provider_config_with_env_vars(self):
        """Test loading provider config with environment variable expansion."""
        yaml_content = """
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    timeout: 60
"""

        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", mock_open(read_data=yaml_content)):
                with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env_key_456"}):
                    config = Settings.load_provider_config("anthropic")

                    assert config is not None
                    assert config.api_key == "env_key_456"

    def test_load_provider_config_error_handling(self):
        """Test error handling when loading provider config fails."""
        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", side_effect=IOError("Read error")):
                config = Settings.load_provider_config("anthropic")

                # Should return None on error
                assert config is None

    def test_get_provider_settings_anthropic(self):
        """Test getting settings for Anthropic provider."""
        settings = Settings(anthropic_api_key="test_anthropic_key")

        provider_settings = settings.get_provider_settings("anthropic")

        assert provider_settings["api_key"] == "test_anthropic_key"
        assert provider_settings["base_url"] == "https://api.anthropic.com"

    def test_get_provider_settings_openai(self):
        """Test getting settings for OpenAI provider."""
        settings = Settings(openai_api_key="test_openai_key")

        provider_settings = settings.get_provider_settings("openai")

        assert provider_settings["api_key"] == "test_openai_key"
        assert provider_settings["base_url"] == "https://api.openai.com/v1"

    def test_get_provider_settings_google(self):
        """Test getting settings for Google provider."""
        settings = Settings(google_api_key="test_google_key")

        provider_settings = settings.get_provider_settings("google")

        assert provider_settings["api_key"] == "test_google_key"

    def test_get_provider_settings_ollama(self):
        """Test getting settings for Ollama provider."""
        settings = Settings(ollama_base_url="http://localhost:11434")

        provider_settings = settings.get_provider_settings("ollama")

        assert provider_settings["base_url"] == "http://localhost:11434"

    def test_get_provider_settings_lmstudio(self):
        """Test getting settings for LMStudio provider."""
        settings = Settings(lmstudio_base_url="http://localhost:1234")

        provider_settings = settings.get_provider_settings("lmstudio")

        assert provider_settings["base_url"] == "http://localhost:1234"

    def test_get_provider_settings_vllm(self):
        """Test getting settings for vLLM provider."""
        settings = Settings(vllm_base_url="http://localhost:8000")

        provider_settings = settings.get_provider_settings("vllm")

        assert provider_settings["base_url"] == "http://localhost:8000"

    def test_get_provider_settings_with_config(self):
        """Test getting settings with provider config from YAML."""
        yaml_content = """
providers:
  anthropic:
    timeout: 120
    max_retries: 10
"""

        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", mock_open(read_data=yaml_content)):
                settings = Settings(anthropic_api_key="test_key")
                provider_settings = settings.get_provider_settings("anthropic")

                assert provider_settings["api_key"] == "test_key"
                assert provider_settings["timeout"] == 120
                assert provider_settings["max_retries"] == 10


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_settings(self):
        """Test loading settings."""
        settings = load_settings()

        assert isinstance(settings, Settings)
        assert settings.default_provider == "ollama"
