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

"""Tests for provider configuration and API key resolution."""

import os
import pytest
from unittest.mock import patch

from victor.providers.provider_factory import (
    ProviderConfig,
    resolve_api_key,
    create_provider_config,
    get_env_var_names_for_provider,
    register_provider_env_patterns,
    is_local_provider,
    needs_api_key,
    PROVIDER_ENV_VAR_PATTERNS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def clear_env():
    """Clear environment variables before each test."""
    old_env = os.environ.copy()
    os.environ.clear()
    yield
    os.environ.update(old_env)


# =============================================================================
# PROVIDER CONFIG TESTS
# =============================================================================


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_create_config(self):
        """Test creating a provider config."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            timeout=120,
            max_retries=5,
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 120
        assert config.max_retries == 5

    def test_config_defaults(self):
        """Test default values for ProviderConfig."""
        config = ProviderConfig(api_key="key")

        assert config.api_key == "key"
        assert config.base_url is None
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.extra_config == {}

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            timeout=90,
            extra_config={"custom_param": "value"},
        )

        result = config.to_dict()

        assert result["api_key"] == "test-key"
        assert result["base_url"] == "https://api.example.com"
        assert result["timeout"] == 90
        assert result["max_retries"] == 3
        assert result["custom_param"] == "value"


# =============================================================================
# API KEY RESOLUTION TESTS
# =============================================================================


class TestAPIKeyResolution:
    """Tests for API key resolution."""

    def test_provided_api_key_takes_priority(self, clear_env):
        """Test that explicitly provided API key is used first."""
        result = resolve_api_key("explicit-key", "testprovider")

        assert result == "explicit-key"

    def test_env_var_resolution(self, clear_env):
        """Test resolving API key from environment variable."""
        os.environ["TESTPROVIDER_API_KEY"] = "env-key"

        result = resolve_api_key(None, "testprovider")

        assert result == "env-key"

    def test_env_var_resolution_standard_patterns(self, clear_env):
        """Test standard provider environment variable patterns."""
        # Test OpenAI
        os.environ["OPENAI_API_KEY"] = "openai-key"
        assert resolve_api_key(None, "openai") == "openai-key"

        # Test Anthropic
        os.environ["ANTHROPIC_API_KEY"] = "anthropic-key"
        assert resolve_api_key(None, "anthropic") == "anthropic-key"

        # Test Groq (with alternative)
        os.environ["GROQCLOUD_API_KEY"] = "groq-key"
        assert resolve_api_key(None, "groq") == "groq-key"

    def test_custom_env_var_names(self, clear_env):
        """Test custom environment variable names."""
        os.environ["CUSTOM_KEY"] = "custom-value"

        result = resolve_api_key(
            None,
            "myprovider",
            env_var_names=["CUSTOM_KEY"],
        )

        assert result == "custom-value"

    def test_multiple_env_var_names_first_wins(self, clear_env):
        """Test that first matching env var is used."""
        os.environ["FIRST_KEY"] = "first-value"
        os.environ["SECOND_KEY"] = "second-value"

        result = resolve_api_key(
            None,
            "testprovider",
            env_var_names=["FIRST_KEY", "SECOND_KEY"],
        )

        assert result == "first-value"

    def test_keyring_fallback(self, clear_env):
        """Test keyring as fallback when env var not set."""
        with patch("victor.config.api_keys.get_api_key") as mock_get:
            mock_get.return_value = "keyring-key"

            result = resolve_api_key(None, "testprovider")

            assert result == "keyring-key"
            mock_get.assert_called_once_with("testprovider")

    def test_keyring_disabled(self, clear_env):
        """Test that keyring can be disabled."""
        with patch("victor.config.api_keys.get_api_key") as mock_get:
            mock_get.return_value = "keyring-key"

            result = resolve_api_key(None, "testprovider", use_keyring=False)

            assert result == ""
            mock_get.assert_not_called()

    def test_keyring_import_error_handled(self, clear_env):
        """Test that ImportError from keyring is handled gracefully."""
        with patch(
            "victor.config.api_keys.get_api_key",
            side_effect=ImportError,
        ):
            result = resolve_api_key(None, "testprovider")

            assert result == ""

    def test_keyring_exception_handled(self, clear_env):
        """Test that exceptions from keyring are handled gracefully."""
        with patch(
            "victor.config.api_keys.get_api_key",
            side_effect=Exception("Keyring error"),
        ):
            result = resolve_api_key(None, "testprovider")

            assert result == ""

    def test_empty_returned_when_not_found(self, clear_env):
        """Test that empty string is returned when key not found."""
        result = resolve_api_key(None, "unknownprovider")

        assert result == ""

    def test_raise_when_not_found_and_allow_empty_false(self, clear_env):
        """Test that ValueError is raised when allow_empty=False."""
        with pytest.raises(ValueError) as exc_info:
            resolve_api_key(None, "testprovider", allow_empty=False)

        assert "testprovider" in str(exc_info.value)
        assert "API key not found" in str(exc_info.value)

    def test_warning_logging_control(self, clear_env, caplog):
        """Test that warning logging can be controlled."""
        with caplog.at_level("WARNING"):
            # With log_warning=False, should not log
            resolve_api_key(None, "testprovider", log_warning=False)

            # Check that no warning was logged
            assert not any("API key not provided" in record.message for record in caplog.records)

    def test_case_insensitive_provider_name(self, clear_env):
        """Test that provider name lookup is case-insensitive."""
        os.environ["OPENAI_API_KEY"] = "test-key"

        # All should resolve to the same env var
        assert resolve_api_key(None, "openai") == "test-key"
        assert resolve_api_key(None, "OpenAI") == "test-key"
        assert resolve_api_key(None, "OPENAI") == "test-key"


# =============================================================================
# CREATE PROVIDER CONFIG TESTS
# =============================================================================


class TestCreateProviderConfig:
    """Tests for create_provider_config function."""

    def test_create_config_with_api_key(self):
        """Test creating config with explicit API key."""
        config = create_provider_config(
            "testprovider",
            api_key="explicit-key",
            timeout=120,
        )

        assert config.api_key == "explicit-key"
        assert config.timeout == 120

    def test_create_config_resolves_api_key(self):
        """Test that create_provider_config resolves API key."""
        with patch.dict(os.environ, {"TESTPROVIDER_API_KEY": "env-key"}):
            config = create_provider_config("testprovider")

            assert config.api_key == "env-key"

    def test_create_config_with_all_options(self):
        """Test creating config with all options."""
        config = create_provider_config(
            "myprovider",
            base_url="https://custom.url",
            timeout=90,
            max_retries=5,
            custom_param="value",
        )

        assert config.base_url == "https://custom.url"
        assert config.timeout == 90
        assert config.max_retries == 5
        assert config.extra_config == {"custom_param": "value"}


# =============================================================================
# ENV VAR PATTERN TESTS
# =============================================================================


class TestEnvVarPatterns:
    """Tests for environment variable pattern registry."""

    def test_get_env_var_names_known_provider(self):
        """Test getting env var names for known provider."""
        names = get_env_var_names_for_provider("groq")

        assert "GROQ_API_KEY" in names
        assert "GROQCLOUD_API_KEY" in names

    def test_get_env_var_names_unknown_provider(self):
        """Test getting env var names for unknown provider."""
        names = get_env_var_names_for_provider("unknownprovider")

        assert names == ["UNKNOWNPROVIDER_API_KEY"]

    def test_register_custom_patterns(self):
        """Test registering custom env var patterns."""
        custom_patterns = {
            "myprovider": ["MYPROVIDER_KEY", "MYPROVIDER_TOKEN"],
        }

        register_provider_env_patterns(custom_patterns)

        names = get_env_var_names_for_provider("myprovider")
        assert names == ["MYPROVIDER_KEY", "MYPROVIDER_TOKEN"]

    def test_standard_patterns_exist(self):
        """Test that standard providers have patterns defined."""
        # Check a few key providers
        assert "OPENAI_API_KEY" in get_env_var_names_for_provider("openai")
        assert "ANTHROPIC_API_KEY" in get_env_var_names_for_provider("anthropic")
        assert "DEEPSEEK_API_KEY" in get_env_var_names_for_provider("deepseek")


# =============================================================================
# LOCAL PROVIDER TESTS
# =============================================================================


class TestLocalProviderDetection:
    """Tests for local provider detection."""

    def test_ollama_is_local(self):
        """Test that Ollama is detected as local."""
        assert is_local_provider("ollama")

    def test_lmstudio_is_local(self):
        """Test that LM Studio is detected as local."""
        assert is_local_provider("lmstudio")

    def test_vllm_is_local(self):
        """Test that vLLM is detected as local."""
        assert is_local_provider("vllm")

    def test_llamacpp_is_local(self):
        """Test that llama.cpp is detected as local."""
        assert is_local_provider("llamacpp")

    def test_cloud_provider_not_local(self):
        """Test that cloud providers are not local."""
        assert not is_local_provider("openai")
        assert not is_local_provider("anthropic")
        assert not is_local_provider("groq")

    def test_case_insensitive_local_check(self):
        """Test that local provider check is case-insensitive."""
        assert is_local_provider("OLLAMA")
        assert is_local_provider("Ollama")

    def test_needs_api_key_for_cloud_providers(self):
        """Test that cloud providers need API keys."""
        assert needs_api_key("openai")
        assert needs_api_key("anthropic")
        assert needs_api_key("groq")

    def test_needs_api_key_false_for_local(self):
        """Test that local providers don't need API keys."""
        assert not needs_api_key("ollama")
        assert not needs_api_key("lmstudio")
        assert not needs_api_key("vllm")


# =============================================================================
# PROVIDER SPECIFIC TESTS
# =============================================================================


class TestProviderSpecificPatterns:
    """Tests for provider-specific environment variable patterns."""

    def test_huggingface_has_multiple_patterns(self):
        """Test HuggingFace has multiple env var options."""
        patterns = get_env_var_names_for_provider("huggingface")

        assert "HUGGINGFACE_API_KEY" in patterns
        assert "HF_TOKEN" in patterns

    def test_google_has_multiple_patterns(self):
        """Test Google has multiple env var options."""
        patterns = get_env_var_names_for_provider("google")

        assert "GOOGLE_API_KEY" in patterns
        assert "GEMINI_API_KEY" in patterns

    def test_xai_has_multiple_patterns(self):
        """Test xAI has multiple env var options."""
        patterns = get_env_var_names_for_provider("xai")

        assert "XAI_API_KEY" in patterns
        assert "GROK_API_KEY" in patterns

    def test_azure_has_multiple_patterns(self):
        """Test Azure has multiple env var options."""
        patterns = get_env_var_names_for_provider("azure")

        assert "AZURE_OPENAI_API_KEY" in patterns
        assert "AZURE_API_KEY" in patterns

    def test_replicate_uses_token_var(self):
        """Test Replicate uses TOKEN instead of API_KEY."""
        patterns = get_env_var_names_for_provider("replicate")

        assert "REPLICATE_API_TOKEN" in patterns


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in provider factory."""

    def test_empty_string_api_key_is_valid(self):
        """Test that empty string is a valid API key (for testing)."""
        result = resolve_api_key("", "testprovider")

        assert result == ""

    def test_none_api_key_triggers_resolution(self):
        """Test that None triggers resolution logic."""
        with patch.dict(os.environ, {"TESTPROVIDER_API_KEY": "from-env"}):
            result = resolve_api_key(None, "testprovider")

            assert result == "from-env"

    def test_whitespace_api_key_is_preserved(self):
        """Test that whitespace in API key is preserved."""
        result = resolve_api_key("  key-with-spaces  ", "testprovider")

        assert result == "  key-with-spaces  "

    def test_extra_config_preserved_in_to_dict(self):
        """Test that extra_config is merged in to_dict output."""
        config = ProviderConfig(
            api_key="key",
            extra_config={"param1": "value1", "param2": 42},
        )

        result = config.to_dict()

        assert result["param1"] == "value1"
        assert result["param2"] == 42
        assert "extra_config" not in result  # Should be merged, not nested


# =============================================================================
# IMMUTABILITY TESTS
# =============================================================================


class TestImmutability:
    """Tests for data immutability and thread safety."""

    def test_provider_config_is_mutable(self):
        """Test that ProviderConfig fields can be modified."""
        config = ProviderConfig(api_key="key")

        config.api_key = "new-key"
        config.timeout = 120

        assert config.api_key == "new-key"
        assert config.timeout == 120

    def test_extra_config_mutable(self):
        """Test that extra_config dict can be modified."""
        config = ProviderConfig(api_key="key")

        config.extra_config["new_field"] = "value"

        assert config.extra_config["new_field"] == "value"


# =============================================================================
# ENV VAR PATTERN REGISTRY TESTS
# =============================================================================


class TestEnvVarPatternRegistry:
    """Tests for the global env var pattern registry."""

    def test_registry_is_mutable(self):
        """Test that the registry can be modified."""
        original_patterns = PROVIDER_ENV_VAR_PATTERNS.copy()

        # Register a new pattern
        register_provider_env_patterns({"newprovider": ["NEWPROVIDER_KEY"]})

        # Check it was registered
        assert "NEWPROVIDER_KEY" in get_env_var_names_for_provider("newprovider")

        # Restore original patterns
        PROVIDER_ENV_VAR_PATTERNS.clear()
        PROVIDER_ENV_VAR_PATTERNS.update(original_patterns)

    def test_register_overwrites_existing(self):
        """Test that registering overwrites existing patterns."""
        original = get_env_var_names_for_provider("testprovider")

        register_provider_env_patterns({"testprovider": ["NEW_PATTERN"]})

        updated = get_env_var_names_for_provider("testprovider")
        assert updated == ["NEW_PATTERN"]
        assert updated != original

    def test_registry_case_insensitivity(self):
        """Test that registry is case-insensitive for provider names."""
        register_provider_env_patterns({"MyProvider": ["MYPROVIDER_KEY"]})

        assert "MYPROVIDER_KEY" in get_env_var_names_for_provider("myprovider")
        assert "MYPROVIDER_KEY" in get_env_var_names_for_provider("MYPROVIDER")
        assert "MYPROVIDER_KEY" in get_env_var_names_for_provider("MyProvider")
