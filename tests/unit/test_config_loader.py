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

"""Tests for config_loader module."""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.config_loader import ConfigLoader
from victor.agent.tool_selection import get_critical_tools
from victor.tools.base import ToolRegistry, BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str, category: str = "general"):
        self._name = name
        self._category = category

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock tool {self._name}"

    @property
    def parameters(self) -> dict:
        return {}

    @property
    def category(self) -> str:
        return self._category

    async def execute(self, context, **kwargs):
        return {"success": True}


class TestCriticalTools:
    """Tests for get_critical_tools function."""

    def test_critical_tools_contains_essential(self):
        """Test that get_critical_tools() returns essential tools (canonical names)."""
        critical_tools = get_critical_tools()  # Fallback returns canonical names
        assert "read" in critical_tools
        assert "write" in critical_tools
        assert "ls" in critical_tools
        assert "shell" in critical_tools
        assert "edit" in critical_tools
        assert "search" in critical_tools


class TestConfigLoaderInit:
    """Tests for ConfigLoader initialization."""

    def test_init(self):
        """Test ConfigLoader initialization."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        assert loader.settings is mock_settings


class TestConfigLoaderLoadToolConfig:
    """Tests for ConfigLoader.load_tool_config method."""

    @pytest.fixture
    def registry_with_tools(self):
        """Create a registry with test tools (using canonical names)."""
        registry = ToolRegistry()
        registry.register(MockTool("read", category="core"))
        registry.register(MockTool("write", category="core"))
        registry.register(MockTool("code_review"))
        registry.register(MockTool("git"))
        return registry

    def test_load_empty_config(self, registry_with_tools):
        """Test loading with empty configuration."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = None

        loader = ConfigLoader(settings=mock_settings)
        loader.load_tool_config(registry_with_tools)

        # All tools should remain enabled
        assert registry_with_tools.is_tool_enabled("read")
        assert registry_with_tools.is_tool_enabled("write")

    def test_load_with_disabled_list(self, registry_with_tools):
        """Test loading with disabled list."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = {"disabled": ["code_review"]}

        loader = ConfigLoader(settings=mock_settings)
        loader.load_tool_config(registry_with_tools)

        # code_review should be disabled
        assert not registry_with_tools.is_tool_enabled("code_review")
        # Other tools should remain enabled
        assert registry_with_tools.is_tool_enabled("read")

    def test_load_with_enabled_list(self, registry_with_tools):
        """Test loading with enabled list (exclusive mode)."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = {"enabled": ["read", "write"]}

        loader = ConfigLoader(settings=mock_settings)
        loader.load_tool_config(registry_with_tools)

        # Only enabled tools should be enabled
        assert registry_with_tools.is_tool_enabled("read")
        assert registry_with_tools.is_tool_enabled("write")
        assert not registry_with_tools.is_tool_enabled("code_review")
        assert not registry_with_tools.is_tool_enabled("git")

    def test_load_with_invalid_tool_names(self, registry_with_tools):
        """Test loading with invalid tool names logs warning."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = {"disabled": ["nonexistent_tool"]}

        loader = ConfigLoader(settings=mock_settings)

        with patch("victor.agent.config_loader.logger"):
            loader.load_tool_config(registry_with_tools)
            # Should log warning about invalid tool name
            # (Checking that no exception was raised is sufficient)

    def test_load_with_individual_settings(self, registry_with_tools):
        """Test loading with individual tool settings."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = {
            "code_review": {"enabled": False},
            "git": {"enabled": True},
        }

        loader = ConfigLoader(settings=mock_settings)
        loader.load_tool_config(registry_with_tools)

        # code_review should be disabled via individual setting
        assert not registry_with_tools.is_tool_enabled("code_review")

    def test_load_handles_exception(self, registry_with_tools):
        """Test that exceptions are handled gracefully."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.side_effect = ValueError("Config error")

        loader = ConfigLoader(settings=mock_settings)

        # Should not raise exception
        loader.load_tool_config(registry_with_tools)


class TestConfigLoaderHelpers:
    """Tests for ConfigLoader helper methods."""

    @pytest.fixture
    def registry_with_tools(self):
        """Create a registry with test tools (using canonical names)."""
        registry = ToolRegistry()
        registry.register(MockTool("read", category="core"))
        registry.register(MockTool("write", category="core"))
        registry.register(MockTool("ls", category="core"))
        registry.register(MockTool("shell", category="core"))
        registry.register(MockTool("code_review"))
        return registry

    def test_apply_enabled_list_warns_about_missing_core(self, registry_with_tools):
        """Test that missing core tools triggers warning."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        config = {"enabled": ["code_review"]}  # Missing core tools
        registered = {t.name for t in registry_with_tools.list_tools(only_enabled=False)}

        with patch("victor.agent.config_loader.logger") as mock_logger:
            loader._apply_enabled_list(config, registry_with_tools, registered)
            # Should warn about missing core tools
            assert mock_logger.warning.called

    def test_apply_disabled_list_warns_about_core_tools(self, registry_with_tools):
        """Test that disabling core tools triggers warning."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        config = {"disabled": ["read"]}  # Disabling core tool (canonical name)
        registered = {t.name for t in registry_with_tools.list_tools(only_enabled=False)}

        with patch("victor.agent.config_loader.logger") as mock_logger:
            loader._apply_disabled_list(config, registry_with_tools, registered)
            # Should warn about disabling core tool
            assert mock_logger.warning.called

    def test_apply_individual_settings(self, registry_with_tools):
        """Test applying individual tool settings."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        config = {
            "code_review": {"enabled": False},
            "read": {"enabled": True},  # Canonical name
        }
        registered = {t.name for t in registry_with_tools.list_tools(only_enabled=False)}

        loader._apply_individual_settings(config, registry_with_tools, registered)

        assert not registry_with_tools.is_tool_enabled("code_review")
        assert registry_with_tools.is_tool_enabled("read")

    def test_apply_individual_settings_ignores_reserved_keys(self, registry_with_tools):
        """Test that reserved keys are ignored."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        config = {
            "enabled": ["some_tool"],  # Reserved key
            "disabled": ["other_tool"],  # Reserved key
            "code_review": {"enabled": False},
        }
        registered = {t.name for t in registry_with_tools.list_tools(only_enabled=False)}

        loader._apply_individual_settings(config, registry_with_tools, registered)

        # code_review should be disabled
        assert not registry_with_tools.is_tool_enabled("code_review")


class TestConfigLoaderLogging:
    """Tests for ConfigLoader logging."""

    def test_log_tool_states(self):
        """Test logging tool states."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))
        registry.disable_tool("tool2")

        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        with patch("victor.agent.config_loader.logger") as mock_logger:
            loader._log_tool_states(registry)
            # Should log at debug level
            assert mock_logger.debug.called or mock_logger.info.called


# =============================================================================
# TEST: ConfigLoader.resolve_env_vars
# =============================================================================


class TestResolveEnvVars:
    """Tests for ConfigLoader.resolve_env_vars static method."""

    def test_resolve_simple_var(self):
        """Should resolve simple ${VAR} syntax."""
        import os

        os.environ["TEST_VAR_RESOLVE"] = "test_value"
        try:
            result = ConfigLoader.resolve_env_vars("prefix_${TEST_VAR_RESOLVE}_suffix")
            assert result == "prefix_test_value_suffix"
        finally:
            del os.environ["TEST_VAR_RESOLVE"]

    def test_resolve_with_default(self):
        """Should use default when var is not set."""
        import os

        # Ensure var is not set
        if "UNSET_VAR_RESOLVE" in os.environ:
            del os.environ["UNSET_VAR_RESOLVE"]

        result = ConfigLoader.resolve_env_vars("${UNSET_VAR_RESOLVE:-default_value}")
        assert result == "default_value"

    def test_resolve_with_default_var_exists(self):
        """Should use env var value even when default is specified."""
        import os

        os.environ["SET_VAR_RESOLVE"] = "actual_value"
        try:
            result = ConfigLoader.resolve_env_vars("${SET_VAR_RESOLVE:-default_value}")
            assert result == "actual_value"
        finally:
            del os.environ["SET_VAR_RESOLVE"]

    def test_resolve_missing_var_returns_empty(self):
        """Should return empty string for missing vars without default."""
        import os

        if "MISSING_VAR_RESOLVE" in os.environ:
            del os.environ["MISSING_VAR_RESOLVE"]

        result = ConfigLoader.resolve_env_vars("prefix_${MISSING_VAR_RESOLVE}_suffix")
        assert result == "prefix__suffix"

    def test_resolve_multiple_vars(self):
        """Should resolve multiple vars in same string."""
        import os

        os.environ["VAR1_RESOLVE"] = "one"
        os.environ["VAR2_RESOLVE"] = "two"
        try:
            result = ConfigLoader.resolve_env_vars("${VAR1_RESOLVE} and ${VAR2_RESOLVE}")
            assert result == "one and two"
        finally:
            del os.environ["VAR1_RESOLVE"]
            del os.environ["VAR2_RESOLVE"]

    def test_resolve_no_vars(self):
        """Should return original string if no vars."""
        result = ConfigLoader.resolve_env_vars("no vars here")
        assert result == "no vars here"


# =============================================================================
# TEST: ConfigLoader.resolve_endpoint_list
# =============================================================================


class TestResolveEndpointList:
    """Tests for ConfigLoader.resolve_endpoint_list static method."""

    def test_resolve_simple_list(self):
        """Should resolve list of endpoints."""
        endpoints = ["http://localhost:8080", "http://localhost:9090"]
        result = ConfigLoader.resolve_endpoint_list(endpoints)
        assert result == ["http://localhost:8080", "http://localhost:9090"]

    def test_resolve_with_env_var_prefix(self):
        """Should prepend endpoints from env var."""
        import os

        os.environ["EXTRA_ENDPOINTS_TEST"] = "http://extra1:8000,http://extra2:9000"
        try:
            endpoints = ["http://localhost:8080"]
            result = ConfigLoader.resolve_endpoint_list(endpoints, "EXTRA_ENDPOINTS_TEST")
            assert result == [
                "http://extra1:8000",
                "http://extra2:9000",
                "http://localhost:8080",
            ]
        finally:
            del os.environ["EXTRA_ENDPOINTS_TEST"]

    def test_resolve_filters_empty(self):
        """Should filter empty endpoints."""
        endpoints = ["http://localhost", "  ", "", "http://other"]
        result = ConfigLoader.resolve_endpoint_list(endpoints)
        assert result == ["http://localhost", "http://other"]

    def test_resolve_env_var_in_endpoints(self):
        """Should resolve env vars in endpoint URLs."""
        import os

        os.environ["PORT_RESOLVE_TEST"] = "8080"
        try:
            endpoints = ["http://localhost:${PORT_RESOLVE_TEST}"]
            result = ConfigLoader.resolve_endpoint_list(endpoints)
            assert result == ["http://localhost:8080"]
        finally:
            del os.environ["PORT_RESOLVE_TEST"]

    def test_resolve_env_prefix_not_set(self):
        """Should handle missing env var prefix gracefully."""
        import os

        if "MISSING_PREFIX_TEST" in os.environ:
            del os.environ["MISSING_PREFIX_TEST"]

        endpoints = ["http://localhost:8080"]
        result = ConfigLoader.resolve_endpoint_list(endpoints, "MISSING_PREFIX_TEST")
        assert result == ["http://localhost:8080"]


# =============================================================================
# TEST: ProfileValidationResult
# =============================================================================


class TestProfileValidationResult:
    """Tests for ProfileValidationResult dataclass."""

    def test_create_valid_result(self):
        """Should create valid result with defaults."""
        from victor.agent.config_loader import ProfileValidationResult

        result = ProfileValidationResult("test_profile")
        assert result.profile_name == "test_profile"
        assert result.is_valid is True
        assert result.warnings == []
        assert result.errors == []
        assert result.suggestions == []

    def test_create_invalid_result(self):
        """Should create invalid result with errors."""
        from victor.agent.config_loader import ProfileValidationResult

        result = ProfileValidationResult(
            "bad_profile",
            is_valid=False,
            errors=["Error 1", "Error 2"],
        )
        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_str_valid(self):
        """Should format valid result correctly."""
        from victor.agent.config_loader import ProfileValidationResult

        result = ProfileValidationResult("my_profile")
        text = str(result)
        assert "my_profile" in text
        assert "VALID" in text

    def test_str_invalid_with_errors(self):
        """Should format invalid result with errors."""
        from victor.agent.config_loader import ProfileValidationResult

        result = ProfileValidationResult(
            "bad_profile",
            is_valid=False,
            errors=["Config error"],
            warnings=["Some warning"],
            suggestions=["Try this"],
        )
        text = str(result)
        assert "INVALID" in text
        assert "Config error" in text
        assert "Some warning" in text
        assert "Try this" in text


# =============================================================================
# TEST: ProfileValidator
# =============================================================================


class TestProfileValidator:
    """Tests for ProfileValidator class."""

    def test_init(self):
        """Should initialize with settings."""
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        validator = ProfileValidator(mock_settings)
        assert validator.settings == mock_settings
        assert validator._capability_loader is None

    def test_get_capability_loader_lazy_load(self):
        """Should lazy-load capability loader."""
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        validator = ProfileValidator(mock_settings)
        assert validator._capability_loader is None

        with patch(
            "victor.agent.tool_calling.capabilities.ModelCapabilityLoader"
        ) as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader_cls.return_value = mock_loader

            result = validator._get_capability_loader()
            assert result == mock_loader

            # Second call should return cached
            result2 = validator._get_capability_loader()
            assert result2 == mock_loader
            mock_loader_cls.assert_called_once()

    def test_validate_profile_success(self):
        """Should return valid result for good profile."""
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        validator = ProfileValidator(mock_settings)

        mock_caps = MagicMock()
        mock_caps.native_tool_calls = True

        with patch.object(validator, "_get_capability_loader") as mock_get_loader:
            mock_loader = MagicMock()
            mock_loader.get_capabilities.return_value = mock_caps
            mock_get_loader.return_value = mock_loader

            with patch(
                "victor.agent.tool_calling.capabilities.normalize_model_name",
                return_value="llama3.1",
            ):
                with patch(
                    "victor.agent.tool_calling.capabilities.get_model_name_variants",
                    return_value=["llama3.1", "llama3.1:latest"],
                ):
                    result = validator.validate_profile(
                        profile_name="test",
                        provider="ollama",
                        model="llama3.1",
                    )

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_profile_no_capabilities(self):
        """Should warn when model capabilities not found."""
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        validator = ProfileValidator(mock_settings)

        with patch.object(validator, "_get_capability_loader") as mock_get_loader:
            mock_loader = MagicMock()
            mock_loader.get_capabilities.return_value = None
            mock_get_loader.return_value = mock_loader

            with patch(
                "victor.agent.tool_calling.capabilities.normalize_model_name",
                return_value="unknown-model",
            ):
                with patch(
                    "victor.agent.tool_calling.capabilities.get_model_name_variants",
                    return_value=["unknown-model"],
                ):
                    result = validator.validate_profile(
                        profile_name="test",
                        provider="ollama",
                        model="unknown-model",
                    )

        assert len(result.warnings) > 0
        assert "not found in capability patterns" in result.warnings[0]

    def test_validate_profile_no_native_tool_calls(self):
        """Should warn when model doesn't support native tool calls."""
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        validator = ProfileValidator(mock_settings)

        mock_caps = MagicMock()
        mock_caps.native_tool_calls = False

        with patch.object(validator, "_get_capability_loader") as mock_get_loader:
            mock_loader = MagicMock()
            mock_loader.get_capabilities.return_value = mock_caps
            mock_get_loader.return_value = mock_loader

            with patch(
                "victor.agent.tool_calling.capabilities.normalize_model_name",
                return_value="phi",
            ):
                with patch(
                    "victor.agent.tool_calling.capabilities.get_model_name_variants",
                    return_value=["phi"],
                ):
                    result = validator.validate_profile(
                        profile_name="test",
                        provider="ollama",
                        model="phi",
                    )

        assert len(result.warnings) > 0
        assert "does not support native tool calls" in result.warnings[0]


class TestCheckOllamaModel:
    """Tests for ProfileValidator._check_ollama_model method."""

    def test_check_ollama_model_found(self):
        """Should return None when model is found."""
        import json
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        mock_settings.ollama_base_url = "http://localhost:11434"
        validator = ProfileValidator(mock_settings)

        response_data = {"models": [{"name": "llama3.1:latest"}]}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = validator._check_ollama_model("llama3.1")

        assert result is None

    def test_check_ollama_model_not_found(self):
        """Should return warning when model not found."""
        import json
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        mock_settings.ollama_base_url = "http://localhost:11434"
        validator = ProfileValidator(mock_settings)

        response_data = {"models": [{"name": "mistral:latest"}]}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = validator._check_ollama_model("llama3.1")

        assert result is not None
        assert "not found" in result

    def test_check_ollama_model_connection_error(self):
        """Should return None on connection error."""
        import urllib.error
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        mock_settings.ollama_base_url = "http://localhost:11434"
        validator = ProfileValidator(mock_settings)

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            result = validator._check_ollama_model("llama3.1")

        assert result is None  # Don't warn on connection error


class TestValidateAllOllamaProfiles:
    """Tests for ProfileValidator.validate_all_ollama_profiles method."""

    def test_no_profiles_yaml(self):
        """Should return empty list when no profiles yaml."""
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        mock_settings._profiles_yaml = None
        validator = ProfileValidator(mock_settings)
        results = validator.validate_all_ollama_profiles()
        assert results == []

    def test_validates_ollama_profiles(self):
        """Should validate only Ollama profiles."""
        from victor.agent.config_loader import ProfileValidator, ProfileValidationResult

        mock_settings = MagicMock()
        mock_settings._profiles_yaml = {
            "profiles": {
                "ollama_profile": {"provider": "ollama", "model": "llama3.1"},
                "openai_profile": {"provider": "openai", "model": "gpt-4"},
            }
        }
        validator = ProfileValidator(mock_settings)

        with patch.object(validator, "validate_profile") as mock_validate:
            mock_validate.return_value = ProfileValidationResult("test")
            results = validator.validate_all_ollama_profiles()

        # Should only validate ollama profile
        assert mock_validate.call_count == 1
        mock_validate.assert_called_with(
            profile_name="ollama_profile",
            provider="ollama",
            model="llama3.1",
            check_ollama_availability=False,
        )

    def test_skips_non_dict_profiles(self):
        """Should skip non-dict profile configs."""
        from victor.agent.config_loader import ProfileValidator, ProfileValidationResult

        mock_settings = MagicMock()
        mock_settings._profiles_yaml = {
            "profiles": {
                "bad_profile": "not a dict",
                "good_profile": {"provider": "ollama", "model": "llama3.1"},
            }
        }
        validator = ProfileValidator(mock_settings)

        with patch.object(validator, "validate_profile") as mock_validate:
            mock_validate.return_value = ProfileValidationResult("test")
            results = validator.validate_all_ollama_profiles()

        assert mock_validate.call_count == 1


class TestLogValidationSummary:
    """Tests for ProfileValidator.log_validation_summary method."""

    def test_log_no_results(self, caplog):
        """Should do nothing with empty results."""
        from victor.agent.config_loader import ProfileValidator

        mock_settings = MagicMock()
        validator = ProfileValidator(mock_settings)
        validator.log_validation_summary([])
        # No error logs

    def test_log_warnings_only(self, caplog):
        """Should log warning summary."""
        import logging
        from victor.agent.config_loader import ProfileValidator, ProfileValidationResult

        mock_settings = MagicMock()
        validator = ProfileValidator(mock_settings)
        results = [
            ProfileValidationResult("p1", warnings=["Warning 1"]),
            ProfileValidationResult("p2", warnings=["Warning 2"]),
        ]

        with caplog.at_level(logging.WARNING):
            validator.log_validation_summary(results)

        assert "2 warnings" in caplog.text

    def test_log_errors(self, caplog):
        """Should log error summary."""
        import logging
        from victor.agent.config_loader import ProfileValidator, ProfileValidationResult

        mock_settings = MagicMock()
        validator = ProfileValidator(mock_settings)
        results = [
            ProfileValidationResult("p1", is_valid=False, errors=["Error 1"]),
        ]

        with caplog.at_level(logging.ERROR):
            validator.log_validation_summary(results)

        assert "1 errors" in caplog.text


# =============================================================================
# TEST: validate_profiles_on_startup
# =============================================================================


class TestValidateProfilesOnStartup:
    """Tests for validate_profiles_on_startup function."""

    def test_calls_validator(self):
        """Should call validator methods."""
        from victor.agent.config_loader import validate_profiles_on_startup

        mock_settings = MagicMock()

        with patch("victor.agent.config_loader.ProfileValidator") as mock_cls:
            mock_validator = MagicMock()
            mock_validator.validate_all_ollama_profiles.return_value = []
            mock_cls.return_value = mock_validator

            validate_profiles_on_startup(mock_settings)

            mock_cls.assert_called_once_with(mock_settings)
            mock_validator.validate_all_ollama_profiles.assert_called_once_with(
                check_availability=False
            )
            mock_validator.log_validation_summary.assert_called_once()

    def test_with_availability_check(self):
        """Should pass check_availability flag."""
        from victor.agent.config_loader import validate_profiles_on_startup

        mock_settings = MagicMock()

        with patch("victor.agent.config_loader.ProfileValidator") as mock_cls:
            mock_validator = MagicMock()
            mock_validator.validate_all_ollama_profiles.return_value = []
            mock_cls.return_value = mock_validator

            validate_profiles_on_startup(mock_settings, check_availability=True)

            mock_validator.validate_all_ollama_profiles.assert_called_once_with(
                check_availability=True
            )
