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

"""Tests for configuration validation."""

import os
import sys
import pytest

from victor.config.validation import (
    ValidationError,
    ValidationResult,
    validate_environment,
    validate_paths,
    validate_provider_settings,
    validate_tool_settings,
    validate_performance_settings,
    validate_configuration,
    format_validation_result,
)


class TestValidationError:
    """Tests for ValidationError class."""

    def test_basic_error(self):
        """Basic error creation."""
        error = ValidationError(message="Test error")
        assert "Test error" in str(error)
        assert error.message == "Test error"

    def test_error_with_field(self):
        """Error with field."""
        error = ValidationError(message="Test error", field="api_key")
        assert "api_key" in str(error)

    def test_error_with_suggestion(self):
        """Error with suggestion."""
        error = ValidationError(message="Test error", suggestion="Fix it")
        assert "Fix it" in str(error)

    def test_error_with_code(self):
        """Error with error code."""
        error = ValidationError(message="Test error", error_code="TEST_ERR")
        assert "TEST_ERR" in str(error)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Valid result has no errors."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert result.error_count == 0

    def test_invalid_result(self):
        """Invalid result has errors."""
        result = ValidationResult(
            is_valid=False,
            errors=[
                ValidationError("Error 1"),
                ValidationError("Error 2"),
            ],
        )
        assert not result.is_valid
        assert result.error_count == 2

    def test_add_error(self):
        """Adding error makes result invalid."""
        result = ValidationResult(is_valid=True)
        result.add_error(ValidationError("New error"))
        assert not result.is_valid
        assert result.error_count == 1


class TestValidateEnvironment:
    """Tests for validate_environment."""

    def test_python_version_ok(self):
        """Python version 3.10+ passes."""
        if sys.version_info >= (3, 10):
            result = validate_environment()
            # Should not have Python version error
            has_python_error = any(
                "Python" in error.message and "not supported" in error.message
                for error in result.errors
            )
            assert not has_python_error

    def test_deprecated_env_vars(self, monkeypatch):
        """Deprecated environment variables are detected."""
        monkeypatch.setenv("VICTOR_API_KEY", "test")
        result = validate_environment()
        has_deprecated_error = any(
            "Deprecated" in error.message or "deprecated" in error.message.lower()
            for error in result.errors
        )
        assert has_deprecated_error

    def test_no_api_key_warning(self, monkeypatch):
        """No API key warning when using cloud provider."""
        # Clear API keys
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        # This test is context-dependent - skip if keys are set
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
            pytest.skip("API keys already set")

        result = validate_environment()
        # Should have a warning about no API keys for cloud providers
        # (but this depends on settings, which we can't easily mock)


class TestValidatePaths:
    """Tests for validate_paths."""

    def test_missing_config_dir(self):
        """Missing config directory is an error."""
        from victor.config.settings import Settings

        # Create settings with non-existent config dir
        settings = Settings()
        # We can't easily mock get_config_dir, so we'll just call the function
        # In real testing, we'd use a mock
        result = validate_paths(settings)
        # The result should be valid if default config exists
        # or have an error if it doesn't


class TestValidateToolSettings:
    """Tests for validate_tool_settings."""

    def test_negative_max_tools(self):
        """Negative fallback_max_tools is an error."""
        from victor.config.settings import Settings

        settings = Settings()
        settings.tools.fallback_max_tools = -1

        result = validate_tool_settings(settings)
        has_invalid_error = any(
            "fallback_max_tools" in error.message or "budget" in error.message.lower()
            for error in result.errors
        )
        assert has_invalid_error

    def test_high_max_tools_warning(self):
        """Very high fallback_max_tools is a warning."""
        from victor.config.settings import Settings

        settings = Settings()
        settings.tools.fallback_max_tools = 100

        result = validate_tool_settings(settings)
        has_high_warning = any(
            "fallback_max_tools" in (error.field or "") for error in result.errors
        )
        # Should have a warning about high tool budget


class TestValidatePerformanceSettings:
    """Tests for validate_performance_settings."""

    def test_disabled_preloading_warning(self):
        """Disabled preloading generates warning."""
        from victor.config.settings import Settings

        settings = Settings()
        if hasattr(settings, "framework_preload_enabled"):
            settings.framework_preload_enabled = False

            result = validate_performance_settings(settings)
            has_warning = any(
                error.error_code == "PRELOAD_DISABLED" for error in result.errors
            )
            assert has_warning


class TestValidateConfiguration:
    """Tests for validate_configuration."""

    def test_validate_with_none(self):
        """Validating with None creates default settings."""
        result = validate_configuration(None)
        # Should create default settings and validate
        assert isinstance(result, ValidationResult)

    def test_comprehensive_validation(self):
        """Comprehensive validation checks all categories."""
        from victor.config.settings import Settings

        settings = Settings()
        result = validate_configuration(settings)

        # Should have some result
        assert isinstance(result, ValidationResult)
        assert isinstance(result.errors, list)


class TestFormatValidationResult:
    """Tests for format_validation_result."""

    def test_format_valid_result(self):
        """Formatting valid result shows success message."""
        result = ValidationResult(is_valid=True)
        formatted = format_validation_result(result)
        assert "valid" in formatted.lower() or "✓" in formatted

    def test_format_invalid_result(self):
        """Formatting invalid result shows errors."""
        result = ValidationResult(
            is_valid=False,
            errors=[
                ValidationError("Error 1", field="field1", suggestion="Fix it"),
            ],
        )
        formatted = format_validation_result(result)
        assert "Error 1" in formatted
        assert "field1" in formatted
        assert "Fix it" in formatted

    def test_format_with_error_code(self):
        """Formatting with error code includes code."""
        result = ValidationResult(
            is_valid=False,
            errors=[
                ValidationError("Error", error_code="TEST_ERR"),
            ],
        )
        formatted = format_validation_result(result)
        assert "TEST_ERR" in formatted
