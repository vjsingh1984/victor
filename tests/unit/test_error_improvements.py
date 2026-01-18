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

"""Tests for improved error messages.

Tests Phase 1 Quick Wins:
- Provider initialization errors with available providers list
- Tool execution errors with tool name and arguments
- Configuration validation errors with field names and recovery hints
"""

import pytest
from unittest.mock import Mock, patch

from victor.core.errors import (
    ProviderNotFoundError,
    ProviderInitializationError,
    ToolExecutionError,
    ConfigurationError,
    ConfigurationValidationError,
)


class TestProviderErrors:
    """Test improved provider error messages."""

    def test_provider_not_found_shows_available_providers(self):
        """Test ProviderNotFoundError shows list of available providers."""
        available_providers = ["anthropic", "openai", "ollama", "deepseek"]

        error = ProviderNotFoundError(
            provider="nonexistent",
            available_providers=available_providers,
        )

        # Check message includes available providers
        assert "nonexistent" in str(error)
        assert any(p in str(error) for p in available_providers[:3])

        # Check details include available providers
        assert error.available_providers == available_providers
        assert error.details["available_providers"] == available_providers

        # Check recovery hint
        assert error.recovery_hint is not None
        assert "victor providers" in error.recovery_hint.lower()

    def test_provider_initialization_error_with_config_key(self):
        """Test ProviderInitializationError includes config key in recovery hint."""
        error = ProviderInitializationError(
            message="Failed to initialize provider",
            provider="anthropic",
            config_key="ANTHROPIC_API_KEY",
        )

        # Check message
        assert "anthropic" in str(error).lower()
        assert "Failed to initialize" in str(error)

        # Check config key is stored
        assert error.config_key == "ANTHROPIC_API_KEY"
        assert error.details["config_key"] == "ANTHROPIC_API_KEY"

        # Check recovery hint includes config key
        assert error.recovery_hint is not None
        assert "ANTHROPIC_API_KEY" in error.recovery_hint

    def test_provider_initialization_error_without_config_key(self):
        """Test ProviderInitializationError generates generic recovery hint."""
        error = ProviderInitializationError(
            message="Failed to initialize provider",
            provider="openai",
        )

        # Check recovery hint is generic
        assert error.recovery_hint is not None
        assert "API credentials" in error.recovery_hint or "configuration" in error.recovery_hint

    def test_provider_error_correlation_id(self):
        """Test provider errors include correlation IDs."""
        error = ProviderInitializationError(
            message="Test error",
            provider="test",
        )

        # Check correlation ID exists
        assert error.correlation_id is not None
        assert len(error.correlation_id) == 8  # Short format
        assert error.correlation_id in str(error)


class TestToolExecutionErrors:
    """Test improved tool execution error messages."""

    def test_tool_execution_error_includes_tool_name(self):
        """Test ToolExecutionError includes tool name."""
        error = ToolExecutionError(
            message="Tool 'code_search' execution failed",
            tool_name="code_search",
            arguments={"query": "test"},
        )

        # Check tool name in message
        assert "code_search" in str(error)

        # Check tool name in details
        assert error.tool_name == "code_search"
        assert error.details["tool_name"] == "code_search"

    def test_tool_execution_error_includes_arguments(self):
        """Test ToolExecutionError includes tool arguments."""
        args = {"query": "test", "file_pattern": "*.py"}
        error = ToolExecutionError(
            message="Tool failed",
            tool_name="grep",
            arguments=args,
        )

        # Check arguments are stored
        assert error.arguments == args
        assert error.details["arguments"] == args

    def test_tool_execution_error_recovery_hint(self):
        """Test ToolExecutionError includes recovery hint."""
        error = ToolExecutionError(
            message="Tool 'read' failed",
            tool_name="read",
            arguments={"path": "nonexistent.txt"},
            recovery_hint="Check the file path and try again",
        )

        # Check recovery hint
        assert error.recovery_hint is not None
        assert "file path" in error.recovery_hint.lower() or "check" in error.recovery_hint.lower()

    def test_tool_execution_error_correlation_id(self):
        """Test ToolExecutionError includes correlation ID."""
        error = ToolExecutionError(
            message="Tool failed",
            tool_name="test_tool",
        )

        # Check correlation ID
        assert error.correlation_id is not None
        assert len(error.correlation_id) == 8


class TestConfigurationErrors:
    """Test improved configuration error messages."""

    def test_configuration_error_with_invalid_fields(self):
        """Test ConfigurationError includes list of invalid fields."""
        invalid_fields = ["timeout", "max_iterations", "tool_budget"]
        error = ConfigurationError(
            message="Validation failed",
            config_key="workflow.my_workflow",
            invalid_fields=invalid_fields,
        )

        # Check invalid fields are stored
        assert error.invalid_fields == invalid_fields
        assert error.details["invalid_fields"] == invalid_fields

        # Check config key
        assert error.config_key == "workflow.my_workflow"
        assert error.details["config_key"] == "workflow.my_workflow"

    def test_configuration_error_recovery_hint(self):
        """Test ConfigurationError includes actionable recovery hint."""
        error = ConfigurationError(
            message="YAML validation failed",
            config_key="workflow.test",
            recovery_hint="Use 'victor workflow validate <path>' to check",
        )

        # Check recovery hint
        assert error.recovery_hint is not None
        assert "victor workflow validate" in error.recovery_hint

    def test_configuration_error_empty_invalid_fields(self):
        """Test ConfigurationError handles empty invalid_fields list."""
        error = ConfigurationError(
            message="Validation failed",
            config_key="workflow.test",
            invalid_fields=[],
        )

        # Should not crash
        assert error.invalid_fields == []
        assert error.details["invalid_fields"] == []

    def test_configuration_error_none_invalid_fields(self):
        """Test ConfigurationError handles None invalid_fields."""
        error = ConfigurationError(
            message="Validation failed",
            config_key="workflow.test",
            invalid_fields=None,
        )

        # Should default to empty list
        assert error.invalid_fields == []
        assert error.details["invalid_fields"] == []


class TestIntegrationScenarios:
    """Test integration scenarios for improved error messages."""

    def test_extract_error_info_from_provider_error(self):
        """Test extracting structured info from provider error."""
        error = ProviderNotFoundError(
            provider="bad_provider",
            available_providers=["good_provider1", "good_provider2"],
        )

        # Check to_dict() includes all info
        error_dict = error.to_dict()
        assert "error" in error_dict
        assert "correlation_id" in error_dict
        assert "recovery_hint" in error_dict
        assert "details" in error_dict
        assert error_dict["details"]["provider"] == "bad_provider"

    def test_extract_error_info_from_tool_error(self):
        """Test extracting structured info from tool error."""
        error = ToolExecutionError(
            message="Tool failed",
            tool_name="test_tool",
            arguments={"param": "value"},
        )

        # Check to_dict() includes all info
        error_dict = error.to_dict()
        assert "error" in error_dict
        assert "correlation_id" in error_dict
        assert error_dict["details"]["tool_name"] == "test_tool"
        assert error_dict["details"]["arguments"] == {"param": "value"}

    def test_extract_error_info_from_config_error(self):
        """Test extracting structured info from config error."""
        error = ConfigurationError(
            message="Config validation failed",
            config_key="workflow.test",
            invalid_fields=["field1", "field2"],
        )

        # Check to_dict() includes all info
        error_dict = error.to_dict()
        assert "error" in error_dict
        assert error_dict["details"]["config_key"] == "workflow.test"
        assert error_dict["details"]["invalid_fields"] == ["field1", "field2"]

    def test_error_string_formatting(self):
        """Test error __str__ includes recovery hint."""
        error = ProviderInitializationError(
            message="Init failed",
            provider="test",
            config_key="TEST_KEY",
        )

        error_str = str(error)
        assert "Init failed" in error_str
        # The __str__ method includes recovery hint in format: [correlation_id] message\nRecovery hint: hint
        assert "Recovery hint:" in error_str or "recovery_hint" in error_str.lower()


class TestConfigurationValidationErrors:
    """Test ConfigurationValidationError improvements."""

    def test_validation_error_with_field_errors(self):
        """Test ConfigurationValidationError includes field errors."""
        field_errors = {
            "start_node": "Start node 'init' not found in workflow",
            "tool_budget": "Tool budget must be positive integer",
        }
        invalid_fields = ["start_node", "tool_budget"]
        validation_errors = list(field_errors.values())

        error = ConfigurationValidationError(
            message="Workflow validation failed with 2 errors",
            config_key="test_workflow.yaml",
            invalid_fields=invalid_fields,
            field_errors=field_errors,
            validation_errors=validation_errors,
        )

        # Verify error attributes
        assert error.config_key == "test_workflow.yaml"
        assert error.invalid_fields == invalid_fields
        assert error.field_errors == field_errors
        assert error.validation_errors == validation_errors

        # Verify in details
        assert error.details["field_errors"] == field_errors
        assert error.details["validation_errors"] == validation_errors

    def test_validation_error_with_line_numbers(self):
        """Test ConfigurationValidationError includes line numbers."""
        line_numbers = {"start_node": 15, "tool_budget": 23}

        error = ConfigurationValidationError(
            message="Workflow validation failed",
            config_key="workflow.yaml",
            line_numbers=line_numbers,
        )

        assert error.line_numbers == line_numbers
        assert "line_numbers" in error.details

    def test_validation_error_get_field_error(self):
        """Test get_field_error method."""
        field_errors = {
            "start_node": "Start node 'init' not found",
            "timeout": "Timeout must be positive",
        }

        error = ConfigurationValidationError(
            message="Validation failed",
            field_errors=field_errors,
        )

        # Test getting specific field errors
        assert error.get_field_error("start_node") == "Start node 'init' not found"
        assert error.get_field_error("timeout") == "Timeout must be positive"
        assert error.get_field_error("nonexistent") is None

    def test_validation_error_get_line_number(self):
        """Test get_line_number method."""
        line_numbers = {"start_node": 15, "timeout": 30}

        error = ConfigurationValidationError(
            message="Validation failed",
            line_numbers=line_numbers,
        )

        # Test getting line numbers
        assert error.get_line_number("start_node") == 15
        assert error.get_line_number("timeout") == 30
        assert error.get_line_number("nonexistent") is None

    def test_validation_error_enhances_message(self):
        """Test ConfigurationValidationError enhances short messages."""
        validation_errors = [
            "Workflow must have a name",
            "Start node 'init' not found",
            "Tool budget must be positive",
        ]

        error = ConfigurationValidationError(
            message="Validation failed",  # Short message
            validation_errors=validation_errors,
        )

        # Message should be enhanced with validation errors
        assert "3 error" in error.message
        assert "Workflow must have a name" in error.message

    def test_validation_error_yaml_recovery_hint(self):
        """Test ConfigurationValidationError generates YAML-specific recovery hint."""
        error = ConfigurationValidationError(
            message="Validation failed",
            config_key="workflow.yaml",
        )

        # Recovery hint should mention victor workflow validate
        assert error.recovery_hint is not None
        assert "victor workflow validate" in error.recovery_hint.lower()

    def test_validation_error_to_dict(self):
        """Test ConfigurationValidationError serialization."""
        field_errors = {"field1": "error1"}
        line_numbers = {"field1": 10}
        validation_errors = ["error1", "error2"]

        error = ConfigurationValidationError(
            message="Validation failed",
            config_key="test.yaml",
            field_errors=field_errors,
            line_numbers=line_numbers,
            validation_errors=validation_errors,
        )

        error_dict = error.to_dict()

        # Should include all validation details
        assert "field_errors" in error_dict
        assert "line_numbers" in error_dict
        assert "validation_errors" in error_dict
        assert error_dict["field_errors"] == field_errors
        assert error_dict["line_numbers"] == line_numbers
        assert error_dict["validation_errors"] == validation_errors

    def test_validation_error_generates_recovery_hint_for_non_yaml(self):
        """Test ConfigurationValidationError auto-generates recovery hint for non-YAML."""
        error = ConfigurationValidationError(
            message="Validation failed",
            config_key="config.json",
        )

        # Recovery hint should not mention victor workflow validate for non-YAML
        assert error.recovery_hint is not None
        assert "config.json" in error.recovery_hint
        assert "victor workflow validate" not in error.recovery_hint
