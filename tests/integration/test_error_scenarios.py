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

"""Integration tests for error scenarios.

These tests verify the complete user experience when errors occur,
including error handling, reporting, and recovery.
"""

import pytest
from pathlib import Path

from victor.core.errors import (
    ProviderNotFoundError,
    ProviderInitializationError,
    ToolNotFoundError,
    ToolExecutionError,
    ConfigurationError,
    ValidationError,
)
from victor.observability.error_tracker import get_error_tracker, reset_error_tracker


@pytest.fixture(autouse=True)
def reset_error_tracker_fixture():
    """Reset error tracker before each test."""
    reset_error_tracker()
    yield
    reset_error_tracker()


class TestProviderErrorScenarios:
    """Test user experience when provider errors occur."""

    def test_provider_not_found_scenario(self):
        """Test user experience when provider not found."""
        # Simulate user attempting to use non-existent provider
        with pytest.raises(ProviderNotFoundError) as exc_info:
            # This would typically be called from provider registry
            raise ProviderNotFoundError(
                provider="nonexistent_provider",
                available_providers=["anthropic", "openai", "ollama"],
            )

        error = exc_info.value

        # Verify error has all required fields
        assert "nonexistent_provider" in str(error)
        assert error.provider == "nonexistent_provider"
        assert error.available_providers == ["anthropic", "openai", "ollama"]
        assert error.correlation_id is not None
        assert len(error.correlation_id) == 8

        # Verify recovery hint is actionable
        assert error.recovery_hint is not None
        assert "victor providers" in error.recovery_hint.lower()

        # Verify error was tracked
        tracker = get_error_tracker()
        summary = tracker.get_error_summary()
        assert "ProviderNotFoundError" in summary["error_counts"]

    def test_provider_initialization_error_scenario(self):
        """Test user experience when provider fails to initialize."""
        missing_key = "ANTHROPIC_API_KEY"

        with pytest.raises(ProviderInitializationError) as exc_info:
            raise ProviderInitializationError(
                "Provider initialization failed",
                provider="anthropic",
                config_key=missing_key,
            )

        error = exc_info.value

        # Verify error has context
        assert error.provider == "anthropic"
        assert error.config_key == missing_key
        assert error.correlation_id is not None

        # Verify recovery hint mentions the config key
        assert missing_key in error.recovery_hint
        assert error.correlation_id is not None


class ToolErrorScenarios:
    """Test user experience when tool errors occur."""

    def test_tool_not_found_scenario(self):
        """Test user experience when tool not found."""
        with pytest.raises(ToolNotFoundError) as exc_info:
            raise ToolNotFoundError(tool_name="nonexistent_tool")

        error = exc_info.value

        # Verify error details
        assert "nonexistent_tool" in str(error)
        assert error.tool_name == "nonexistent_tool"
        assert error.correlation_id is not None

        # Verify recovery hint
        assert error.recovery_hint is not None
        assert "list_tools" in error.recovery_hint.lower()

    def test_tool_execution_error_scenario(self):
        """Test user experience when tool execution fails."""
        tool_name = "read"
        arguments = {"path": "/nonexistent/file.txt"}
        error_message = "File not found"

        with pytest.raises(ToolExecutionError) as exc_info:
            raise ToolExecutionError(
                error_message,
                tool_name=tool_name,
                arguments=arguments,
            )

        error = exc_info.value

        # Verify error context
        assert error.tool_name == tool_name
        assert error.arguments == arguments
        assert error_message in str(error)
        assert error.correlation_id is not None

        # Verify arguments captured in details
        assert "arguments" in error.details
        assert error.details["arguments"] == arguments


class ConfigurationErrorScenarios:
    """Test user experience when configuration errors occur."""

    def test_configuration_error_scenario(self):
        """Test user experience when configuration is invalid."""
        config_key = "workflow.max_iterations"
        invalid_value = -1

        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(
                f"Invalid value for {config_key}: {invalid_value}",
                config_key=config_key,
                invalid_fields=["max_iterations"],
            )

        error = exc_info.value

        # Verify error details
        assert error.config_key == config_key
        assert "max_iterations" in error.invalid_fields
        assert error.correlation_id is not None

        # Verify recovery hint
        assert error.recovery_hint is not None

    def test_validation_error_scenario(self):
        """Test user experience when input validation fails."""
        field = "mode"
        invalid_value = "invalid_mode"

        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(
                f"Invalid mode: {invalid_value}",
                field=field,
                value=invalid_value,
            )

        error = exc_info.value

        # Verify error details
        assert error.field == field
        assert error.value == invalid_value
        assert error.correlation_id is not None

        # Verify field in details
        assert error.details["field"] == field
        assert error.details["value"] == invalid_value


class ErrorRecoveryScenarios:
    """Test error recovery workflows."""

    def test_error_recovery_with_correlation_id(self):
        """Test using correlation ID to find error details."""
        # Create an error
        with pytest.raises(ProviderNotFoundError) as exc_info:
            raise ProviderNotFoundError(
                provider="test_provider",
                available_providers=["anthropic", "openai"],
            )

        error = exc_info.value
        correlation_id = error.correlation_id

        # Use correlation ID to find error in tracker
        tracker = get_error_tracker()
        summary = tracker.get_error_summary()

        # Find error with matching correlation ID
        found = False
        for error_record in summary["recent_errors"]:
            if error_record["correlation_id"] == correlation_id:
                found = True
                assert error_record["error_type"] == "ProviderNotFoundError"
                assert "test_provider" in error_record["error_message"]
                break

        assert found, "Error not found in tracker"

    def test_error_stats_aggregation(self):
        """Test error statistics aggregation."""
        tracker = get_error_tracker()

        # Create multiple errors of different types
        for i in range(5):
            try:
                raise ProviderNotFoundError(provider_name=f"provider_{i}")
            except ProviderNotFoundError:
                pass  # Error tracked automatically

        for i in range(3):
            try:
                raise ToolNotFoundError(tool_name=f"tool_{i}")
            except ToolNotFoundError:
                pass

        # Check stats
        summary = tracker.get_error_summary()

        assert summary["total_errors"] == 8
        assert summary["error_counts"]["ProviderNotFoundError"] == 5
        assert summary["error_counts"]["ToolNotFoundError"] == 3

        # Most common should be ProviderNotFoundError
        most_common = summary["most_common"][0]
        assert most_common[0] == "ProviderNotFoundError"
        assert most_common[1] == 5


class ErrorUserExperience:
    """Test overall error user experience."""

    def test_error_message_is_helpful(self):
        """Test that error messages guide users to solutions."""
        with pytest.raises(ProviderNotFoundError) as exc_info:
            raise ProviderNotFoundError(
                provider_name="xyz",
                available_providers=["anthropic", "openai"],
            )

        error = exc_info.value

        # Message should be clear
        error_str = str(error)
        assert "xyz" in error_str  # Shows what went wrong
        assert "anthropic" in error_str or "openai" in error_str  # Shows alternatives

        # Recovery hint should be actionable
        assert "victor providers" in error.recovery_hint.lower()
        assert "list" in error.recovery_hint.lower()

    def test_error_provides_context(self):
        """Test that errors provide sufficient context."""
        with pytest.raises(ToolExecutionError) as exc_info:
            raise ToolExecutionError(
                "Tool execution failed",
                tool_name="write",
                arguments={"path": "/tmp/file.txt", "content": "test"},
            )

        error = exc_info.value

        # Should have tool name
        assert error.tool_name == "write"

        # Should have arguments
        assert error.arguments == {"path": "/tmp/file.txt", "content": "test"}

        # Should have correlation ID for tracking
        assert error.correlation_id is not None

    def test_error_serialization_for_logging(self):
        """Test that errors can be serialized for logging."""
        error = ConfigurationError(
            "Invalid configuration",
            config_key="workflow.max_iterations",
            invalid_fields=["max_iterations"],
        )

        # Should be serializable
        error_dict = error.to_dict()

        # All important fields present
        assert "error" in error_dict
        assert "category" in error_dict
        assert "severity" in error_dict
        assert "correlation_id" in error_dict
        assert "recovery_hint" in error_dict
        assert "details" in error_dict
        assert "timestamp" in error_dict

    def test_error_chain_preserved(self):
        """Test that error chains are preserved for debugging."""
        original_error = ValueError("Original error message")

        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(
                "Configuration failed",
                config_key="test",
            ) from original_error

        error = exc_info.value

        # Check that __cause__ is preserved
        assert error.__cause__ is not None
        assert isinstance(error.__cause__, ValueError)
        assert str(error.__cause__) == "Original error message"


class ErrorCLIIntegration:
    """Test error handling integration with CLI."""

    def test_error_export_functionality(self, tmp_path):
        """Test error metrics export."""
        tracker = get_error_tracker()

        # Create some errors
        for i in range(3):
            try:
                raise ProviderNotFoundError(provider_name=f"provider_{i}")
            except ProviderNotFoundError:
                pass

        # Export to file
        export_path = tmp_path / "error_metrics.json"
        tracker.export_metrics(str(export_path))

        # Verify file exists
        assert export_path.exists()

        # Verify content
        import json

        with open(export_path) as f:
            metrics = json.load(f)

        assert "summary" in metrics
        assert "error_rates" in metrics
        assert "exported_at" in metrics
        assert metrics["summary"]["total_errors"] == 3

    def test_error_filtering_by_timeframe(self):
        """Test filtering errors by timeframe."""
        tracker = get_error_tracker()

        # Create errors
        for i in range(5):
            try:
                raise ProviderNotFoundError(provider_name=f"provider_{i}")
            except ProviderNotFoundError:
                pass

        # Get all errors
        all_errors = tracker.get_errors_by_timeframe(hours=24)
        assert len(all_errors) == 5

        # Get specific type
        provider_errors = tracker.get_errors_by_type("ProviderNotFoundError")
        assert len(provider_errors) == 5

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        tracker = get_error_tracker()

        # Create multiple errors of same type
        for i in range(10):
            try:
                raise ProviderNotFoundError(provider_name=f"provider_{i}")
            except ProviderNotFoundError:
                pass

        # Get error rate
        rate = tracker.get_error_rate("ProviderNotFoundError")

        # Should be 10 errors per hour (since all occurred within last hour)
        assert rate == 10.0
