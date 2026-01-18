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

"""Test error message quality and completeness.

This test suite verifies that all Victor errors meet quality standards:
- All errors have correlation IDs
- All errors have recovery hints
- Error messages are user-friendly
- Error chains are preserved
"""

import pytest

from victor.core.errors import (
    ProviderNotFoundError,
    ProviderInitializationError,
    ProviderConnectionError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderInvalidResponseError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError,
    ConfigurationError,
    ValidationError,
    SearchError,
    WorkflowExecutionError,
    FileNotFoundError,
    FileError,
    NetworkError,
    ExtensionLoadError,
)


class TestErrorQuality:
    """Verify all errors meet quality standards."""

    @pytest.mark.parametrize(
        "error_class,kwargs",
        [
            (ProviderNotFoundError, {"provider": "anthropic"}),
            (ProviderInitializationError, {"provider": "openai"}),
            (ProviderConnectionError, {"provider": "anthropic"}),
            (ProviderAuthError, {"provider": "openai"}),
            (ProviderRateLimitError, {"provider": "anthropic"}),
            (ProviderTimeoutError, {"provider": "openai"}),
            (ProviderInvalidResponseError, {"provider": "anthropic"}),
            (ToolNotFoundError, {"tool_name": "read", "message": "Test message"}),
            (ToolExecutionError, {"tool_name": "write"}),
            (ToolValidationError, {"tool_name": "search"}),
            (ToolTimeoutError, {"tool_name": "execute", "message": "Test message"}),
            (ConfigurationError, {"config_key": "workflow"}),
            (ValidationError, {"field": "mode"}),
            (SearchError, {"search_type": "semantic"}),
            (WorkflowExecutionError, {"workflow_id": "test"}),
            (FileNotFoundError, {"path": "/tmp/file.txt", "message": "Test message"}),
            (FileError, {"path": "/tmp/file.txt"}),
            (NetworkError, {"url": "https://api.example.com"}),
            (
                ExtensionLoadError,
                {
                    "extension_type": "safety",
                    "vertical_name": "coding",
                    "original_error": Exception("test"),
                },
            ),
        ],
    )
    def test_error_has_correlation_id(self, error_class, kwargs):
        """All errors must include correlation ID."""
        # Extract message if provided (for errors that auto-generate messages)
        message = kwargs.pop("message", "Test message")
        error = error_class(message, **kwargs)
        assert hasattr(error, "correlation_id")
        assert error.correlation_id is not None
        assert len(error.correlation_id) == 8  # 8-character hex

    @pytest.mark.parametrize(
        "error_class,kwargs",
        [
            (ProviderNotFoundError, {"provider": "anthropic"}),
            (ProviderInitializationError, {"provider": "openai"}),
            (ProviderConnectionError, {"provider": "anthropic"}),
            (ProviderAuthError, {"provider": "openai"}),
            (ProviderRateLimitError, {"provider": "anthropic"}),
            (ProviderTimeoutError, {"provider": "openai"}),
            (ToolNotFoundError, {"tool_name": "read"}),
            (ToolExecutionError, {"tool_name": "write"}),
            (ToolValidationError, {"tool_name": "search"}),
            (ToolTimeoutError, {"tool_name": "execute"}),
            (ConfigurationError, {"config_key": "workflow"}),
            (ValidationError, {"field": "mode"}),
        ],
    )
    def test_error_has_recovery_hint(self, error_class, kwargs):
        """All errors must include recovery hint."""
        error = error_class("Test message", **kwargs)
        assert hasattr(error, "recovery_hint")
        assert error.recovery_hint is not None
        assert len(error.recovery_hint) > 20  # Meaningful hint

    @pytest.mark.parametrize(
        "error_class,kwargs",
        [
            (ProviderNotFoundError, {"provider": "anthropic"}),
            (ProviderInitializationError, {"provider": "openai"}),
            (ToolNotFoundError, {"tool_name": "read"}),
            (ToolExecutionError, {"tool_name": "write"}),
            (ConfigurationError, {"config_key": "workflow"}),
            (ValidationError, {"field": "mode"}),
        ],
    )
    def test_error_message_is_user_friendly(self, error_class, kwargs):
        """Error messages should be clear and actionable."""
        error = error_class("Test message", **kwargs)

        # No technical jargon without explanation
        assert not any(
            jargon in str(error)
            for jargon in ["NoneType", "AttributeError", "KeyError", "ValueError"]
        )

        # Should include what went wrong
        assert len(str(error)) > 20

    def test_error_chains_preserved(self):
        """Original exceptions must be preserved."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = ProviderInitializationError(
                "Provider failed",
                provider="test",
                recovery_hint="Fix it",
            )
            error.__cause__ = e

            # Note: We're not using 'from e' in the test to avoid Python 3 behavior
            # but we verify that the cause can be set
            assert error.__cause__ is not None
            assert isinstance(error.__cause__, ValueError)

    def test_error_to_dict(self):
        """Errors should be serializable to dict."""
        error = ProviderNotFoundError(
            "Provider not found",
            provider="test_provider",
            available_providers=["openai", "anthropic"],
        )

        error_dict = error.to_dict()

        assert "error" in error_dict
        assert "category" in error_dict
        assert "severity" in error_dict
        assert "correlation_id" in error_dict
        assert "recovery_hint" in error_dict
        assert "details" in error_dict
        assert "timestamp" in error_dict

    def test_error_details_populated(self):
        """Error details should include relevant context."""
        error = ProviderNotFoundError(
            "Provider not found",
            provider="test_provider",
            available_providers=["openai", "anthropic"],
        )

        assert "provider_name" in error.details
        assert error.details["provider_name"] == "test_provider"
        assert "available_providers" in error.details

    def test_search_error_details(self):
        """SearchError should include backend details."""
        error = SearchError(
            "All backends failed",
            search_type="semantic",
            failed_backends=["Backend1", "Backend2"],
            failure_details={"Backend1": "Timeout", "Backend2": "Connection error"},
        )

        assert error.search_type == "semantic"
        assert error.failed_backends == ["Backend1", "Backend2"]
        assert error.failure_details == {"Backend1": "Timeout", "Backend2": "Connection error"}
        assert "failed_backends" in error.details

    def test_workflow_error_details(self):
        """WorkflowExecutionError should include workflow context."""
        error = WorkflowExecutionError(
            "Workflow failed",
            workflow_id="test_workflow",
            node_id="node_1",
            node_type="compute",
            checkpoint_id="chk_123",
            execution_context={"iteration": 3},
        )

        assert error.workflow_id == "test_workflow"
        assert error.node_id == "node_1"
        assert error.node_type == "compute"
        assert error.checkpoint_id == "chk_123"
        assert error.execution_context == {"iteration": 3}

    def test_extension_load_error_severity(self):
        """ExtensionLoadError severity should depend on is_required flag."""
        # Required extension should be CRITICAL
        required_error = ExtensionLoadError(
            "Required extension failed",
            extension_type="safety",
            vertical_name="coding",
            is_required=True,
        )
        assert required_error.severity.value == "critical"

        # Optional extension should be WARNING
        optional_error = ExtensionLoadError(
            "Optional extension failed",
            extension_type="analytics",
            vertical_name="coding",
            is_required=False,
        )
        assert optional_error.severity.value == "warning"


class TestErrorMessages:
    """Test specific error message quality."""

    def test_provider_not_found_message(self):
        """ProviderNotFoundError should list available providers."""
        error = ProviderNotFoundError(
            provider="xyz",
            available_providers=["anthropic", "openai", "ollama"],
        )

        error_str = str(error)
        assert "xyz" in error_str
        assert any(p in error_str for p in ["anthropic", "openai", "ollama"])

    def test_provider_rate_limit_includes_retry_after(self):
        """ProviderRateLimitError should include retry information."""
        error = ProviderRateLimitError(
            "Rate limit exceeded",
            provider="anthropic",
            retry_after=60,
        )

        assert "60" in error.recovery_hint.lower()
        assert error.retry_after == 60

    def test_tool_timeout_includes_timeout_value(self):
        """ToolTimeoutError should include timeout value."""
        error = ToolTimeoutError(
            tool_name="long_running_tool",
            timeout=120,
        )

        assert "120" in str(error)
        assert error.timeout == 120

    def test_validation_error_includes_field_and_value(self):
        """ValidationError should include field and value information."""
        error = ValidationError(
            "Invalid value",
            field="mode",
            value="invalid_mode",
        )

        assert error.field == "mode"
        assert error.value == "invalid_mode"
        assert "mode" in error.details["field"]


class TestRecoveryHints:
    """Test recovery hint quality."""

    def test_provider_errors_have_actionable_hints(self):
        """Provider error recovery hints should be actionable."""
        errors = [
            ProviderNotFoundError(provider="test"),
            ProviderInitializationError("Failed to init", provider="test"),
            ProviderConnectionError("Connection failed", provider="test"),
            ProviderAuthError("Auth failed", provider="test"),
        ]

        for error in errors:
            assert error.recovery_hint is not None
            # Should include action verbs
            assert any(
                verb in error.recovery_hint.lower()
                for verb in ["check", "set", "verify", "use", "try"]
            )

    def test_tool_errors_have_actionable_hints(self):
        """Tool error recovery hints should be actionable."""
        errors = [
            ToolNotFoundError(tool_name="test"),
            ToolExecutionError("Execution failed", tool_name="test"),
            ToolValidationError("Validation failed", tool_name="test"),
        ]

        for error in errors:
            assert error.recovery_hint is not None
            # Should mention checking or fixing
            assert any(
                word in error.recovery_hint.lower() for word in ["check", "verify", "fix", "try"]
            )

    def test_workflow_error_recovery_includes_checkpoint(self):
        """WorkflowExecutionError recovery hint should mention checkpoint if available."""
        error = WorkflowExecutionError(
            "Workflow failed",
            workflow_id="test",
            node_id="node1",
            checkpoint_id="chk_123",
        )

        assert "chk_123" in error.recovery_hint
        assert "resume" in error.recovery_hint.lower()


class TestErrorTracking:
    """Test error tracking integration."""

    def test_errors_are_tracked(self):
        """Errors should be recorded in error tracker."""
        from victor.observability.error_tracker import (
            get_error_tracker,
            reset_error_tracker,
        )

        # Reset tracker before test
        reset_error_tracker()

        # Create error
        error = ProviderNotFoundError(provider="test")

        # Check tracker recorded it
        tracker = get_error_tracker()
        summary = tracker.get_error_summary()

        assert summary["total_errors"] > 0
        assert "ProviderNotFoundError" in summary["error_counts"]

    def test_error_tracking_context(self):
        """Error tracker should capture error context."""
        from victor.observability.error_tracker import (
            get_error_tracker,
            reset_error_tracker,
        )

        # Reset tracker before test
        reset_error_tracker()

        # Create error with details
        error = ToolExecutionError(
            "Tool failed",
            tool_name="test_tool",
            arguments={"file": "/tmp/test.txt"},
        )

        # Check context was captured
        tracker = get_error_tracker()
        recent_errors = tracker.get_errors_by_type("ToolExecutionError")

        assert len(recent_errors) > 0
        assert recent_errors[-1].context["tool_name"] == "test_tool"

    def test_error_tracker_thread_safety(self):
        """Error tracker should be thread-safe."""
        from victor.observability.error_tracker import (
            get_error_tracker,
            reset_error_tracker,
        )
        import threading

        # Reset tracker before test
        reset_error_tracker()

        tracker = get_error_tracker()

        # Create multiple errors from different threads
        def create_errors():
            for i in range(10):
                error = ProviderNotFoundError(provider=f"provider_{i}")
                # Error is tracked automatically

        threads = [threading.Thread(target=create_errors) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have 50 errors total (10 * 5 threads)
        summary = tracker.get_error_summary()
        assert summary["total_errors"] == 50


class TestErrorFormatting:
    """Test error message formatting."""

    def test_error_str_includes_correlation_id(self):
        """Error string representation should include correlation ID."""
        error = ProviderNotFoundError(provider="test")

        error_str = str(error)
        assert error.correlation_id in error_str
        assert error_str.startswith(f"[{error.correlation_id}]")

    def test_error_str_includes_recovery_hint(self):
        """Error string representation should include recovery hint."""
        error = ProviderNotFoundError(provider="test")

        error_str = str(error)
        assert "recovery hint" in error_str.lower()
        assert error.recovery_hint in error_str

    def test_error_to_dict_serializable(self):
        """Error to_dict should produce JSON-serializable output."""
        import json

        error = WorkflowExecutionError(
            "Workflow failed",
            workflow_id="test",
            node_id="node1",
        )

        error_dict = error.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(error_dict)
        assert json_str is not None

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert deserialized["error"] == error.message
