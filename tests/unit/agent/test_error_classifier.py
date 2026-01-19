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

"""Unit tests for error_classifier module."""

import pytest

from victor.agent.error_classifier import (
    ErrorType,
    ToolCallSignature,
    ToolErrorClassifier,
    get_error_classifier,
    reset_error_classifier,
)


class TestToolCallSignature:
    """Tests for ToolCallSignature dataclass."""

    def test_from_call_simple_arguments(self) -> None:
        """Test creating signature from simple arguments."""
        sig = ToolCallSignature.from_call(
            "read_file", {"path": "/tmp/test.txt", "encoding": "utf-8"}
        )

        assert sig.tool_name == "read_file"
        assert isinstance(sig.arguments_hash, int)

    def test_from_call_empty_arguments(self) -> None:
        """Test creating signature with no arguments."""
        sig = ToolCallSignature.from_call("tool_name", {})

        assert sig.tool_name == "tool_name"
        assert isinstance(sig.arguments_hash, int)

    def test_from_call_unhashable_values(self) -> None:
        """Test creating signature with unhashable values (lists, dicts)."""
        # Should fallback to str hashing
        sig = ToolCallSignature.from_call(
            "complex_tool", {"nested": {"key": "value"}, "list": [1, 2, 3]}
        )

        assert sig.tool_name == "complex_tool"
        assert isinstance(sig.arguments_hash, int)

    def test_from_call_with_unhashable_key(self) -> None:
        """Test creating signature with unhashable keys (triggers exception handler)."""

        # Using a complex object as key that can't be sorted will trigger the exception handler
        # This tests lines 58-60 which are the except block
        class CustomObject:
            def __str__(self) -> str:
                raise TypeError("Cannot convert to string")

        # Create arguments with a value that will fail sorting
        args = {"key": CustomObject()}

        # This should trigger the exception handler and use str(arguments) fallback
        sig = ToolCallSignature.from_call("tool", args)

        assert sig.tool_name == "tool"
        assert isinstance(sig.arguments_hash, int)

    def test_hash_consistency(self) -> None:
        """Test that same arguments produce same hash."""
        args = {"path": "/tmp/test.txt", "encoding": "utf-8"}
        sig1 = ToolCallSignature.from_call("read_file", args)
        sig2 = ToolCallSignature.from_call("read_file", args)

        assert hash(sig1) == hash(sig2)

    def test_hash_different_order(self) -> None:
        """Test that argument order doesn't affect hash."""
        sig1 = ToolCallSignature.from_call("tool", {"a": "1", "b": "2"})
        sig2 = ToolCallSignature.from_call("tool", {"b": "2", "a": "1"})

        assert hash(sig1) == hash(sig2)

    def test_equality_same_signature(self) -> None:
        """Test equality for identical signatures."""
        sig1 = ToolCallSignature.from_call("tool", {"arg": "value"})
        sig2 = ToolCallSignature.from_call("tool", {"arg": "value"})

        assert sig1 == sig2

    def test_equality_different_tool(self) -> None:
        """Test inequality for different tool names."""
        sig1 = ToolCallSignature.from_call("tool_a", {"arg": "value"})
        sig2 = ToolCallSignature.from_call("tool_b", {"arg": "value"})

        assert sig1 != sig2

    def test_equality_different_arguments(self) -> None:
        """Test inequality for different arguments."""
        sig1 = ToolCallSignature.from_call("tool", {"arg": "value1"})
        sig2 = ToolCallSignature.from_call("tool", {"arg": "value2"})

        assert sig1 != sig2

    def test_equality_non_signature_object(self) -> None:
        """Test inequality with non-signature objects."""
        sig = ToolCallSignature.from_call("tool", {"arg": "value"})

        assert sig != "not_a_signature"
        assert sig != 123
        assert sig is not None

    def test_signature_in_set(self) -> None:
        """Test that signatures work correctly in sets."""
        sig1 = ToolCallSignature.from_call("tool", {"arg": "value"})
        sig2 = ToolCallSignature.from_call("tool", {"arg": "value"})  # Duplicate
        sig3 = ToolCallSignature.from_call("tool", {"arg": "other"})

        signature_set = {sig1, sig2, sig3}

        # Should have 2 unique signatures
        assert len(signature_set) == 2
        assert sig1 in signature_set
        assert sig3 in signature_set


class TestErrorType:
    """Tests for ErrorType enum."""

    def test_error_type_values(self) -> None:
        """Test ErrorType enum values."""
        assert ErrorType.PERMANENT.value == "permanent"
        assert ErrorType.TRANSIENT.value == "transient"
        assert ErrorType.RETRYABLE.value == "retryable"

    def test_error_type_uniqueness(self) -> None:
        """Test that all error types are unique."""
        types = {ErrorType.PERMANENT, ErrorType.TRANSIENT, ErrorType.RETRYABLE}
        assert len(types) == 3


class TestToolErrorClassifier:
    """Tests for ToolErrorClassifier class."""

    @pytest.fixture
    def classifier(self) -> ToolErrorClassifier:
        """Create a fresh classifier for each test."""
        return ToolErrorClassifier()

    def test_init_empty_state(self, classifier: ToolErrorClassifier) -> None:
        """Test classifier initializes with empty state."""
        assert classifier.failed_call_count == 0
        assert len(classifier._failed_calls) == 0

    def test_classify_permanent_file_not_found(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of file not found error."""
        error = "No such file or directory: /tmp/missing.txt"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_permission_denied(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of permission denied error."""
        error = "Permission denied: /root/secret.txt"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_module_not_found(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of module not found error."""
        error = "ModuleNotFoundError: No module named 'missing_module'"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_import_error(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of import error."""
        error = "ImportError: No module named 'nonexistent'"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_command_not_found(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of command not found error."""
        error = "command not found: invalid_cmd"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_is_a_directory_error(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of IsADirectoryError."""
        error = "IsADirectoryError: /tmp/folder"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_not_a_directory_error(
        self, classifier: ToolErrorClassifier
    ) -> None:
        """Test classification of NotADirectoryError."""
        error = "NotADirectoryError: /tmp/file.txt"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_permission_error(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of PermissionError."""
        error = "PermissionError: [Errno 13] Permission denied"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_directory_not_empty(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of directory not empty error."""
        error = "directory not empty: /tmp/folder"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_file_exists(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of file exists error."""
        error = "File exists: /tmp/existing.txt"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_permanent_read_only_filesystem(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of read-only filesystem error."""
        error = "read-only file system"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_transient_connection_refused(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of connection refused error."""
        error = "Connection refused: localhost:8080"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_transient_connection_timeout(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of connection timeout error."""
        error = "Connection timed out"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_transient_network_unreachable(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of network unreachable error."""
        error = "Network is unreachable"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_transient_rate_limit(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of rate limit error."""
        error = "rate limit exceeded, try again later"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_transient_too_many_requests(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of too many requests error."""
        error = "429 Too Many Requests"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_transient_service_unavailable(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of service unavailable error."""
        error = "503 Service Unavailable"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_transient_gateway_timeout(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of gateway timeout error."""
        error = "504 Gateway Timeout"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_transient_temporary_error(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of temporary error."""
        error = "temporary failure, please retry"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_transient_try_again(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of try again error."""
        error = "try again in a few minutes"
        result = classifier.classify(error)

        assert result == ErrorType.TRANSIENT

    def test_classify_retryable_syntax_error(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of syntax error (default retryable)."""
        error = "SyntaxError: invalid syntax"
        result = classifier.classify(error)

        assert result == ErrorType.RETRYABLE

    def test_classify_retryable_unknown_error(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of unknown error (default retryable)."""
        error = "Some unknown error occurred"
        result = classifier.classify(error)

        assert result == ErrorType.RETRYABLE

    def test_classify_case_insensitive(self, classifier: ToolErrorClassifier) -> None:
        """Test that classification is case-insensitive."""
        error1 = "No such file or directory"
        error2 = "NO SUCH FILE OR DIRECTORY"
        error3 = "No Such File Or Directory"

        assert classifier.classify(error1) == ErrorType.PERMANENT
        assert classifier.classify(error2) == ErrorType.PERMANENT
        assert classifier.classify(error3) == ErrorType.PERMANENT

    def test_record_failure_permanent(self, classifier: ToolErrorClassifier) -> None:
        """Test recording a permanent failure."""
        error_type = classifier.record_failure(
            "read_file", {"path": "/tmp/missing.txt"}, "No such file or directory"
        )

        assert error_type == ErrorType.PERMANENT
        assert classifier.failed_call_count == 1

    def test_record_failure_transient(self, classifier: ToolErrorClassifier) -> None:
        """Test recording a transient failure (not stored)."""
        error_type = classifier.record_failure(
            "api_call", {"endpoint": "/api/data"}, "Connection timed out"
        )

        assert error_type == ErrorType.TRANSIENT
        assert classifier.failed_call_count == 0  # Transient errors not stored

    def test_record_failure_retryable(self, classifier: ToolErrorClassifier) -> None:
        """Test recording a retryable failure (not stored)."""
        error_type = classifier.record_failure(
            "parse_code", {"code": "invalid"}, "SyntaxError: invalid syntax"
        )

        assert error_type == ErrorType.RETRYABLE
        assert classifier.failed_call_count == 0  # Retryable errors not stored

    def test_should_skip_not_recorded(self, classifier: ToolErrorClassifier) -> None:
        """Test should_skip returns False for unrecorded calls."""
        result = classifier.should_skip("read_file", {"path": "/tmp/test.txt"})

        assert result is False

    def test_should_skip_after_permanent_failure(self, classifier: ToolErrorClassifier) -> None:
        """Test should_skip returns True after permanent failure."""
        # Record permanent failure
        classifier.record_failure(
            "read_file", {"path": "/tmp/missing.txt"}, "No such file or directory"
        )

        # Same call should be skipped
        result = classifier.should_skip("read_file", {"path": "/tmp/missing.txt"})

        assert result is True

    def test_should_skip_different_arguments(self, classifier: ToolErrorClassifier) -> None:
        """Test should_skip returns False for different arguments."""
        # Record failure
        classifier.record_failure(
            "read_file", {"path": "/tmp/missing.txt"}, "No such file or directory"
        )

        # Different arguments should not be skipped
        result = classifier.should_skip("read_file", {"path": "/tmp/other.txt"})

        assert result is False

    def test_should_skip_different_tool(self, classifier: ToolErrorClassifier) -> None:
        """Test should_skip returns False for different tool."""
        # Record failure
        classifier.record_failure(
            "read_file", {"path": "/tmp/test.txt"}, "No such file or directory"
        )

        # Different tool should not be skipped
        result = classifier.should_skip("write_file", {"path": "/tmp/test.txt"})

        assert result is False

    def test_should_skip_after_transient_failure(self, classifier: ToolErrorClassifier) -> None:
        """Test should_skip returns False after transient failure."""
        # Record transient failure (not stored)
        classifier.record_failure("api_call", {"endpoint": "/api"}, "Connection timed out")

        # Should not skip transient failures
        result = classifier.should_skip("api_call", {"endpoint": "/api"})

        assert result is False

    def test_get_skip_reason_no_skip(self, classifier: ToolErrorClassifier) -> None:
        """Test get_skip_reason returns None when not skipping."""
        reason = classifier.get_skip_reason("read_file", {"path": "/tmp/test.txt"})

        assert reason is None

    def test_get_skip_reason_with_skip(self, classifier: ToolErrorClassifier) -> None:
        """Test get_skip_reason returns reason when skipping."""
        # Record permanent failure
        classifier.record_failure(
            "read_file", {"path": "/tmp/missing.txt"}, "No such file or directory"
        )

        # Get skip reason
        reason = classifier.get_skip_reason("read_file", {"path": "/tmp/missing.txt"})

        assert reason is not None
        assert "Skipping read_file" in reason
        assert "permanent error" in reason

    def test_get_skip_reason_no_skip_different_args(self, classifier: ToolErrorClassifier) -> None:
        """Test get_skip_reason returns None for different arguments."""
        # Record failure
        classifier.record_failure(
            "read_file", {"path": "/tmp/missing.txt"}, "No such file or directory"
        )

        # Different arguments - no skip
        reason = classifier.get_skip_reason("read_file", {"path": "/tmp/other.txt"})

        assert reason is None

    def test_reset_clears_failures(self, classifier: ToolErrorClassifier) -> None:
        """Test that reset clears all recorded failures."""
        # Record multiple failures
        classifier.record_failure("tool1", {"arg": "val1"}, "No such file or directory")
        classifier.record_failure("tool2", {"arg": "val2"}, "Permission denied")
        classifier.record_failure("tool3", {"arg": "val3"}, "File exists")

        assert classifier.failed_call_count == 3

        # Reset
        classifier.reset()

        assert classifier.failed_call_count == 0

    def test_failed_call_count_property(self, classifier: ToolErrorClassifier) -> None:
        """Test failed_call_count property."""
        assert classifier.failed_call_count == 0

        classifier.record_failure("tool", {"arg": "val"}, "No such file or directory")
        assert classifier.failed_call_count == 1

        classifier.record_failure("tool2", {"arg": "val2"}, "Permission denied")
        assert classifier.failed_call_count == 2

    def test_multiple_failures_same_signature(self, classifier: ToolErrorClassifier) -> None:
        """Test recording same failure multiple times doesn't duplicate."""
        # Record same failure twice
        classifier.record_failure("tool", {"arg": "val"}, "No such file or directory")
        classifier.record_failure("tool", {"arg": "val"}, "No such file or directory")

        # Should only count once
        assert classifier.failed_call_count == 1

    def test_classify_partial_pattern_match(self, classifier: ToolErrorClassifier) -> None:
        """Test classification with partial pattern matches."""
        # Should match even if pattern is part of larger message
        error = "Error: No such file or directory in /tmp/path"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_permanent_patterns_take_precedence(self, classifier: ToolErrorClassifier) -> None:
        """Test that permanent patterns are checked before transient."""
        # This matches both "temporary" (transient) and could be related to permanent
        # But permanent patterns should be checked first
        error = "No such file or directory"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_classify_empty_string(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of empty string (default retryable)."""
        result = classifier.classify("")

        assert result == ErrorType.RETRYABLE

    def test_signature_argument_order_independence(self, classifier: ToolErrorClassifier) -> None:
        """Test that argument order doesn't affect signature matching."""
        # Record failure with one order
        classifier.record_failure(
            "tool", {"a": "1", "b": "2", "c": "3"}, "No such file or directory"
        )

        # Check with different order - should still skip
        result = classifier.should_skip("tool", {"c": "3", "a": "1", "b": "2"})

        assert result is True


class TestGlobalClassifier:
    """Tests for global error classifier functions."""

    def test_get_error_classifier_singleton(self) -> None:
        """Test that get_error_classifier returns singleton instance."""
        # Reset to ensure clean state
        reset_error_classifier()

        classifier1 = get_error_classifier()
        classifier2 = get_error_classifier()

        assert classifier1 is classifier2

    def test_get_error_classifier_initialization(self) -> None:
        """Test that get_error_classifier initializes on first call."""
        # Reset to ensure clean state
        reset_error_classifier()

        classifier = get_error_classifier()

        assert isinstance(classifier, ToolErrorClassifier)
        assert classifier.failed_call_count == 0

    def test_reset_error_classifier(self) -> None:
        """Test reset_error_classifier clears global instance."""
        # Get classifier and add some failures
        classifier = get_error_classifier()
        classifier.record_failure("tool", {"arg": "val"}, "No such file or directory")
        assert classifier.failed_call_count == 1

        # Reset
        reset_error_classifier()

        # Get new instance - should be fresh
        new_classifier = get_error_classifier()
        assert new_classifier.failed_call_count == 0
        # Note: May be same instance but cleared, or new instance depending on implementation

    def test_global_classifier_persists(self) -> None:
        """Test that global classifier persists across calls."""
        # Reset to ensure clean state
        reset_error_classifier()

        classifier1 = get_error_classifier()
        classifier1.record_failure("tool", {"arg": "val"}, "No such file or directory")

        classifier2 = get_error_classifier()
        assert classifier2.failed_call_count == 1
        assert classifier2.should_skip("tool", {"arg": "val"})

    def test_reset_before_first_get(self) -> None:
        """Test reset before first get doesn't cause errors."""
        # This should not raise any errors
        reset_error_classifier()

        classifier = get_error_classifier()
        assert isinstance(classifier, ToolErrorClassifier)


class TestEdgeCases:
    """Tests for edge cases and complex scenarios."""

    @pytest.fixture
    def classifier(self) -> ToolErrorClassifier:
        """Create a fresh classifier for each test."""
        return ToolErrorClassifier()

    def test_unicode_error_messages(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of unicode error messages."""
        error = "FileNotFoundError: 文件不存在.txt"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_very_long_error_message(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of very long error messages."""
        error = "No such file or directory: " + "/very/long/path/" * 1000
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_special_characters_in_error(self, classifier: ToolErrorClassifier) -> None:
        """Test classification of errors with special characters."""
        error = "No such file or directory: /tmp/file_with_特殊_chars.txt\n\t\r"
        result = classifier.classify(error)

        assert result == ErrorType.PERMANENT

    def test_arguments_with_none_values(self, classifier: ToolErrorClassifier) -> None:
        """Test signature creation with None values."""
        sig = ToolCallSignature.from_call("tool", {"arg": None, "other": "value"})

        assert isinstance(sig.arguments_hash, int)

    def test_arguments_with_numeric_values(self, classifier: ToolErrorClassifier) -> None:
        """Test signature creation with numeric values."""
        sig = ToolCallSignature.from_call("tool", {"count": 42, "ratio": 3.14})

        assert isinstance(sig.arguments_hash, int)

    def test_arguments_with_boolean_values(self, classifier: ToolErrorClassifier) -> None:
        """Test signature creation with boolean values."""
        sig = ToolCallSignature.from_call("tool", {"enabled": True, "disabled": False})

        assert isinstance(sig.arguments_hash, int)

    def test_mixed_error_type_detection(self, classifier: ToolErrorClassifier) -> None:
        """Test classifying multiple different error types."""
        permanent_errors = [
            "No such file or directory",
            "Permission denied",
            "ModuleNotFoundError: test",
        ]

        transient_errors = [
            "Connection refused",
            "rate limit exceeded",
            "Service Unavailable",
        ]

        retryable_errors = [
            "SyntaxError: invalid syntax",
            "ValueError: bad value",
            "Unknown error",
        ]

        for error in permanent_errors:
            assert classifier.classify(error) == ErrorType.PERMANENT

        for error in transient_errors:
            assert classifier.classify(error) == ErrorType.TRANSIENT

        for error in retryable_errors:
            assert classifier.classify(error) == ErrorType.RETRYABLE

    def test_signature_with_nested_arguments(self, classifier: ToolErrorClassifier) -> None:
        """Test signature creation with deeply nested arguments."""
        sig = ToolCallSignature.from_call(
            "tool", {"config": {"nested": {"deep": "value"}}, "list": [1, 2, [3, 4]]}
        )

        assert isinstance(sig.arguments_hash, int)
        # Creating same signature should produce same hash
        sig2 = ToolCallSignature.from_call(
            "tool", {"config": {"nested": {"deep": "value"}}, "list": [1, 2, [3, 4]]}
        )
        assert sig == sig2

    def test_concurrent_failure_recording(self, classifier: ToolErrorClassifier) -> None:
        """Test recording multiple failures in sequence."""
        failures = [
            ("tool1", {"arg": "val1"}, "No such file or directory"),
            ("tool2", {"arg": "val2"}, "Permission denied"),
            ("tool3", {"arg": "val3"}, "File exists"),
            ("tool1", {"arg": "val1"}, "No such file or directory"),  # Duplicate
            ("tool4", {"arg": "val4"}, "IsADirectoryError"),
        ]

        for tool, args, error in failures:
            classifier.record_failure(tool, args, error)

        # Should have 4 unique failures (one duplicate)
        assert classifier.failed_call_count == 4

    def test_should_skip_allows_different_permutations(
        self, classifier: ToolErrorClassifier
    ) -> None:
        """Test that different argument permutations are not skipped."""
        # Record failure
        classifier.record_failure("tool", {"a": "1", "b": "2"}, "No such file or directory")

        # These should not be skipped (different arguments)
        assert not classifier.should_skip("tool", {"a": "1"})
        assert not classifier.should_skip("tool", {"b": "2"})
        assert not classifier.should_skip("tool", {"a": "1", "b": "2", "c": "3"})

        # This should be skipped (same arguments)
        assert classifier.should_skip("tool", {"a": "1", "b": "2"})
