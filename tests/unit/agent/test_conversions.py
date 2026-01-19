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

"""Unit tests for conversion utilities."""

import pytest
from dataclasses import dataclass

from victor.agent.utils.conversions import (
    token_usage_to_dict,
    validation_result_to_dict,
    message_to_dict,
    stream_metrics_to_dict,
    tool_result_to_dict,
)


@dataclass
class MockTokenUsage:
    """Mock TokenUsage for testing."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class MockValidationResult:
    """Mock ValidationResult for testing."""
    is_valid: bool
    errors: list
    warnings: list


@dataclass
class MockMessage:
    """Mock Message for testing."""
    role: str
    content: str


@dataclass
class MockStreamMetrics:
    """Mock StreamMetrics for testing."""
    duration_ms: int
    tokens_per_second: float
    total_chunks: int


@dataclass
class MockToolResult:
    """Mock ToolResult for testing."""
    success: bool
    output: str
    error: str | None


class TestTokenUsageConversions:
    """Tests for token usage conversion utilities."""

    def test_token_usage_to_dict_with_none(self):
        """Test conversion with None input."""
        result = token_usage_to_dict(None)
        assert result == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_token_usage_to_dict_with_dict(self):
        """Test conversion with dict input."""
        input_dict = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        result = token_usage_to_dict(input_dict)
        assert result == input_dict

    def test_token_usage_to_dict_with_dict_missing_total(self):
        """Test conversion with dict missing total_tokens."""
        input_dict = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }
        result = token_usage_to_dict(input_dict)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150  # Calculated

    def test_token_usage_to_dict_with_object(self):
        """Test conversion with TokenUsage object."""
        usage = MockTokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        result = token_usage_to_dict(usage)
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_token_usage_to_dict_with_object_missing_total(self):
        """Test conversion with object missing total_tokens attribute."""
        # When total_tokens is provided as 0, it stays 0 (not calculated)
        usage = MockTokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=0,
        )
        result = token_usage_to_dict(usage)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        # total_tokens is taken from the object, not calculated
        assert result["total_tokens"] == 0


class TestValidationResultConversions:
    """Tests for validation result conversion utilities."""

    def test_validation_result_to_dict_with_none(self):
        """Test conversion with None input."""
        result = validation_result_to_dict(None)
        assert result == {
            "is_valid": False,
            "errors": [],
            "warnings": [],
        }

    def test_validation_result_to_dict_with_object(self):
        """Test conversion with ValidationResult object."""
        validation = MockValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor issue"],
        )
        result = validation_result_to_dict(validation)
        assert result == {
            "is_valid": True,
            "errors": [],
            "warnings": ["Minor issue"],
        }

    def test_validation_result_to_dict_with_errors(self):
        """Test conversion with validation errors."""
        validation = MockValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=[],
        )
        result = validation_result_to_dict(validation)
        assert result["is_valid"] is False
        assert result["errors"] == ["Error 1", "Error 2"]
        assert result["warnings"] == []

    def test_validation_result_to_dict_without_warnings_attribute(self):
        """Test conversion with object missing warnings attribute."""
        class SimpleValidation:
            is_valid = True
            errors = []

        validation = SimpleValidation()
        result = validation_result_to_dict(validation)
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert result["warnings"] == []  # Default


class TestMessageConversions:
    """Tests for message conversion utilities."""

    def test_message_to_dict_with_none(self):
        """Test conversion with None input."""
        result = message_to_dict(None)
        assert result == {
            "role": "system",
            "content": "",
        }

    def test_message_to_dict_with_dict(self):
        """Test conversion with dict input."""
        input_dict = {"role": "user", "content": "Hello"}
        result = message_to_dict(input_dict)
        assert result == input_dict
        # Ensure it's a copy
        assert result is not input_dict

    def test_message_to_dict_with_object(self):
        """Test conversion with Message object."""
        message = MockMessage(role="user", content="Hello")
        result = message_to_dict(message)
        assert result == {
            "role": "user",
            "content": "Hello",
        }

    def test_message_to_dict_with_additional_fields(self):
        """Test conversion with message object with extra fields."""
        class ExtendedMessage:
            role = "assistant"
            content = "Response"
            timestamp = 123456
            tool_calls = []

        message = ExtendedMessage()
        result = message_to_dict(message)
        assert result["role"] == "assistant"
        assert result["content"] == "Response"
        assert result["timestamp"] == 123456
        assert result["tool_calls"] == []


class TestStreamMetricsConversions:
    """Tests for stream metrics conversion utilities."""

    def test_stream_metrics_to_dict_with_none(self):
        """Test conversion with None input."""
        result = stream_metrics_to_dict(None)
        assert result == {
            "duration_ms": 0,
            "tokens_per_second": 0.0,
            "total_chunks": 0,
        }

    def test_stream_metrics_to_dict_with_dict(self):
        """Test conversion with dict input."""
        input_dict = {
            "duration_ms": 1000,
            "tokens_per_second": 50.5,
            "total_chunks": 10,
        }
        result = stream_metrics_to_dict(input_dict)
        assert result == input_dict

    def test_stream_metrics_to_dict_with_object(self):
        """Test conversion with StreamMetrics object."""
        metrics = MockStreamMetrics(
            duration_ms=1000,
            tokens_per_second=50.5,
            total_chunks=10,
        )
        result = stream_metrics_to_dict(metrics)
        # Check required fields
        assert result["duration_ms"] == 1000
        assert result["tokens_per_second"] == 50.5
        assert result["total_chunks"] == 10
        # Optional fields are added with defaults
        assert "first_chunk_time_ms" in result

    def test_stream_metrics_to_dict_with_optional_fields(self):
        """Test conversion with optional fields."""
        class ExtendedMetrics:
            duration_ms = 500
            tokens_per_second = 100.0
            total_chunks = 5
            first_chunk_time_ms = 50

        metrics = ExtendedMetrics()
        result = stream_metrics_to_dict(metrics)
        assert result["duration_ms"] == 500
        assert result["tokens_per_second"] == 100.0
        assert result["total_chunks"] == 5
        assert result["first_chunk_time_ms"] == 50


class TestToolResultConversions:
    """Tests for tool result conversion utilities."""

    def test_tool_result_to_dict_with_none(self):
        """Test conversion with None input."""
        result = tool_result_to_dict(None)
        assert result == {
            "success": False,
            "output": None,
            "error": None,
        }

    def test_tool_result_to_dict_with_dict(self):
        """Test conversion with dict input."""
        input_dict = {
            "success": True,
            "output": "Result text",
            "error": None,
        }
        result = tool_result_to_dict(input_dict)
        assert result == input_dict

    def test_tool_result_to_dict_with_object(self):
        """Test conversion with ToolResult object."""
        tool_result = MockToolResult(
            success=True,
            output="Result text",
            error=None,
        )
        result = tool_result_to_dict(tool_result)
        assert result == {
            "success": True,
            "output": "Result text",
            "error": None,
        }

    def test_tool_result_to_dict_with_error(self):
        """Test conversion with failed tool result."""
        tool_result = MockToolResult(
            success=False,
            output=None,
            error="Error occurred",
        )
        result = tool_result_to_dict(tool_result)
        assert result["success"] is False
        assert result["output"] is None
        assert result["error"] == "Error occurred"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_token_usage_to_dict_with_zero_values(self):
        """Test with zero token values."""
        result = token_usage_to_dict({"prompt_tokens": 0, "completion_tokens": 0})
        assert result["total_tokens"] == 0

    def test_validation_result_to_dict_empty_lists(self):
        """Test with empty error and warning lists."""
        validation = MockValidationResult(is_valid=True, errors=[], warnings=[])
        result = validation_result_to_dict(validation)
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_message_to_dict_empty_content(self):
        """Test with empty message content."""
        message = MockMessage(role="user", content="")
        result = message_to_dict(message)
        assert result["content"] == ""

    def test_tool_result_to_dict_success_false_without_error(self):
        """Test failed result without error message."""
        class IncompleteResult:
            success = False

        result = tool_result_to_dict(IncompleteResult())
        assert result["success"] is False
        assert result["error"] is None
