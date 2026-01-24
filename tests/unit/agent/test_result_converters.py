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

"""Unit tests for ResultConverters adapter.

Tests conversion between different result types used in the orchestrator.
"""

import pytest
from dataclasses import dataclass, field

from victor.agent.adapters.result_converters import ResultConverters


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockValidationResult:
    """Mock ValidationResult for testing."""

    is_valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class MockIntelligentValidationResult:
    """Mock IntelligentValidationResult for testing."""

    quality_score: float
    grounding_score: float
    is_grounded: bool
    is_valid: bool
    grounding_issues: list = field(default_factory=list)
    should_finalize: bool = False
    should_retry: bool = False
    finalize_reason: str = ""
    grounding_feedback: str = ""


@dataclass
class MockTokenUsage:
    """Mock TokenUsage for testing."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class MockToolExecutionResult:
    """Mock tool execution result for testing."""

    success: bool
    output: str = ""
    error: str = ""
    duration: float = 0.0


# ============================================================================
# ResultConverters Tests
# ============================================================================


class TestResultConverters:
    """Test suite for ResultConverters static methods."""

    def test_validation_result_to_dict_with_valid_result(self):
        """Test converting valid ValidationResult to dict."""
        result = MockValidationResult(is_valid=True)
        result_dict = ResultConverters.validation_result_to_dict(result)

        assert result_dict["is_valid"] is True
        assert result_dict["errors"] == []
        assert result_dict["warnings"] == []

    def test_validation_result_to_dict_with_errors(self):
        """Test converting ValidationResult with errors to dict."""
        result = MockValidationResult(
            is_valid=False, errors=["Error 1", "Error 2"], warnings=["Warning 1"]
        )
        result_dict = ResultConverters.validation_result_to_dict(result)

        assert result_dict["is_valid"] is False
        assert result_dict["errors"] == ["Error 1", "Error 2"]
        assert result_dict["warnings"] == ["Warning 1"]

    def test_validation_result_to_dict_with_none(self):
        """Test converting None ValidationResult to dict."""
        result_dict = ResultConverters.validation_result_to_dict(None)

        assert result_dict == {}

    def test_intelligent_validation_to_dict_complete(self):
        """Test converting complete IntelligentValidationResult to dict."""
        result = MockIntelligentValidationResult(
            quality_score=0.85,
            grounding_score=0.92,
            is_grounded=True,
            is_valid=True,
            grounding_issues=[],
            should_finalize=False,
            should_retry=False,
            finalize_reason="",
            grounding_feedback="",
        )
        result_dict = ResultConverters.intelligent_validation_to_dict(result)

        assert result_dict is not None
        assert result_dict["quality_score"] == 0.85
        assert result_dict["grounding_score"] == 0.92
        assert result_dict["is_grounded"] is True
        assert result_dict["is_valid"] is True
        assert result_dict["grounding_issues"] == []
        assert result_dict["should_finalize"] is False
        assert result_dict["should_retry"] is False

    def test_intelligent_validation_to_dict_with_issues(self):
        """Test converting IntelligentValidationResult with grounding issues."""
        result = MockIntelligentValidationResult(
            quality_score=0.45,
            grounding_score=0.60,
            is_grounded=False,
            is_valid=False,
            grounding_issues=["Unverified claim", "Missing citation"],
            should_finalize=True,
            should_retry=True,
            finalize_reason="Low quality",
            grounding_feedback="Add citations",
        )
        result_dict = ResultConverters.intelligent_validation_to_dict(result)

        assert result_dict["quality_score"] == 0.45
        assert result_dict["grounding_score"] == 0.60
        assert result_dict["is_grounded"] is False
        assert result_dict["is_valid"] is False
        assert result_dict["grounding_issues"] == ["Unverified claim", "Missing citation"]
        assert result_dict["should_finalize"] is True
        assert result_dict["should_retry"] is True
        assert result_dict["finalize_reason"] == "Low quality"
        assert result_dict["grounding_feedback"] == "Add citations"

    def test_intelligent_validation_to_dict_with_none(self):
        """Test converting None IntelligentValidationResult to dict."""
        result_dict = ResultConverters.intelligent_validation_to_dict(None)

        assert result_dict is None

    def test_token_usage_to_dict_with_object(self):
        """Test converting TokenUsage object to dict."""
        usage = MockTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        usage_dict = ResultConverters.token_usage_to_dict(usage)

        assert usage_dict["prompt_tokens"] == 1000
        assert usage_dict["completion_tokens"] == 500
        assert usage_dict["total_tokens"] == 1500

    def test_token_usage_to_dict_with_dict(self):
        """Test converting dict TokenUsage to dict (passthrough)."""
        usage = {
            "prompt_tokens": 2000,
            "completion_tokens": 1000,
            "total_tokens": 3000,
        }
        usage_dict = ResultConverters.token_usage_to_dict(usage)

        assert usage_dict == usage

    def test_token_usage_to_dict_with_none(self):
        """Test converting None TokenUsage to dict."""
        usage_dict = ResultConverters.token_usage_to_dict(None)

        assert usage_dict["prompt_tokens"] == 0
        assert usage_dict["completion_tokens"] == 0
        assert usage_dict["total_tokens"] == 0

    def test_tool_execution_to_dict_with_object(self):
        """Test converting tool execution result to dict."""
        result = MockToolExecutionResult(success=True, output="Tool output", error="", duration=1.5)
        result_dict = ResultConverters.tool_execution_to_dict(result)

        assert result_dict["success"] is True
        assert result_dict["output"] == "Tool output"
        assert result_dict["error"] == ""
        assert result_dict["duration"] == 1.5

    def test_tool_execution_to_dict_with_dict(self):
        """Test converting dict tool execution result (passthrough)."""
        result = {"success": False, "error": "Tool failed"}
        result_dict = ResultConverters.tool_execution_to_dict(result)

        assert result_dict == result

    def test_tool_execution_to_dict_with_none(self):
        """Test converting None tool execution result to dict."""
        result_dict = ResultConverters.tool_execution_to_dict(None)

        assert result_dict["success"] is False
        assert result_dict["error"] == "No result"

    def test_checkpoint_state_to_dict_with_dict(self):
        """Test converting dict checkpoint state (passthrough)."""
        state = {
            "stage": "ANALYZING",
            "tool_history": ["read", "write"],
            "observed_files": ["file1.py"],
        }
        state_dict = ResultConverters.checkpoint_state_to_dict(state)

        assert state_dict == state

    def test_checkpoint_state_to_dict_with_dataclass(self):
        """Test converting dataclass checkpoint state to dict."""

        @dataclass
        class MockCheckpointState:
            stage: str
            tool_history: list
            observed_files: list

        state = MockCheckpointState(stage="EXECUTING", tool_history=["test"], observed_files=[])
        state_dict = ResultConverters.checkpoint_state_to_dict(state)

        assert state_dict["stage"] == "EXECUTING"
        assert state_dict["tool_history"] == ["test"]
        assert state_dict["observed_files"] == []

    def test_checkpoint_state_to_dict_with_none(self):
        """Test converting None checkpoint state to dict."""
        state_dict = ResultConverters.checkpoint_state_to_dict(None)

        assert state_dict == {}

    def test_checkpoint_state_to_dict_with_attributes(self):
        """Test converting object with attributes to dict."""

        class MockState:
            def __init__(self):
                self.stage = "PLANNING"
                self.tool_history = ["read"]
                self.observed_files = ["test.py"]
                self.modified_files = ["main.py"]

        state = MockState()
        state_dict = ResultConverters.checkpoint_state_to_dict(state)

        assert state_dict["stage"] == "PLANNING"
        assert state_dict["tool_history"] == ["read"]
        assert state_dict["observed_files"] == ["test.py"]
        assert state_dict["modified_files"] == ["main.py"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
