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

"""Tests for common tool validators."""

import pytest
from victor.tools.validators.common import (
    CombinedToolValidator,
    ToolAvailabilityValidator,
    ToolBudgetValidator,
    ValidationResult,
    validate_budget,
    validate_tools,
)


# =============================================================================
# Test Doubles
# =============================================================================


class MockToolRegistry:
    """Mock tool registry for testing."""

    def __init__(self, available_tools=None):
        self._available = set(available_tools or [])

    def get(self, name):
        return name if name in self._available else None

    def has_tool(self, name):
        return name in self._available

    def list_tools(self, only_enabled=True):
        return list(self._available)


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_result_is_valid(self):
        """Default ValidationResult should be valid."""
        result = ValidationResult()
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_makes_invalid(self):
        """Adding an error should mark result as invalid."""
        result = ValidationResult()
        result.add_error("Test error")
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"

    def test_add_warning_keeps_valid(self):
        """Adding a warning should keep result valid."""
        result = ValidationResult()
        result.add_warning("Test warning")
        assert result.valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"

    def test_merge_results(self):
        """Merging should combine errors and warnings."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        result1.add_warning("Warning 1")

        result2 = ValidationResult()
        result2.add_error("Error 2")
        result2.add_warning("Warning 2")

        result1.merge(result2)
        assert result1.valid is False
        assert len(result1.errors) == 2
        assert len(result1.warnings) == 2

    def test_summary(self):
        """Summary should provide human-readable message."""
        # Valid result
        result = ValidationResult()
        assert "passed" in result.summary().lower()

        # Invalid result
        result.add_error("Error 1")
        result.add_error("Error 2")
        assert "failed" in result.summary().lower()
        assert "2 error" in result.summary()

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = ValidationResult()
        result.add_error("Error")
        result.add_warning("Warning")
        result.context["test_key"] = "test_value"

        data = result.to_dict()
        assert data["valid"] is False
        assert len(data["errors"]) == 1
        assert len(data["warnings"]) == 1
        assert data["context"]["test_key"] == "test_value"


# =============================================================================
# ToolAvailabilityValidator Tests
# =============================================================================


class TestToolAvailabilityValidator:
    """Tests for ToolAvailabilityValidator."""

    def test_all_tools_available(self):
        """Should validate successfully when all tools exist."""
        registry = MockToolRegistry(["read", "write", "search"])
        validator = ToolAvailabilityValidator(registry)

        result = validator.validate_tools_available(["read", "write"])
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_tools_in_non_strict_mode(self):
        """Missing tools should generate warnings in non-strict mode."""
        registry = MockToolRegistry(["read", "write"])
        validator = ToolAvailabilityValidator(registry, strict=False)

        result = validator.validate_tools_available(["read", "missing_tool"])
        assert result.valid is True  # Still valid in non-strict mode
        assert len(result.warnings) == 1
        assert "missing_tool" in result.warnings[0]

    def test_missing_tools_in_strict_mode(self):
        """Missing tools should generate errors in strict mode."""
        registry = MockToolRegistry(["read", "write"])
        validator = ToolAvailabilityValidator(registry, strict=True)

        result = validator.validate_tools_available(["read", "missing_tool"])
        assert result.valid is False
        assert len(result.errors) == 1
        assert "missing_tool" in result.errors[0]

    def test_validate_single_tool(self):
        """Should validate a single tool."""
        registry = MockToolRegistry(["read"])
        validator = ToolAvailabilityValidator(registry)

        result = validator.validate_tool_available("read")
        assert result.valid is True

        result = validator.validate_tool_available("missing")
        assert result.valid is True  # Non-strict mode default
        assert len(result.warnings) == 1

    def test_is_tool_available(self):
        """Quick check should return boolean."""
        registry = MockToolRegistry(["read"])
        validator = ToolAvailabilityValidator(registry)

        assert validator.is_tool_available("read") is True
        assert validator.is_tool_available("missing") is False

    def test_list_missing_tools(self):
        """Should list only missing tools."""
        registry = MockToolRegistry(["read", "write"])
        validator = ToolAvailabilityValidator(registry)

        missing = validator.list_missing_tools(["read", "missing1", "write", "missing2"])
        assert len(missing) == 2
        assert "missing1" in missing
        assert "missing2" in missing


# =============================================================================
# ToolBudgetValidator Tests
# =============================================================================


class TestToolBudgetValidator:
    """Tests for ToolBudgetValidator."""

    def test_valid_budget(self):
        """Should accept valid budget."""
        validator = ToolBudgetValidator()
        result = validator.validate_budget(50)
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_budget_below_minimum(self):
        """Should reject budget below minimum."""
        validator = ToolBudgetValidator(min_budget=5)
        result = validator.validate_budget(3)
        assert result.valid is False
        assert len(result.errors) == 1
        assert "below minimum" in result.errors[0]

    def test_budget_above_maximum(self):
        """Should reject budget above maximum."""
        validator = ToolBudgetValidator(max_budget=100)
        result = validator.validate_budget(150)
        assert result.valid is False
        assert len(result.errors) == 1
        assert "exceeds maximum" in result.errors[0]

    def test_budget_below_recommended(self):
        """Should warn when budget below recommended."""
        validator = ToolBudgetValidator(recommended_min=10)
        result = validator.validate_budget(5)
        assert result.valid is True  # Still valid, just warned
        assert len(result.warnings) == 1
        assert "below recommended" in result.warnings[0]

    def test_budget_above_recommended(self):
        """Should warn when budget above recommended."""
        validator = ToolBudgetValidator(recommended_max=50)
        result = validator.validate_budget(100)
        assert result.valid is True  # Still valid, just warned
        assert len(result.warnings) == 1
        assert "above recommended" in result.warnings[0]

    def test_invalid_budget_type(self):
        """Should reject non-integer budget."""
        validator = ToolBudgetValidator()
        result = validator.validate_budget("not_an_int")
        assert result.valid is False
        assert len(result.errors) == 1
        assert "integer" in result.errors[0].lower()

    def test_validate_task_budgets(self):
        """Should validate multiple task budgets."""
        validator = ToolBudgetValidator(min_budget=5, max_budget=50)

        budgets = {
            "task1": 10,  # Valid
            "task2": 3,  # Below min
            "task3": 60,  # Above max
        }

        result = validator.validate_task_budgets(budgets)
        assert result.valid is False
        assert len(result.errors) == 2  # task2 and task3

    def test_is_valid_budget(self):
        """Quick check should return boolean."""
        validator = ToolBudgetValidator(min_budget=5, max_budget=50)

        assert validator.is_valid_budget(10) is True
        assert validator.is_valid_budget(3) is False
        assert validator.is_valid_budget(60) is False

    def test_clamp_budget(self):
        """Should clamp budget to valid range."""
        validator = ToolBudgetValidator(min_budget=5, max_budget=50)

        assert validator.clamp_budget(10) == 10
        assert validator.clamp_budget(3) == 5  # Clamped to min
        assert validator.clamp_budget(60) == 50  # Clamped to max
        assert validator.clamp_budget("invalid") == validator._recommended_min


# =============================================================================
# CombinedToolValidator Tests
# =============================================================================


class TestCombinedToolValidator:
    """Tests for CombinedToolValidator."""

    def test_validate_both_tools_and_budget(self):
        """Should validate both tools and budget."""
        registry = MockToolRegistry(["read", "write"])
        validator = CombinedToolValidator(registry, min_budget=5, max_budget=50)

        result = validator.validate(
            tool_names=["read", "write"],
            tool_budget=20,
        )

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_tools_only(self):
        """Should validate only tools when budget not provided."""
        registry = MockToolRegistry(["read", "write"])
        validator = CombinedToolValidator(registry)

        result = validator.validate(tool_names=["read", "write"])
        assert result.valid is True

    def test_validate_budget_only(self):
        """Should validate only budget when tools not provided."""
        registry = MockToolRegistry(["read"])
        validator = CombinedToolValidator(registry, min_budget=5, max_budget=50)

        result = validator.validate(tool_budget=20)
        assert result.valid is True

    def test_combined_validation_errors(self):
        """Should aggregate errors from both validators."""
        registry = MockToolRegistry(["read"])
        validator = CombinedToolValidator(
            registry,
            min_budget=10,
            max_budget=50,
            strict_tool_validation=True,
        )

        result = validator.validate(
            tool_names=["read", "missing_tool"],
            tool_budget=5,  # Below min
        )

        assert result.valid is False
        assert len(result.errors) >= 1  # At least one error

    def test_get_component_validators(self):
        """Should expose component validators."""
        registry = MockToolRegistry(["read"])
        validator = CombinedToolValidator(registry)

        assert isinstance(
            validator.get_availability_validator(), ToolAvailabilityValidator
        )
        assert isinstance(validator.get_budget_validator(), ToolBudgetValidator)


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_tools_function(self):
        """Convenience function should validate tools."""
        registry = MockToolRegistry(["read", "write"])
        result = validate_tools(["read", "write"], registry)
        assert result.valid is True

    def test_validate_budget_function(self):
        """Convenience function should validate budget."""
        result = validate_budget(50, min_budget=0, max_budget=100)
        assert result.valid is True

    def test_validate_budget_function_invalid(self):
        """Convenience function should reject invalid budget."""
        result = validate_budget(150, min_budget=0, max_budget=100)
        assert result.valid is False
