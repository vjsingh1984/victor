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

"""Common tool validators for use across all verticals.

This module provides reusable validation components for tool-related checks,
eliminating duplicate validation logic across verticals.

Phase 2.3: Tool Validation Unification (SOLID Refactoring)

Validators:
- ToolAvailabilityValidator: Check if tools exist in registry
- ToolBudgetValidator: Validate tool budget values
- CombinedToolValidator: Validate both availability and budget
- ValidationResult: Common validation result dataclass

Design Pattern: Strategy + Template Method
- Protocol-based dependency injection for tool registry
- Configurable validation rules
- Consistent error reporting across all verticals

Example:
    from victor.tools.validators.common import ToolAvailabilityValidator
    from victor.tools.registry import ToolRegistry

    # Get tool registry
    registry = ToolRegistry.get_shared_instance()

    # Create validator
    validator = ToolAvailabilityValidator(registry)

    # Validate tools
    result = validator.validate_tools_available(["read", "write", "search"])
    if not result.valid:
        for error in result.errors:
            print(f"Error: {error}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class ToolRegistryProtocol(Protocol):
    """Protocol for tool registry abstraction.

    This protocol enables dependency injection and testing without
    coupling to a specific ToolRegistry implementation.
    """

    def get(self, name: str) -> Optional[Any]:
        """Get tool by name.

        Args:
            name: Tool name to look up

        Returns:
            Tool instance or None if not found
        """
        ...

    def has_tool(self, name: str) -> bool:
        """Check if tool exists in registry.

        Args:
            name: Tool name to check

        Returns:
            True if tool exists, False otherwise
        """
        ...

    def list_tools(self, only_enabled: bool = True) -> List[Any]:
        """List all available tools.

        Args:
            only_enabled: If True, only return enabled tools

        Returns:
            List of tool instances
        """
        ...


# =============================================================================
# Validation Results
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        valid: True if validation passed, False otherwise
        errors: List of error messages
        warnings: List of warning messages
        context: Additional context about the validation
    """

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Add an error message and mark validation as failed.

        Args:
            error: Error message to add
        """
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message.

        Args:
            warning: Warning message to add
        """
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one.

        Args:
            other: Another ValidationResult to merge
        """
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.context.update(other.context)

    def summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Summary string
        """
        if self.valid:
            msg = "Validation passed"
            if self.warnings:
                msg += f" with {len(self.warnings)} warning(s)"
            return msg

        return (
            f"Validation failed: {len(self.errors)} error(s), " f"{len(self.warnings)} warning(s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "context": self.context,
        }


# =============================================================================
# Tool Availability Validator
# =============================================================================


class ToolAvailabilityValidator:
    """Validates that tools exist in the tool registry.

    This validator checks if requested tools are available in the registry
    before attempting to use them. Useful for pre-flight validation in
    workflows, agent initialization, and tool configuration.

    Example:
        validator = ToolAvailabilityValidator(tool_registry)
        result = validator.validate_tools_available(["read", "write", "search"])
        if not result.valid:
            print(f"Missing tools: {result.errors}")
    """

    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        strict: bool = False,
    ):
        """Initialize the validator.

        Args:
            tool_registry: Tool registry to check against
            strict: If True, validation fails on any missing tool.
                    If False, missing tools are warnings only.
        """
        self._tool_registry = tool_registry
        self._strict = strict

    def validate_tools_available(
        self,
        tool_names: List[str],
    ) -> ValidationResult:
        """Validate that all tools are available in the registry.

        Args:
            tool_names: List of tool names to validate

        Returns:
            ValidationResult with any missing tools reported
        """
        result = ValidationResult()

        missing_tools: List[str] = []
        for name in tool_names:
            if not self._tool_registry.has_tool(name):
                missing_tools.append(name)

        if missing_tools:
            msg = f"Tool(s) not found in registry: {', '.join(missing_tools)}"
            if self._strict:
                result.add_error(msg)
            else:
                result.add_warning(msg)

        result.context["checked_tools"] = tool_names
        result.context["missing_tools"] = missing_tools

        return result

    def validate_tool_available(
        self,
        tool_name: str,
    ) -> ValidationResult:
        """Validate that a single tool is available.

        Args:
            tool_name: Tool name to validate

        Returns:
            ValidationResult
        """
        return self.validate_tools_available([tool_name])

    def is_tool_available(self, tool_name: str) -> bool:
        """Quick check if tool is available (no result object).

        Args:
            tool_name: Tool name to check

        Returns:
            True if tool exists, False otherwise
        """
        return self._tool_registry.has_tool(tool_name)

    def list_missing_tools(
        self,
        tool_names: List[str],
    ) -> List[str]:
        """Get list of missing tools from a list.

        Args:
            tool_names: List of tool names to check

        Returns:
            List of tools that are not available
        """
        result = self.validate_tools_available(tool_names)
        return result.context.get("missing_tools", [])


# =============================================================================
# Tool Budget Validator
# =============================================================================


class ToolBudgetValidator:
    """Validates tool budget values.

    Tool budgets control how many tool calls an agent can make in a session.
    This validator ensures budgets are within acceptable ranges.

    Default ranges:
    - Minimum: 0 (unlimited/disabled)
    - Maximum: 500 (framework-wide limit)
    - Recommended: 10-50 for typical tasks

    Example:
        validator = ToolBudgetValidator()
        result = validator.validate_budget(100)
        if not result.valid:
            print(f"Invalid budget: {result.errors}")
    """

    # Framework-wide budget limits
    DEFAULT_MIN_BUDGET = 0
    DEFAULT_MAX_BUDGET = 500
    DEFAULT_RECOMMENDED_MIN = 5
    DEFAULT_RECOMMENDED_MAX = 50

    def __init__(
        self,
        min_budget: int = DEFAULT_MIN_BUDGET,
        max_budget: int = DEFAULT_MAX_BUDGET,
        recommended_min: int = DEFAULT_RECOMMENDED_MIN,
        recommended_max: int = DEFAULT_RECOMMENDED_MAX,
    ):
        """Initialize the validator.

        Args:
            min_budget: Minimum allowed budget (default: 0)
            max_budget: Maximum allowed budget (default: 500)
            recommended_min: Minimum recommended budget (default: 5)
            recommended_max: Maximum recommended budget (default: 50)
        """
        self._min_budget = min_budget
        self._max_budget = max_budget
        self._recommended_min = recommended_min
        self._recommended_max = recommended_max

    def validate_budget(
        self,
        budget: int,
    ) -> ValidationResult:
        """Validate a tool budget value.

        Args:
            budget: Budget value to validate

        Returns:
            ValidationResult with any validation issues
        """
        result = ValidationResult()
        result.context["budget"] = budget

        # Type check
        if not isinstance(budget, int):
            result.add_error(f"Budget must be an integer, got {type(budget).__name__}")
            return result

        # Range check
        if budget < self._min_budget:
            result.add_error(f"Budget ({budget}) is below minimum ({self._min_budget})")

        if budget > self._max_budget:
            result.add_error(f"Budget ({budget}) exceeds maximum ({self._max_budget})")

        # Recommendation check (warning only)
        if budget < self._recommended_min:
            result.add_warning(
                f"Budget ({budget}) is below recommended minimum ({self._recommended_min}). "
                "Low budgets may limit agent effectiveness."
            )

        if budget > self._recommended_max:
            result.add_warning(
                f"Budget ({budget}) is above recommended maximum ({self._recommended_max}). "
                "High budgets may lead to excessive tool usage."
            )

        return result

    def validate_task_budgets(
        self,
        budgets: Dict[str, int],
    ) -> ValidationResult:
        """Validate multiple task budgets.

        Args:
            budgets: Dictionary mapping task names to budgets

        Returns:
            ValidationResult with aggregated issues
        """
        result = ValidationResult()
        result.context["task_count"] = len(budgets)

        for task_name, budget in budgets.items():
            task_result = self.validate_budget(budget)
            if not task_result.valid:
                for error in task_result.errors:
                    result.add_error(f"{task_name}: {error}")
            for warning in task_result.warnings:
                result.add_warning(f"{task_name}: {warning}")

        return result

    def is_valid_budget(self, budget: int) -> bool:
        """Quick check if budget is valid (no result object).

        Args:
            budget: Budget value to check

        Returns:
            True if budget is within valid range
        """
        return isinstance(budget, int) and self._min_budget <= budget <= self._max_budget

    def clamp_budget(self, budget: int) -> int:
        """Clamp budget to valid range.

        Args:
            budget: Budget value to clamp

        Returns:
            Budget clamped to [min_budget, max_budget]
        """
        if not isinstance(budget, int):
            return self._recommended_min

        return max(self._min_budget, min(budget, self._max_budget))


# =============================================================================
# Combined Validator
# =============================================================================


class CombinedToolValidator:
    """Combined validator for tools and budgets.

    This validator combines multiple validators for comprehensive tool
    validation. Useful for agent initialization and workflow setup.

    Example:
        validator = CombinedToolValidator(
            tool_registry=registry,
            min_budget=0,
            max_budget=100,
        )

        # Validate tools and budget
        result = validator.validate(
            tool_names=["read", "write"],
            tool_budget=50,
        )
        if not result.valid:
            print(f"Validation failed: {result.summary()}")
    """

    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        min_budget: int = ToolBudgetValidator.DEFAULT_MIN_BUDGET,
        max_budget: int = ToolBudgetValidator.DEFAULT_MAX_BUDGET,
        strict_tool_validation: bool = False,
    ):
        """Initialize the combined validator.

        Args:
            tool_registry: Tool registry for availability checks
            min_budget: Minimum allowed tool budget
            max_budget: Maximum allowed tool budget
            strict_tool_validation: If True, fail on missing tools
        """
        self._availability_validator = ToolAvailabilityValidator(
            tool_registry,
            strict=strict_tool_validation,
        )
        self._budget_validator = ToolBudgetValidator(
            min_budget=min_budget,
            max_budget=max_budget,
        )

    def validate(
        self,
        tool_names: Optional[List[str]] = None,
        tool_budget: Optional[int] = None,
        task_budgets: Optional[Dict[str, int]] = None,
    ) -> ValidationResult:
        """Validate tools and/or budgets.

        Args:
            tool_names: Optional list of tool names to validate
            tool_budget: Optional tool budget to validate
            task_budgets: Optional task budgets to validate

        Returns:
            ValidationResult with aggregated issues
        """
        result = ValidationResult()

        # Validate tool availability
        if tool_names:
            availability_result = self._availability_validator.validate_tools_available(tool_names)
            result.merge(availability_result)

        # Validate tool budget
        if tool_budget is not None:
            budget_result = self._budget_validator.validate_budget(tool_budget)
            result.merge(budget_result)

        # Validate task budgets
        if task_budgets:
            task_budgets_result = self._budget_validator.validate_task_budgets(task_budgets)
            result.merge(task_budgets_result)

        return result

    def get_availability_validator(self) -> ToolAvailabilityValidator:
        """Get the availability validator component.

        Returns:
            ToolAvailabilityValidator instance
        """
        return self._availability_validator

    def get_budget_validator(self) -> ToolBudgetValidator:
        """Get the budget validator component.

        Returns:
            ToolBudgetValidator instance
        """
        return self._budget_validator


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_tools(
    tool_names: List[str],
    tool_registry: ToolRegistryProtocol,
    strict: bool = False,
) -> ValidationResult:
    """Convenience function for tool availability validation.

    Args:
        tool_names: List of tool names to validate
        tool_registry: Tool registry to check against
        strict: If True, fail on missing tools

    Returns:
        ValidationResult

    Example:
        from victor.tools.validators.common import validate_tools
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry.get_shared_instance()
        result = validate_tools(["read", "write"], registry)
    """
    validator = ToolAvailabilityValidator(tool_registry, strict=strict)
    return validator.validate_tools_available(tool_names)


def validate_budget(
    budget: int,
    min_budget: int = ToolBudgetValidator.DEFAULT_MIN_BUDGET,
    max_budget: int = ToolBudgetValidator.DEFAULT_MAX_BUDGET,
) -> ValidationResult:
    """Convenience function for budget validation.

    Args:
        budget: Budget value to validate
        min_budget: Minimum allowed budget
        max_budget: Maximum allowed budget

    Returns:
        ValidationResult

    Example:
        from victor.tools.validators.common import validate_budget

        result = validate_budget(100, min_budget=0, max_budget=500)
    """
    validator = ToolBudgetValidator(
        min_budget=min_budget,
        max_budget=max_budget,
    )
    return validator.validate_budget(budget)


__all__ = [
    # Protocols
    "ToolRegistryProtocol",
    # Results
    "ValidationResult",
    # Validators
    "ToolAvailabilityValidator",
    "ToolBudgetValidator",
    "CombinedToolValidator",
    # Convenience functions
    "validate_tools",
    "validate_budget",
]
