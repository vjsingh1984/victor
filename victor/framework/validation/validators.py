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

"""Common validators for the validation framework.

This module provides reusable validators that can be used across
all verticals to eliminate duplicate validation logic.

Validators:
- ThresholdValidator: Check if value meets threshold (min/max)
- RangeValidator: Check if value is within range
- PresenceValidator: Check if value exists
- PatternValidator: Regex pattern matching
- TypeValidator: Type checking
- LengthValidator: String/collection length validation
- CompositeValidator: Combine multiple validators

Design Pattern: Strategy + Composite
"""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Union,
    cast,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.framework.validation.pipeline import (
        ValidationContext,
        ValidationResult,
        ValidationSeverity,
    )

logger = __import__("logging").getLogger(__name__)


# =============================================================================
# Helper to avoid circular imports
# =============================================================================


def _get_validation_result() -> type[Any]:
    """Get ValidationResult at runtime to avoid circular import."""
    from victor.framework.validation.pipeline import ValidationResult

    return ValidationResult


def _get_validation_context() -> type[Any]:
    """Get ValidationContext at runtime to avoid circular import."""
    from victor.framework.validation.pipeline import ValidationContext

    return ValidationContext


# =============================================================================
# Validator Protocol
# =============================================================================


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for validators.

    All validators must implement this protocol to be compatible
    with the ValidationPipeline.
    """

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate the data.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult with any issues found
        """
        ...

    @property
    def name(self) -> str:
        """Get the validator name."""
        ...


# =============================================================================
# Base Validator
# =============================================================================


class BaseValidator(ABC):
    """Base class for validators with common functionality.

    Provides helper methods for extracting values from data
    and creating validation results.
    """

    def __init__(
        self,
        field: Optional[str] = None,
        error_code: Optional[str] = None,
        custom_message: Optional[str] = None,
    ):
        """Initialize the base validator.

        Args:
            field: Field to validate (None = validate entire data)
            error_code: Error code for issues
            custom_message: Custom error message template
        """
        self._field = field
        self._error_code = error_code
        self._custom_message = custom_message
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        """Get the validator name."""
        return self.__class__.__name__

    def _get_value(self, data: Dict[str, Any]) -> tuple[str, Any]:
        """Get the value to validate.

        Returns:
            Tuple of (field_path, value)
        """
        if self._field is None:
            return "", data

        # Support nested paths with dot notation
        if "." in self._field:
            parts = self._field.split(".")
            value: Any = data
            path: list[str] = []
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                    path.append(part)
                else:
                    return self._field, None
            return ".".join(path), value

        return self._field, data.get(self._field)

    def _create_result(
        self,
        is_valid: bool = True,
    ) -> "ValidationResult":
        """Create a validation result.

        Args:
            is_valid: Whether validation passed

        Returns:
            ValidationResult
        """
        from victor.framework.validation.pipeline import ValidationResult

        return ValidationResult(is_valid=is_valid)

    def _add_error(
        self,
        result: ValidationResult,
        message: str,
        value: Optional[Any] = None,
    ) -> None:
        """Add an error to the result.

        Args:
            result: Result to add error to
            message: Error message
            value: The problematic value
        """
        path = self._field or ""
        if self._custom_message:
            message = self._custom_message.format(
                field=path,
                value=value,
                message=message,
            )

        result.add_error(
            path=path,
            message=message,
            code=self._error_code,
            value=value,
        )

    def _add_warning(
        self,
        result: ValidationResult,
        message: str,
    ) -> None:
        """Add a warning to the result.

        Args:
            result: Result to add warning to
            message: Warning message
        """
        path = self._field or ""
        result.add_warning(path=path, message=message, code=self._error_code)

    def _add_info(
        self,
        result: ValidationResult,
        message: str,
    ) -> None:
        """Add an info message to the result.

        Args:
            result: Result to add info to
            message: Info message
        """
        path = self._field or ""
        result.add_info(path=path, message=message, code=self._error_code)

    @abstractmethod
    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate the data.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult with any issues found
        """
        ...


# =============================================================================
# Threshold Validator
# =============================================================================


class ThresholdValidator(BaseValidator):
    """Validate that a numeric value meets threshold requirements.

    Checks if a value is within min/max bounds.

    Example:
        validator = ThresholdValidator(
            field="score",
            min_value=0,
            max_value=100,
        )
        result = validator.validate({"score": 85}, context)
    """

    def __init__(
        self,
        field: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        error_code: Optional[str] = None,
    ):
        """Initialize the threshold validator.

        Args:
            field: Field to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            min_inclusive: Whether min is inclusive
            max_inclusive: Whether max is inclusive
            error_code: Error code for issues
        """
        super().__init__(field=field, error_code=error_code)
        self._min_value = min_value
        self._max_value = max_value
        self._min_inclusive = min_inclusive
        self._max_inclusive = max_inclusive

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate the threshold.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        result = self._create_result()
        path, value = self._get_value(data)

        if value is None:
            if self._min_value is not None:
                self._add_error(
                    result,
                    f"Field '{path}' is required but missing",
                )
            return result

        # Check if numeric
        if not isinstance(value, (int, float)):
            self._add_error(
                result,
                f"Field '{path}' must be numeric, got {type(value).__name__}",
            )
            return result

        # Check minimum
        if self._min_value is not None:
            if self._min_inclusive:
                if value < self._min_value:
                    self._add_error(
                        result,
                        f"Value {value} is below minimum {self._min_value}",
                        value=value,
                    )
            else:
                if value <= self._min_value:
                    self._add_error(
                        result,
                        f"Value {value} is at or below minimum {self._min_value}",
                        value=value,
                    )

        # Check maximum
        if self._max_value is not None:
            if self._max_inclusive:
                if value > self._max_value:
                    self._add_error(
                        result,
                        f"Value {value} exceeds maximum {self._max_value}",
                        value=value,
                    )
            else:
                if value >= self._max_value:
                    self._add_error(
                        result,
                        f"Value {value} is at or above maximum {self._max_value}",
                        value=value,
                    )

        return result


# =============================================================================
# Range Validator
# =============================================================================


class RangeValidator(BaseValidator):
    """Validate that a value is within a specific range.

    Similar to ThresholdValidator but supports both numeric ranges
    and sequence ranges (like checking if an index is valid).

    Example:
        validator = RangeValidator(
            field="age",
            min_value=18,
            max_value=65,
        )
        result = validator.validate({"age": 25}, context)
    """

    def __init__(
        self,
        field: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        exclusive_min: bool = False,
        exclusive_max: bool = False,
        error_code: Optional[str] = None,
    ):
        """Initialize the range validator.

        Args:
            field: Field to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            exclusive_min: Exclude minimum value
            exclusive_max: Exclude maximum value
            error_code: Error code for issues
        """
        super().__init__(field=field, error_code=error_code)
        self._min_value = min_value
        self._max_value = max_value
        self._exclusive_min = exclusive_min
        self._exclusive_max = exclusive_max

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate the range.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        result = self._create_result()
        path, value = self._get_value(data)

        if value is None:
            return result  # Skip validation for None

        if not isinstance(value, (int, float)):
            self._add_error(
                result,
                f"Field '{path}' must be numeric for range validation",
            )
            return result

        if self._min_value is not None:
            if self._exclusive_min:
                if value <= self._min_value:
                    self._add_error(
                        result,
                        f"Value {value} must be greater than {self._min_value}",
                        value=value,
                    )
            else:
                if value < self._min_value:
                    self._add_error(
                        result,
                        f"Value {value} must be at least {self._min_value}",
                        value=value,
                    )

        if self._max_value is not None:
            if self._exclusive_max:
                if value >= self._max_value:
                    self._add_error(
                        result,
                        f"Value {value} must be less than {self._max_value}",
                        value=value,
                    )
            else:
                if value > self._max_value:
                    self._add_error(
                        result,
                        f"Value {value} must be at most {self._max_value}",
                        value=value,
                    )

        return result


# =============================================================================
# Presence Validator
# =============================================================================


class PresenceValidator(BaseValidator):
    """Validate that a value exists (is not None/empty).

    Can check for:
    - Field existence
    - Non-None values
    - Non-empty strings/collections

    Example:
        validator = PresenceValidator(
            field="email",
            allow_empty=False,
        )
        result = validator.validate({"email": "test@example.com"}, context)
    """

    def __init__(
        self,
        field: str,
        required: bool = True,
        allow_empty: bool = True,
        check_truthiness: bool = False,
        error_code: Optional[str] = None,
    ):
        """Initialize the presence validator.

        Args:
            field: Field to validate
            required: Whether field must exist
            allow_empty: Whether empty strings/collections are allowed
            check_truthiness: Check if value is truthy
            error_code: Error code for issues
        """
        super().__init__(field=field, error_code=error_code)
        self._required = required
        self._allow_empty = allow_empty
        self._check_truthiness = check_truthiness

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate the presence.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        result = self._create_result()
        path, value = self._get_value(data)

        # _get_value returns None for missing fields (including nested paths)
        # Check for None
        if value is None:
            if self._required or self._check_truthiness:
                self._add_error(
                    result,
                    f"Field '{path}' is missing",
                )
            return result

        # Check for empty
        if not self._allow_empty:
            if isinstance(value, (str, list, dict, set, tuple)):
                if len(value) == 0:
                    self._add_error(
                        result,
                        f"Field '{path}' is empty",
                    )

        # Check truthiness
        if self._check_truthiness and not value:
            self._add_error(
                result,
                f"Field '{path}' is not truthy",
            )

        return result


# =============================================================================
# Pattern Validator
# =============================================================================


class PatternValidator(BaseValidator):
    r"""Validate that a string matches a regex pattern.

    Supports both regex patterns and common patterns like email, URL, etc.

    Example:
        validator = PatternValidator(
            field="email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        result = validator.validate({"email": "test@example.com"}, context)

        # Or use built-in pattern
        validator = PatternValidator(
            field="email",
            pattern_type="email",
        )
    """

    class PatternType(str, Enum):
        """Common pattern types."""

        EMAIL = "email"
        URL = "url"
        UUID = "uuid"
        HEX_COLOR = "hex_color"
        IP_ADDRESS = "ip_address"
        MAC_ADDRESS = "mac_address"
        SEMVER = "semver"
        SLUG = "slug"
        USERNAME = "username"

    # Common regex patterns
    PATTERNS: Dict[PatternType, str] = {
        PatternType.EMAIL: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        PatternType.URL: r"^https?://[^\s/$.?#].[^\s]*$",
        PatternType.UUID: r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        PatternType.HEX_COLOR: r"^#?([a-f0-9]{6}|[a-f0-9]{3})$",
        PatternType.IP_ADDRESS: r"^(\d{1,3}\.){3}\d{1,3}$",
        PatternType.MAC_ADDRESS: r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$",
        PatternType.SEMVER: r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$",
        PatternType.SLUG: r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        PatternType.USERNAME: r"^[a-zA-Z0-9_-]{3,20}$",
    }

    _pattern: str = ""

    def __init__(
        self,
        field: str,
        pattern: Optional[str] = None,
        pattern_type: Optional[Union[str, PatternType]] = None,
        flags: int = 0,
        error_code: Optional[str] = None,
    ):
        """Initialize the pattern validator.

        Args:
            field: Field to validate
            pattern: Custom regex pattern
            pattern_type: Built-in pattern type
            flags: Regex flags (re.IGNORECASE, etc.)
            error_code: Error code for issues
        """
        super().__init__(field=field, error_code=error_code)

        if pattern_type:
            if isinstance(pattern_type, str):
                pattern_type = self.PatternType(pattern_type)
            self._pattern = self.PATTERNS[pattern_type]
        else:
            self._pattern = pattern or ""

        self._flags = flags
        self._compiled_pattern: Optional[re.Pattern[str]] = None
        self._compile_pattern()

    def _compile_pattern(self) -> None:
        """Compile the regex pattern."""
        if self._pattern:
            try:
                self._compiled_pattern = re.compile(self._pattern, self._flags)
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {e}")
                self._compiled_pattern = None

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate the pattern.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        result = self._create_result()
        path, value = self._get_value(data)

        if value is None:
            return result  # Skip None values

        if not isinstance(value, str):
            self._add_error(
                result,
                f"Field '{path}' must be a string for pattern validation",
            )
            return result

        if self._compiled_pattern is None:
            self._add_warning(
                result,
                f"Pattern validator for '{path}' has invalid pattern",
            )
            return result

        if not self._compiled_pattern.match(value):
            self._add_error(
                result,
                f"Value '{value}' does not match required pattern",
                value=value,
            )

        return result


# =============================================================================
# Type Validator
# =============================================================================


class TypeValidator(BaseValidator):
    """Validate that a value is of the expected type.

    Supports:
    - Single type checking
    - Union type checking (one of multiple types)
    - Custom type check functions

    Example:
        validator = TypeValidator(
            field="count",
            expected_type=int,
        )
        result = validator.validate({"count": 10}, context)

        # Union type
        validator = TypeValidator(
            field="value",
            expected_type=(int, float),
        )
    """

    def __init__(
        self,
        field: str,
        expected_type: Union[type, tuple[type, ...], str],
        coerce: bool = False,
        error_code: Optional[str] = None,
    ):
        """Initialize the type validator.

        Args:
            field: Field to validate
            expected_type: Expected type(s) or type name string
            coerce: Whether to coerce to expected type
            error_code: Error code for issues
        """
        super().__init__(field=field, error_code=error_code)

        # Handle string type names
        if isinstance(expected_type, str):
            self._expected_type = self._resolve_type_name(expected_type)
        else:
            self._expected_type = expected_type

        self._coerce = coerce

    def _resolve_type_name(self, type_name: str) -> Union[type, tuple[type, ...]]:
        """Resolve a type name to the actual type.

        Args:
            type_name: Name of the type

        Returns:
            The resolved type
        """
        type_map = {
            "int": int,
            "integer": int,
            "float": float,
            "str": str,
            "string": str,
            "bool": bool,
            "boolean": bool,
            "list": list[Any],
            "dict": dict[str, Any],
            "tuple": tuple[Any, ...],
            "set": set[Any],
        }
        return type_map.get(type_name.lower(), str)

    def _coerce_value(
        self,
        value: Any,
        target_type: type,
    ) -> tuple[bool, Any]:
        """Coerce a value to the target type.

        Args:
            value: Value to coerce
            target_type: Target type

        Returns:
            Tuple of (success, coerced_value)
        """
        try:
            if target_type is int:
                return True, int(float(value)) if isinstance(value, str) else int(value)
            if target_type is float:
                return True, float(value)
            if target_type is str:
                return True, str(value)
            if target_type is bool:
                if isinstance(value, str):
                    return True, value.lower() in ("true", "1", "yes", "on")
                return True, bool(value)
        except (ValueError, TypeError):
            pass

        return False, value

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate the type.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        result = self._create_result()
        path, value = self._get_value(data)

        if value is None:
            return result  # Skip None values

        # Check type
        if not isinstance(value, self._expected_type):
            if self._coerce:
                # Try to coerce - handle tuple of types
                expected_type = (
                    self._expected_type[0]
                    if isinstance(self._expected_type, tuple)
                    else self._expected_type
                )
                success, coerced = self._coerce_value(value, expected_type)
                if success:
                    # Update the data with coerced value
                    if self._field:
                        # Update the path in the original data
                        if "." in self._field:
                            parts = self._field.split(".")
                            current = data
                            for part in parts[:-1]:
                                if part in current:
                                    current = current[part]
                            current[parts[-1]] = coerced
                        else:
                            data[self._field] = coerced
                    result.add_info(
                        path,
                        f"Coerced '{path}' from {type(value).__name__} to {self._type_name()}",
                    )
                    return result

            self._add_error(
                result,
                f"Field '{path}' must be of type {self._type_name()}, got {type(value).__name__}",
            )

        return result

    def _type_name(self) -> str:
        """Get the expected type name."""
        if isinstance(self._expected_type, tuple):
            return " or ".join(t.__name__ for t in self._expected_type)
        return self._expected_type.__name__


# =============================================================================
# Length Validator
# =============================================================================


class LengthValidator(BaseValidator):
    """Validate the length of strings or collections.

    Example:
        validator = LengthValidator(
            field="username",
            min_length=3,
            max_length=20,
        )
        result = validator.validate({"username": "john_doe"}, context)
    """

    def __init__(
        self,
        field: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        exact_length: Optional[int] = None,
        error_code: Optional[str] = None,
    ):
        """Initialize the length validator.

        Args:
            field: Field to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            exact_length: Exact required length
            error_code: Error code for issues
        """
        super().__init__(field=field, error_code=error_code)
        self._min_length = min_length
        self._max_length = max_length
        self._exact_length = exact_length

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate the length.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        result = self._create_result()
        path, value = self._get_value(data)

        if value is None:
            return result  # Skip None values

        # Get length
        try:
            length = len(value)
        except TypeError:
            self._add_error(
                result,
                f"Field '{path}' does not have a measurable length",
            )
            return result

        # Check exact length
        if self._exact_length is not None:
            if length != self._exact_length:
                self._add_error(
                    result,
                    f"Length {length} does not match required length {self._exact_length}",
                )
            return result

        # Check minimum
        if self._min_length is not None and length < self._min_length:
            self._add_error(
                result,
                f"Length {length} is below minimum {self._min_length}",
            )

        # Check maximum
        if self._max_length is not None and length > self._max_length:
            self._add_error(
                result,
                f"Length {length} exceeds maximum {self._max_length}",
            )

        return result


# =============================================================================
# Composite Validator
# =============================================================================


class CompositeLogic(str, Enum):
    """Logic for combining validators."""

    ALL = "all"  # All validators must pass
    ANY = "any"  # At least one validator must pass
    NONE = "none"  # No validators should pass (negation)
    ONE = "one"  # Exactly one validator must pass


class CompositeValidator(BaseValidator):
    """Combine multiple validators with configurable logic.

    Supports:
    - ALL: All validators must pass (AND)
    - ANY: At least one validator must pass (OR)
    - NONE: No validators should pass (NOT)
    - ONE: Exactly one validator must pass (XOR)

    Example:
        validator = CompositeValidator(
            validators=[
                PatternValidator(field="email", pattern_type="email"),
                PresenceValidator(field="email"),
            ],
            logic=CompositeLogic.ALL,
        )
    """

    def __init__(
        self,
        validators: List[ValidatorProtocol],
        logic: Union[CompositeLogic, str] = CompositeLogic.ALL,
        field: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        """Initialize the composite validator.

        Args:
            validators: List of validators to combine
            logic: How to combine the results
            field: Optional field (for BaseValidator)
            error_code: Error code for issues
        """
        super().__init__(field=field, error_code=error_code)

        if isinstance(logic, str):
            logic = CompositeLogic(logic)

        self._validators = validators
        self._logic = logic
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        """Get the validator name."""
        return f"CompositeValidator({self._logic.value})"

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate using the composite logic.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        results = []

        with self._lock:
            validators = list(self._validators)  # Copy for thread safety

        for validator in validators:
            result = validator.validate(data, context)
            results.append(result)

        return self._combine_results(results)

    def _combine_results(
        self,
        results: List[ValidationResult],
    ) -> ValidationResult:
        """Combine validation results using the configured logic.

        Args:
            results: List of validation results

        Returns:
            Combined ValidationResult
        """
        if self._logic == CompositeLogic.ALL:
            return self._all_logic(results)
        elif self._logic == CompositeLogic.ANY:
            return self._any_logic(results)
        elif self._logic == CompositeLogic.NONE:
            return self._none_logic(results)
        elif self._logic == CompositeLogic.ONE:
            return self._one_logic(results)
        else:
            # Default to ALL
            return self._all_logic(results)

    def _all_logic(self, results: List[ValidationResult]) -> ValidationResult:
        """ALL logic: All validators must pass."""
        from victor.framework.validation.pipeline import ValidationResult

        combined = ValidationResult()
        for result in results:
            combined.merge(result)
        return combined

    def _any_logic(self, results: List[ValidationResult]) -> ValidationResult:
        """ANY logic: At least one validator must pass."""
        # Check if any passed
        if any(r.is_valid for r in results):
            # Return the first valid result
            for result in results:
                if result.is_valid:
                    return result

        # All failed, combine all errors
        from victor.framework.validation.pipeline import ValidationResult

        combined = ValidationResult()
        for result in results:
            combined.merge(result)
        return combined

    def _none_logic(self, results: List[ValidationResult]) -> ValidationResult:
        """NONE logic: No validators should pass."""
        from victor.framework.validation.pipeline import ValidationResult

        passed = [r for r in results if r.is_valid]

        if passed:
            result = ValidationResult()
            result.add_error(
                self._field or "",
                "Validation passed but should have failed (none logic)",
            )
            return result

        return ValidationResult(is_valid=True)

    def _one_logic(self, results: List[ValidationResult]) -> ValidationResult:
        """ONE logic: Exactly one validator must pass."""
        from victor.framework.validation.pipeline import ValidationResult

        passed = [r for r in results if r.is_valid]

        if len(passed) == 1:
            return passed[0]

        result = ValidationResult()
        if len(passed) == 0:
            result.add_error(
                self._field or "",
                "No validators passed (expected exactly one)",
            )
        else:
            result.add_error(
                self._field or "",
                f"Multiple validators passed (expected exactly one): {len(passed)}",
            )

        return result


# =============================================================================
# Conditional Validator
# =============================================================================


class ConditionalValidator(BaseValidator):
    """Validator that runs conditionally based on a predicate.

    Example:
        validator = ConditionalValidator(
            validator=ThresholdValidator("age", min_value=18),
            condition=lambda data: data.get("requires_age_verification", False),
        )
    """

    def __init__(
        self,
        validator: ValidatorProtocol,
        condition: Callable[[Dict[str, Any]], bool],
        field: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        """Initialize the conditional validator.

        Args:
            validator: Validator to run conditionally
            condition: Function that returns True if validator should run
            field: Optional field
            error_code: Optional error code
        """
        super().__init__(field=field, error_code=error_code)
        self._validator = validator
        self._condition = condition

    @property
    def name(self) -> str:
        """Get the validator name."""
        return f"ConditionalValidator({self._validator.name})"

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate only if condition is met.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        if not self._condition(data):
            from victor.framework.validation.pipeline import ValidationResult

            return ValidationResult(is_valid=True)

        return self._validator.validate(data, context)


# =============================================================================
# Transforming Validator
# =============================================================================


class TransformingValidator(BaseValidator):
    """Validator that can transform values before validation.

    Example:
        validator = TransformingValidator(
            validator=PatternValidator(
                field="email",
                pattern_type="email",
            ),
            transform=lambda v: v.strip().lower(),
        )
    """

    def __init__(
        self,
        validator: BaseValidator,
        transform: Callable[[Any], Any],
        field: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        """Initialize the transforming validator.

        Args:
            validator: Validator to run after transformation
            transform: Function to transform the value
            field: Optional field
            error_code: Optional error code
        """
        super().__init__(field=field, error_code=error_code)
        self._validator = validator
        self._transform = transform

    @property
    def name(self) -> str:
        """Get the validator name."""
        return f"TransformingValidator({self._validator.name})"

    def validate(
        self,
        data: Dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate after transforming the value.

        Args:
            data: The data to validate
            context: The validation context

        Returns:
            ValidationResult
        """
        ValidationResult = _get_validation_result()

        # Get the value
        path, value = self._get_value(data)

        if value is None:
            from victor.framework.validation.pipeline import ValidationResult

            return ValidationResult(is_valid=True)  # type: ignore[no-any-return]

        # Transform the value
        try:
            transformed = self._transform(value)
        except Exception as e:
            from victor.framework.validation.pipeline import ValidationResult

            result = ValidationResult()
            self._add_error(
                result,
                f"Transform failed: {e}",
            )
            return result  # type: ignore[no-any-return]

        # Create a copy of data with transformed value
        if path:
            if "." in path:
                parts = path.split(".")
                current = data
                for part in parts[:-1]:
                    if part in current:
                        current = current[part]
                if parts[-1] in current:
                    current[parts[-1]] = transformed
            else:
                data[path] = transformed

        # Run the validator on the transformed data
        return self._validator.validate(data, context)


__all__ = [
    # Protocol
    "ValidatorProtocol",
    # Base
    "BaseValidator",
    # Validators
    "ThresholdValidator",
    "RangeValidator",
    "PresenceValidator",
    "PatternValidator",
    "TypeValidator",
    "LengthValidator",
    "CompositeValidator",
    "ConditionalValidator",
    "TransformingValidator",
    # Enums
    "CompositeLogic",
    "PatternValidator",
]
