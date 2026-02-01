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

"""Validation pipeline for workflow validation.

This module implements the generic validation flow:
    validate -> check -> handle -> retry

Design Pattern: Pipeline + Chain of Responsibility
- Protocol-based interfaces for extensibility
- Thread-safe validation execution
- Composable handlers for different failure scenarios
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from collections.abc import Callable, Iterator

from victor.framework.validation.validators import ValidatorProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Enums
# =============================================================================


class ValidationStage(str, Enum):
    """Stages in the validation pipeline."""

    VALIDATE = "validate"
    CHECK = "check"
    HANDLE = "handle"
    RETRY = "retry"


class ValidationAction(str, Enum):
    """Actions to take after validation."""

    CONTINUE = "continue"  # Continue with the validated data
    RETRY = "retry"  # Retry the validation
    HALT = "halt"  # Stop execution
    SKIP = "skip"  # Skip this validation step


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Validation Result
# =============================================================================


@dataclass
class ValidationIssue:
    """A single validation issue.

    Attributes:
        path: Dot-separated path to the field with the issue
        message: Human-readable error message
        severity: Severity level of the issue
        code: Optional error code for programmatic handling
        value: The problematic value (optional)
    """

    path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    code: Optional[str] = None
    value: Optional[Any] = None

    def __str__(self) -> str:
        """Format issue as string."""
        prefix = f"[{self.severity.value.upper()}]"
        if self.code:
            return f"{prefix} {self.path}: {self.message} (code: {self.code})"
        return f"{prefix} {self.path}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "value": self.value,
        }


@dataclass
class ValidationResult:
    """Result of a validation operation.

    This is distinct from ValidationResult in other modules:
    - victor.tools.validators.common: Tool-specific validation
    - victor.core.validation: Configuration validation with Pydantic
    - This module: Generic workflow validation with handlers

    Attributes:
        is_valid: True if no errors were found
        issues: List of validation issues
        data: The validated data (possibly modified)
        context: Additional context about the validation
        duration_seconds: Time taken for validation
        retry_count: Number of retries attempted
    """

    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    retry_count: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get all error-level issues."""
        return [
            i
            for i in self.issues
            if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def info(self) -> list[ValidationIssue]:
        """Get all info-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def add_error(
        self,
        path: str,
        message: str,
        code: Optional[str] = None,
        value: Optional[Any] = None,
    ) -> None:
        """Add an error and mark validation as failed."""
        self.issues.append(
            ValidationIssue(
                path=path,
                message=message,
                severity=ValidationSeverity.ERROR,
                code=code,
                value=value,
            )
        )
        self.is_valid = False

    def add_warning(
        self,
        path: str,
        message: str,
        code: Optional[str] = None,
    ) -> None:
        """Add a warning."""
        self.issues.append(
            ValidationIssue(
                path=path,
                message=message,
                severity=ValidationSeverity.WARNING,
                code=code,
            )
        )

    def add_info(
        self,
        path: str,
        message: str,
        code: Optional[str] = None,
    ) -> None:
        """Add an info message."""
        self.issues.append(
            ValidationIssue(
                path=path,
                message=message,
                severity=ValidationSeverity.INFO,
                code=code,
            )
        )

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.issues.extend(other.issues)
        self.context.update(other.context)
        self.duration_seconds += other.duration_seconds

    def summary(self) -> str:
        """Get a human-readable summary."""
        if self.is_valid:
            msg = "Validation passed"
            if self.warnings:
                msg += f" with {len(self.warnings)} warning(s)"
            return msg

        return (
            f"Validation failed: {len(self.errors)} error(s), " f"{len(self.warnings)} warning(s)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "issues": [i.to_dict() for i in self.issues],
            "data": self.data,
            "context": self.context,
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
        }


# =============================================================================
# Validation Context
# =============================================================================


@dataclass
class ValidationContext:
    """Context passed through the validation pipeline.

    Attributes:
        data: The data being validated
        state: Current workflow state
        metadata: Additional metadata
        retry_count: Current retry attempt
        max_retries: Maximum allowed retries
    """

    data: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the data."""
        self.data[key] = value

    def get_nested(self, path: str, default: Any = None) -> Any:
        """Get a nested value using dot notation."""
        parts = path.split(".")
        current: Any = self.data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return default
        return current if current is not None else default

    def can_retry(self) -> bool:
        """Check if retries are available."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment the retry counter."""
        self.retry_count += 1

    def copy(self) -> "ValidationContext":
        """Create a copy of the context."""
        return ValidationContext(
            data=self.data.copy(),
            state=self.state.copy(),
            metadata=self.metadata.copy(),
            retry_count=self.retry_count,
            max_retries=self.max_retries,
        )


# =============================================================================
# Validation Handler Protocol
# =============================================================================


@runtime_checkable
class ValidationHandler(Protocol):
    """Protocol for validation failure handlers.

    Handlers determine what action to take when validation fails.
    """

    def handle(
        self,
        result: ValidationResult,
        context: ValidationContext,
    ) -> ValidationAction:
        """Handle a validation failure.

        Args:
            result: The validation result
            context: The validation context

        Returns:
            The action to take
        """
        ...

    def can_handle(self, result: ValidationResult) -> bool:
        """Check if this handler can handle the result.

        Args:
            result: The validation result

        Returns:
            True if this handler can handle the result
        """
        ...


# =============================================================================
# Built-in Handlers
# =============================================================================


class RetryHandler:
    """Handler that retries on failure.

    Attributes:
        max_retries: Maximum number of retries
        backoff_factor: Multiplier for retry delay
        retry_on_warnings: Whether to retry on warnings too
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        retry_on_warnings: bool = False,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_on_warnings = retry_on_warnings

    def can_handle(self, result: ValidationResult) -> bool:
        """Can handle if there are issues."""
        # Return bool, not ValidationResult
        has_issues = not result.is_valid or (self.retry_on_warnings and result.warnings)
        return bool(has_issues)

    def handle(
        self,
        result: ValidationResult,
        context: ValidationContext,
    ) -> ValidationAction:
        """Handle by retrying if retries available."""
        if context.can_retry():
            # Apply exponential backoff
            delay = self.backoff_factor * (2**context.retry_count)
            if delay > 0:
                time.sleep(delay)
            context.increment_retry()
            return ValidationAction.RETRY
        return ValidationAction.HALT


class HaltHandler:
    """Handler that halts on any failure."""

    def can_handle(self, result: ValidationResult) -> bool:
        """Always handles."""
        return True

    def handle(
        self,
        result: ValidationResult,
        context: ValidationContext,
    ) -> ValidationAction:
        """Halt on failure."""
        return ValidationAction.HALT


class SkipHandler:
    """Handler that skips validation failures."""

    def __init__(self, log_warnings: bool = True):
        self.log_warnings = log_warnings

    def can_handle(self, result: ValidationResult) -> bool:
        """Always handles."""
        return True

    def handle(
        self,
        result: ValidationResult,
        context: ValidationContext,
    ) -> ValidationAction:
        """Log and skip."""
        if self.log_warnings and result.errors:
            logger.warning(f"Skipping validation failures: {result.summary()}")
        return ValidationAction.SKIP


class ConditionalHandler:
    """Handler that chooses action based on condition."""

    def __init__(
        self,
        condition: Callable[[ValidationResult], bool],
        action: ValidationAction,
        fallback_handler: Optional[ValidationHandler] = None,
    ):
        self.condition = condition
        self.action = action
        self.fallback_handler = fallback_handler or HaltHandler()

    def can_handle(self, result: ValidationResult) -> bool:
        """Always handles."""
        return True

    def handle(
        self,
        result: ValidationResult,
        context: ValidationContext,
    ) -> ValidationAction:
        """Choose action based on condition."""
        if self.condition(result):
            return self.action
        if self.fallback_handler:
            return self.fallback_handler.handle(result, context)
        return ValidationAction.HALT


class ChainHandler:
    """Chain multiple handlers in sequence."""

    def __init__(self, handlers: list[ValidationHandler]):
        self.handlers = handlers

    def can_handle(self, result: ValidationResult) -> bool:
        """Can handle if any handler can."""
        return any(h.can_handle(result) for h in self.handlers)

    def handle(
        self,
        result: ValidationResult,
        context: ValidationContext,
    ) -> ValidationAction:
        """Try each handler in sequence."""
        for handler in self.handlers:
            if handler.can_handle(result):
                action = handler.handle(result, context)
                if action != ValidationAction.HALT or not context.can_retry():
                    return action
        return ValidationAction.HALT


# =============================================================================
# Validation Configuration
# =============================================================================


@dataclass
class ValidationConfig:
    """Configuration for the validation pipeline.

    Attributes:
        validators: List of validators to apply
        handler: Handler for validation failures
        halt_on_error: Whether to halt on first error
        collect_all_errors: Whether to collect all errors before stopping
        max_retries: Maximum retry attempts
        timeout_seconds: Optional timeout for validation
        enable_logging: Whether to log validation steps
    """

    validators: list[ValidatorProtocol] = field(default_factory=list)
    handler: Optional[ValidationHandler] = None
    halt_on_error: bool = True
    collect_all_errors: bool = True
    max_retries: int = 3
    timeout_seconds: Optional[float] = None
    enable_logging: bool = True

    def with_validator(self, validator: ValidatorProtocol) -> "ValidationConfig":
        """Add a validator (for builder pattern)."""
        self.validators.append(validator)
        return self

    def with_handler(self, handler: ValidationHandler) -> "ValidationConfig":
        """Set the handler (for builder pattern)."""
        self.handler = handler
        return self


# =============================================================================
# Validation Pipeline
# =============================================================================


class ValidationPipeline:
    """Generic validation pipeline with validate -> check -> handle -> retry flow.

    This class provides a thread-safe, composable validation pipeline that:
    1. Validates data using configured validators
    2. Checks the result against thresholds
    3. Handles failures using configured handlers
    4. Retries if allowed

    Example:
        pipeline = ValidationPipeline(
            validators=[
                ThresholdValidator(min_value=0, max_value=100),
            ],
            halt_on_error=True,
        )

        result = pipeline.validate({"score": 85})
        if result.is_valid:
            print("Valid!")
    """

    def __init__(
        self,
        validators: Optional[list[ValidatorProtocol]] = None,
        handler: Optional[ValidationHandler] = None,
        halt_on_error: bool = True,
        collect_all_errors: bool = True,
        max_retries: int = 3,
        timeout_seconds: Optional[float] = None,
        enable_logging: bool = True,
    ):
        """Initialize the validation pipeline.

        Args:
            validators: List of validators to apply
            handler: Handler for validation failures
            halt_on_error: Whether to halt on first error
            collect_all_errors: Whether to collect all errors
            max_retries: Maximum retry attempts
            timeout_seconds: Optional timeout
            enable_logging: Whether to log validation steps
        """
        self._validators = validators or []
        self._handler = handler or HaltHandler()
        self._halt_on_error = halt_on_error
        self._collect_all_errors = collect_all_errors
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._enable_logging = enable_logging
        self._lock = threading.RLock()

    def add_validator(self, validator: ValidatorProtocol) -> "ValidationPipeline":
        """Add a validator to the pipeline.

        Args:
            validator: Validator to add

        Returns:
            Self for chaining
        """
        with self._lock:
            self._validators.append(validator)
        return self

    def remove_validator(self, validator: ValidatorProtocol) -> "ValidationPipeline":
        """Remove a validator from the pipeline.

        Args:
            validator: Validator to remove

        Returns:
            Self for chaining
        """
        with self._lock:
            if validator in self._validators:
                self._validators.remove(validator)
        return self

    def set_handler(self, handler: ValidationHandler) -> "ValidationPipeline":
        """Set the failure handler.

        Args:
            handler: Handler to use

        Returns:
            Self for chaining
        """
        with self._lock:
            self._handler = handler
        return self

    def validate(
        self,
        data: dict[str, Any],
        state: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate the data using the pipeline.

        This implements the validate -> check -> handle -> retry flow:
        1. Run all validators against the data
        2. Check if validation passed
        3. If failed, invoke the handler
        4. If handler returns RETRY, go back to step 1

        Args:
            data: The data to validate
            state: Optional workflow state
            metadata: Optional metadata

        Returns:
            ValidationResult with the outcome
        """
        start_time = time.time()
        state = state or {}
        metadata = metadata or {}

        context = ValidationContext(
            data=data.copy(),
            state=state.copy(),
            metadata=metadata.copy(),
            max_retries=self._max_retries,
        )

        while True:
            result = self._validate(context)

            if result.is_valid:
                result.duration_seconds = time.time() - start_time
                result.data = context.data.copy()
                return result

            # Check if handler can handle this result
            if not self._handler.can_handle(result):
                result.duration_seconds = time.time() - start_time
                return result

            # Handle the failure
            action = self._handler.handle(result, context)

            if action == ValidationAction.CONTINUE:
                result.is_valid = True
                result.duration_seconds = time.time() - start_time
                return result

            if action == ValidationAction.SKIP:
                result.is_valid = True  # Force valid
                result.duration_seconds = time.time() - start_time
                return result

            if action == ValidationAction.HALT:
                result.duration_seconds = time.time() - start_time
                return result

            # ValidationAction.RETRY
            if not context.can_retry():
                result.add_error(
                    "",
                    "Maximum retries exceeded",
                    code="max_retries_exceeded",
                )
                result.duration_seconds = time.time() - start_time
                return result
            continue

    def _validate(self, context: ValidationContext) -> ValidationResult:
        """Internal validation method.

        Args:
            context: Validation context

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        with self._lock:
            validators = list(self._validators)  # Copy for thread safety

        for validator in validators:
            if self._enable_logging:
                logger.debug(f"Running validator: {validator.__class__.__name__}")

            validator_result = validator.validate(context.data, context)

            # Merge results
            result.merge(validator_result)

            # Halt on first error if configured
            if not result.is_valid and self._halt_on_error and not self._collect_all_errors:
                break

        result.retry_count = context.retry_count
        return result

    def validate_stream(
        self,
        data_stream: Iterator[dict[str, Any]],
        state: Optional[dict[str, Any]] = None,
    ) -> Iterator[ValidationResult]:
        """Validate a stream of data items.

        Args:
            data_stream: Iterator of data items
            state: Optional workflow state

        Yields:
            ValidationResult for each item
        """
        for data in data_stream:
            yield self.validate(data, state)

    def validate_batch(
        self,
        data_batch: list[dict[str, Any]],
        state: Optional[dict[str, Any]] = None,
    ) -> list[ValidationResult]:
        """Validate a batch of data items.

        Args:
            data_batch: List of data items
            state: Optional workflow state

        Returns:
            List of ValidationResult
        """
        return [self.validate(data, state) for data in data_batch]

    def aggregate_results(
        self,
        results: list[ValidationResult],
    ) -> ValidationResult:
        """Aggregate multiple validation results.

        Args:
            results: List of validation results

        Returns:
            Aggregated ValidationResult
        """
        if not results:
            return ValidationResult()

        aggregated = ValidationResult()
        total_duration = 0.0

        for result in results:
            aggregated.merge(result)
            total_duration += result.duration_seconds

        aggregated.duration_seconds = total_duration
        aggregated.data = {
            "total_results": len(results),
            "valid_count": sum(1 for r in results if r.is_valid),
            "invalid_count": sum(1 for r in results if not r.is_valid),
        }

        return aggregated


# =============================================================================
# Convenience Functions
# =============================================================================


def create_validation_pipeline(
    validators: Optional[list[ValidatorProtocol]] = None,
    **kwargs: Any,
) -> ValidationPipeline:
    """Create a validation pipeline with the given configuration.

    Args:
        validators: List of validators
        **kwargs: Additional configuration for ValidationPipeline

    Returns:
        Configured ValidationPipeline

    Example:
        pipeline = create_validation_pipeline(
            validators=[
                ThresholdValidator("score", min_value=0, max_value=100),
            ],
            halt_on_error=True,
            max_retries=3,
        )
    """
    return ValidationPipeline(
        validators=validators,
        **kwargs,
    )


def validate_and_get_errors(
    data: dict[str, Any],
    validators: list[ValidatorProtocol],
) -> list[str]:
    """Validate data and return error messages.

    Args:
        data: Data to validate
        validators: List of validators

    Returns:
        List of error messages (empty if valid)
    """
    pipeline = ValidationPipeline(validators=validators)
    result = pipeline.validate(data)
    return [str(issue) for issue in result.errors]


def is_valid(
    data: dict[str, Any],
    validators: list[ValidatorProtocol],
) -> bool:
    """Check if data is valid.

    Args:
        data: Data to validate
        validators: List of validators

    Returns:
        True if valid, False otherwise
    """
    pipeline = ValidationPipeline(validators=validators)
    result = pipeline.validate(data)
    return result.is_valid


__all__ = [
    # Enums
    "ValidationStage",
    "ValidationAction",
    "ValidationSeverity",
    # Result
    "ValidationResult",
    "ValidationIssue",
    # Context
    "ValidationContext",
    # Handler Protocol
    "ValidationHandler",
    # Built-in Handlers
    "RetryHandler",
    "HaltHandler",
    "SkipHandler",
    "ConditionalHandler",
    "ChainHandler",
    # Config
    "ValidationConfig",
    # Pipeline
    "ValidationPipeline",
    # Convenience functions
    "create_validation_pipeline",
    "validate_and_get_errors",
    "is_valid",
]
