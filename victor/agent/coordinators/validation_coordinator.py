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

"""Validation Coordinator - Coordinates validation logic across the orchestrator.

This module extracts validation-related logic from AgentOrchestrator into a
focused coordinator following SOLID principles (Single Responsibility).

Responsibilities:
- Response validation (quality scoring, grounding verification)
- Tool call validation (name format, enabled status)
- Cancellation checking
- Context overflow checking
- Input parameter validation

Design Patterns:
- Single Responsibility: Focuses only on validation logic
- Facade Pattern: Provides unified interface for validation operations
- Delegation Pattern: Delegates to specialized validators when needed

Usage:
    coordinator = ValidationCoordinator(
        intelligent_integration=integration,
        context_manager=context_manager,
        response_coordinator=response_coordinator,
    )

    # Validate response
    result = await coordinator.validate_intelligent_response(
        response="The code analysis shows...",
        query="Analyze code",
        tool_calls=3,
        task_type="analysis",
    )

    # Check cancellation
    if coordinator.is_cancelled():
        # Handle cancellation

    # Check context overflow
    if coordinator.check_context_overflow(max_chars=200000):
        # Handle overflow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    pass

from victor.agent.coordinators.base_config import BaseCoordinatorConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Result Types
# ============================================================================


@dataclass
class ValidationResult:
    """Base class for validation results."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class IntelligentValidationResult(ValidationResult):
    """Result of intelligent response validation.

    Attributes:
        is_valid: Whether response meets quality threshold
        quality_score: Response quality score (0.0-1.0)
        grounding_score: Grounding verification score (0.0-1.0)
        is_grounded: Whether response is properly grounded
        grounding_issues: List of detected grounding issues
        should_finalize: Whether to force finalize response
        should_retry: Whether to retry the response
        finalize_reason: Reason for finalization
        grounding_feedback: Feedback for grounding correction
    """

    quality_score: float = 0.5
    grounding_score: float = 0.5
    is_grounded: bool = True
    grounding_issues: list[str] = field(default_factory=list)
    should_finalize: bool = False
    should_retry: bool = False
    finalize_reason: str = ""
    grounding_feedback: str = ""

    def meets_quality_threshold(self, threshold: float = 0.5) -> bool:
        """Check if quality score meets threshold."""
        return self.quality_score >= threshold

    def meets_grounding_threshold(self, threshold: float = 0.7) -> bool:
        """Check if grounding score meets threshold."""
        return self.grounding_score >= threshold


@dataclass
class ToolCallValidationResult(ValidationResult):
    """Result of tool call validation.

    Attributes:
        tool_calls: Validated tool calls (filtered)
        filtered_count: Number of tool calls filtered out
        remaining_content: Content after tool call extraction
    """

    tool_calls: Optional[list[dict[str, Any]]] = None
    filtered_count: int = 0
    remaining_content: str = ""


@dataclass
class ContextValidationResult(ValidationResult):
    """Result of context validation.

    Attributes:
        is_overflow: Whether context is at risk of overflow
        current_size: Current context size in characters
        max_size: Maximum allowed context size
        utilization_percent: Context utilization as percentage
    """

    is_overflow: bool = False
    current_size: int = 0
    max_size: int = 0
    utilization_percent: float = 0.0


# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class IIntelligentValidationProtocol(Protocol):
    """Protocol for intelligent response validation."""

    async def validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[dict[str, Any]]:
        """Validate response using intelligent pipeline.

        Args:
            response: The model's response content
            query: Original user query
            tool_calls: Number of tool calls made so far
            task_type: Task type for context

        Returns:
            Dictionary with quality/grounding scores, or None if pipeline disabled
        """
        ...


@runtime_checkable
class IContextValidationProtocol(Protocol):
    """Protocol for context validation."""

    def check_context_overflow(self, max_context_chars: int = 200000) -> bool:
        """Check if context is at risk of overflow.

        Args:
            max_context_chars: Maximum allowed context size in chars

        Returns:
            True if context is dangerously large
        """
        ...

    def get_context_metrics(self) -> Any:
        """Get detailed context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        ...

    def get_max_context_chars(self) -> int:
        """Get maximum context size in characters.

        Returns:
            Maximum context size
        """
        ...


@runtime_checkable
class IResponseValidationProtocol(Protocol):
    """Protocol for response validation."""

    def is_valid_tool_name(self, name: str) -> bool:
        """Validate a tool name.

        Args:
            name: Tool name to validate

        Returns:
            True if tool name is valid
        """
        ...

    def parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[list[dict[str, Any]]],
        content: str,
        enabled_tools: Optional[set[str]] = None,
    ) -> Any:
        """Parse and validate tool calls from response.

        Args:
            tool_calls: Native tool calls from provider
            content: Response content for fallback parsing
            enabled_tools: Set of enabled tool names for validation

        Returns:
            ToolCallValidationResult with validated calls and remaining content
        """
        ...


@runtime_checkable
class ICancellationProtocol(Protocol):
    """Protocol for cancellation checking."""

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancelled, False otherwise
        """
        ...

    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested (alternative name).

        Returns:
            True if cancelled, False otherwise
        """
        ...


# ============================================================================
# Validation Coordinator
# ============================================================================


@dataclass
class ValidationCoordinatorConfig(BaseCoordinatorConfig):
    """Configuration for ValidationCoordinator.

    Inherits common configuration from BaseCoordinatorConfig:
        enabled: Whether the coordinator is enabled
        timeout: Default timeout in seconds for operations
        max_retries: Maximum number of retry attempts for failed operations
        retry_enabled: Whether retry logic is enabled
        log_level: Logging level for coordinator messages
        enable_metrics: Whether to collect metrics

    Attributes:
        enable_intelligent_validation: Enable intelligent pipeline validation
        enable_tool_call_validation: Enable tool call validation
        enable_context_validation: Enable context overflow checking
        min_response_length: Minimum response length for validation
        quality_threshold: Default quality threshold
        grounding_threshold: Default grounding threshold
        max_garbage_chunks: Max consecutive garbage chunks before stopping
    """

    enable_intelligent_validation: bool = True
    enable_tool_call_validation: bool = True
    enable_context_validation: bool = True
    min_response_length: int = 50
    quality_threshold: float = 0.5
    grounding_threshold: float = 0.7
    max_garbage_chunks: int = 3


class ValidationCoordinator:
    """Coordinator for validation logic across the orchestrator.

    This coordinator consolidates validation-related logic that was
    previously scattered across the orchestrator:

    1. Intelligent Response Validation:
       - Quality scoring (coherence, completeness, relevance)
       - Grounding verification (hallucination detection)
       - Delegates to OrchestratorIntegration

    2. Tool Call Validation:
       - Tool name format validation
       - Tool enabled status checking
       - Delegates to ResponseCoordinator

    3. Context Validation:
       - Context overflow checking
       - Context metrics reporting
       - Delegates to ContextCoordinator

    4. Cancellation Checking:
       - Check if cancellation was requested
       - Delegates to cancel event

    Example:
        coordinator = ValidationCoordinator(
            intelligent_integration=integration,
            context_manager=context_manager,
            response_coordinator=response_coordinator,
        )

        # Validate response
        result = await coordinator.validate_intelligent_response(
            response="The code shows...",
            query="Analyze code",
            tool_calls=3,
            task_type="analysis",
        )

        if not.result.is_valid:
            # Handle validation failure
            pass
    """

    def __init__(
        self,
        intelligent_integration: Optional[IIntelligentValidationProtocol] = None,
        context_manager: Optional[IContextValidationProtocol] = None,
        response_coordinator: Optional[IResponseValidationProtocol] = None,
        cancel_event: Optional[Any] = None,
        metrics_coordinator: Optional[ICancellationProtocol] = None,
        config: Optional[ValidationCoordinatorConfig] = None,
    ):
        """Initialize the ValidationCoordinator.

        Args:
            intelligent_integration: Optional intelligent pipeline integration
            context_manager: Optional context manager for overflow checking
            response_coordinator: Optional response coordinator for tool validation
            cancel_event: Optional event for cancellation checking (legacy)
            metrics_coordinator: Optional metrics coordinator for cancellation checking
            config: Optional configuration settings
        """
        self._intelligent_integration = intelligent_integration
        self._context_manager = context_manager
        self._response_coordinator = response_coordinator
        self._cancel_event = cancel_event
        self._metrics_coordinator = metrics_coordinator
        self._config = config or ValidationCoordinatorConfig()

        logger.debug(
            f"ValidationCoordinator initialized with "
            f"intelligent={'enabled' if self._intelligent_integration else 'disabled'}, "
            f"context={'enabled' if self._context_manager else 'disabled'}, "
            f"response={'enabled' if self._response_coordinator else 'disabled'}, "
            f"metrics={'enabled' if self._metrics_coordinator else 'disabled'}"
        )

    # ========================================================================
    # Intelligent Response Validation
    # ========================================================================

    async def validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[IntelligentValidationResult]:
        """Validate response using intelligent pipeline integration.

        This method:
        1. Skips validation for very short responses (< 50 chars)
        2. Delegates to OrchestratorIntegration if available
        3. Returns None if intelligent validation disabled (for backward compatibility)

        Args:
            response: The model's response content
            query: Original user query
            tool_calls: Number of tool calls made so far
            task_type: Task type for context

        Returns:
            IntelligentValidationResult with quality and grounding scores, or None
        """
        # Skip validation for very short responses (backward compatibility)
        if not response or len(response.strip()) < self._config.min_response_length:
            return None

        # Return None if intelligent validation disabled or not available
        if not self._config.enable_intelligent_validation or not self._intelligent_integration:
            return None

        result = IntelligentValidationResult(is_valid=True)

        try:
            validation_dict = await self._intelligent_integration.validate_intelligent_response(
                response=response,
                query=query,
                tool_calls=tool_calls,
                task_type=task_type,
            )

            # Return None if integration returns None (error case)
            if validation_dict is None:
                return None

            result.quality_score = validation_dict.get("quality_score", 0.5)
            result.grounding_score = validation_dict.get("grounding_score", 0.5)
            result.is_grounded = validation_dict.get("is_grounded", True)
            result.grounding_issues = validation_dict.get("grounding_issues", [])
            result.should_finalize = validation_dict.get("should_finalize", False)
            result.should_retry = validation_dict.get("should_retry", False)
            result.finalize_reason = validation_dict.get("finalize_reason", "")
            result.grounding_feedback = validation_dict.get("grounding_feedback", "")

            # Update validity based on scores
            result.is_valid = result.meets_quality_threshold(
                self._config.quality_threshold
            ) and result.meets_grounding_threshold(self._config.grounding_threshold)

            if not result.is_valid:
                result.add_error(
                    f"Response below threshold: quality={result.quality_score:.2f}, "
                    f"grounding={result.grounding_score:.2f}"
                )

            logger.debug(
                "Intelligent validation: quality=%.2f, grounded=%s, valid=%s",
                result.quality_score,
                result.is_grounded,
                result.is_valid,
            )
        except Exception as e:
            logger.warning("Intelligent validation failed: %s", e)
            return None

        return result

    # ========================================================================
    # Tool Call Validation
    # ========================================================================

    def validate_tool_call_structure(
        self,
        tool_call: Any,
    ) -> ValidationResult:
        """Validate the structure of a single tool call.

        Args:
            tool_call: The tool call to validate

        Returns:
            ValidationResult with any errors found
        """
        result = ValidationResult(is_valid=True)

        # Check if tool call is a dict
        if not isinstance(tool_call, dict):
            result.add_error(f"Tool call is not a dict: {type(tool_call).__name__}")
            return result

        # Check for name field
        tool_name = tool_call.get("name")
        if not tool_name:
            result.add_error("Tool call missing 'name' field")

        return result

    def validate_tool_name(self, tool_name: str) -> ValidationResult:
        """Validate a tool name format.

        Args:
            tool_name: Tool name to validate

        Returns:
            ValidationResult with any errors found
        """
        result = ValidationResult(is_valid=True)

        if not tool_name:
            result.add_error("Tool name is empty")
            return result

        # Check if tool name format is valid
        if self._response_coordinator:
            if not self._response_coordinator.is_valid_tool_name(tool_name):
                result.add_error(f"Invalid tool name format: '{tool_name}'")
        else:
            # Basic validation: reject obviously malformed names
            if any(char in tool_name for char in ['"', "'", "\n", "\t", "\\"]):
                result.add_error(f"Tool name contains invalid characters: '{tool_name}'")

        return result

    def validate_and_filter_tool_calls(
        self,
        tool_calls: Optional[list[dict[str, Any]]],
        content: str,
        enabled_tools: Optional[set[str]] = None,
    ) -> ToolCallValidationResult:
        """Parse and validate tool calls from response.

        Args:
            tool_calls: Native tool calls from provider
            content: Response content for fallback parsing
            enabled_tools: Set of enabled tool names for validation

        Returns:
            ToolCallValidationResult with validated calls and remaining content
        """
        result = ToolCallValidationResult(is_valid=True, remaining_content=content)

        if not tool_calls:
            return result

        # Validate tool call structure
        validated_calls = []
        for tc in tool_calls:
            structure_result = self.validate_tool_call_structure(tc)
            if structure_result.is_valid:
                tool_name = tc.get("name", "")
                name_result = self.validate_tool_name(tool_name)
                if name_result.is_valid:
                    validated_calls.append(tc)
                else:
                    result.add_error(
                        name_result.errors[0] if name_result.errors else "Invalid tool name"
                    )
                    result.filtered_count += 1
            else:
                result.errors.extend(structure_result.errors)
                result.filtered_count += 1

        result.tool_calls = validated_calls if validated_calls else None

        # Delegate to response coordinator for full parsing if available
        if self._response_coordinator and self._config.enable_tool_call_validation:
            try:
                parse_result = self._response_coordinator.parse_and_validate_tool_calls(
                    tool_calls, content, enabled_tools
                )
                result.tool_calls = parse_result.tool_calls
                result.remaining_content = parse_result.remaining_content
                result.filtered_count += parse_result.filtered_count
            except Exception as e:
                logger.warning(f"Response coordinator validation failed: {e}")
                result.add_warning(f"Response coordinator validation failed: {e}")

        return result

    # ========================================================================
    # Context Validation
    # ========================================================================

    def check_context_overflow(self, max_context_chars: int = 200000) -> ContextValidationResult:
        """Check if context is at risk of overflow.

        Args:
            max_context_chars: Maximum allowed context size in chars

        Returns:
            ContextValidationResult with overflow status
        """
        result = ContextValidationResult(is_valid=True, max_size=max_context_chars)

        if not self._context_manager or not self._config.enable_context_validation:
            return result

        try:
            is_overflow = self._context_manager.check_context_overflow(max_context_chars)
            metrics = self._context_manager.get_context_metrics()

            result.is_overflow = is_overflow
            result.current_size = getattr(metrics, "total_chars", 0)
            result.utilization_percent = (
                (result.current_size / max_context_chars * 100) if max_context_chars > 0 else 0
            )

            if is_overflow:
                result.add_warning(
                    f"Context overflow risk: {result.current_size}/{max_context_chars} chars "
                    f"({result.utilization_percent:.1f}%)"
                )
        except Exception as e:
            logger.warning(f"Context overflow check failed: {e}")
            result.add_warning(f"Context overflow check failed: {e}")

        return result

    def get_max_context_chars(self) -> int:
        """Get maximum context size in characters.

        Returns:
            Maximum context size, or default if not configured
        """
        if self._context_manager:
            try:
                return self._context_manager.get_max_context_chars()
            except Exception:
                pass
        return 200000  # Default

    # ========================================================================
    # Cancellation Checking
    # ========================================================================

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancelled, False otherwise
        """
        # First try metrics coordinator (preferred method)
        if self._metrics_coordinator:
            try:
                # Try both methods for compatibility
                if hasattr(self._metrics_coordinator, "is_cancellation_requested"):
                    return self._metrics_coordinator.is_cancellation_requested()
                elif hasattr(self._metrics_coordinator, "is_cancelled"):
                    return self._metrics_coordinator.is_cancelled()
            except Exception:
                pass

        # Fall back to cancel event
        if self._cancel_event:
            try:
                is_set: bool = self._cancel_event.is_set()
                return is_set
            except Exception:
                pass

        return False

    # ========================================================================
    # Input Parameter Validation
    # ========================================================================

    def validate_query(self, query: str) -> ValidationResult:
        """Validate user query input.

        Args:
            query: The user query to validate

        Returns:
            ValidationResult with any errors found
        """
        result = ValidationResult(is_valid=True)

        if not query:
            result.add_error("Query is empty")
            return result

        if not isinstance(query, str):
            result.add_error(f"Query must be string, got {type(query).__name__}")  # type: ignore[unreachable]
            return result

        # Check for excessively long queries
        if len(query) > 100000:
            result.add_warning(f"Query is very long ({len(query)} chars)")

        return result

    def validate_task_type(self, task_type: str) -> ValidationResult:
        """Validate task type input.

        Args:
            task_type: The task type to validate

        Returns:
            ValidationResult with any errors found
        """
        result = ValidationResult(is_valid=True)

        if not task_type:
            result.add_warning("Task type is empty, using 'general'")
            return result

        # Known task types
        known_types = {
            "general",
            "analysis",
            "edit",
            "debug",
            "test",
            "refactor",
            "documentation",
            "planning",
        }

        if task_type.lower() not in known_types:
            result.add_warning(f"Unknown task type: '{task_type}'")

        return result

    # ========================================================================
    # Composite Validation
    # ========================================================================

    async def validate_request(
        self,
        query: str,
        task_type: str,
        max_context_chars: int = 200000,
    ) -> ValidationResult:
        """Validate all inputs before processing request.

        Args:
            query: The user query
            task_type: The task type
            max_context_chars: Maximum context size for overflow check

        Returns:
            Combined ValidationResult with all validation errors
        """
        result = ValidationResult(is_valid=True)

        # Validate query
        query_result = self.validate_query(query)
        result.errors.extend(query_result.errors)
        result.warnings.extend(query_result.warnings)

        # Validate task type
        task_result = self.validate_task_type(task_type)
        result.warnings.extend(task_result.warnings)

        # Check context overflow
        context_result = self.check_context_overflow(max_context_chars)
        result.warnings.extend(context_result.warnings)

        # Check cancellation
        if self.is_cancelled():
            result.add_error("Operation cancelled")

        # Update validity
        result.is_valid = not result.errors

        return result

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def config(self) -> ValidationCoordinatorConfig:
        """Get the coordinator configuration."""
        return self._config

    @property
    def intelligent_integration(self) -> Optional[IIntelligentValidationProtocol]:
        """Get the intelligent integration."""
        return self._intelligent_integration

    @property
    def context_manager(self) -> Optional[IContextValidationProtocol]:
        """Get the context manager."""
        return self._context_manager

    @property
    def response_coordinator(self) -> Optional[IResponseValidationProtocol]:
        """Get the response coordinator."""
        return self._response_coordinator


__all__ = [
    "ValidationCoordinator",
    "ValidationCoordinatorConfig",
    "ValidationResult",
    "IntelligentValidationResult",
    "ToolCallValidationResult",
    "ContextValidationResult",
    "IIntelligentValidationProtocol",
    "IContextValidationProtocol",
    "IResponseValidationProtocol",
    "ICancellationProtocol",
]
