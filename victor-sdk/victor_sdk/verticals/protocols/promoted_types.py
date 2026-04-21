"""Promoted data types from victor.core for SDK-level protocol definitions.

These types were originally defined in victor.core.vertical_types and
victor.security.safety.types. They are promoted here so that external
verticals can use SDK protocols without importing from victor.core.*

All types in this module are pure data structures with ZERO runtime
dependencies on the victor package.

NOTE: StageValidationResult and ValidationError are duplicated from
victor.core.verticals.protocols.stages. This duplication is INTENTIONAL
and NECESSARY to avoid circular imports between victor.core and victor_sdk.
When updating either definition, please update BOTH to maintain consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# =============================================================================
# Safety Types (promoted from victor.security.safety.types)
# =============================================================================


@dataclass
class SafetyPatternData:
    """A safety pattern for detecting dangerous operations.

    Attributes:
        pattern: Regex pattern to match
        description: Human-readable description
        risk_level: Risk level (use string for flexibility)
        category: Category of the pattern (e.g., "git", "filesystem")
    """

    pattern: str
    description: str
    risk_level: str = "HIGH"  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    category: str = "general"


# =============================================================================
# Middleware Types (promoted from victor.core.vertical_types)
# =============================================================================


class MiddlewarePriority(Enum):
    """Priority levels for middleware execution order.

    Middleware executes in priority order - lower values execute first
    in before_tool_call, higher values execute first in after_tool_call.

    Levels:
        CRITICAL (0): Security validation, permission checks
        HIGH (25): Core functionality, format validation
        NORMAL (50): Standard processing, transformations
        LOW (75): Logging, metrics collection
        DEFERRED (100): Cleanup, finalization tasks
    """

    CRITICAL = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    DEFERRED = 100


@dataclass
class MiddlewareResult:
    """Result from middleware processing.

    Attributes:
        proceed: Whether to proceed with the operation (False blocks execution)
        modified_arguments: Modified arguments to pass downstream (if any)
        error_message: Error message if proceed is False
        metadata: Additional metadata for downstream processing
    """

    proceed: bool = True
    modified_arguments: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Task Type Hints (promoted from victor.core.vertical_types)
# =============================================================================


@dataclass
class TaskTypeHintData:
    """Hint for a specific task type.

    Attributes:
        task_type: Task type identifier (e.g., "edit", "search")
        hint: Prompt hint text to include in system prompt
        tool_budget: Recommended tool budget for this task type
        priority_tools: Tools to prioritize for this task
        token_budget: Token budget for responses (optimization hint)
        context_budget: Context window budget for this task
        skip_planning: Skip planning phase for this task type
        skip_evaluation: Skip evaluation phase for this task type
        temperature_override: LLM temperature for this task type; None = provider default
    """

    task_type: str
    hint: str
    tool_budget: Optional[int] = None
    priority_tools: List[str] = field(default_factory=list)
    token_budget: Optional[int] = None
    context_budget: Optional[int] = None
    skip_planning: bool = False
    skip_evaluation: bool = False
    temperature_override: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the hint to the legacy serializable mapping."""

        return {
            "task_type": self.task_type,
            "hint": self.hint,
            "tool_budget": self.tool_budget,
            "priority_tools": self.priority_tools.copy(),
            "token_budget": self.token_budget,
            "context_budget": self.context_budget,
            "skip_planning": self.skip_planning,
            "skip_evaluation": self.skip_evaluation,
            "temperature_override": self.temperature_override,
        }


# =============================================================================
# Mode Config (promoted from victor.core.verticals.protocols.mode_provider)
# =============================================================================


@dataclass
class ModeConfig:
    """Configuration for an operational mode.

    Attributes:
        name: Mode name (e.g., "fast", "thorough")
        tool_budget: Tool call budget
        max_iterations: Maximum iterations
        temperature: Temperature setting
        description: Human-readable description
    """

    name: str
    tool_budget: int
    max_iterations: int
    temperature: float = 0.7
    description: str = ""


# =============================================================================
# Tool Selection Types (promoted from victor.core.verticals.protocols.tool_provider)
# =============================================================================


@dataclass
class ToolSelectionContext:
    """Context for tool selection decisions.

    Attributes:
        task_type: Detected task type (e.g., "edit", "debug", "refactor")
        user_message: The user's message/query
        conversation_stage: Current conversation stage
        available_tools: Set of currently available tool names
        recent_tools: List of recently used tools (for context)
        metadata: Additional context metadata
    """

    task_type: str
    user_message: str
    conversation_stage: str = "exploration"
    available_tools: Set[str] = field(default_factory=set)
    recent_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSelectionResult:
    """Result of vertical-specific tool selection.

    Attributes:
        priority_tools: Tools to prioritize (ordered by priority)
        excluded_tools: Tools to exclude from selection
        tool_weights: Custom weights for tool scoring (0.0-1.0)
        budget_override: Optional budget override for this selection
        reasoning: Optional explanation for selection decisions
    """

    priority_tools: List[str] = field(default_factory=list)
    excluded_tools: Set[str] = field(default_factory=set)
    tool_weights: Dict[str, float] = field(default_factory=dict)
    budget_override: Optional[int] = None
    reasoning: Optional[str] = None


# =============================================================================
# Stage Contract Types (promoted from victor.core.verticals.protocols.stages)
# =============================================================================

# NOTE: These definitions are duplicated from victor.core.verticals.protocols.stages
# to maintain SDK independence. When updating, update BOTH files.


class ValidationError(Enum):
    """Types of validation errors for stage definitions.

    NOTE: Duplicated from victor.core.verticals.protocols.stages
    to avoid circular imports. Keep both definitions in sync.
    """

    MISSING_REQUIRED_STAGE = "missing_required_stage"
    INVALID_TRANSITION = "invalid_transition"
    INVALID_STAGE_NAME = "invalid_stage_name"
    MISSING_NEXT_STAGES = "missing_next_stages"
    CIRCULAR_TRANSITION = "circular_transition"
    INVALID_KEYWORDS = "invalid_keywords"
    INVALID_DESCRIPTION = "invalid_description"


@dataclass
class StageValidationResult:
    """Result of stage contract validation.

    Attributes:
        is_valid: Whether the stage definition passes validation
        errors: List of validation errors
        warnings: List of validation warnings
        details: Additional validation details

    NOTE: Duplicated from victor.core.verticals.protocols.stages
    to avoid circular imports. Keep both definitions in sync.
    """

    is_valid: bool
    errors: List[tuple[ValidationError, str]]
    warnings: List[str]
    details: Dict[str, Any]

    def add_error(self, error_type: ValidationError, message: str) -> None:
        """Add a validation error."""
        self.errors.append((error_type, message))
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [(e.value, m) for e, m in self.errors],
            "warnings": self.warnings,
            "details": self.details,
        }


# Type aliases for backward compatibility
SafetyPattern = SafetyPatternData
TaskTypeHint = TaskTypeHintData


__all__ = [
    # Safety
    "SafetyPatternData",
    "SafetyPattern",
    # Middleware
    "MiddlewarePriority",
    "MiddlewareResult",
    # Task hints
    "TaskTypeHintData",
    "TaskTypeHint",
    # Mode config
    "ModeConfig",
    # Tool selection
    "ToolSelectionContext",
    "ToolSelectionResult",
    # Stage validation
    "ValidationError",
    "StageValidationResult",
]
