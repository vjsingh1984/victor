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

"""Unified Budget Manager.

This module provides centralized budget management with consistent multiplier
composition across all budget types.

Design:
- Single source of truth for all budget tracking
- Consistent multiplier composition: effective_max = base × model × mode × productivity
- Separate tracking for exploration vs action operations
- Mode-specific early exit criteria for graceful completion
- Integration with UnifiedTaskTracker and prompt builder

Usage:
    manager = BudgetManager(config=BudgetConfig())
    manager.set_mode_multiplier(2.5)  # PLAN mode

    if manager.consume(BudgetType.EXPLORATION):
        # Performed exploration operation
        pass

    if manager.is_exhausted(BudgetType.EXPLORATION):
        # Force synthesis/completion
        pass

    # Check for mode-based early exit
    if manager.should_early_exit("PLAN", response_text):
        # Mode objectives met, can stop
        pass

Issue Reference: workflow-test-issues-v2.md Issue #6
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable

from victor.agent.protocols import (
    BudgetConfig,
    BudgetStatus,
    BudgetType,
    IBudgetManager,
)
from victor.protocols.mode_aware import ModeAwareMixin

logger = logging.getLogger(__name__)


# =============================================================================
# Mode Completion Criteria (Fix #6)
# =============================================================================


class ModeObjective(Enum):
    """Defines primary objectives for each mode."""

    EXPLORE = "understand_codebase"
    PLAN = "provide_implementation_plan"
    BUILD = "create_or_modify_files"


@dataclass
class ModeCompletionConfig:
    """Configuration for mode-specific completion criteria.

    Attributes:
        min_files_read: Minimum files to read before considering complete
        min_files_written: Minimum files to write (BUILD mode)
        max_iterations: Maximum iterations before forcing completion
        completion_signals: Phrases indicating task completion
        required_sections: Required sections in output (PLAN mode)
    """

    min_files_read: int = 1
    min_files_written: int = 0
    max_iterations: int = 20
    completion_signals: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)


@runtime_checkable
class IModeCompletionCriteria(Protocol):
    """Protocol for mode completion detection."""

    def get_criteria(self, mode: str) -> ModeCompletionConfig:
        """Get completion criteria for a mode."""
        ...

    def check_early_exit(
        self,
        mode: str,
        files_read: int,
        files_written: int,
        iterations: int,
        response_text: str,
    ) -> tuple[bool, str]:
        """Check if mode objectives are met for early exit."""
        ...


class ModeCompletionCriteria:
    """Defines and checks completion criteria per mode.

    Each mode has specific objectives and exit criteria:
    - EXPLORE: Understand codebase, provide explanation
    - PLAN: Analyze requirements, provide implementation plan
    - BUILD: Create/modify files, complete implementation

    Usage:
        criteria = ModeCompletionCriteria()

        # Check if can exit early
        should_exit, reason = criteria.check_early_exit(
            mode="PLAN",
            files_read=3,
            files_written=0,
            iterations=10,
            response_text="Here's the implementation plan..."
        )
    """

    # Default criteria per mode
    CRITERIA = {
        ModeObjective.EXPLORE: ModeCompletionConfig(
            min_files_read=1,
            min_files_written=0,
            max_iterations=15,
            completion_signals=[
                "here's what",
                "the file",
                "this is",
                "here's an overview",
                "this module",
                "the codebase",
                "i found",
                "the structure",
            ],
            required_sections=[],
        ),
        ModeObjective.PLAN: ModeCompletionConfig(
            min_files_read=1,
            min_files_written=0,
            max_iterations=20,
            completion_signals=[
                "implementation plan",
                "steps to",
                "here's how",
                "here's the plan",
                "proposed approach",
                "implementation steps",
                "the plan",
            ],
            required_sections=[
                "step",
                "file",
            ],
        ),
        ModeObjective.BUILD: ModeCompletionConfig(
            min_files_read=0,
            min_files_written=1,
            max_iterations=30,
            completion_signals=[
                "created",
                "implemented",
                "written",
                "has been created",
                "successfully created",
                "file created",
                "implementation complete",
            ],
            required_sections=[],
        ),
    }

    def __init__(self, custom_criteria: Optional[Dict[str, ModeCompletionConfig]] = None):
        """Initialize with optional custom criteria.

        Args:
            custom_criteria: Override default criteria for specific modes
        """
        self._custom_criteria = custom_criteria or {}
        self._iteration_counts: Dict[str, int] = {}

    def get_criteria(self, mode: str) -> ModeCompletionConfig:
        """Get completion criteria for mode.

        Args:
            mode: Mode name (EXPLORE, PLAN, BUILD)

        Returns:
            Completion configuration for the mode
        """
        # Check custom criteria first
        if mode.upper() in self._custom_criteria:
            return self._custom_criteria[mode.upper()]

        # Get from default criteria
        try:
            objective = ModeObjective[mode.upper()]
            return self.CRITERIA.get(objective, ModeCompletionConfig())
        except KeyError:
            logger.warning(f"Unknown mode: {mode}, using default criteria")
            return ModeCompletionConfig()

    def check_early_exit(
        self,
        mode: str,
        files_read: int,
        files_written: int,
        iterations: int,
        response_text: str,
    ) -> tuple[bool, str]:
        """Check if mode objectives are met for early exit.

        Args:
            mode: Current mode (EXPLORE, PLAN, BUILD)
            files_read: Number of files read so far
            files_written: Number of files written so far
            iterations: Current iteration count
            response_text: Agent's response text

        Returns:
            Tuple of (should_exit, reason)
        """
        criteria = self.get_criteria(mode)

        # Track iterations
        self._iteration_counts[mode] = iterations

        # Check maximum iterations exceeded
        if iterations >= criteria.max_iterations:
            logger.info(f"Mode {mode}: max iterations ({criteria.max_iterations}) reached")
            return True, f"Maximum iterations ({criteria.max_iterations}) reached"

        # Check minimum requirements by mode
        mode_upper = mode.upper()

        if mode_upper == "BUILD":
            # BUILD mode requires file(s) to be written
            if files_written < criteria.min_files_written:
                return False, f"Need {criteria.min_files_written - files_written} more file(s) written"
        else:
            # EXPLORE and PLAN require files to be read
            if files_read < criteria.min_files_read:
                return False, f"Need {criteria.min_files_read - files_read} more file(s) read"

        # Check for completion signals in response
        response_lower = response_text.lower()
        signals = criteria.completion_signals

        found_signal = None
        for signal in signals:
            if signal in response_lower:
                found_signal = signal
                break

        if not found_signal:
            return False, "No completion signal detected"

        # For PLAN mode, check required sections
        if mode_upper == "PLAN" and criteria.required_sections:
            missing_sections = []
            for section in criteria.required_sections:
                # Check for section headers or keywords
                if not re.search(rf"\b{section}\b", response_lower):
                    missing_sections.append(section)

            if missing_sections:
                return False, f"Missing required sections: {missing_sections}"

        reason = f"Mode objectives complete: '{found_signal}' signal detected"
        logger.info(f"Mode {mode}: early exit - {reason}")
        return True, reason

    def reset(self, mode: Optional[str] = None) -> None:
        """Reset iteration counts.

        Args:
            mode: Specific mode to reset, or None for all
        """
        if mode is None:
            self._iteration_counts.clear()
        elif mode.upper() in self._iteration_counts:
            del self._iteration_counts[mode.upper()]

    def get_progress(self, mode: str) -> Dict[str, Any]:
        """Get progress towards mode completion.

        Args:
            mode: Mode to check

        Returns:
            Progress information dictionary
        """
        criteria = self.get_criteria(mode)
        iterations = self._iteration_counts.get(mode.upper(), 0)

        return {
            "mode": mode,
            "iterations": iterations,
            "max_iterations": criteria.max_iterations,
            "progress_pct": min(100, (iterations / criteria.max_iterations) * 100),
            "min_files_read": criteria.min_files_read,
            "min_files_written": criteria.min_files_written,
        }


# =============================================================================
# Write Tools Detection
# =============================================================================

# Tools that are considered write/action operations
WRITE_TOOLS: Set[str] = frozenset(
    {
        "write_file",
        "write",
        "edit_files",
        "edit",
        "shell",
        "bash",
        "execute_bash",
        "git_commit",
        "git_push",
        "delete_file",
        "create_directory",
        "mkdir",
    }
)


def is_write_tool(tool_name: str) -> bool:
    """Check if a tool is a write/action operation.

    Args:
        tool_name: Name of the tool

    Returns:
        True if this is a write/modify operation
    """
    return tool_name.lower() in WRITE_TOOLS


# =============================================================================
# Budget State
# =============================================================================


@dataclass
class BudgetState:
    """Internal state for a budget type.

    Attributes:
        current: Current usage count
        base_maximum: Base maximum before multipliers
        last_tool: Last tool that consumed this budget
    """
    current: int = 0
    base_maximum: int = 0
    last_tool: Optional[str] = None


# =============================================================================
# Budget Manager Implementation
# =============================================================================


@dataclass
class BudgetManager(IBudgetManager, ModeAwareMixin):
    """Unified budget management with multiplier composition.

    Centralizes all budget tracking with consistent multiplier application:
    effective_max = base × model_multiplier × mode_multiplier × productivity_multiplier

    Replaces scattered budget tracking in:
    - unified_task_tracker.py: exploration_iterations, action_iterations
    - orchestrator.py: tool_budget, complexity_tool_budget
    - intelligent_prompt_builder.py: recommended_tool_budget

    Attributes:
        config: Budget configuration with base values
        _budgets: Internal state for each budget type
        _model_multiplier: Model-specific multiplier (1.0-1.5)
        _mode_multiplier: Mode-specific multiplier (1.0-3.0)
        _productivity_multiplier: Productivity-based multiplier (0.8-2.0)
        _on_exhausted: Optional callback when budget exhausted
    """

    config: BudgetConfig = field(default_factory=BudgetConfig)
    _budgets: Dict[BudgetType, BudgetState] = field(default_factory=dict)
    _model_multiplier: float = 1.0
    _mode_multiplier: float = 1.0
    _productivity_multiplier: float = 1.0
    _on_exhausted: Optional[Callable[[BudgetType], None]] = None

    def __post_init__(self) -> None:
        """Initialize budget states from config."""
        self._initialize_budgets()

    def _initialize_budgets(self) -> None:
        """Set up initial budget states from config."""
        self._budgets = {
            BudgetType.TOOL_CALLS: BudgetState(
                current=0, base_maximum=self.config.base_tool_calls
            ),
            BudgetType.ITERATIONS: BudgetState(
                current=0, base_maximum=self.config.base_iterations
            ),
            BudgetType.EXPLORATION: BudgetState(
                current=0, base_maximum=self.config.base_exploration
            ),
            BudgetType.ACTION: BudgetState(
                current=0, base_maximum=self.config.base_action
            ),
        }

    def _calculate_effective_max(self, budget_type: BudgetType) -> int:
        """Calculate effective maximum with all multipliers applied.

        Formula: effective_max = base × model × mode × productivity

        Args:
            budget_type: Type of budget

        Returns:
            Effective maximum after multipliers
        """
        state = self._budgets.get(budget_type)
        if state is None:
            return 0

        base = state.base_maximum
        combined = self._model_multiplier * self._mode_multiplier * self._productivity_multiplier
        return max(1, int(base * combined))

    def get_status(self, budget_type: BudgetType) -> BudgetStatus:
        """Get current status of a budget.

        Args:
            budget_type: Type of budget to check

        Returns:
            BudgetStatus with current usage and limits
        """
        state = self._budgets.get(budget_type)
        if state is None:
            return BudgetStatus(
                budget_type=budget_type,
                is_exhausted=True,
            )

        effective_max = self._calculate_effective_max(budget_type)
        current = state.current
        is_exhausted = current >= effective_max

        return BudgetStatus(
            budget_type=budget_type,
            current=current,
            base_maximum=state.base_maximum,
            effective_maximum=effective_max,
            is_exhausted=is_exhausted,
            model_multiplier=self._model_multiplier,
            mode_multiplier=self._mode_multiplier,
            productivity_multiplier=self._productivity_multiplier,
        )

    def consume(self, budget_type: BudgetType, amount: int = 1) -> bool:
        """Consume budget for an operation.

        Args:
            budget_type: Type of budget to consume
            amount: Amount to consume (default 1)

        Returns:
            True if budget was available, False if exhausted
        """
        state = self._budgets.get(budget_type)
        if state is None:
            logger.warning(f"Unknown budget type: {budget_type}")
            return False

        effective_max = self._calculate_effective_max(budget_type)
        was_available = state.current < effective_max

        state.current += amount

        is_now_exhausted = state.current >= effective_max
        if is_now_exhausted and was_available:
            logger.info(
                f"Budget {budget_type.value} exhausted: "
                f"{state.current}/{effective_max} (base={state.base_maximum})"
            )
            if self._on_exhausted:
                self._on_exhausted(budget_type)

        return was_available

    def is_exhausted(self, budget_type: BudgetType) -> bool:
        """Check if a budget is exhausted.

        Args:
            budget_type: Type of budget to check

        Returns:
            True if budget is fully consumed
        """
        status = self.get_status(budget_type)
        return status.is_exhausted

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set the model-specific multiplier.

        Model multipliers vary by model capability:
        - GPT-4o: 1.0 (baseline)
        - Claude Opus: 1.2 (more capable, fewer retries)
        - DeepSeek: 1.3 (needs more exploration)
        - Ollama local: 1.5 (needs more attempts)

        Args:
            multiplier: Model multiplier value
        """
        old_multiplier = self._model_multiplier
        self._model_multiplier = max(0.5, min(3.0, multiplier))

        if old_multiplier != self._model_multiplier:
            logger.debug(
                f"BudgetManager: model_multiplier={self._model_multiplier}"
            )

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set the mode-specific multiplier.

        Mode multipliers:
        - BUILD: 2.0 (reading before writing)
        - PLAN: 2.5 (thorough analysis)
        - EXPLORE: 3.0 (exploration is primary goal)

        Args:
            multiplier: Mode multiplier value
        """
        old_multiplier = self._mode_multiplier
        self._mode_multiplier = max(0.5, min(5.0, multiplier))

        if old_multiplier != self._mode_multiplier:
            logger.debug(
                f"BudgetManager: mode_multiplier={self._mode_multiplier}"
            )

    def set_productivity_multiplier(self, multiplier: float) -> None:
        """Set the productivity multiplier.

        Productivity multipliers (from RL learning):
        - High productivity session: 0.8 (less budget needed)
        - Normal: 1.0
        - Low productivity: 1.2-2.0 (more attempts needed)

        Args:
            multiplier: Productivity multiplier value
        """
        old_multiplier = self._productivity_multiplier
        self._productivity_multiplier = max(0.5, min(3.0, multiplier))

        if old_multiplier != self._productivity_multiplier:
            logger.debug(
                f"BudgetManager: productivity_multiplier={self._productivity_multiplier}"
            )

    def reset(self, budget_type: Optional[BudgetType] = None) -> None:
        """Reset budget(s) to initial state.

        Args:
            budget_type: Specific budget to reset, or None for all
        """
        if budget_type is None:
            self._initialize_budgets()
            logger.debug("BudgetManager: all budgets reset")
        else:
            state = self._budgets.get(budget_type)
            if state:
                state.current = 0
                state.last_tool = None
                logger.debug(f"BudgetManager: {budget_type.value} budget reset")

    def get_prompt_budget_info(self) -> Dict[str, Any]:
        """Get budget information for system prompts.

        Returns:
            Dictionary with budget info for prompt building
        """
        tool_status = self.get_status(BudgetType.TOOL_CALLS)
        exploration_status = self.get_status(BudgetType.EXPLORATION)
        action_status = self.get_status(BudgetType.ACTION)

        return {
            "tool_budget": tool_status.effective_maximum,
            "tool_calls_used": tool_status.current,
            "tool_calls_remaining": tool_status.remaining,
            "exploration_budget": exploration_status.effective_maximum,
            "exploration_used": exploration_status.current,
            "exploration_remaining": exploration_status.remaining,
            "action_budget": action_status.effective_maximum,
            "action_used": action_status.current,
            "model_multiplier": self._model_multiplier,
            "mode_multiplier": self._mode_multiplier,
            "productivity_multiplier": self._productivity_multiplier,
        }

    def record_tool_call(
        self, tool_name: str, is_write_operation: bool = False
    ) -> bool:
        """Record a tool call and consume appropriate budget.

        Automatically routes to EXPLORATION or ACTION budget based
        on whether the operation is a write operation.

        Args:
            tool_name: Name of the tool called
            is_write_operation: Whether this is a write/modify operation
                              (if not specified, auto-detected from tool name)

        Returns:
            True if budget was available
        """
        # Auto-detect write operation if not specified
        if not is_write_operation:
            is_write_operation = is_write_tool(tool_name)

        # Always consume from tool calls budget
        tool_available = self.consume(BudgetType.TOOL_CALLS)

        # Track in appropriate category
        if is_write_operation:
            category_available = self.consume(BudgetType.ACTION)
            budget_type = BudgetType.ACTION
        else:
            category_available = self.consume(BudgetType.EXPLORATION)
            budget_type = BudgetType.EXPLORATION

        # Update last tool
        state = self._budgets.get(budget_type)
        if state:
            state.last_tool = tool_name

        logger.debug(
            f"BudgetManager: tool_call={tool_name}, "
            f"is_write={is_write_operation}, "
            f"tool_available={tool_available}, "
            f"category_available={category_available}"
        )

        return tool_available and category_available

    def set_base_budget(self, budget_type: BudgetType, base: int) -> None:
        """Set the base budget for a type.

        Useful for task-specific budget adjustments.

        Args:
            budget_type: Type of budget to adjust
            base: New base value
        """
        state = self._budgets.get(budget_type)
        if state:
            state.base_maximum = max(1, base)
            logger.debug(
                f"BudgetManager: {budget_type.value} base set to {base}"
            )

    def set_on_exhausted(self, callback: Callable[[BudgetType], None]) -> None:
        """Set callback for when a budget is exhausted.

        Args:
            callback: Function called with budget type when exhausted
        """
        self._on_exhausted = callback

    def update_from_mode(self) -> None:
        """Update multiplier from current mode controller.

        Convenience method that reads mode from ModeAwareMixin.
        """
        multiplier = self.exploration_multiplier
        self.set_mode_multiplier(multiplier)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about all budgets.

        Returns:
            Dictionary with detailed budget state
        """
        diagnostics: Dict[str, Any] = {
            "multipliers": {
                "model": self._model_multiplier,
                "mode": self._mode_multiplier,
                "productivity": self._productivity_multiplier,
                "combined": (
                    self._model_multiplier
                    * self._mode_multiplier
                    * self._productivity_multiplier
                ),
            },
            "budgets": {},
        }

        for budget_type in BudgetType:
            status = self.get_status(budget_type)
            state = self._budgets.get(budget_type)
            diagnostics["budgets"][budget_type.value] = {
                "current": status.current,
                "base_maximum": status.base_maximum,
                "effective_maximum": status.effective_maximum,
                "remaining": status.remaining,
                "utilization": f"{status.utilization:.1%}",
                "is_exhausted": status.is_exhausted,
                "last_tool": state.last_tool if state else None,
            }

        return diagnostics


# =============================================================================
# Extended Budget Manager with Mode Completion
# =============================================================================


@dataclass
class ExtendedBudgetManager(BudgetManager):
    """BudgetManager with integrated mode completion criteria.

    Extends BudgetManager to include mode-specific early exit detection,
    providing a unified interface for both budget tracking and mode
    completion checking.

    Attributes:
        _mode_criteria: Mode completion criteria checker
        _files_read: Count of files read in current session
        _files_written: Count of files written in current session
    """

    _mode_criteria: Optional[ModeCompletionCriteria] = None
    _files_read: int = 0
    _files_written: int = 0
    _current_mode: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize with mode criteria."""
        super().__post_init__()
        if self._mode_criteria is None:
            self._mode_criteria = ModeCompletionCriteria()

    def set_mode(self, mode: str) -> None:
        """Set current operating mode.

        Args:
            mode: Mode name (EXPLORE, PLAN, BUILD)
        """
        self._current_mode = mode.upper()
        logger.debug(f"ExtendedBudgetManager: mode set to {self._current_mode}")

    def record_file_read(self) -> None:
        """Record a file read operation."""
        self._files_read += 1

    def record_file_write(self) -> None:
        """Record a file write operation."""
        self._files_written += 1

    def record_tool_call(
        self, tool_name: str, is_write_operation: bool = False
    ) -> bool:
        """Record tool call with file tracking.

        Extends parent to track file read/write operations.

        Args:
            tool_name: Name of the tool called
            is_write_operation: Whether this is a write operation

        Returns:
            True if budget was available
        """
        # Track file operations
        tool_lower = tool_name.lower()
        if tool_lower in {"read", "read_file"}:
            self.record_file_read()
        elif tool_lower in WRITE_TOOLS:
            self.record_file_write()

        return super().record_tool_call(tool_name, is_write_operation)

    def should_early_exit(
        self,
        response_text: str,
        mode: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Check if mode objectives are met for early exit.

        Args:
            response_text: Agent's response text to analyze
            mode: Mode to check (uses current mode if not specified)

        Returns:
            Tuple of (should_exit, reason)
        """
        check_mode = mode or self._current_mode
        if not check_mode:
            return False, "No mode set"

        if self._mode_criteria is None:
            return False, "No mode criteria configured"

        # Get current iteration from exploration budget
        exploration_status = self.get_status(BudgetType.EXPLORATION)
        iterations = exploration_status.current

        return self._mode_criteria.check_early_exit(
            mode=check_mode,
            files_read=self._files_read,
            files_written=self._files_written,
            iterations=iterations,
            response_text=response_text,
        )

    def get_mode_progress(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get progress towards mode completion.

        Args:
            mode: Mode to check (uses current mode if not specified)

        Returns:
            Progress information dictionary
        """
        check_mode = mode or self._current_mode
        if not check_mode or self._mode_criteria is None:
            return {}

        progress = self._mode_criteria.get_progress(check_mode)
        progress["files_read"] = self._files_read
        progress["files_written"] = self._files_written
        return progress

    def reset(self, budget_type: Optional[BudgetType] = None) -> None:
        """Reset budgets and file counters.

        Args:
            budget_type: Specific budget to reset, or None for all
        """
        super().reset(budget_type)
        if budget_type is None:
            self._files_read = 0
            self._files_written = 0
            if self._mode_criteria:
                self._mode_criteria.reset()


# =============================================================================
# Factory Functions
# =============================================================================


def create_budget_manager(
    config: Optional[BudgetConfig] = None,
    model_multiplier: float = 1.0,
    mode_multiplier: float = 1.0,
    productivity_multiplier: float = 1.0,
) -> BudgetManager:
    """Create a configured BudgetManager instance.

    Factory function for DI registration.

    Args:
        config: Budget configuration with base values
        model_multiplier: Initial model multiplier
        mode_multiplier: Initial mode multiplier
        productivity_multiplier: Initial productivity multiplier

    Returns:
        Configured BudgetManager instance
    """
    manager = BudgetManager(config=config or BudgetConfig())
    manager.set_model_multiplier(model_multiplier)
    manager.set_mode_multiplier(mode_multiplier)
    manager.set_productivity_multiplier(productivity_multiplier)
    return manager


def create_extended_budget_manager(
    config: Optional[BudgetConfig] = None,
    mode: Optional[str] = None,
    model_multiplier: float = 1.0,
    mode_multiplier: float = 1.0,
) -> ExtendedBudgetManager:
    """Create a configured ExtendedBudgetManager with mode completion support.

    Factory function for DI registration.

    Args:
        config: Budget configuration with base values
        mode: Initial operating mode (EXPLORE, PLAN, BUILD)
        model_multiplier: Initial model multiplier
        mode_multiplier: Initial mode multiplier

    Returns:
        Configured ExtendedBudgetManager instance
    """
    manager = ExtendedBudgetManager(config=config or BudgetConfig())
    manager.set_model_multiplier(model_multiplier)
    manager.set_mode_multiplier(mode_multiplier)
    if mode:
        manager.set_mode(mode)
    return manager


def create_mode_completion_criteria(
    custom_criteria: Optional[Dict[str, ModeCompletionConfig]] = None,
) -> ModeCompletionCriteria:
    """Create a ModeCompletionCriteria instance.

    Factory function for DI registration.

    Args:
        custom_criteria: Override default criteria for specific modes

    Returns:
        Configured ModeCompletionCriteria instance
    """
    return ModeCompletionCriteria(custom_criteria=custom_criteria)
