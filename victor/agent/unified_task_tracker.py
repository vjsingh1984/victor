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

"""Unified Task Tracker - Single source of truth for task progress and loop detection.

This module consolidates TaskMilestoneMonitor and LoopDetector into a single
unified system that provides:

- Unified task types (7 fine-grained types)
- Combined milestone and loop detection
- Model-aware exploration settings
- Single configuration source
- Clear, testable interface

Usage:
    tracker = UnifiedTaskTracker()
    tracker.set_task_type(TaskType.EDIT)
    tracker.set_model_capabilities(tool_calling_caps)

    # In the main loop:
    tracker.record_tool_call("read_file", {"path": "test.py"})

    decision = tracker.should_stop()
    if decision.should_stop:
        # Force completion with decision.hint
        break
"""

import hashlib
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import yaml

from victor.tools.tool_names import ToolNames, get_canonical_name
from victor.agent.loop_detector import (
    get_progress_params_for_tool,
    is_progressive_tool,
)

if TYPE_CHECKING:
    from victor.agent.unified_classifier import ClassificationResult

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class TaskType(Enum):
    """Unified task types combining fine-grained and coarse classifications."""

    # Action tasks (modify/create files)
    EDIT = "edit"  # Modify existing files
    CREATE = "create"  # Create new files with context
    CREATE_SIMPLE = "create_simple"  # Create files directly (no context needed)

    # Analysis tasks (read/search/understand)
    SEARCH = "search"  # Find/locate code or files
    ANALYZE = "analyze"  # Count, measure, analyze code
    RESEARCH = "research"  # Web research tasks

    # Other tasks
    DESIGN = "design"  # Conceptual/planning (no tools)
    GENERAL = "general"  # Ambiguous or mixed tasks


class Milestone(Enum):
    """Task milestones for progress tracking."""

    TARGET_IDENTIFIED = "target_identified"
    TARGET_READ = "target_read"
    CHANGE_MADE = "change_made"
    CHANGE_VERIFIED = "change_verified"
    SEARCH_COMPLETE = "search_complete"


class ConversationStage(Enum):
    """Conversation stages for tool availability."""

    INITIAL = "initial"
    READING = "reading"
    EXECUTING = "executing"
    VERIFYING = "verifying"


class StopReason(Enum):
    """Reason for stopping execution."""

    NONE = "none"
    TOOL_BUDGET = "tool_budget"
    LOOP_DETECTED = "loop_detected"
    MAX_ITERATIONS = "max_iterations"
    GOAL_FORCING = "goal_forcing"
    MANUAL_STOP = "manual_stop"


# Tools that indicate research activity
RESEARCH_TOOLS = frozenset({"web_search", "web_fetch", "tavily_search", "search_web", "fetch_url"})


# Default limit for read_file when not specified
DEFAULT_READ_LIMIT = 500

# Canonical tool name for file reading (from centralized registry)
CANONICAL_READ_TOOL = ToolNames.READ


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FileReadRange:
    """Represents a range of lines read from a file."""

    offset: int
    limit: int

    @property
    def end(self) -> int:
        """End line (exclusive) of this range."""
        return self.offset + self.limit

    def overlaps(self, other: "FileReadRange") -> bool:
        """Check if this range overlaps with another."""
        return self.offset < other.end and other.offset < self.end

    def __hash__(self) -> int:
        return hash((self.offset, self.limit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileReadRange):
            return False
        return self.offset == other.offset and self.limit == other.limit


@dataclass
class TaskConfig:
    """Configuration for a specific task type."""

    max_exploration_iterations: int = 8
    force_action_after_target_read: bool = False
    tool_budget: int = 50
    loop_repeat_threshold: int = 3
    needs_tools: bool = True
    required_tools: List[str] = field(default_factory=list)
    stage_tools: Dict[str, List[str]] = field(default_factory=dict)
    force_action_hints: Dict[str, str] = field(default_factory=dict)


@dataclass
class StopDecision:
    """Decision about whether to stop execution."""

    should_stop: bool
    reason: StopReason = StopReason.NONE
    hint: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    is_warning: bool = False


@dataclass
class UnifiedTaskProgress:
    """Single source of truth for task progress."""

    # Task classification
    task_type: TaskType = TaskType.GENERAL

    # Iteration tracking
    iteration_count: int = 0  # Productive iterations (tool calls)
    total_turns: int = 0  # All turns including continuations
    low_output_iterations: int = 0

    # Tool budget
    tool_calls: int = 0
    tool_budget: int = 50

    # Milestone tracking
    milestones: Set[Milestone] = field(default_factory=set)
    target_files: Set[str] = field(default_factory=set)
    target_entities: Set[str] = field(default_factory=set)
    files_read: Set[str] = field(default_factory=set)
    files_modified: Set[str] = field(default_factory=set)

    # Loop detection
    unique_resources: Set[str] = field(default_factory=set)
    file_read_ranges: Dict[str, List[FileReadRange]] = field(default_factory=dict)
    signature_history: deque = field(default_factory=lambda: deque(maxlen=10))
    base_resource_counts: Counter = field(default_factory=Counter)
    loop_warning_given: bool = False
    warned_signature: Optional[str] = None
    consecutive_research_calls: int = 0

    # Stage tracking
    stage: ConversationStage = ConversationStage.INITIAL

    # Manual stop
    forced_stop: Optional[str] = None

    # Log deduplication - only log forcing completion once
    completion_forcing_logged: bool = False

    # Response loop detection - tracks last response content to detect repeated responses
    last_response_content: str = ""
    response_loop_detected: bool = False


# =============================================================================
# Configuration Loader
# =============================================================================


class UnifiedTaskConfigLoader:
    """Loads unified task configuration from YAML."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "task_types": {
            "edit": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": True,
                "tool_budget": 25,
                "loop_repeat_threshold": 4,
                "required_tools": ["edit_files", "read_file"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["edit_files", "read_file"],
                    "verifying": ["read_file", "run_tests"],
                },
                "force_action_hints": {
                    "after_target_read": "Use edit_files to make the change.",
                    "max_iterations": "Please make the change or explain blockers.",
                },
            },
            "create": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": False,
                "tool_budget": 25,
                "loop_repeat_threshold": 4,
                "required_tools": ["write_file"],
                "stage_tools": {
                    "initial": ["list_directory", "read_file"],
                    "reading": ["read_file"],
                    "executing": ["write_file", "edit_files"],
                    "verifying": ["read_file", "run_tests"],
                },
                "force_action_hints": {
                    "max_iterations": "Please create the file.",
                },
            },
            "create_simple": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": False,
                "tool_budget": 20,
                "loop_repeat_threshold": 3,
                "required_tools": ["write_file"],
                "stage_tools": {
                    "initial": ["write_file"],
                    "reading": [],
                    "executing": ["write_file"],
                    "verifying": ["read_file"],
                },
                "force_action_hints": {
                    "immediate": "Create the code directly using write_file.",
                },
            },
            "search": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": False,
                "tool_budget": 25,
                "loop_repeat_threshold": 3,
                "required_tools": ["code_search", "read_file"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["read_file"],
                    "verifying": ["read_file"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize your findings.",
                },
            },
            "analyze": {
                "max_exploration_iterations": 20,
                "force_action_after_target_read": False,
                "tool_budget": 40,
                "loop_repeat_threshold": 5,
                "required_tools": ["read_file", "execute_bash"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search", "execute_bash"],
                    "executing": ["execute_bash"],
                    "verifying": ["read_file"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize your analysis.",
                },
            },
            "research": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": False,
                "tool_budget": 20,
                "loop_repeat_threshold": 3,
                "required_tools": ["web_search", "web_fetch"],
                "stage_tools": {
                    "initial": ["web_search"],
                    "reading": ["web_fetch", "web_search"],
                    "executing": ["web_fetch"],
                    "verifying": [],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize your research findings.",
                },
            },
            "design": {
                # Architecture/design questions require codebase exploration
                # to understand key components, structure, and patterns.
                # Increased limits from 3/5 to match "analyze" task type.
                "max_exploration_iterations": 20,
                "force_action_after_target_read": False,
                "tool_budget": 40,
                "loop_repeat_threshold": 5,
                "needs_tools": True,
                "required_tools": ["read_file", "list_directory", "code_search"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search", "read_file"],
                    "reading": ["read_file", "code_search", "list_directory"],
                    "executing": [],
                    "verifying": ["read_file"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize the architecture and provide your recommendations.",
                },
            },
            "general": {
                "max_exploration_iterations": 15,
                "force_action_after_target_read": False,
                "tool_budget": 35,
                "loop_repeat_threshold": 4,
                "required_tools": ["read_file", "list_directory"],
                "stage_tools": {
                    "initial": ["list_directory", "code_search", "read_file"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["edit_files", "write_file", "execute_bash"],
                    "verifying": ["read_file", "run_tests"],
                },
                "force_action_hints": {
                    "max_iterations": "Please complete the task or explain blockers.",
                },
            },
        },
        "model_overrides": {
            "deepseek*": {
                "exploration_multiplier": 1.5,
                "continuation_patience": 5,
            },
            "qwen*": {
                "exploration_multiplier": 1.3,
                "continuation_patience": 4,
            },
        },
        "global": {
            "max_total_iterations": 50,
            "min_content_threshold": 150,
            "signature_history_size": 10,
            "max_overlapping_reads_per_file": 3,
            "max_searches_per_query_prefix": 2,
        },
    }

    _instance: Optional["UnifiedTaskConfigLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "UnifiedTaskConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from existing YAML files.

        Uses:
        - task_tool_config.yaml for task type configurations
        - model_capabilities.yaml for model-specific settings (via capabilities loader)
        """
        # Load task config from existing task_tool_config.yaml
        task_config_path = Path(__file__).parent.parent / "config" / "task_tool_config.yaml"

        if task_config_path.exists():
            try:
                with open(task_config_path) as f:
                    task_config = yaml.safe_load(f) or {}

                # Restructure to match our expected format
                self._config = {
                    "task_types": task_config.get("task_types", {}),
                    "global": {
                        "max_total_iterations": 50,
                        "min_content_threshold": 150,
                        "signature_history_size": 10,
                        "max_overlapping_reads_per_file": 3,
                        "max_searches_per_query_prefix": 2,
                    },
                    # Model overrides come from model_capabilities.yaml via capabilities loader
                    "model_overrides": self.DEFAULT_CONFIG.get("model_overrides", {}),
                }
                logger.debug(f"Loaded task config from {task_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load task config: {e}, using defaults")
                self._config = self.DEFAULT_CONFIG
        else:
            logger.debug("Using default task config (task_tool_config.yaml not found)")
            self._config = self.DEFAULT_CONFIG

    def reload(self) -> None:
        """Force reload configuration."""
        self._config = None
        self._load_config()

    def get_task_config(self, task_type: TaskType) -> TaskConfig:
        """Get configuration for a specific task type."""
        config = self._config or self.DEFAULT_CONFIG
        task_configs = config.get("task_types", {})
        task_data = task_configs.get(task_type.value, task_configs.get("general", {}))

        return TaskConfig(
            max_exploration_iterations=task_data.get("max_exploration_iterations", 8),
            force_action_after_target_read=task_data.get("force_action_after_target_read", False),
            tool_budget=task_data.get("tool_budget", 50),
            loop_repeat_threshold=task_data.get("loop_repeat_threshold", 3),
            needs_tools=task_data.get("needs_tools", True),
            required_tools=task_data.get("required_tools", []),
            stage_tools=task_data.get("stage_tools", {}),
            force_action_hints=task_data.get("force_action_hints", {}),
        )

    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration settings."""
        config = self._config or self.DEFAULT_CONFIG
        return config.get("global", {})

    def get_model_override(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific override settings."""
        config = self._config or self.DEFAULT_CONFIG
        overrides = config.get("model_overrides", {})

        model_lower = model_name.lower()
        for pattern, settings in overrides.items():
            # Simple glob matching
            if pattern.endswith("*"):
                if model_lower.startswith(pattern[:-1].lower()):
                    return settings
            elif model_lower == pattern.lower():
                return settings

        return {}


# =============================================================================
# Main Tracker Class
# =============================================================================


class UnifiedTaskTracker:
    """Unified task tracking combining milestones and loop detection.

    This class is the single source of truth for:
    - Task type and configuration
    - Iteration counting and limits
    - Milestone tracking
    - Loop detection
    - Stop decisions

    Usage:
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TaskType.EDIT)

        # Record tool calls
        tracker.record_tool_call("read_file", {"path": "test.py"})

        # Check if should stop
        decision = tracker.should_stop()
        if decision.should_stop:
            print(decision.hint)
    """

    def __init__(self) -> None:
        """Initialize the unified task tracker."""
        self._config_loader = UnifiedTaskConfigLoader()
        self._progress = UnifiedTaskProgress()
        self._task_config: Optional[TaskConfig] = None
        self._sticky_user_budget: bool = False
        self._allow_budget_override: bool = False
        self._sticky_user_iterations: bool = False
        self._allow_iteration_override: bool = False

        # Model-specific settings
        self._exploration_multiplier: float = 1.0
        self._continuation_patience: int = 3

        # Global settings
        global_config = self._config_loader.get_global_config()
        self._max_total_iterations = global_config.get("max_total_iterations", 50)
        self._min_content_threshold = global_config.get("min_content_threshold", 150)
        self._max_overlapping_reads = global_config.get("max_overlapping_reads_per_file", 3)
        self._max_searches_per_prefix = global_config.get("max_searches_per_query_prefix", 2)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def progress(self) -> UnifiedTaskProgress:
        """Get current progress state."""
        return self._progress

    @property
    def task_type(self) -> TaskType:
        """Get current task type."""
        return self._progress.task_type

    @property
    def iteration_count(self) -> int:
        """Get productive iteration count."""
        return self._progress.iteration_count

    @property
    def tool_calls(self) -> int:
        """Get total tool calls."""
        return self._progress.tool_calls

    @property
    def remaining_budget(self) -> int:
        """Get remaining tool budget."""
        return max(0, self._progress.tool_budget - self._progress.tool_calls)

    @property
    def milestones(self) -> Set[Milestone]:
        """Get achieved milestones."""
        return self._progress.milestones.copy()

    @property
    def stage(self) -> ConversationStage:
        """Get current conversation stage."""
        return self._progress.stage

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_task_type(self, task_type: TaskType) -> None:
        """Set the task type and load corresponding configuration."""
        self._progress.task_type = task_type
        self._task_config = self._config_loader.get_task_config(task_type)
        self._progress.tool_budget = self._task_config.tool_budget

        logger.info(
            f"UnifiedTaskTracker: task_type={task_type.value}, "
            f"max_iterations={self._task_config.max_exploration_iterations}, "
            f"tool_budget={self._task_config.tool_budget}"
        )

    def set_model_capabilities(
        self,
        exploration_multiplier: float = 1.0,
        continuation_patience: int = 3,
    ) -> None:
        """Set model-specific exploration settings."""
        self._exploration_multiplier = exploration_multiplier
        self._continuation_patience = continuation_patience

        logger.info(
            f"UnifiedTaskTracker: model settings - "
            f"multiplier={exploration_multiplier}, patience={continuation_patience}"
        )

    def set_target_files(self, files: Set[str]) -> None:
        """Set target files for the task."""
        self._progress.target_files = files

    def set_tool_budget(self, budget: int, user_override: bool = False) -> None:
        """Set the tool budget for this task.

        Args:
            budget: Maximum number of tool calls allowed
            user_override: Mark this budget as sticky (prevents auto-adjustment)
        """
        if self._sticky_user_budget and not user_override and not self._allow_budget_override:
            logger.debug("UnifiedTaskTracker: sticky user budget set; skipping auto-adjustment")
            return

        if user_override:
            self._allow_budget_override = True
            self._sticky_user_budget = True

        self._progress.tool_budget = budget
        logger.debug(f"UnifiedTaskTracker: tool_budget set to {budget}")
        self._allow_budget_override = False

    def set_max_iterations(self, iterations: int, user_override: bool = False) -> None:
        """Set the maximum total iterations allowed.

        Args:
            iterations: Maximum iterations
            user_override: Mark this limit as sticky (prevents auto-adjustment)
        """
        if self._sticky_user_iterations and not user_override and not self._allow_iteration_override:
            logger.debug("UnifiedTaskTracker: sticky user max iterations set; skipping auto-adjustment")
            return

        if user_override:
            self._allow_iteration_override = True
            self._sticky_user_iterations = True

        self._max_total_iterations = iterations
        logger.debug(f"UnifiedTaskTracker: max_total_iterations set to {iterations}")
        self._allow_iteration_override = False

    def set_target_entities(self, entities: Set[str]) -> None:
        """Set target entities (functions, classes) for the task."""
        self._progress.target_entities = entities

    def get_required_tools(self) -> Set[str]:
        """Get the required tools for the current task type.

        Returns:
            Set of tool names required for this task
        """
        return set(self._task_config.required_tools)

    @property
    def max_exploration_iterations(self) -> int:
        """Get max exploration iterations for current task type."""
        return self._task_config.max_exploration_iterations

    # =========================================================================
    # Recording
    # =========================================================================

    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Record a tool call - updates milestones, loops, and budgets."""
        self._progress.tool_calls += 1
        self._progress.iteration_count += 1
        self._progress.total_turns += 1

        # Update milestones
        self._update_milestones(tool_name, arguments)

        # Update loop detection
        self._update_loop_state(tool_name, arguments)

        # Track resources
        self._track_resource(tool_name, arguments)

        # Update stage
        self._update_stage(tool_name)

        logger.debug(
            f"UnifiedTaskTracker: tool_call={tool_name}, "
            f"iteration={self._progress.iteration_count}, "
            f"tool_calls={self._progress.tool_calls}"
        )

    def increment_turn(self) -> None:
        """Record a turn without tool call (continuation prompt)."""
        self._progress.total_turns += 1
        logger.debug(
            f"UnifiedTaskTracker: turn without tool call, "
            f"total_turns={self._progress.total_turns}, "
            f"productive={self._progress.iteration_count}"
        )

    def record_iteration(self, content_length: int) -> None:
        """Record an iteration completion with content length."""
        if content_length < self._min_content_threshold:
            self._progress.low_output_iterations += 1

    def force_stop(self, reason: str) -> None:
        """Force the tracker to recommend stopping."""
        self._progress.forced_stop = reason

    def reset(self) -> None:
        """Reset all state for a new conversation."""
        self._progress = UnifiedTaskProgress()
        self._task_config = None

    # =========================================================================
    # Stop Decision
    # =========================================================================

    def should_stop(self) -> StopDecision:
        """Single unified check for whether to stop.

        Checks in order:
        1. Manual force stop
        2. Tool budget exceeded
        3. True loop detected
        4. Max iterations exceeded
        5. Goal-based forcing (milestone-aware)
        """
        details = self._get_details()

        # Manual stop
        if self._progress.forced_stop:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MANUAL_STOP,
                hint=f"Manual stop: {self._progress.forced_stop}",
                details=details,
            )

        # Tool budget check
        if self._progress.tool_calls >= self._progress.tool_budget:
            return StopDecision(
                should_stop=True,
                reason=StopReason.TOOL_BUDGET,
                hint=f"Tool budget exceeded ({self._progress.tool_calls}/{self._progress.tool_budget})",
                details=details,
            )

        # Loop check
        loop = self._check_loop()
        if loop:
            return StopDecision(
                should_stop=True,
                reason=StopReason.LOOP_DETECTED,
                hint=f"Loop detected: {loop}",
                details=details,
            )

        # Hard iteration limit
        if self._progress.iteration_count >= self._max_total_iterations:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_ITERATIONS,
                hint=f"Max iterations reached ({self._progress.iteration_count}/{self._max_total_iterations})",
                details=details,
            )

        # Task-specific iteration limit (with model multiplier)
        effective_max = self._calculate_effective_max()
        if self._progress.iteration_count > effective_max:
            hint = self._get_completion_hint()
            # Only log once to avoid duplicate messages
            if not self._progress.completion_forcing_logged:
                self._progress.completion_forcing_logged = True
                logger.info(
                    f"UnifiedTaskTracker: Forcing completion at iteration {self._progress.iteration_count} "
                    f"(task_type={self._progress.task_type.value}, "
                    f"base_max={self._get_base_max()}, effective_max={effective_max}, "
                    f"total_turns={self._progress.total_turns}, "
                    f"multiplier={self._exploration_multiplier})"
                )
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_ITERATIONS,
                hint=hint,
                details=details,
            )

        # Goal-based forcing (EDIT tasks after target read)
        goal_decision = self._check_goal_forcing()
        if goal_decision:
            return goal_decision

        return StopDecision(should_stop=False, details=details)

    def check_loop_warning(self) -> Optional[str]:
        """Check if approaching loop threshold (warning before hard stop)."""
        if self._progress.loop_warning_given:
            return None

        if len(self._progress.signature_history) < 3:
            return None

        threshold = self._get_loop_threshold()
        recent = list(self._progress.signature_history)[
            -min(len(self._progress.signature_history), 8) :
        ]

        if recent:
            sig_counts = Counter(recent)
            for sig, count in sig_counts.items():
                if count == threshold - 1:
                    self._progress.loop_warning_given = True
                    self._progress.warned_signature = sig
                    return f"Approaching loop ({count}/{threshold}): {sig[:80]}"

        return None

    def is_blocked_after_warning(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Check if a tool call is blocked due to matching warned signature."""
        if not self._progress.loop_warning_given or not self._progress.warned_signature:
            return None

        proposed_sig = self._get_signature(tool_name, arguments)
        if proposed_sig == self._progress.warned_signature:
            return f"Blocked: same operation after warning ({proposed_sig[:50]})"

        return None

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get current progress metrics for logging/monitoring."""
        return {
            "task_type": self._progress.task_type.value,
            "iteration_count": self._progress.iteration_count,
            "total_turns": self._progress.total_turns,
            "tool_calls": self._progress.tool_calls,
            "tool_budget": self._progress.tool_budget,
            "remaining_budget": self.remaining_budget,
            "unique_resources": len(self._progress.unique_resources),
            "files_read": len(self._progress.files_read),
            "files_modified": len(self._progress.files_modified),
            "milestones": [m.value for m in self._progress.milestones],
            "stage": self._progress.stage.value,
            "exploration_multiplier": self._exploration_multiplier,
            "effective_max": self._calculate_effective_max(),
        }

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _update_milestones(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Update milestones based on tool call."""
        if tool_name in {"list_directory", "code_search", "semantic_code_search"}:
            self._progress.milestones.add(Milestone.TARGET_IDENTIFIED)

        elif get_canonical_name(tool_name) == CANONICAL_READ_TOOL:
            path = arguments.get("path", "")
            if path:
                self._progress.files_read.add(path)
                if path in self._progress.target_files:
                    self._progress.milestones.add(Milestone.TARGET_READ)

        elif tool_name in {"edit_files", "write_file"}:
            self._progress.milestones.add(Milestone.CHANGE_MADE)
            if tool_name == "edit_files":
                files = arguments.get("files", [])
                if isinstance(files, list):
                    for f in files:
                        if isinstance(f, dict):
                            self._progress.files_modified.add(f.get("path", ""))
            else:
                path = arguments.get("path", "")
                if path:
                    self._progress.files_modified.add(path)

        elif tool_name in {"run_tests", "execute_bash"}:
            if Milestone.CHANGE_MADE in self._progress.milestones:
                self._progress.milestones.add(Milestone.CHANGE_VERIFIED)

    def _update_loop_state(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Update loop detection state."""
        # Track file reads with offset-aware detection
        if get_canonical_name(tool_name) == CANONICAL_READ_TOOL:
            self._track_file_read(arguments)
        else:
            # Track base resources for non-file operations
            base_key = self._get_base_resource_key(tool_name, arguments)
            if base_key:
                self._progress.base_resource_counts[base_key] += 1

        # Track signature
        signature = self._get_signature(tool_name, arguments)

        # Clear warning if truly different signature
        if self._progress.loop_warning_given:
            recent = list(self._progress.signature_history)[-3:]
            if signature not in recent:
                self._progress.loop_warning_given = False
                self._progress.warned_signature = None

        self._progress.signature_history.append(signature)

        # Track research calls
        if tool_name in RESEARCH_TOOLS:
            self._progress.consecutive_research_calls += 1
        else:
            self._progress.consecutive_research_calls = 0

    def _track_file_read(self, arguments: Dict[str, Any]) -> None:
        """Track file read with offset-aware overlap detection."""
        path = arguments.get("path", "")
        if not path:
            return

        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", DEFAULT_READ_LIMIT)
        new_range = FileReadRange(offset=offset, limit=limit)

        if path not in self._progress.file_read_ranges:
            self._progress.file_read_ranges[path] = []

        self._progress.file_read_ranges[path].append(new_range)

    def _track_resource(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Track unique resource access."""
        resource_key = self._get_resource_key(tool_name, arguments)
        if resource_key:
            self._progress.unique_resources.add(resource_key)

    def _update_stage(self, tool_name: str) -> None:
        """Update conversation stage based on tool usage."""
        canonical = get_canonical_name(tool_name)
        if canonical in {CANONICAL_READ_TOOL, ToolNames.GREP, ToolNames.SEARCH}:
            if self._progress.stage == ConversationStage.INITIAL:
                self._progress.stage = ConversationStage.READING

        elif tool_name in {"edit_files", "write_file"}:
            self._progress.stage = ConversationStage.EXECUTING

        elif tool_name in {"run_tests", "execute_bash"}:
            if Milestone.CHANGE_MADE in self._progress.milestones:
                self._progress.stage = ConversationStage.VERIFYING

    def _check_loop(self) -> Optional[str]:
        """Check for loop patterns."""
        # Check file read overlaps
        file_loop = self._check_file_read_loops()
        if file_loop:
            return file_loop

        # Check search query repetition
        for base_key, count in self._progress.base_resource_counts.items():
            if base_key.startswith("search:") and count > self._max_searches_per_prefix:
                return f"Similar search repeated {count} times: {base_key}"

        # Check signature repetition
        if len(self._progress.signature_history) < 3:
            return None

        threshold = self._get_loop_threshold()
        recent = list(self._progress.signature_history)[
            -min(len(self._progress.signature_history), 8) :
        ]

        if recent:
            sig_counts = Counter(recent)
            for sig, count in sig_counts.items():
                if count >= threshold:
                    return f"Same signature repeated {count} times: {sig[:50]}"

        return None

    def _check_file_read_loops(self) -> Optional[str]:
        """Check for overlapping file read loops."""
        for path, ranges in self._progress.file_read_ranges.items():
            if len(ranges) < 2:
                continue

            for i, current in enumerate(ranges):
                overlap_count = sum(
                    1 for j, other in enumerate(ranges) if i != j and current.overlaps(other)
                )
                total = overlap_count + 1
                if total > self._max_overlapping_reads:
                    return (
                        f"Same file region read {total} times: {path} " f"[offset={current.offset}]"
                    )

        return None

    def check_response_loop(self, content: str, similarity_threshold: float = 0.9) -> bool:
        """Check if response content is a repeat of the previous response.

        This detects when the model keeps responding with similar text but makes
        no tool calls - a different type of loop than tool call loops.

        NOTE: Threshold raised from 0.7 to 0.9 to reduce false positives.
        When exploring directories, responses like "Let me examine dir1" and
        "Let me examine dir2" have high word overlap (~80%) but represent progress.
        A 0.9 threshold catches near-verbatim repeats while allowing variation.

        Args:
            content: The current response content to check
            similarity_threshold: Word overlap ratio to consider as repeated (default 0.9)

        Returns:
            True if a response loop is detected, False otherwise
        """
        content_for_comparison = (content or "").strip()[:500]
        last_content = self._progress.last_response_content

        is_repeated = False
        if last_content and content_for_comparison:
            # Simple word overlap similarity check
            current_words = set(content_for_comparison.lower().split())
            last_words = set(last_content.lower().split())
            if current_words and last_words:
                overlap = len(current_words & last_words)
                max_words = max(len(current_words), len(last_words))
                similarity = overlap / max_words if max_words > 0 else 0
                if similarity > similarity_threshold:
                    is_repeated = True
                    self._progress.response_loop_detected = True
                    logger.warning(
                        f"Response loop detected (similarity={similarity:.2f}), "
                        "forcing completion to prevent infinite loop"
                    )

        # Track content for next iteration's comparison
        self._progress.last_response_content = content_for_comparison
        return is_repeated

    @property
    def response_loop_detected(self) -> bool:
        """Check if a response loop has been detected."""
        return self._progress.response_loop_detected

    def _check_goal_forcing(self) -> Optional[StopDecision]:
        """Check for goal-based forcing (EDIT after target read)."""
        if not self._task_config:
            return None

        if (
            self._task_config.force_action_after_target_read
            and Milestone.TARGET_READ in self._progress.milestones
            and Milestone.CHANGE_MADE not in self._progress.milestones
            and self._progress.iteration_count >= self._calculate_effective_max()
        ):
            hint = self._task_config.force_action_hints.get(
                "after_target_read",
                "You have read the target. Please make the change.",
            )
            return StopDecision(
                should_stop=True,
                reason=StopReason.GOAL_FORCING,
                hint=hint,
                details=self._get_details(),
            )

        return None

    def _calculate_effective_max(self) -> int:
        """Calculate effective max iterations with model adjustments."""
        base_max = self._get_base_max()

        # Apply model multiplier
        model_adjusted = int(base_max * self._exploration_multiplier)

        # Apply productivity ratio adjustment
        total = self._progress.total_turns
        productive = self._progress.iteration_count

        if total > 0 and productive > 0:
            productivity = productive / total
            if productivity > 0:
                productivity_mult = min(2.0, max(1.0, 1.0 / productivity))
                return int(model_adjusted * productivity_mult)

        return model_adjusted

    def _get_base_max(self) -> int:
        """Get base max iterations from task config."""
        if self._task_config:
            return self._task_config.max_exploration_iterations
        return 8

    def _get_loop_threshold(self) -> int:
        """Get loop repeat threshold from task config."""
        if self._task_config:
            return self._task_config.loop_repeat_threshold
        return 3

    def _get_completion_hint(self) -> str:
        """Get appropriate completion hint based on task type."""
        if self._task_config:
            return self._task_config.force_action_hints.get(
                "max_iterations",
                "Please complete the task or explain what's blocking you.",
            )

        # Fallback hints by task type
        if self._progress.task_type == TaskType.SEARCH:
            return "Please summarize your findings and provide the answer."
        elif self._progress.task_type == TaskType.ANALYZE:
            return "Please summarize your analysis."
        elif self._progress.task_type in {TaskType.EDIT, TaskType.CREATE}:
            return "Please complete the change or explain what's blocking you."
        else:
            return "Please complete the task or explain what's blocking you."

    def _get_signature(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate signature for loop detection.

        For progressive tools (with progress_params defined via @tool decorator),
        only the specified parameters are used in the signature - this allows
        different queries/paths to be treated as exploration, not loops.
        """
        params = get_progress_params_for_tool(tool_name)
        if params:
            sig_parts = [tool_name]
            for param in params:
                value = arguments.get(param, "")
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100]
                sig_parts.append(str(value))
            return "|".join(sig_parts)
        else:
            args_str = str(sorted(arguments.items()))
            return f"{tool_name}:{hashlib.md5(args_str.encode()).hexdigest()[:8]}"

    def _get_resource_key(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Generate resource key for tracking unique resources."""
        canonical = get_canonical_name(tool_name)
        if canonical == CANONICAL_READ_TOOL:
            path = arguments.get("path", "")
            offset = arguments.get("offset", 0)
            return f"file:{path}:{offset}" if path else None
        elif canonical == ToolNames.LS:
            path = arguments.get("path", "")
            return f"dir:{path}" if path else None
        elif canonical in {ToolNames.GREP, ToolNames.SEARCH}:
            query = arguments.get("query", "")
            directory = arguments.get("directory", ".")
            return f"search:{directory}:{query[:50]}" if query else None
        elif canonical == ToolNames.SHELL:
            command = arguments.get("command", "")
            return f"bash:{command[:50]}" if command else None
        return None

    def _get_base_resource_key(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Generate base resource key for loop detection."""
        if tool_name == "list_directory":
            path = arguments.get("path", "")
            return f"dir:{path}" if path else None
        elif tool_name in {"code_search", "semantic_code_search"}:
            query = arguments.get("query", "")
            directory = arguments.get("directory", ".")
            return f"search:{directory}:{query[:20]}" if query else None
        elif tool_name == "execute_bash":
            command = arguments.get("command", "")
            if command:
                parts = command.strip().split()
                if parts:
                    base = parts[0]
                    if base == "git" and len(parts) > 1:
                        base = f"git {parts[1]}"
                    return f"bash:{base}"
        return None

    def _get_details(self) -> Dict[str, Any]:
        """Get details dict for stop decision."""
        return {
            "task_type": self._progress.task_type.value,
            "iteration_count": self._progress.iteration_count,
            "total_turns": self._progress.total_turns,
            "tool_calls": self._progress.tool_calls,
            "tool_budget": self._progress.tool_budget,
            "milestones": [m.value for m in self._progress.milestones],
            "effective_max": self._calculate_effective_max(),
        }

    # =========================================================================
    # Backward Compatibility Methods (for orchestrator integration)
    # =========================================================================

    @property
    def config(self) -> "CompatConfig":
        """Return config-like object for backward compatibility with LoopDetector."""
        return CompatConfig(self)

    @property
    def iterations(self) -> int:
        """Alias for iteration_count (LoopDetector compatibility)."""
        return self._progress.iteration_count

    @property
    def unique_resources(self) -> Set[str]:
        """Get unique resources (LoopDetector compatibility)."""
        return self._progress.unique_resources.copy()

    def detect_task_type(self, message: str) -> TaskType:
        """Detect task type from message and configure tracker.

        Args:
            message: User message to classify

        Returns:
            Detected TaskType
        """
        from victor.embeddings.task_classifier import TaskType as ClassifierTaskType
        from victor.embeddings.task_classifier import TaskTypeClassifier

        # Use the singleton classifier instance
        classifier = TaskTypeClassifier.get_instance()
        # Initialize synchronously if not already initialized
        classifier.initialize_sync()
        result = classifier.classify_sync(message)
        classifier_type = result.task_type

        type_map = {
            ClassifierTaskType.EDIT: TaskType.EDIT,
            ClassifierTaskType.CREATE: TaskType.CREATE,
            ClassifierTaskType.CREATE_SIMPLE: TaskType.CREATE_SIMPLE,
            ClassifierTaskType.SEARCH: TaskType.SEARCH,
            ClassifierTaskType.ANALYZE: TaskType.ANALYZE,
            ClassifierTaskType.DESIGN: TaskType.DESIGN,
            ClassifierTaskType.GENERAL: TaskType.GENERAL,
            # Map additional types to closest match
            ClassifierTaskType.ACTION: TaskType.GENERAL,  # Actions use general limits
            ClassifierTaskType.ANALYSIS_DEEP: TaskType.ANALYZE,  # Deep analysis uses analyze limits
        }

        task_type = type_map.get(classifier_type, TaskType.GENERAL)
        self.set_task_type(task_type)
        return task_type

    def detect_task_type_with_negation(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[TaskType, "ClassificationResult"]:
        """Detect task type using negation-aware keyword classification.

        Uses UnifiedTaskClassifier for robust classification that handles:
        - Negation patterns ("don't analyze" won't classify as analysis)
        - Positive overrides ("don't analyze, just run" classifies as action)
        - Context boosting from conversation history
        - Confidence scoring for reliability assessment

        Args:
            message: User message to classify
            history: Optional conversation history for context boosting

        Returns:
            Tuple of (TaskType, ClassificationResult) for detailed inspection
        """
        from victor.agent.unified_classifier import (
            TaskType as UnifiedTaskType,
            get_unified_classifier,
        )

        classifier = get_unified_classifier()

        # Use context-aware classification if history provided
        if history:
            result = classifier.classify_with_context(message, history)
        else:
            result = classifier.classify(message)

        # Map UnifiedTaskClassifier.TaskType to UnifiedTaskTracker.TaskType
        type_map = {
            UnifiedTaskType.EDIT: TaskType.EDIT,
            UnifiedTaskType.SEARCH: TaskType.SEARCH,
            UnifiedTaskType.ANALYSIS: TaskType.ANALYZE,
            UnifiedTaskType.GENERATION: TaskType.CREATE,
            UnifiedTaskType.ACTION: TaskType.GENERAL,  # ACTION maps to GENERAL for broad tool access
            UnifiedTaskType.DEFAULT: TaskType.GENERAL,
        }

        task_type = type_map.get(result.task_type, TaskType.GENERAL)

        # Use recommended tool budget from classifier
        self.set_task_type(task_type)
        if result.recommended_tool_budget:
            self.set_tool_budget(result.recommended_tool_budget)

        # Log negation info for debugging
        if result.negated_keywords:
            negated_kws = [m.keyword for m in result.negated_keywords]
            logger.info(
                f"UnifiedTaskTracker: Negated keywords detected: {negated_kws}, "
                f"final task_type={task_type.value}, confidence={result.confidence:.2f}"
            )

        return task_type, result

    def classify_with_negation_awareness(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Classify message with full negation awareness and return detailed results.

        Combines UnifiedTaskClassifier (keyword + negation) with UnifiedTaskTracker
        (budget + milestone tracking) for comprehensive task analysis.

        Args:
            message: User message to classify
            history: Optional conversation history

        Returns:
            Dictionary with classification details:
            - task_type: TaskType enum value
            - confidence: Classification confidence (0-1)
            - is_action_task: Whether task involves actions
            - is_analysis_task: Whether task involves analysis
            - negated_keywords: List of negated keyword strings
            - recommended_budget: Tool call budget
            - source: Classification source (keyword/semantic/context)
        """
        task_type, result = self.detect_task_type_with_negation(message, history)

        return {
            "task_type": task_type.value,
            "confidence": result.confidence,
            "is_action_task": result.is_action_task,
            "is_analysis_task": result.is_analysis_task,
            "is_generation_task": result.is_generation_task,
            "needs_execution": result.needs_execution,
            "negated_keywords": [m.keyword for m in result.negated_keywords],
            "matched_keywords": [m.keyword for m in result.matched_keywords],
            "recommended_budget": result.recommended_tool_budget,
            "source": result.source,
            "context_boost": result.context_boost,
        }

    def update_from_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update tracker from tool call (TaskMilestoneMonitor compatibility).

        Args:
            tool_name: Name of the tool called
            arguments: Tool arguments
            result: Optional tool result (unused, for compatibility)
        """
        # This is called by orchestrator, but record_tool_call already handles everything
        # Just update milestones if needed (record_tool_call handles most of it)
        pass  # record_tool_call already called separately

    def increment_iteration(self) -> None:
        """Increment iteration count (TaskMilestoneMonitor compatibility).

        Note: This is typically called after record_tool_call, but record_tool_call
        already increments iteration_count, so this is a no-op to avoid double counting.
        """
        # Already handled in record_tool_call
        pass

    def should_force_action(self) -> Tuple[bool, Optional[str]]:
        """Check if action should be forced (TaskMilestoneMonitor compatibility).

        Returns:
            Tuple of (should_force, hint_message)
        """
        decision = self.should_stop()
        if decision.should_stop and decision.reason in (
            StopReason.MAX_ITERATIONS,
            StopReason.GOAL_FORCING,
        ):
            return True, decision.hint
        return False, None

    def set_model_exploration_settings(
        self,
        exploration_multiplier: float = 1.0,
        continuation_patience: int = 3,
    ) -> None:
        """Set model exploration settings (TaskMilestoneMonitor compatibility).

        Alias for set_model_capabilities.
        """
        self.set_model_capabilities(exploration_multiplier, continuation_patience)


class CompatConfig:
    """Compatibility wrapper for LoopDetector's config attribute."""

    def __init__(self, tracker: UnifiedTaskTracker) -> None:
        self._tracker = tracker

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (dict-like interface).

        Args:
            key: Configuration key name
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if key == "max_total_iterations":
            return self._tracker._max_total_iterations
        elif key == "tool_budget":
            return self._tracker._progress.tool_budget
        return default

    @property
    def max_total_iterations(self) -> int:
        return self._tracker._max_total_iterations

    @max_total_iterations.setter
    def max_total_iterations(self, value: int) -> None:
        if (
            self._tracker._sticky_user_iterations
            and not self._tracker._allow_iteration_override
        ):
            logger.debug("UnifiedTaskTracker: sticky user max iterations set; skipping auto-adjustment")
            return
        self._tracker._max_total_iterations = value

    @property
    def tool_budget(self) -> int:
        return self._tracker._progress.tool_budget

    @tool_budget.setter
    def tool_budget(self, value: int) -> None:
        if (
            self._tracker._sticky_user_budget
            and not self._tracker._allow_budget_override
        ):
            logger.debug("UnifiedTaskTracker: sticky user budget set; skipping auto-adjustment")
            return
        self._tracker._progress.tool_budget = value


# =============================================================================
# Factory Functions
# =============================================================================


def create_tracker_for_task(task_type: TaskType) -> UnifiedTaskTracker:
    """Create a tracker configured for a specific task type.

    Args:
        task_type: The task type to configure for

    Returns:
        Configured UnifiedTaskTracker
    """
    tracker = UnifiedTaskTracker()
    tracker.set_task_type(task_type)
    return tracker


def create_tracker_from_message(message: str) -> Tuple[UnifiedTaskTracker, TaskType]:
    """Create a tracker by classifying a message.

    Args:
        message: User message to classify

    Returns:
        Tuple of (tracker, detected_task_type)
    """
    # Import here to avoid circular dependency
    from victor.embeddings.task_classifier import TaskType as ClassifierTaskType
    from victor.embeddings.task_classifier import classify_task_type

    # Classify the message
    classifier_type = classify_task_type(message)

    # Map classifier types to unified types
    type_map = {
        ClassifierTaskType.EDIT: TaskType.EDIT,
        ClassifierTaskType.CREATE: TaskType.CREATE,
        ClassifierTaskType.CREATE_SIMPLE: TaskType.CREATE_SIMPLE,
        ClassifierTaskType.SEARCH: TaskType.SEARCH,
        ClassifierTaskType.ANALYZE: TaskType.ANALYZE,
        ClassifierTaskType.DESIGN: TaskType.DESIGN,
        ClassifierTaskType.GENERAL: TaskType.GENERAL,
    }

    task_type = type_map.get(classifier_type, TaskType.GENERAL)

    tracker = UnifiedTaskTracker()
    tracker.set_task_type(task_type)

    return tracker, task_type


def create_tracker_with_negation_awareness(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[UnifiedTaskTracker, TaskType, Dict[str, Any]]:
    """Create a tracker using negation-aware keyword classification.

    This is the recommended way to create a tracker when negation detection
    is important (e.g., "don't analyze, just run the tests").

    Args:
        message: User message to classify
        history: Optional conversation history for context boosting

    Returns:
        Tuple of (tracker, task_type, classification_details)

    Example:
        tracker, task_type, details = create_tracker_with_negation_awareness(
            "Don't analyze the code, just run the tests"
        )
        print(task_type)  # TaskType.GENERAL (not ANALYZE)
        print(details["negated_keywords"])  # ["analyze"]
        print(details["matched_keywords"])  # ["run"]
    """
    tracker = UnifiedTaskTracker()
    task_type, result = tracker.detect_task_type_with_negation(message, history)

    details = {
        "task_type": task_type.value,
        "confidence": result.confidence,
        "is_action_task": result.is_action_task,
        "is_analysis_task": result.is_analysis_task,
        "is_generation_task": result.is_generation_task,
        "needs_execution": result.needs_execution,
        "negated_keywords": [m.keyword for m in result.negated_keywords],
        "matched_keywords": [m.keyword for m in result.matched_keywords],
        "recommended_budget": result.recommended_tool_budget,
        "source": result.source,
    }

    return tracker, task_type, details
