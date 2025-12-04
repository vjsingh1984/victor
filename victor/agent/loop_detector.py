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

"""Loop detection and budget enforcement for the orchestrator.

This module provides loop detection and tool budget enforcement:
- Tool call counting and budget enforcement
- True loop detection (repeated signatures)
- Progress tracking (unique files read)
- Task-type aware thresholds (analysis vs action vs default)
- Research loop detection

The primary class is `LoopDetector`.

For goal-aware milestone tracking, see `victor.agent.milestone_monitor.TaskMilestoneMonitor`.

Design principles:
- Single source of truth for loop detection state
- Clear, testable interface
- Task-type aware configuration
- Extensible for future needs

"""

import hashlib
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

if TYPE_CHECKING:
    from victor.agent.complexity_classifier import TaskClassification

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type for configuring progress thresholds."""

    DEFAULT = "default"
    ANALYSIS = "analysis"
    ACTION = "action"
    RESEARCH = "research"


# Tools that indicate research activity
RESEARCH_TOOLS = frozenset({"web_search", "web_fetch", "tavily_search", "search_web", "fetch_url"})

# Progressive tools - different params indicate progress, not loops
# Format: tool_name -> list of params that indicate progress
PROGRESSIVE_PARAMS = {
    "read_file": ["path", "offset", "limit"],
    "code_search": ["query", "directory"],
    "semantic_code_search": ["query", "directory"],
    "list_directory": ["path", "recursive"],
    "execute_bash": ["command"],
    "web_search": ["query"],
    "web_fetch": ["url"],
}


@dataclass
class ProgressConfig:
    """Configuration for progress tracking thresholds.

    All thresholds are configurable to allow tuning for different use cases.
    """

    # Tool budget
    tool_budget: int = 50

    # Iteration limits by task type
    max_iterations_default: int = 8
    max_iterations_analysis: int = 50
    max_iterations_action: int = 12
    max_iterations_research: int = 6

    # Loop detection
    repeat_threshold_default: int = 3
    repeat_threshold_analysis: int = 5
    signature_history_size: int = 10

    # Progress detection
    min_content_threshold: int = 150

    # Hard limits
    max_total_iterations: int = 20

    # Same-resource access limits (Gap 2 fix)
    # Tracks reads of the same file regardless of offset
    max_reads_per_file: int = 3
    max_searches_per_query_prefix: int = 2


@dataclass
class StopReason:
    """Reason why progress tracker recommends stopping.

    Attributes:
        should_stop: Whether execution should stop
        reason: Human-readable reason for stopping
        details: Dictionary with additional context for logging/debugging
    """

    should_stop: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class LoopDetector:
    """Loop detection and budget enforcement for the orchestrator.

    This class handles reactive loop detection and budget enforcement:
    - Tool call counting and budget enforcement
    - True loop detection (repeated signatures)
    - Progress tracking (unique files read)
    - Task-type aware thresholds
    - Research loop detection

    For goal-aware milestone tracking and proactive nudging, see TaskMilestoneMonitor.

    Usage:
        detector = LoopDetector(task_type=TaskType.ANALYSIS)

        # In the main loop:
        detector.record_tool_call("read_file", {"path": "test.py"})
        detector.record_iteration(content_length=len(response))

        if detector.should_stop().should_stop:
            # Force completion
            break

    """

    def __init__(
        self,
        config: Optional[ProgressConfig] = None,
        task_type: TaskType = TaskType.DEFAULT,
    ):
        """Initialize the progress tracker.

        Args:
            config: Configuration for thresholds. Uses defaults if not provided.
            task_type: Type of task (affects thresholds).
        """
        self.config = config or ProgressConfig()
        self.task_type = task_type

        # Core counters
        self._tool_calls = 0
        self._iterations = 0
        self._low_output_iterations = 0

        # Resource tracking
        self._unique_resources: Set[str] = set()

        # Base resource counting (tracks same file regardless of offset)
        # Key: base resource (e.g., "file:orchestrator.py"), Value: count
        self._base_resource_counts: Counter[str] = Counter()

        # Loop detection
        self._signature_history: deque = deque(maxlen=self.config.signature_history_size)
        self._consecutive_research_calls = 0

        # Manual stop
        self._forced_stop: Optional[str] = None

    def reset(self) -> None:
        """Reset all state for a new conversation turn."""
        self._tool_calls = 0
        self._iterations = 0
        self._low_output_iterations = 0
        self._unique_resources.clear()
        self._base_resource_counts.clear()
        self._signature_history.clear()
        self._consecutive_research_calls = 0
        self._forced_stop = None

    @property
    def tool_calls(self) -> int:
        """Number of tool calls made."""
        return self._tool_calls

    @property
    def iterations(self) -> int:
        """Number of iterations completed."""
        return self._iterations

    @property
    def low_output_iterations(self) -> int:
        """Number of iterations with low content output."""
        return self._low_output_iterations

    @property
    def unique_resources(self) -> Set[str]:
        """Set of unique resources accessed."""
        return self._unique_resources.copy()

    @property
    def remaining_budget(self) -> int:
        """Remaining tool call budget."""
        return max(0, self.config.tool_budget - self._tool_calls)

    @property
    def progress_percentage(self) -> float:
        """Progress through tool budget as percentage."""
        if self.config.tool_budget <= 0:
            return 100.0
        return (self._tool_calls / self.config.tool_budget) * 100.0

    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Record a tool call for tracking and loop detection.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
        """
        self._tool_calls += 1

        # Track unique resources (includes offset for granular tracking)
        resource_key = self._get_resource_key(tool_name, arguments)
        if resource_key:
            self._unique_resources.add(resource_key)

        # Track base resources (ignores offset - for same-file loop detection)
        base_key = self._get_base_resource_key(tool_name, arguments)
        if base_key:
            self._base_resource_counts[base_key] += 1

        # Track signature for loop detection
        signature = self._get_signature(tool_name, arguments)
        self._signature_history.append(signature)

        # Track research calls
        if tool_name in RESEARCH_TOOLS:
            self._consecutive_research_calls += 1
        else:
            self._consecutive_research_calls = 0

        logger.debug(
            f"LoopDetector: tool_call={tool_name}, "
            f"total={self._tool_calls}, unique_resources={len(self._unique_resources)}, "
            f"base_key={base_key}, base_count={self._base_resource_counts.get(base_key, 0) if base_key else 0}"
        )

    def record_iteration(self, content_length: int) -> None:
        """Record an iteration completion.

        Args:
            content_length: Length of content produced in this iteration
        """
        self._iterations += 1

        if content_length < self.config.min_content_threshold:
            self._low_output_iterations += 1

        logger.debug(
            f"LoopDetector: iteration={self._iterations}, "
            f"content_length={content_length}, "
            f"low_output_count={self._low_output_iterations}"
        )

    def force_stop(self, reason: str) -> None:
        """Force the tracker to recommend stopping.

        Args:
            reason: Reason for forcing stop
        """
        self._forced_stop = reason

    def should_stop(self) -> StopReason:
        """Check if execution should stop.

        Returns:
            StopReason with should_stop flag, reason, and details
        """
        details = self._get_base_details()

        # Check forced stop
        if self._forced_stop:
            return StopReason(
                should_stop=True,
                reason=f"Manual stop: {self._forced_stop}",
                details=details,
            )

        # Check tool budget
        if self._tool_calls >= self.config.tool_budget:
            return StopReason(
                should_stop=True,
                reason=f"Tool budget exceeded ({self._tool_calls}/{self.config.tool_budget})",
                details=details,
            )

        # Check for true loop (repeated signatures)
        loop_result = self._check_loop()
        if loop_result:
            return StopReason(
                should_stop=True,
                reason=f"True loop detected: {loop_result}",
                details={**details, "loop_type": loop_result},
            )

        # Check iteration limits (hard limit regardless of task type)
        if self._iterations >= self.config.max_total_iterations:
            return StopReason(
                should_stop=True,
                reason=f"Max total iterations reached ({self._iterations}/{self.config.max_total_iterations})",
                details=details,
            )

        # Check research loop
        if self.task_type == TaskType.RESEARCH:
            if self._consecutive_research_calls >= self.config.max_iterations_research:
                return StopReason(
                    should_stop=True,
                    reason=f"Research loop detected ({self._consecutive_research_calls} consecutive research calls)",
                    details=details,
                )

        return StopReason(should_stop=False, reason="", details=details)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current progress metrics.

        Returns:
            Dictionary of metrics for logging/monitoring
        """
        return {
            "tool_calls": self._tool_calls,
            "iterations": self._iterations,
            "low_output_iterations": self._low_output_iterations,
            "unique_resources": len(self._unique_resources),
            "remaining_budget": self.remaining_budget,
            "progress_percentage": self.progress_percentage,
            "task_type": self.task_type.value,
        }

    def _get_signature(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate a signature for loop detection.

        For progressive tools, includes key parameters to distinguish
        different work from repeated identical calls.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            String signature for the tool call
        """
        if tool_name in PROGRESSIVE_PARAMS:
            # Include progressive params in signature
            params = PROGRESSIVE_PARAMS[tool_name]
            sig_parts = [tool_name]
            for param in params:
                value = arguments.get(param, "")
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100]
                sig_parts.append(str(value))
            return "|".join(sig_parts)
        else:
            # For other tools, hash the full arguments
            args_str = str(sorted(arguments.items()))
            return f"{tool_name}:{hashlib.md5(args_str.encode()).hexdigest()[:8]}"

    def _get_resource_key(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Generate a resource key for tracking unique resources.

        Includes offset/query details for granular tracking.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Resource key string or None if not trackable
        """
        if tool_name == "read_file":
            path = arguments.get("path", "")
            offset = arguments.get("offset", 0)
            if path:
                return f"file:{path}:{offset}"
        elif tool_name == "list_directory":
            path = arguments.get("path", "")
            if path:
                return f"dir:{path}"
        elif tool_name in {"code_search", "semantic_code_search"}:
            query = arguments.get("query", "")
            directory = arguments.get("directory", ".")
            if query:
                return f"search:{directory}:{query[:50]}"
        return None

    def _get_base_resource_key(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Generate a base resource key for same-resource loop detection.

        Ignores offset/limit to track repeated access to same file regardless of chunk.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Base resource key string or None if not trackable
        """
        if tool_name == "read_file":
            path = arguments.get("path", "")
            if path:
                return f"file:{path}"
        elif tool_name == "list_directory":
            path = arguments.get("path", "")
            if path:
                return f"dir:{path}"
        elif tool_name in {"code_search", "semantic_code_search"}:
            query = arguments.get("query", "")
            directory = arguments.get("directory", ".")
            if query:
                # Use first 20 chars as prefix to detect similar queries
                return f"search:{directory}:{query[:20]}"
        return None

    def _check_loop(self) -> Optional[str]:
        """Check for loop patterns in recent tool calls.

        Returns:
            Description of loop type if detected, None otherwise
        """
        # Check for same-file overuse (Gap 2 fix)
        for base_key, count in self._base_resource_counts.items():
            if base_key.startswith("file:") and count > self.config.max_reads_per_file:
                return f"same file read {count} times (max: {self.config.max_reads_per_file}): {base_key}"
            if base_key.startswith("search:") and count > self.config.max_searches_per_query_prefix:
                return f"similar search repeated {count} times (max: {self.config.max_searches_per_query_prefix}): {base_key}"

        if len(self._signature_history) < 3:
            return None

        # Get repeat threshold based on task type
        threshold = (
            self.config.repeat_threshold_analysis
            if self.task_type == TaskType.ANALYSIS
            else self.config.repeat_threshold_default
        )

        # Check for exact repeated signatures
        recent = list(self._signature_history)[-min(len(self._signature_history), 6) :]
        if recent:
            last_sig = recent[-1]
            repeat_count = sum(1 for s in recent if s == last_sig)
            if repeat_count >= threshold:
                return f"same signature repeated {repeat_count} times"

        return None

    def _get_max_iterations(self) -> int:
        """Get maximum iterations based on task type."""
        if self.task_type == TaskType.ANALYSIS:
            return self.config.max_iterations_analysis
        elif self.task_type == TaskType.ACTION:
            return self.config.max_iterations_action
        elif self.task_type == TaskType.RESEARCH:
            return self.config.max_iterations_research
        return self.config.max_iterations_default

    def _get_base_details(self) -> Dict[str, Any]:
        """Get base details for stop reason."""
        return {
            "tool_calls": self._tool_calls,
            "tool_budget": self.config.tool_budget,
            "iterations": self._iterations,
            "unique_resources": len(self._unique_resources),
            "task_type": self.task_type.value,
        }


def create_tracker_from_classification(
    classification: "TaskClassification",
    base_config: Optional[ProgressConfig] = None,
) -> Tuple[LoopDetector, str]:
    """Create a LoopDetector configured for a specific task classification.

    This factory function bridges the task classifier with loop detection,
    setting appropriate tool budgets based on the classified task complexity.

    Args:
        classification: TaskClassification from ComplexityClassifier
        base_config: Optional base configuration to override defaults

    Returns:
        Tuple of (LoopDetector, prompt_hint) where prompt_hint should be
        injected into the system prompt

    Example:
        from victor.agent.complexity_classifier import classify_task
        from victor.agent.loop_detector import create_tracker_from_classification

        classification = classify_task("List files in the directory")
        detector, hint = create_tracker_from_classification(classification)
        # Use detector in orchestrator loop
        # Inject hint into system prompt
    """
    from victor.agent.complexity_classifier import TaskComplexity

    # Map TaskComplexity to TaskType
    complexity_to_task_type = {
        TaskComplexity.SIMPLE: TaskType.DEFAULT,  # Simple uses default, but with lower budget
        TaskComplexity.MEDIUM: TaskType.DEFAULT,
        TaskComplexity.COMPLEX: TaskType.ANALYSIS,
        TaskComplexity.GENERATION: TaskType.ACTION,
    }

    task_type = complexity_to_task_type.get(classification.complexity, TaskType.DEFAULT)

    # Create config with appropriate budget
    config = base_config or ProgressConfig()
    config.tool_budget = classification.tool_budget

    # Adjust max iterations based on complexity
    if classification.complexity == TaskComplexity.SIMPLE:
        config.max_total_iterations = 3
    elif classification.complexity == TaskComplexity.GENERATION:
        config.max_total_iterations = 2

    tracker = LoopDetector(config=config, task_type=task_type)

    logger.debug(
        f"Created detector for {classification.complexity.value} task: "
        f"budget={classification.tool_budget}, type={task_type.value}"
    )

    return tracker, classification.prompt_hint


def classify_and_create_tracker(
    message: str,
    base_config: Optional[ProgressConfig] = None,
) -> Tuple[LoopDetector, str, "TaskClassification"]:
    """Convenience function to classify a message and create appropriate detector.

    Args:
        message: User message to classify
        base_config: Optional base configuration

    Returns:
        Tuple of (LoopDetector, prompt_hint, TaskClassification)

    Example:
        detector, hint, classification = classify_and_create_tracker(
            "List all Python files"
        )
        print(classification.complexity)  # SIMPLE
        print(detector.remaining_budget)   # 2
    """
    from victor.agent.complexity_classifier import classify_task

    classification = classify_task(message)
    tracker, hint = create_tracker_from_classification(classification, base_config)

    return tracker, hint, classification
