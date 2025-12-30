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

.. deprecated::
    This module is deprecated. Use `victor.agent.unified_task_tracker.UnifiedTaskTracker`
    instead, which consolidates TaskMilestoneMonitor and LoopDetector into a single
    unified system.

This module provides loop detection and tool budget enforcement:
- Tool call counting and budget enforcement
- True loop detection (repeated signatures)
- Progress tracking (unique files read)
- Task-type aware thresholds (analysis vs action vs default)
- Research loop detection
- **Unified offset-aware file read tracking** - allows paginated reading
  of large files while catching true loops (same region read repeatedly)

The primary class is `LoopDetector`.

For goal-aware milestone tracking, see `victor.agent.milestone_monitor.TaskMilestoneMonitor`.

Design principles:
- Single source of truth for loop detection state
- Clear, testable interface
- Task-type aware configuration
- Extensible for future needs
- **Offset-aware detection** - reads to different parts of the same file
  are NOT counted as loops; only overlapping reads trigger detection

"""

import hashlib
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from victor.tools.tool_names import ToolNames, get_canonical_name
from victor.tools.metadata_registry import get_progress_params as registry_get_progress_params

# Import native extensions with fallback
try:
    from victor.processing.native import (
        compute_signature as native_compute_signature,
        is_native_available,
    )

    _NATIVE_AVAILABLE = is_native_available()
except ImportError:
    _NATIVE_AVAILABLE = False

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


def get_progress_params_for_tool(tool_name: str) -> List[str]:
    """Get progress parameters for a tool from the decorator-driven registry.

    Progress parameters indicate exploration progress (e.g., different query,
    different file offset) rather than loops. Tools define these via
    @tool(progress_params=["path", "offset"]) decorator.

    Args:
        tool_name: Name of the tool

    Returns:
        List of parameter names that indicate progress, or empty list
    """
    registry_params = registry_get_progress_params(tool_name)
    return list(registry_params) if registry_params else []


def is_progressive_tool(tool_name: str) -> bool:
    """Check if a tool has progress parameters defined in the registry.

    Args:
        tool_name: Name of the tool

    Returns:
        True if tool has progress params defined via @tool decorator
    """
    return bool(registry_get_progress_params(tool_name))


# Default limit for read_file when not specified (matches tool default)
DEFAULT_READ_LIMIT = 500

# Canonical tool name for file reading (from centralized registry)
# The loop detector normalizes tool names using get_canonical_name() before checking
CANONICAL_READ_TOOL = ToolNames.READ


@dataclass
class FileReadRange:
    """Represents a range of lines read from a file.

    Used for smart overlap detection in loop prevention.
    Two reads are considered duplicates only if they access
    overlapping line ranges.
    """

    offset: int
    limit: int

    @property
    def end(self) -> int:
        """End line (exclusive) of this range."""
        return self.offset + self.limit

    def overlaps(self, other: "FileReadRange") -> bool:
        """Check if this range overlaps with another.

        Args:
            other: Another FileReadRange to compare against

        Returns:
            True if ranges overlap, False otherwise
        """
        # Ranges overlap if neither is completely before the other
        return self.offset < other.end and other.offset < self.end

    def __hash__(self) -> int:
        return hash((self.offset, self.limit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileReadRange):
            return False
        return self.offset == other.offset and self.limit == other.limit


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
    repeat_threshold_action: int = 4  # Action tasks need more flexibility for file ops
    signature_history_size: int = 10
    signature_lookback_for_warning: int = 3  # Check last N sigs when clearing warning

    # Progress detection
    min_content_threshold: int = 150

    # Hard limits - increased to allow complex multi-step tasks
    # Loop detection will catch actual loops; this is just a safety net
    max_total_iterations: int = 50

    # Same-resource access limits (unified offset-aware tracking)
    # For file reads: only counts reads with overlapping offset ranges as duplicates
    # This allows paginated reading of large files while catching true loops
    max_overlapping_reads_per_file: int = 3
    max_searches_per_query_prefix: int = 2


@dataclass
class StopReason:
    """Reason why progress tracker recommends stopping.

    Attributes:
        should_stop: Whether execution should stop
        reason: Human-readable reason for stopping
        details: Dictionary with additional context for logging/debugging
        is_warning: Whether this is a soft warning (not yet a hard stop)
    """

    should_stop: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    is_warning: bool = False  # True for soft warnings before hard stop


class LoopDetector:
    """Loop detection and budget enforcement for the orchestrator.

    This class handles reactive loop detection and budget enforcement:
    - Tool call counting and budget enforcement
    - True loop detection (repeated signatures)
    - **Content loop detection (repeated text phrases in streaming)**
    - Progress tracking (unique files read)
    - Task-type aware thresholds
    - Research loop detection

    For goal-aware milestone tracking and proactive nudging, see TaskMilestoneMonitor.

    Usage:
        detector = LoopDetector(task_type=TaskType.ANALYSIS)

        # In the main loop:
        detector.record_tool_call("read_file", {"path": "test.py"})
        detector.record_iteration(content_length=len(response))

        # For streaming content loop detection:
        detector.record_content_chunk("Let me analyze...")
        if detector.check_content_loop():
            # Content loop detected, stop streaming
            break

        if detector.should_stop().should_stop:
            # Force completion
            break

    """

    # Default content loop detection settings
    CONTENT_PHRASE_MIN_LENGTH = 15  # Minimum phrase length to track
    CONTENT_PHRASE_MAX_LENGTH = 80  # Maximum phrase length to track
    CONTENT_REPEAT_THRESHOLD = 5  # Number of repetitions to detect loop
    CONTENT_BUFFER_SIZE = 5000  # Rolling buffer size for content analysis

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

        # File read range tracking (unified offset-aware detection)
        # Key: file path, Value: list of FileReadRange objects
        # Counts overlapping reads as duplicates, allows non-overlapping pagination
        self._file_read_ranges: Dict[str, List[FileReadRange]] = {}

        # Base resource counting for non-file resources (searches, directories)
        # Key: base resource (e.g., "search:.:query_prefix"), Value: count
        self._base_resource_counts: Counter[str] = Counter()

        # Loop detection
        self._signature_history: deque = deque(maxlen=self.config.signature_history_size)
        self._consecutive_research_calls = 0
        self._loop_warning_given: bool = False  # Track if we've warned about impending loop
        self._warned_signature: Optional[str] = None  # The signature we warned about

        # Content loop detection (for streaming "thinking" loops)
        self._content_buffer: str = ""
        self._content_loop_detected: bool = False
        self._content_loop_phrase: Optional[str] = None

        # Manual stop
        self._forced_stop: Optional[str] = None

    def reset(self) -> None:
        """Reset all state for a new conversation turn."""
        self._tool_calls = 0
        self._iterations = 0
        self._low_output_iterations = 0
        self._unique_resources.clear()
        self._file_read_ranges.clear()
        self._base_resource_counts.clear()
        self._signature_history.clear()
        self._consecutive_research_calls = 0
        self._loop_warning_given = False
        self._warned_signature = None
        self._content_buffer = ""
        self._content_loop_detected = False
        self._content_loop_phrase = None
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

        # Track file reads with offset-aware overlap detection
        # Normalize tool name to canonical form (e.g., "read_file" → "read")
        canonical_name = get_canonical_name(tool_name)
        if canonical_name == CANONICAL_READ_TOOL:
            self._track_file_read(arguments)
        else:
            # Track base resources for non-file operations (searches, etc.)
            base_key = self._get_base_resource_key(tool_name, arguments)
            if base_key:
                self._base_resource_counts[base_key] += 1

        # Track signature for loop detection
        signature = self._get_signature(tool_name, arguments)

        # Clear loop warning only if signature is TRULY different
        # Check against last N signatures to prevent gaming by alternating (A, B, A, B...)
        if self._loop_warning_given and self._signature_history:
            lookback = self.config.signature_lookback_for_warning
            recent_sigs = list(self._signature_history)[-lookback:]
            if signature not in recent_sigs:
                logger.debug(
                    f"LoopDetector: Signature not in last {lookback} history, clearing loop warning"
                )
                self._loop_warning_given = False

        self._signature_history.append(signature)

        # Track research calls
        if tool_name in RESEARCH_TOOLS:
            self._consecutive_research_calls += 1
        else:
            self._consecutive_research_calls = 0

        logger.debug(
            f"LoopDetector: tool_call={tool_name}, "
            f"total={self._tool_calls}, unique_resources={len(self._unique_resources)}"
        )

    def _track_file_read(self, arguments: Dict[str, Any]) -> None:
        """Track a file read operation with offset-aware overlap detection.

        Args:
            arguments: Arguments from the read_file tool call
        """
        path = arguments.get("path", "")
        if not path:
            return

        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", DEFAULT_READ_LIMIT)
        new_range = FileReadRange(offset=offset, limit=limit)

        if path not in self._file_read_ranges:
            self._file_read_ranges[path] = []

        self._file_read_ranges[path].append(new_range)

        # Count overlapping reads
        overlapping_count = self._count_overlapping_reads(path, new_range)

        logger.debug(
            f"LoopDetector: file_read path={path}, offset={offset}, limit={limit}, "
            f"overlapping_reads={overlapping_count}, total_reads={len(self._file_read_ranges[path])}"
        )

    def _count_overlapping_reads(self, path: str, current_range: FileReadRange) -> int:
        """Count how many previous reads to this file overlap with the current range.

        Args:
            path: File path being read
            current_range: The current read range to check against

        Returns:
            Number of overlapping reads (including the current one)
        """
        if path not in self._file_read_ranges:
            return 0

        count = 0
        for prev_range in self._file_read_ranges[path]:
            if prev_range.overlaps(current_range):
                count += 1
        return count

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
            "files_read": len(self._file_read_ranges),
            "total_file_reads": sum(len(ranges) for ranges in self._file_read_ranges.values()),
            "remaining_budget": self.remaining_budget,
            "progress_percentage": self.progress_percentage,
            "task_type": self.task_type.value,
        }

    def _get_signature(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate a signature for loop detection.

        Uses native Rust implementation when available for ~10x speedup.

        For progressive tools, includes key parameters to distinguish
        different work from repeated identical calls.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            String signature for the tool call
        """
        # Check for progress params (registry with static fallback)
        progress_params = get_progress_params_for_tool(tool_name)
        if progress_params:
            # Include progressive params in signature
            sig_parts = [tool_name]
            for param in progress_params:
                value = arguments.get(param, "")
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100]
                sig_parts.append(str(value))
            return "|".join(sig_parts)
        else:
            # For other tools, hash the full arguments
            if _NATIVE_AVAILABLE:
                # Use native Rust xxHash3 implementation
                return native_compute_signature(tool_name, arguments)
            else:
                # Python fallback with MD5
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
        # Normalize tool name to canonical form
        canonical_name = get_canonical_name(tool_name)
        if canonical_name == CANONICAL_READ_TOOL:
            path = arguments.get("path", "")
            offset = arguments.get("offset", 0)
            if path:
                return f"file:{path}:{offset}"
        elif canonical_name == ToolNames.LS:
            path = arguments.get("path", "")
            if path:
                return f"dir:{path}"
        elif canonical_name in {ToolNames.GREP, ToolNames.CODE_SEARCH}:
            query = arguments.get("query", "")
            directory = arguments.get("directory", ".")
            if query:
                return f"search:{directory}:{query[:50]}"
        elif canonical_name == ToolNames.SHELL:
            # Track bash commands as resources to show progress
            command = arguments.get("command", "")
            if command:
                # Extract first 50 chars of command for unique identification
                return f"bash:{command[:50]}"
        elif canonical_name in {ToolNames.WEB_FETCH, ToolNames.WEB_SEARCH, ToolNames.SUMMARIZE}:
            # Track web operations as resources (important for ACTION tasks)
            url = arguments.get("url", "")
            query = arguments.get("query", "")
            if url:
                return f"web:{canonical_name}:{url[:80]}"
            elif query:
                return f"web:{canonical_name}:{query[:50]}"
        return None

    def _get_base_resource_key(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Generate a base resource key for same-resource loop detection.

        Note: read_file is handled separately with offset-aware overlap detection
        in _track_file_read() and _check_file_read_loops().

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Base resource key string or None if not trackable
        """
        # Note: read is NOT tracked here - uses unified offset-aware detection
        canonical_name = get_canonical_name(tool_name)
        if canonical_name == ToolNames.LS:
            path = arguments.get("path", "")
            if path:
                return f"dir:{path}"
        elif canonical_name in {ToolNames.GREP, ToolNames.CODE_SEARCH}:
            query = arguments.get("query", "")
            directory = arguments.get("directory", ".")
            if query:
                # Use first 20 chars as prefix to detect similar queries
                return f"search:{directory}:{query[:20]}"
        elif canonical_name == ToolNames.SHELL:
            # Track bash command base (first word/command) for loop detection
            command = arguments.get("command", "")
            if command:
                # Extract the base command (first word or git subcommand)
                parts = command.strip().split()
                if parts:
                    base_cmd = parts[0]
                    if base_cmd == "git" and len(parts) > 1:
                        base_cmd = f"git {parts[1]}"
                    return f"bash:{base_cmd}"
        return None

    def _check_loop(self) -> Optional[str]:
        """Check for loop patterns in recent tool calls.

        Returns:
            Description of loop type if detected, None otherwise
        """
        # Check for overlapping file reads (unified offset-aware detection)
        # Only counts reads to the SAME offset range as loops, allows pagination
        file_loop = self._check_file_read_loops()
        if file_loop:
            return file_loop

        # Check for similar search queries (non-file resources)
        for base_key, count in self._base_resource_counts.items():
            if base_key.startswith("search:") and count > self.config.max_searches_per_query_prefix:
                return f"similar search repeated {count} times (max: {self.config.max_searches_per_query_prefix}): {base_key}"

        if len(self._signature_history) < 3:
            return None

        # Get repeat threshold based on task type
        if self.task_type == TaskType.ANALYSIS:
            threshold = self.config.repeat_threshold_analysis
        elif self.task_type == TaskType.ACTION:
            threshold = self.config.repeat_threshold_action
        else:
            threshold = self.config.repeat_threshold_default

        # Check for repeated signatures (including alternating patterns like A,B,A,B)
        # Look at ALL unique signatures in recent history, not just the last one
        recent = list(self._signature_history)[-min(len(self._signature_history), 8) :]
        if recent:
            # Count occurrences of each signature
            from collections import Counter

            sig_counts = Counter(recent)
            for sig, count in sig_counts.items():
                if count >= threshold:
                    return f"same signature repeated {count} times: {sig[:50]}"

        return None

    def check_loop_warning(self) -> Optional[str]:
        """Check if we're approaching loop threshold (warning at threshold-1).

        This is a soft warning to give the model a chance to correct behavior
        before the hard stop at threshold.

        Returns:
            Warning message if at threshold-1, None otherwise.
            Only returns once per loop pattern (tracked by _loop_warning_given).
        """
        if self._loop_warning_given:
            return None  # Already warned

        if len(self._signature_history) < 3:
            return None

        # Get repeat threshold based on task type
        if self.task_type == TaskType.ANALYSIS:
            threshold = self.config.repeat_threshold_analysis
        elif self.task_type == TaskType.ACTION:
            threshold = self.config.repeat_threshold_action
        else:
            threshold = self.config.repeat_threshold_default

        # Check for approaching repeated signatures (threshold - 1)
        # Look at ALL unique signatures (catches alternating patterns like A,B,A,B)
        recent = list(self._signature_history)[-min(len(self._signature_history), 8) :]
        if recent:
            from collections import Counter

            sig_counts = Counter(recent)
            for sig, count in sig_counts.items():
                if count == threshold - 1:
                    self._loop_warning_given = True
                    self._warned_signature = sig  # Track which signature we warned about
                    logger.info(f"LoopDetector: Setting warned_signature to {sig[:50]}")
                    return f"approaching loop threshold ({count}/{threshold}): {sig[:80]}"

        return None

    def is_blocked_after_warning(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Check if a proposed tool call is blocked because it matches the warned signature.

        After giving a loop warning, if the model tries the SAME operation again,
        block it immediately instead of waiting for the hard loop detection.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments

        Returns:
            Block message if the call matches warned signature, None if allowed.
        """
        if not self._loop_warning_given or not self._warned_signature:
            return None

        # Check if proposed signature matches the warned signature
        proposed_sig = self._get_signature(tool_name, arguments)
        if proposed_sig == self._warned_signature:
            logger.warning(
                f"LoopDetector: Blocking repeated signature after warning: {proposed_sig[:50]}"
            )
            return (
                f"blocked after warning: same operation attempted ({proposed_sig[:50]}). "
                "Try a different approach."
            )

        return None

    def clear_loop_warning(self) -> None:
        """Clear the loop warning flag when model does something different."""
        self._loop_warning_given = False
        self._warned_signature = None

    def _check_file_read_loops(self) -> Optional[str]:
        """Check for file read loops using offset-aware overlap detection.

        Only flags as a loop when the same file region is read multiple times.
        Paginated reads (different offset ranges) are allowed.

        Returns:
            Description of loop if detected, None otherwise
        """
        for path, ranges in self._file_read_ranges.items():
            if len(ranges) < 2:
                continue

            # For each range, count how many other ranges overlap with it
            for i, current_range in enumerate(ranges):
                overlap_count = 0
                for j, other_range in enumerate(ranges):
                    if i != j and current_range.overlaps(other_range):
                        overlap_count += 1

                # +1 to include the current range itself
                total_overlapping = overlap_count + 1
                if total_overlapping > self.config.max_overlapping_reads_per_file:
                    return (
                        f"same file region read {total_overlapping} times "
                        f"(max: {self.config.max_overlapping_reads_per_file}): "
                        f"file:{path} [offset={current_range.offset}, limit={current_range.limit}]"
                    )

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

    # ================================================================
    # Content Loop Detection (for streaming "thinking" loops)
    # ================================================================

    def record_content_chunk(self, chunk: str) -> None:
        """Record a content chunk for loop detection during streaming.

        This method should be called for each content chunk received during
        streaming to detect repetitive "thinking" loops common in reasoning
        models like DeepSeek R1.

        Args:
            chunk: Content chunk from the streaming response
        """
        if not chunk:
            return

        # Append to rolling buffer
        self._content_buffer += chunk

        # Trim buffer if too large (keep last CONTENT_BUFFER_SIZE chars)
        if len(self._content_buffer) > self.CONTENT_BUFFER_SIZE:
            self._content_buffer = self._content_buffer[-self.CONTENT_BUFFER_SIZE :]

    def check_content_loop(self) -> Optional[str]:
        """Check if content shows a repetitive loop pattern.

        Returns:
            Description of the loop if detected, None otherwise
        """
        if self._content_loop_detected:
            return f"Content loop already detected: {self._content_loop_phrase}"

        if (
            len(self._content_buffer)
            < self.CONTENT_PHRASE_MIN_LENGTH * self.CONTENT_REPEAT_THRESHOLD
        ):
            return None

        # Look for repeated phrases in the content buffer
        detected = self._find_repeated_phrase()
        if detected:
            self._content_loop_detected = True
            self._content_loop_phrase = detected
            logger.warning(f"Content loop detected: '{detected[:50]}...' repeated excessively")
            return f"Repetitive content detected: '{detected[:50]}...'"

        return None

    def _find_repeated_phrase(self) -> Optional[str]:
        """Find repeated phrases in the content buffer.

        Uses a sliding window approach to detect phrases that appear
        multiple times in succession.

        Returns:
            The repeated phrase if found, None otherwise
        """
        text = self._content_buffer

        # Try different phrase lengths, starting with longer phrases
        # (longer = more specific = better detection)
        for phrase_len in range(
            self.CONTENT_PHRASE_MAX_LENGTH,
            self.CONTENT_PHRASE_MIN_LENGTH - 1,
            -5,  # Step down by 5 chars
        ):
            if len(text) < phrase_len * self.CONTENT_REPEAT_THRESHOLD:
                continue

            # Extract candidate phrases and count occurrences
            # Use a sliding window with step size to avoid excessive computation
            step = max(1, phrase_len // 4)
            phrase_counts: Counter[str] = Counter()

            for i in range(0, len(text) - phrase_len, step):
                phrase = text[i : i + phrase_len]
                # Skip phrases that are mostly whitespace or punctuation
                if not self._is_meaningful_phrase(phrase):
                    continue
                phrase_counts[phrase] += 1

            # Check if any phrase exceeds the repeat threshold
            for phrase, count in phrase_counts.most_common(5):
                if count >= self.CONTENT_REPEAT_THRESHOLD:
                    return phrase

        return None

    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if a phrase is meaningful (not just whitespace/punctuation).

        Args:
            phrase: The phrase to check

        Returns:
            True if the phrase is meaningful
        """
        # Count alphanumeric characters
        alpha_count = sum(1 for c in phrase if c.isalnum())
        # Phrase should be at least 50% alphanumeric
        return alpha_count >= len(phrase) * 0.5

    @property
    def content_loop_detected(self) -> bool:
        """Whether a content loop has been detected."""
        return self._content_loop_detected

    @property
    def content_loop_phrase(self) -> Optional[str]:
        """The repeated phrase if a content loop was detected."""
        return self._content_loop_phrase

    def reset_content_tracking(self) -> None:
        """Reset just the content tracking state (for new response)."""
        self._content_buffer = ""
        self._content_loop_detected = False
        self._content_loop_phrase = None


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
