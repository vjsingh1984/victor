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

"""Tool loop detection for identifying repetitive tool call patterns.

This module provides enhanced loop detection to identify when LLMs get stuck
in repetitive tool calling patterns (e.g., reading the same file multiple times,
calling the same tool with identical arguments).

Design Pattern: Strategy + Observer
===================================
LoopDetector uses multiple detection strategies (Strategy Pattern) and
notifies observers (Observer Pattern) when loops are detected.

Detection Strategies:
1. Same-argument loops: Same tool + same args N times in a row
2. Cyclical patterns: A→B→A→B repeating cycles
3. Resource contention: Multiple reads of same resource without writes
4. Diminishing returns: Tool calls returning similar/duplicate results

Usage:
    detector = ToolLoopDetector()

    # Record tool executions
    result = detector.record_tool_call(
        tool_name="read_file",
        arguments={"path": "foo.py"},
        result_hash="abc123",
    )

    if result.loop_detected:
        print(f"Loop detected: {result.loop_type}")
        print(f"Recommendation: {result.recommendation}")

GAP-4 Fix:
    This addresses the issue seen with DeepSeek where it reads the same file
    repeatedly (e.g., `read|200|200|investor_homelab/utils/web_search_client.py`
    appearing 6+ times with `[loop] Warning: Approaching loop limit`).
"""

import hashlib
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# Progressive parameter normalization mapping
# Maps various parameter names to canonical forms for semantic comparison
PARAMETER_ALIASES: dict[str, str] = {
    # File path parameters
    "file_path": "path",
    "filepath": "path",
    "file": "path",
    "filename": "path",
    "file_name": "path",
    "target": "path",
    "source": "path",
    "directory": "path",
    "dir": "path",
    "folder": "path",
    # Line/offset parameters
    "start_line": "offset",
    "start": "offset",
    "begin": "offset",
    "from_line": "offset",
    "line_start": "offset",
    # End line parameters
    "end_line": "end_offset",
    "end": "end_offset",
    "to_line": "end_offset",
    "line_end": "end_offset",
    "limit": "end_offset",
    # Query/search parameters
    "search": "query",
    "pattern": "query",
    "regex": "query",
    "keyword": "query",
    "term": "query",
    # Content parameters
    "text": "content",
    "body": "content",
    "data": "content",
}

# Tool groups that access same resource types
TOOL_RESOURCE_GROUPS: dict[str, set[str]] = {
    "file_read": {
        "read",
        "read_file",
        "cat",
        "head",
        "tail",
        "view",
        "show",
        "display",
        "get_file",
    },
    "file_list": {
        "ls",
        "list",
        "list_directory",
        "dir",
        "tree",
        "find",
        "glob",
        "list_files",
    },
    "code_search": {
        "grep",
        "search",
        "code_search",
        "semantic_code_search",
        "find_in_files",
        "ripgrep",
        "ag",
    },
    "symbol_lookup": {
        "symbol",
        "get_symbol",
        "analyze_symbol",
        "find_symbol",
        "definition",
        "references",
        "go_to_definition",
    },
}


class LoopType(Enum):
    """Types of loop patterns detected."""

    NONE = auto()
    SAME_ARGUMENTS = auto()  # Same tool + same args consecutively
    CYCLICAL_PATTERN = auto()  # A→B→A→B repeating
    RESOURCE_CONTENTION = auto()  # Multiple reads without writes
    DIMINISHING_RETURNS = auto()  # Similar results repeatedly
    APPROACHING_LIMIT = auto()  # Approaching configured iteration limit


class LoopSeverity(Enum):
    """Severity levels for loop detection."""

    INFO = auto()  # Informational, may be intentional
    WARNING = auto()  # Approaching problematic patterns
    CRITICAL = auto()  # Definite loop, should break


@dataclass
class LoopDetectionResult:
    """Result of loop detection analysis."""

    loop_detected: bool = False
    loop_type: LoopType = LoopType.NONE
    severity: LoopSeverity = LoopSeverity.INFO
    consecutive_count: int = 0
    max_allowed: int = 0
    recommendation: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def should_warn(self) -> bool:
        """Whether this result warrants a warning."""
        return self.severity in (LoopSeverity.WARNING, LoopSeverity.CRITICAL)

    @property
    def should_break(self) -> bool:
        """Whether this loop is severe enough to suggest breaking."""
        return self.severity == LoopSeverity.CRITICAL


@dataclass
class ToolCallRecord:
    """Record of a single tool call for loop analysis."""

    tool_name: str
    arguments_hash: str
    result_hash: Optional[str] = None
    timestamp: float = 0.0
    resource_key: Optional[str] = None  # e.g., file path for read operations


@dataclass
class LoopDetectorConfig:
    """Configuration for the loop detector.

    Attributes:
        max_same_call_repetitions: Max times same tool+args before warning
        max_cyclical_repetitions: Max cyclical pattern repetitions
        warning_threshold_ratio: Ratio of max before warning (e.g., 0.75 = warn at 3/4)
        window_size: Number of recent calls to analyze
        resource_read_threshold: Max reads of same resource without modification
        enable_result_similarity: Check for similar results (diminishing returns)
        result_similarity_threshold: Hash similarity threshold for diminishing returns
    """

    max_same_call_repetitions: int = 4
    max_cyclical_repetitions: int = 3
    warning_threshold_ratio: float = 0.75
    window_size: int = 20
    resource_read_threshold: int = 4
    enable_result_similarity: bool = True
    result_similarity_threshold: float = 0.8


class LoopObserver(Protocol):
    """Protocol for loop detection observers."""

    def on_loop_detected(self, result: LoopDetectionResult) -> None:
        """Called when a loop is detected."""
        ...


class ToolLoopDetector:
    """Detects repetitive tool calling patterns.

    This class implements multiple detection strategies to identify
    when an LLM is stuck in a loop:

    1. Same-argument detection: Catches `read_file(path="foo.py")` x 5
    2. Cyclical detection: Catches A→B→A→B→A→B patterns
    3. Resource contention: Catches multiple reads without writes
    4. Diminishing returns: Catches similar results repeatedly

    Features:
    - Configurable thresholds
    - Observer pattern for notifications
    - Sliding window analysis
    - Resource-aware tracking
    - Progressive parameter normalization (maps file/path/filepath → path)
    - Tool grouping (ls/list_directory treated as same operation)
    """

    def __init__(self, config: Optional[LoopDetectorConfig] = None):
        """Initialize the loop detector.

        Args:
            config: Optional configuration
        """
        self.config = config or LoopDetectorConfig()

        # Recent tool calls (sliding window)
        self._call_history: deque[ToolCallRecord] = deque(maxlen=self.config.window_size)

        # Track consecutive same calls: (tool, args_hash) → count
        self._consecutive_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._last_call_key: Optional[tuple[str, str]] = None

        # Track resource access: resource_key → list of (tool, is_read)
        self._resource_access: dict[str, list[tuple[str, bool]]] = defaultdict(list)

        # Track result hashes for diminishing returns
        self._result_hashes: deque[str] = deque(maxlen=10)

        # Observers for loop notifications
        self._observers: list[LoopObserver] = []

        # Stats
        self._total_calls: int = 0
        self._loops_detected: int = 0

        logger.debug("ToolLoopDetector initialized")

    def add_observer(self, observer: LoopObserver) -> None:
        """Add an observer for loop detection events."""
        self._observers.append(observer)

    def remove_observer(self, observer: LoopObserver) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self, result: LoopDetectionResult) -> None:
        """Notify all observers of a loop detection."""
        for observer in self._observers:
            try:
                observer.on_loop_detected(result)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")

    def _normalize_parameter_name(self, key: str) -> str:
        """Normalize parameter name to canonical form.

        Maps various parameter names to a standard form:
        - file, filepath, file_path → path
        - start_line, start, begin → offset
        - query, search, pattern → query

        Args:
            key: Original parameter name

        Returns:
            Canonical parameter name
        """
        return PARAMETER_ALIASES.get(key.lower(), key.lower())

    def _normalize_path_value(self, value: Any) -> str:
        """Normalize path values for consistent comparison.

        Handles:
        - Absolute vs relative paths
        - Trailing slashes
        - Case normalization (on case-insensitive systems)

        Args:
            value: Path value

        Returns:
            Normalized path string
        """
        if not isinstance(value, str):
            return str(value)

        # Normalize the path
        normalized = os.path.normpath(value)

        # Remove leading ./ if present
        if normalized.startswith("./"):
            normalized = normalized[2:]

        return normalized

    def _normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Normalize arguments for semantic comparison.

        This applies:
        1. Parameter name normalization (file → path)
        2. Path value normalization (./foo.py → foo.py)
        3. Numeric coercion for line numbers

        Args:
            arguments: Original tool arguments

        Returns:
            Normalized arguments dictionary
        """
        normalized = {}
        for key, value in arguments.items():
            norm_key = self._normalize_parameter_name(key)

            # Normalize path values
            if norm_key == "path":
                value = self._normalize_path_value(value)

            # Normalize numeric values (line numbers, offsets)
            if norm_key in ("offset", "end_offset", "count", "limit"):
                try:
                    value = int(value) if value is not None else 0
                except (ValueError, TypeError):
                    pass

            normalized[norm_key] = value

        return normalized

    def _get_tool_group(self, tool_name: str) -> Optional[str]:
        """Get the tool group for a given tool name.

        Tools in the same group access resources similarly and should
        be considered together for resource contention detection.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool group name or None if not in a group
        """
        tool_lower = tool_name.lower()
        for group_name, tools in TOOL_RESOURCE_GROUPS.items():
            if tool_lower in tools:
                return group_name
        return None

    def _hash_arguments(self, arguments: dict[str, Any]) -> str:
        """Create a stable hash of tool arguments.

        Uses normalized arguments for semantic equivalence detection.
        For example, read(file="foo.py") and read(path="foo.py")
        will produce the same hash.
        """
        # Normalize arguments before hashing
        normalized = self._normalize_arguments(arguments)
        # Sort keys for consistent hashing
        sorted_items = sorted(normalized.items())
        content = str(sorted_items).encode("utf-8")
        # MD5 used for tool loop detection, not security
        return hashlib.md5(content, usedforsecurity=False).hexdigest()[:12]

    def _extract_resource_key(self, tool_name: str, arguments: dict[str, Any]) -> Optional[str]:
        """Extract resource key from tool call (e.g., file path).

        Uses normalized arguments to extract a canonical resource key,
        so both read(file="foo.py") and read(path="foo.py") produce
        the same resource key.
        """
        # Normalize arguments first
        normalized = self._normalize_arguments(arguments)

        # Check for path in normalized arguments
        if "path" in normalized:
            return self._normalize_path_value(str(normalized["path"]))

        # For search tools, use query as resource key
        if "query" in normalized:
            tool_group = self._get_tool_group(tool_name)
            if tool_group == "code_search":
                return f"search:{normalized['query']}"

        return None

    def _is_read_operation(self, tool_name: str) -> bool:
        """Determine if a tool is a read-only operation."""
        read_tools = {
            "read_file",
            "read",
            "list_directory",
            "ls",
            "code_search",
            "semantic_code_search",
            "grep",
            "symbol",
            "get_symbol",
            "analyze_symbol",
            "plan_files",
            "graph_symbol",
            "graph_dependencies",
        }
        return tool_name.lower() in read_tools

    def _detect_same_argument_loop(self, tool_name: str, args_hash: str) -> LoopDetectionResult:
        """Detect consecutive calls with same tool and arguments."""
        call_key = (tool_name, args_hash)

        if call_key == self._last_call_key:
            self._consecutive_counts[call_key] += 1
        else:
            # Reset counter for new call pattern
            self._consecutive_counts[call_key] = 1

        self._last_call_key = call_key
        count = self._consecutive_counts[call_key]
        max_allowed = self.config.max_same_call_repetitions

        # Check if approaching or exceeding limit
        warning_threshold = int(max_allowed * self.config.warning_threshold_ratio)

        if count >= max_allowed:
            return LoopDetectionResult(
                loop_detected=True,
                loop_type=LoopType.SAME_ARGUMENTS,
                severity=LoopSeverity.CRITICAL,
                consecutive_count=count,
                max_allowed=max_allowed,
                recommendation=(
                    f"Tool '{tool_name}' called {count} times with identical arguments. "
                    f"Consider: (1) using a different approach, (2) checking if the "
                    f"previous result was sufficient, or (3) modifying the arguments."
                ),
                details={"tool": tool_name, "repetitions": count},
            )
        elif count >= warning_threshold:
            return LoopDetectionResult(
                loop_detected=True,
                loop_type=LoopType.APPROACHING_LIMIT,
                severity=LoopSeverity.WARNING,
                consecutive_count=count,
                max_allowed=max_allowed,
                recommendation=(
                    f"Approaching loop limit ({count}/{max_allowed}): "
                    f"'{tool_name}' with same arguments. Consider varying your approach."
                ),
                details={"tool": tool_name, "repetitions": count},
            )

        return LoopDetectionResult()

    def _detect_cyclical_pattern(self) -> LoopDetectionResult:
        """Detect A→B→A→B style cyclical patterns."""
        if len(self._call_history) < 4:
            return LoopDetectionResult()

        # Get recent tool names
        recent_tools = [r.tool_name for r in list(self._call_history)[-8:]]

        # Check for 2-element cycles (A→B→A→B)
        if len(recent_tools) >= 4:
            # Check if last 4 form A→B→A→B
            if (
                recent_tools[-4] == recent_tools[-2]
                and recent_tools[-3] == recent_tools[-1]
                and recent_tools[-4] != recent_tools[-3]
            ):

                cycle = f"{recent_tools[-4]}→{recent_tools[-3]}"

                # Count how many times this cycle repeats
                cycle_count = 1
                for i in range(len(recent_tools) - 4, -1, -2):
                    if (
                        i >= 1
                        and recent_tools[i] == recent_tools[-2]
                        and recent_tools[i - 1] == recent_tools[-1]
                    ):
                        cycle_count += 1
                    else:
                        break

                max_allowed = self.config.max_cyclical_repetitions
                if cycle_count >= max_allowed:
                    return LoopDetectionResult(
                        loop_detected=True,
                        loop_type=LoopType.CYCLICAL_PATTERN,
                        severity=LoopSeverity.WARNING,
                        consecutive_count=cycle_count,
                        max_allowed=max_allowed,
                        recommendation=(
                            f"Detected cyclical pattern: {cycle} repeated {cycle_count}x. "
                            f"Consider breaking the cycle with a different tool or approach."
                        ),
                        details={"cycle": cycle, "repetitions": cycle_count},
                    )

        return LoopDetectionResult()

    def _detect_resource_contention(
        self, resource_key: Optional[str], tool_name: str
    ) -> LoopDetectionResult:
        """Detect multiple reads of same resource without writes."""
        if not resource_key:
            return LoopDetectionResult()

        is_read = self._is_read_operation(tool_name)
        self._resource_access[resource_key].append((tool_name, is_read))

        # Only keep recent access history per resource
        if len(self._resource_access[resource_key]) > 10:
            self._resource_access[resource_key] = self._resource_access[resource_key][-10:]

        # Count consecutive reads since last write
        access_list = self._resource_access[resource_key]
        consecutive_reads = 0
        for tool, is_read_op in reversed(access_list):
            if is_read_op:
                consecutive_reads += 1
            else:
                break  # Found a write, stop counting

        threshold = self.config.resource_read_threshold
        if consecutive_reads >= threshold:
            return LoopDetectionResult(
                loop_detected=True,
                loop_type=LoopType.RESOURCE_CONTENTION,
                severity=LoopSeverity.WARNING,
                consecutive_count=consecutive_reads,
                max_allowed=threshold,
                recommendation=(
                    f"Resource '{resource_key}' read {consecutive_reads}x without "
                    f"modification. Consider: (1) caching the result, (2) proceeding "
                    f"with the information you have, or (3) searching for different data."
                ),
                details={"resource": resource_key, "read_count": consecutive_reads},
            )

        return LoopDetectionResult()

    def _detect_diminishing_returns(self, result_hash: Optional[str]) -> LoopDetectionResult:
        """Detect when tool results are becoming similar/identical."""
        if not result_hash or not self.config.enable_result_similarity:
            return LoopDetectionResult()

        self._result_hashes.append(result_hash)

        if len(self._result_hashes) < 3:
            return LoopDetectionResult()

        # Count identical result hashes in recent history
        recent_hashes = list(self._result_hashes)[-5:]
        identical_count = recent_hashes.count(result_hash)

        if identical_count >= 3:
            return LoopDetectionResult(
                loop_detected=True,
                loop_type=LoopType.DIMINISHING_RETURNS,
                severity=LoopSeverity.WARNING,
                consecutive_count=identical_count,
                max_allowed=3,
                recommendation=(
                    f"Last {identical_count} tool results are identical. "
                    f"This suggests the approach isn't yielding new information. "
                    f"Consider trying a different strategy."
                ),
                details={"identical_results": identical_count},
            )

        return LoopDetectionResult()

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_hash: Optional[str] = None,
        timestamp: float = 0.0,
    ) -> LoopDetectionResult:
        """Record a tool call and check for loop patterns.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments
            result_hash: Optional hash of the tool result
            timestamp: Optional execution timestamp

        Returns:
            LoopDetectionResult with analysis
        """
        self._total_calls += 1

        args_hash = self._hash_arguments(arguments)
        resource_key = self._extract_resource_key(tool_name, arguments)

        # Create record
        record = ToolCallRecord(
            tool_name=tool_name,
            arguments_hash=args_hash,
            result_hash=result_hash,
            timestamp=timestamp,
            resource_key=resource_key,
        )
        self._call_history.append(record)

        # Run all detection strategies
        results: list[LoopDetectionResult] = []

        # Strategy 1: Same-argument detection
        results.append(self._detect_same_argument_loop(tool_name, args_hash))

        # Strategy 2: Cyclical pattern detection
        results.append(self._detect_cyclical_pattern())

        # Strategy 3: Resource contention detection
        results.append(self._detect_resource_contention(resource_key, tool_name))

        # Strategy 4: Diminishing returns detection
        results.append(self._detect_diminishing_returns(result_hash))

        # Return the most severe result
        detected_results = [r for r in results if r.loop_detected]
        if detected_results:
            # Sort by severity (CRITICAL > WARNING > INFO)
            severity_order = {
                LoopSeverity.CRITICAL: 0,
                LoopSeverity.WARNING: 1,
                LoopSeverity.INFO: 2,
            }
            detected_results.sort(key=lambda r: severity_order[r.severity])
            worst_result = detected_results[0]

            self._loops_detected += 1
            self._notify_observers(worst_result)

            logger.debug(
                f"Loop detected: {worst_result.loop_type.name} "
                f"(severity: {worst_result.severity.name})"
            )

            return worst_result

        return LoopDetectionResult()

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics.

        Returns:
            Dictionary with detector stats
        """
        return {
            "total_calls": self._total_calls,
            "loops_detected": self._loops_detected,
            "loop_rate": (
                self._loops_detected / self._total_calls if self._total_calls > 0 else 0.0
            ),
            "history_length": len(self._call_history),
            "unique_resources_tracked": len(self._resource_access),
            "active_consecutive_patterns": len(
                [c for c in self._consecutive_counts.values() if c > 1]
            ),
        }

    def reset(self) -> None:
        """Reset all detector state."""
        self._call_history.clear()
        self._consecutive_counts.clear()
        self._last_call_key = None
        self._resource_access.clear()
        self._result_hashes.clear()
        self._total_calls = 0
        self._loops_detected = 0

        logger.debug("ToolLoopDetector reset")

    def clear_history(self) -> None:
        """Clear history but keep configuration."""
        self._call_history.clear()
        self._consecutive_counts.clear()
        self._last_call_key = None
        self._resource_access.clear()
        self._result_hashes.clear()


def create_loop_detector(
    max_repetitions: int = 4,
    window_size: int = 20,
) -> ToolLoopDetector:
    """Factory function to create a configured loop detector.

    Args:
        max_repetitions: Maximum same-argument repetitions before critical
        window_size: Number of calls to track

    Returns:
        Configured ToolLoopDetector instance
    """
    config = LoopDetectorConfig(
        max_same_call_repetitions=max_repetitions,
        window_size=window_size,
    )
    return ToolLoopDetector(config)


class LoggingLoopObserver:
    """Observer that logs loop detection events."""

    def __init__(self, logger_name: str = __name__):
        self._logger = logging.getLogger(logger_name)

    def on_loop_detected(self, result: LoopDetectionResult) -> None:
        """Log the loop detection result."""
        if result.severity == LoopSeverity.CRITICAL:
            self._logger.warning(
                f"[loop] CRITICAL: {result.loop_type.name} - " f"{result.recommendation}"
            )
        elif result.severity == LoopSeverity.WARNING:
            self._logger.warning(
                f"[loop] Warning: {result.loop_type.name} - "
                f"({result.consecutive_count}/{result.max_allowed}): "
                f"{result.recommendation}"
            )
