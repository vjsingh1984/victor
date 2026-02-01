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

"""Tool call deduplication tracker to prevent redundant operations.

This module prevents redundant tool calls by tracking recent operations
and detecting when tools are called with overlapping intent.

Examples of redundancy:
- grep(mode='semantic', query='tool registration') followed by grep(mode='regex', pattern='register_tool')
- read_file(path='foo.py') followed by read_file(path='foo.py')
- list_directory(path='.') followed by list_directory(path='.')
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.agent.protocols import ToolDeduplicationTrackerProtocol

logger = logging.getLogger(__name__)


@dataclass
class TrackedToolCall:
    """Record of a tool call for deduplication tracking.

    Different from victor.agent.tool_calling.base.ToolCall - this is specifically
    for tracking recent calls to detect redundant operations.
    """

    tool_name: str
    args: dict[str, Any]
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class ToolDeduplicationTracker:
    """Tracks recent tool calls to detect and prevent redundant operations.

    Uses semantic similarity and argument overlap to identify redundant calls.
    """

    def __init__(self, window_size: int = 10, similarity_threshold: float = 0.7):
        """Initialize deduplication tracker.

        Args:
            window_size: Number of recent tool calls to track
            similarity_threshold: Threshold for considering calls similar (0.0-1.0)
        """
        from victor.core.utils.content_hasher import ContentHasher

        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.recent_calls: deque[TrackedToolCall] = deque(maxlen=window_size)
        # Use ContentHasher for consistent hashing across components
        # Tool calls need exact matching (no normalization) for precise deduplication
        self._hasher = ContentHasher(
            normalize_whitespace=False,
            case_insensitive=False,
            hash_length=16,
        )

        # Query synonyms for semantic matching
        self.query_synonyms: dict[str, set[str]] = {
            "tool registration": {
                "register tool",
                "tool registry",
                "@tool",
                "register_tool",
                "ToolRegistry",
            },
            "provider": {"llm provider", "model provider", "baseprovider"},
            "error handling": {"exception", "try catch", "try except", "error recovery"},
        }

    def add_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """Add a tool call to the tracker.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
        """
        call = TrackedToolCall(tool_name=tool_name, args=args)
        self.recent_calls.append(call)
        logger.debug(f"Tracking tool call: {tool_name}({self._format_args(args)})")

    def is_redundant(self, tool_name: str, args: dict[str, Any], explain: bool = False) -> bool:
        """Check if a tool call is redundant given recent history.

        Args:
            tool_name: Name of the tool to call
            args: Tool arguments
            explain: If True, log explanation for why call is redundant

        Returns:
            True if the call is likely redundant, False otherwise
        """
        if not self.recent_calls:
            return False

        # Check for exact duplicates
        for recent in self.recent_calls:
            if recent.tool_name == tool_name and recent.args == args:
                if explain:
                    import time

                    logger.info(
                        f"Redundant tool call detected: {tool_name}({self._format_args(args)}) "
                        f"was called {(time.time() - recent.timestamp):.1f}s ago"
                    )
                return True

        # Check for semantic overlap
        if self._has_semantic_overlap(tool_name, args):
            if explain:
                logger.info(
                    f"Semantically redundant tool call: {tool_name}({self._format_args(args)}) "
                    "overlaps with recent call"
                )
            return True

        return False

    def _has_semantic_overlap(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if tool call semantically overlaps with recent calls.

        Args:
            tool_name: Tool name
            args: Tool arguments

        Returns:
            True if there's semantic overlap with recent calls
        """
        # Check for grep/search redundancy
        if tool_name in ("grep", "code_search", "semantic_code_search"):
            return self._check_search_redundancy(tool_name, args)

        # Check for file operation redundancy
        if tool_name in ("read_file", "edit_file", "write_file"):
            return self._check_file_redundancy(tool_name, args)

        # Check for list/ls redundancy
        if tool_name in ("list_directory", "ls", "tree"):
            return self._check_list_redundancy(tool_name, args)

        return False

    def _check_search_redundancy(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if search/grep call is redundant.

        Detects patterns like:
        - semantic grep followed by regex grep for same concept
        - multiple searches with similar queries
        """
        query = args.get("query") or args.get("pattern")
        if not query:
            return False

        query_lower = str(query).lower()

        # Check recent grep/search calls
        for recent in self.recent_calls:
            if recent.tool_name not in ("grep", "code_search", "semantic_code_search"):
                continue

            recent_query = recent.args.get("query") or recent.args.get("pattern")
            if not recent_query:
                continue

            recent_query_lower = str(recent_query).lower()

            # Exact match
            if query_lower == recent_query_lower:
                return True

            # Check if queries are synonyms
            if self._queries_are_synonyms(query_lower, recent_query_lower):
                return True

            # Check if one query contains the other (substring match)
            if len(query_lower) > 3 and (
                query_lower in recent_query_lower or recent_query_lower in query_lower
            ):
                return True

        return False

    def _check_file_redundancy(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if file operation is redundant.

        Detects patterns like:
        - Reading same file twice
        - Writing to file just read
        """
        path = args.get("path") or args.get("file_path") or args.get("file")
        if not path:
            return False

        # Check recent file operations
        for recent in self.recent_calls:
            if recent.tool_name not in ("read_file", "edit_file", "write_file"):
                continue

            recent_path = (
                recent.args.get("path") or recent.args.get("file_path") or recent.args.get("file")
            )
            if not recent_path:
                continue

            # Same file accessed
            if path == recent_path:
                # Read after read is redundant (unless large file with different offsets)
                if tool_name == "read_file" and recent.tool_name == "read_file":
                    # Allow if different offsets specified
                    if args.get("offset") != recent.args.get("offset"):
                        return False
                    return True

        return False

    def _check_list_redundancy(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if list/directory operation is redundant.

        Detects patterns like:
        - Listing same directory multiple times
        """
        path = args.get("path", ".")

        # Check recent list operations
        for recent in self.recent_calls:
            if recent.tool_name not in ("list_directory", "ls", "tree"):
                continue

            recent_path = recent.args.get("path", ".")

            # Same directory listed
            if path == recent_path:
                return True

        return False

    def _queries_are_synonyms(self, query1: str, query2: str) -> bool:
        """Check if two queries are synonyms.

        Args:
            query1: First query (lowercase)
            query2: Second query (lowercase)

        Returns:
            True if queries are synonyms
        """
        for concept, synonyms in self.query_synonyms.items():
            # Check if both queries relate to same concept
            query1_matches = concept in query1 or any(syn in query1 for syn in synonyms)
            query2_matches = concept in query2 or any(syn in query2 for syn in synonyms)

            if query1_matches and query2_matches:
                return True

        return False

    def _format_args(self, args: dict[str, Any]) -> str:
        """Format args for logging.

        Args:
            args: Tool arguments

        Returns:
            Formatted string representation
        """
        formatted = []
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 50:
                value = value[:47] + "..."
            formatted.append(f"{key}={repr(value)}")
        return ", ".join(formatted)

    def get_recent_calls(self, limit: Optional[int] = None) -> list[TrackedToolCall]:
        """Get recent tool calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of recent tool calls (most recent first)
        """
        calls = list(reversed(self.recent_calls))
        if limit:
            calls = calls[:limit]
        return calls

    def clear(self) -> None:
        """Clear all tracked tool calls."""
        self.recent_calls.clear()
        logger.debug("Cleared tool deduplication tracker")

    def get_duplicate_count(self) -> int:
        """Get number of exact duplicate calls in recent history.

        Returns:
            Count of duplicate calls
        """
        seen = set()
        duplicates = 0

        for call in self.recent_calls:
            key = (call.tool_name, frozenset(call.args.items()))
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)

        return duplicates


# Global singleton instance (legacy - prefer DI container)
_deduplication_tracker: Optional[ToolDeduplicationTracker] = None


def get_deduplication_tracker() -> "ToolDeduplicationTrackerProtocol":
    """Get or create the tool deduplication tracker.

    Resolution order:
    1. Check DI container (preferred)
    2. Fall back to module-level singleton (legacy)

    Returns:
        ToolDeduplicationTracker instance
    """
    global _deduplication_tracker

    # Try DI container first
    try:
        from victor.core.container import get_container

        container = get_container()
        # Try to get the concrete class from container
        if container.is_registered(ToolDeduplicationTracker):
            return container.get(ToolDeduplicationTracker)
    except Exception:
        pass  # Fall back to legacy singleton

    # Legacy fallback
    if _deduplication_tracker is None:
        _deduplication_tracker = ToolDeduplicationTracker()
    return _deduplication_tracker


def reset_deduplication_tracker() -> None:
    """Reset the global deduplication tracker (for testing).

    Note: This only resets the legacy module-level singleton. If using DI
    container, use reset_container() as well.
    """
    global _deduplication_tracker
    _deduplication_tracker = None


def is_redundant_call(tool_name: str, args: dict[str, Any]) -> bool:
    """Convenience function to check if a tool call is redundant.

    Args:
        tool_name: Name of the tool
        args: Tool arguments

    Returns:
        True if redundant, False otherwise
    """
    tracker = get_deduplication_tracker()
    return tracker.is_redundant(tool_name, args, explain=True)


def track_call(tool_name: str, args: dict[str, Any]) -> None:
    """Convenience function to track a tool call.

    Args:
        tool_name: Name of the tool
        args: Tool arguments
    """
    tracker = get_deduplication_tracker()
    tracker.add_call(tool_name, args)
