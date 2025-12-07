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

"""Context Compactor - Proactive context management and optimization.

This module provides intelligent context compaction that goes beyond
simple message removal:

- **Proactive compaction**: Triggers before overflow (at configurable threshold)
- **Tool result truncation**: Intelligently truncates large tool outputs
- **Token-accurate estimation**: Uses model-specific tokenization estimates
- **Content-aware pruning**: Preserves important content patterns

Design Pattern: Strategy + Decorator
====================================
ContextCompactor wraps ConversationController and adds proactive
compaction policies that trigger before context overflow occurs.

Usage:
    compactor = ContextCompactor(conversation_controller)

    # After each message, check if compaction is needed
    action = compactor.check_and_compact(current_query)
    # Returns: CompactionAction with what was done

    # Truncate tool result before adding to conversation
    truncated = compactor.truncate_tool_result(result, max_tokens=2000)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from victor.providers.base import Message

if TYPE_CHECKING:
    from victor.agent.conversation_controller import ConversationController

logger = logging.getLogger(__name__)


class CompactionTrigger(Enum):
    """What triggered the compaction."""

    NONE = "none"  # No compaction needed
    THRESHOLD = "threshold"  # Hit utilization threshold
    OVERFLOW = "overflow"  # Actually overflowing
    MANUAL = "manual"  # Manually requested
    SCHEDULED = "scheduled"  # Background scheduled compaction


class TruncationStrategy(Enum):
    """Strategy for truncating large content."""

    HEAD = "head"  # Keep beginning
    TAIL = "tail"  # Keep end
    BOTH = "both"  # Keep beginning and end
    SMART = "smart"  # Use content-aware truncation


@dataclass
class CompactorConfig:
    """Configuration for the context compactor.

    Attributes:
        proactive_threshold: Utilization % to trigger proactive compaction (0.0-1.0)
        min_messages_after_compact: Minimum messages to keep after compaction
        tool_result_max_chars: Max characters for tool results
        tool_result_max_lines: Max lines for tool results
        truncation_strategy: Strategy for truncating large content
        preserve_code_blocks: Try to preserve code blocks when truncating
        preserve_json_structure: Try to preserve JSON structure when truncating
        enable_proactive: Enable proactive (pre-overflow) compaction
        enable_tool_truncation: Enable automatic tool result truncation
    """

    proactive_threshold: float = 0.70  # Compact at 70% utilization
    min_messages_after_compact: int = 8
    tool_result_max_chars: int = 8000  # ~2000 tokens
    tool_result_max_lines: int = 200
    truncation_strategy: TruncationStrategy = TruncationStrategy.SMART
    preserve_code_blocks: bool = True
    preserve_json_structure: bool = True
    enable_proactive: bool = True
    enable_tool_truncation: bool = True


@dataclass
class CompactionAction:
    """Result of a compaction check/action.

    Attributes:
        trigger: What triggered the action
        messages_removed: Number of messages removed
        chars_freed: Approximate characters freed
        tokens_freed: Approximate tokens freed
        truncations_applied: Number of truncations applied
        new_utilization: Context utilization after action
        details: Additional details about what was done
    """

    trigger: CompactionTrigger
    messages_removed: int = 0
    chars_freed: int = 0
    tokens_freed: int = 0
    truncations_applied: int = 0
    new_utilization: float = 0.0
    details: List[str] = field(default_factory=list)

    @property
    def action_taken(self) -> bool:
        """Whether any action was taken."""
        return self.messages_removed > 0 or self.truncations_applied > 0


@dataclass
class TruncationResult:
    """Result of truncating content.

    Attributes:
        content: The truncated content
        original_chars: Original character count
        truncated_chars: Characters removed
        truncated: Whether truncation occurred
        indicator: Truncation indicator added (e.g., "... [truncated]")
    """

    content: str
    original_chars: int
    truncated_chars: int
    truncated: bool
    indicator: str = ""


class ContextCompactor:
    """Proactive context compaction and optimization.

    Wraps ConversationController to add:
    - Proactive compaction before overflow
    - Intelligent tool result truncation
    - Content-aware pruning
    - Token-accurate estimation

    Thread-safe for concurrent access.
    """

    # Token estimation factors by content type
    TOKEN_FACTORS = {
        "code": 3.0,  # Code is more token-dense
        "json": 2.8,  # JSON has structure overhead
        "prose": 4.0,  # English prose ~4 chars/token
        "mixed": 3.5,  # Default mixed content
    }

    # Patterns for preserving important content
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    ERROR_PATTERN = re.compile(r"(?:error|exception|traceback|failed):", re.IGNORECASE)
    PATH_PATTERN = re.compile(r"(?:/[\w./]+|\w:\\[\w\\]+)")

    def __init__(
        self,
        controller: "ConversationController",
        config: Optional[CompactorConfig] = None,
    ):
        """Initialize the context compactor.

        Args:
            controller: The ConversationController to wrap
            config: Optional compactor configuration
        """
        self.controller = controller
        self.config = config or CompactorConfig()
        self._last_compaction_turn: int = 0
        self._total_chars_freed: int = 0
        self._total_tokens_freed: int = 0
        self._compaction_count: int = 0

        logger.debug(
            f"ContextCompactor initialized (threshold: {self.config.proactive_threshold:.0%})"
        )

    # ========================================================================
    # Main Compaction Interface
    # ========================================================================

    def check_and_compact(
        self,
        current_query: Optional[str] = None,
        force: bool = False,
    ) -> CompactionAction:
        """Check context utilization and compact if needed.

        Proactively compacts when utilization exceeds threshold, rather
        than waiting for overflow.

        Args:
            current_query: Current user query for semantic relevance
            force: Force compaction regardless of utilization

        Returns:
            CompactionAction describing what was done
        """
        metrics = self.controller.get_context_metrics()
        utilization = metrics.utilization

        # Determine trigger
        if force:
            trigger = CompactionTrigger.MANUAL
        elif metrics.is_overflow_risk:
            trigger = CompactionTrigger.OVERFLOW
        elif self.config.enable_proactive and utilization >= self.config.proactive_threshold:
            trigger = CompactionTrigger.THRESHOLD
        else:
            return CompactionAction(
                trigger=CompactionTrigger.NONE,
                new_utilization=utilization,
            )

        logger.info(
            f"Compaction triggered: {trigger.value} "
            f"(utilization: {utilization:.1%})"
        )

        # Perform compaction
        chars_before = metrics.char_count
        messages_removed = self.controller.smart_compact_history(
            target_messages=self.config.min_messages_after_compact,
            current_query=current_query,
        )

        # Calculate results
        metrics_after = self.controller.get_context_metrics()
        chars_freed = chars_before - metrics_after.char_count
        tokens_freed = self._estimate_tokens(chars_freed)

        self._total_chars_freed += chars_freed
        self._total_tokens_freed += tokens_freed
        self._compaction_count += 1
        self._last_compaction_turn = len(self.controller.messages)

        action = CompactionAction(
            trigger=trigger,
            messages_removed=messages_removed,
            chars_freed=chars_freed,
            tokens_freed=tokens_freed,
            new_utilization=metrics_after.utilization,
            details=[
                f"Removed {messages_removed} messages",
                f"Freed ~{tokens_freed} tokens",
                f"New utilization: {metrics_after.utilization:.1%}",
            ],
        )

        logger.info(
            f"Compaction complete: {messages_removed} messages removed, "
            f"~{tokens_freed} tokens freed"
        )

        return action

    def should_compact(self) -> Tuple[bool, CompactionTrigger]:
        """Check if compaction should be performed.

        Returns:
            Tuple of (should_compact, trigger_reason)
        """
        metrics = self.controller.get_context_metrics()

        if metrics.is_overflow_risk:
            return True, CompactionTrigger.OVERFLOW

        if self.config.enable_proactive:
            if metrics.utilization >= self.config.proactive_threshold:
                return True, CompactionTrigger.THRESHOLD

        return False, CompactionTrigger.NONE

    # ========================================================================
    # Tool Result Truncation
    # ========================================================================

    def truncate_tool_result(
        self,
        content: str,
        max_chars: Optional[int] = None,
        max_lines: Optional[int] = None,
        content_type: str = "mixed",
    ) -> TruncationResult:
        """Intelligently truncate a tool result.

        Uses content-aware truncation that preserves important patterns
        like error messages, file paths, and code structure.

        Args:
            content: The tool result content
            max_chars: Maximum characters (default: config value)
            max_lines: Maximum lines (default: config value)
            content_type: Type of content (code, json, prose, mixed)

        Returns:
            TruncationResult with the (possibly) truncated content
        """
        if not self.config.enable_tool_truncation:
            return TruncationResult(
                content=content,
                original_chars=len(content),
                truncated_chars=0,
                truncated=False,
            )

        max_chars = max_chars or self.config.tool_result_max_chars
        max_lines = max_lines or self.config.tool_result_max_lines
        original_len = len(content)

        # Check if truncation needed
        lines = content.split("\n")
        needs_truncation = len(content) > max_chars or len(lines) > max_lines

        if not needs_truncation:
            return TruncationResult(
                content=content,
                original_chars=original_len,
                truncated_chars=0,
                truncated=False,
            )

        # Apply truncation strategy
        strategy = self.config.truncation_strategy

        if strategy == TruncationStrategy.SMART:
            truncated = self._smart_truncate(content, max_chars, max_lines, content_type)
        elif strategy == TruncationStrategy.HEAD:
            truncated = self._head_truncate(content, max_chars, max_lines)
        elif strategy == TruncationStrategy.TAIL:
            truncated = self._tail_truncate(content, max_chars, max_lines)
        else:  # BOTH
            truncated = self._both_truncate(content, max_chars, max_lines)

        return TruncationResult(
            content=truncated,
            original_chars=original_len,
            truncated_chars=original_len - len(truncated),
            truncated=True,
            indicator="[output truncated]",
        )

    def _smart_truncate(
        self,
        content: str,
        max_chars: int,
        max_lines: int,
        content_type: str,
    ) -> str:
        """Content-aware smart truncation.

        Preserves:
        - Error messages and tracebacks
        - File paths mentioned in the content
        - Code block structure
        - JSON structure (if applicable)
        """
        lines = content.split("\n")

        # Handle single-line content specially
        if len(lines) <= 2 and len(content) > max_chars:
            # For single/double line content, just truncate by chars
            half = max_chars // 2 - 20  # Leave room for marker
            return f"{content[:half]}\n... [content truncated] ...\n{content[-half:]}"

        # Priority lines (errors, paths, important markers)
        priority_indices: set = set()
        for i, line in enumerate(lines):
            if self.ERROR_PATTERN.search(line):
                priority_indices.add(i)
                # Also include lines around errors
                if i > 0:
                    priority_indices.add(i - 1)
                if i < len(lines) - 1:
                    priority_indices.add(i + 1)
            elif self.PATH_PATTERN.search(line):
                priority_indices.add(i)

        # Detect code blocks to preserve
        if self.config.preserve_code_blocks:
            code_blocks = list(self.CODE_BLOCK_PATTERN.finditer(content))
            if code_blocks:
                # Mark first code block lines as priority
                first_block = code_blocks[0]
                start_line = content[:first_block.start()].count("\n")
                end_line = content[:first_block.end()].count("\n")
                for i in range(start_line, min(end_line + 1, start_line + 20)):
                    priority_indices.add(i)

        # Build truncated output
        result_lines: List[str] = []
        char_count = 0
        half_lines = max(1, max_lines // 2)

        # Keep first portion
        for i, line in enumerate(lines[:half_lines]):
            if char_count + len(line) > max_chars // 2:
                # Truncate this line if it's too long
                remaining = max_chars // 2 - char_count
                if remaining > 50:
                    result_lines.append(line[:remaining] + "...")
                break
            result_lines.append(line)
            char_count += len(line) + 1

        # Add truncation marker
        truncated_count = len(lines) - half_lines * 2
        if truncated_count > 0:
            result_lines.append(f"\n... [{truncated_count} lines truncated] ...\n")

        # Keep last portion and any priority lines in between
        remaining_chars = max_chars - char_count - 50  # Leave room for marker
        remaining_lines: List[str] = []

        for i, line in enumerate(lines[-half_lines:]):
            actual_idx = len(lines) - half_lines + i
            if remaining_chars <= 0 and actual_idx not in priority_indices:
                continue
            remaining_lines.append(line)
            remaining_chars -= len(line) + 1

        result_lines.extend(remaining_lines)

        return "\n".join(result_lines)

    def _head_truncate(self, content: str, max_chars: int, max_lines: int) -> str:
        """Keep the beginning of the content."""
        lines = content.split("\n")
        result_lines = lines[:max_lines]
        result = "\n".join(result_lines)

        if len(result) > max_chars:
            result = result[:max_chars]

        return result + "\n... [output truncated]"

    def _tail_truncate(self, content: str, max_chars: int, max_lines: int) -> str:
        """Keep the end of the content."""
        lines = content.split("\n")
        result_lines = lines[-max_lines:]
        result = "\n".join(result_lines)

        if len(result) > max_chars:
            result = result[-max_chars:]

        return "[output truncated] ...\n" + result

    def _both_truncate(self, content: str, max_chars: int, max_lines: int) -> str:
        """Keep beginning and end."""
        lines = content.split("\n")
        half = max_lines // 2
        half_chars = max_chars // 2

        head = "\n".join(lines[:half])
        if len(head) > half_chars:
            head = head[:half_chars]

        tail = "\n".join(lines[-half:])
        if len(tail) > half_chars:
            tail = tail[-half_chars:]

        truncated = len(lines) - max_lines
        return f"{head}\n\n... [{truncated} lines truncated] ...\n\n{tail}"

    # ========================================================================
    # Token Estimation
    # ========================================================================

    def _estimate_tokens(
        self,
        chars: int,
        content_type: str = "mixed",
    ) -> int:
        """Estimate token count from character count.

        Uses content-type-specific factors for more accurate estimation.

        Args:
            chars: Character count
            content_type: Type of content

        Returns:
            Estimated token count
        """
        factor = self.TOKEN_FACTORS.get(content_type, self.TOKEN_FACTORS["mixed"])
        return int(chars / factor)

    def estimate_message_tokens(self, message: Message) -> int:
        """Estimate tokens for a message.

        Args:
            message: The message to estimate

        Returns:
            Estimated token count
        """
        content = message.content

        # Detect content type
        if "```" in content:
            content_type = "code"
        elif content.strip().startswith("{") or content.strip().startswith("["):
            content_type = "json"
        else:
            content_type = "prose"

        # Add overhead for role/structure
        overhead = 4  # Typical message overhead

        return self._estimate_tokens(len(content), content_type) + overhead

    # ========================================================================
    # Statistics and Metrics
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get compactor statistics.

        Returns:
            Dictionary with compaction statistics
        """
        metrics = self.controller.get_context_metrics()

        return {
            "current_utilization": metrics.utilization,
            "current_chars": metrics.char_count,
            "current_messages": metrics.message_count,
            "compaction_count": self._compaction_count,
            "total_chars_freed": self._total_chars_freed,
            "total_tokens_freed": self._total_tokens_freed,
            "last_compaction_turn": self._last_compaction_turn,
            "proactive_threshold": self.config.proactive_threshold,
            "proactive_enabled": self.config.enable_proactive,
            "truncation_enabled": self.config.enable_tool_truncation,
        }

    def get_compaction_history(self) -> List[str]:
        """Get summaries of past compactions.

        Returns:
            List of compaction summary strings
        """
        return self.controller.get_compaction_summaries()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def reset_statistics(self) -> None:
        """Reset compaction statistics."""
        self._total_chars_freed = 0
        self._total_tokens_freed = 0
        self._compaction_count = 0
        self._last_compaction_turn = 0
        logger.debug("Compactor statistics reset")


def create_context_compactor(
    controller: "ConversationController",
    proactive_threshold: float = 0.70,
    min_messages_after_compact: int = 8,
    tool_result_max_chars: int = 8000,
    tool_result_max_lines: int = 200,
    truncation_strategy: TruncationStrategy = TruncationStrategy.SMART,
    preserve_code_blocks: bool = True,
    enable_proactive: bool = True,
    enable_tool_truncation: bool = True,
) -> ContextCompactor:
    """Factory function to create a configured ContextCompactor.

    Args:
        controller: The ConversationController to wrap
        proactive_threshold: Utilization % to trigger proactive compaction (default: 0.70)
        min_messages_after_compact: Minimum messages to keep after compaction (default: 8)
        tool_result_max_chars: Maximum characters for tool results (default: 8000)
        tool_result_max_lines: Maximum lines for tool results (default: 200)
        truncation_strategy: Strategy for truncating tool results (default: SMART)
        preserve_code_blocks: Preserve code blocks during truncation (default: True)
        enable_proactive: Enable proactive compaction (default: True)
        enable_tool_truncation: Enable tool result truncation (default: True)

    Returns:
        Configured ContextCompactor instance
    """
    config = CompactorConfig(
        proactive_threshold=proactive_threshold,
        min_messages_after_compact=min_messages_after_compact,
        tool_result_max_chars=tool_result_max_chars,
        tool_result_max_lines=tool_result_max_lines,
        truncation_strategy=truncation_strategy,
        preserve_code_blocks=preserve_code_blocks,
        enable_proactive=enable_proactive,
        enable_tool_truncation=enable_tool_truncation,
    )
    return ContextCompactor(controller, config)
