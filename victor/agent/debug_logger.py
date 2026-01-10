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

"""Debug logging utilities for Victor.

Provides clean, scannable debug output that focuses on meaningful events:
- Iteration summaries (not verbose traces)
- Tool calls and results (one-line format)
- Context size tracking
- Progress indicators

Logging Levels (Victor convention):
- TRACE (5): Very verbose per-operation logs (embedding calls, cache lookups)
- DEBUG (10): Detailed internal state (argument normalization, selection steps)
- INFO (20): Key events (iteration start, tool execution, model response)
- WARNING (30): Recoverable issues (empty response, cache miss)
- ERROR (40): Operation failures
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.providers.base import Message

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

# Custom TRACE level for very verbose logging (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log at TRACE level (5) - for very verbose per-operation logs."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Add trace method to Logger class
logging.Logger.trace = trace  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# Third-party loggers to silence (they generate too much noise)
NOISY_LOGGERS = [
    "httpcore",
    "httpx",
    "urllib3",
    "docker",
    "sentence_transformers",
    "asyncio",
    "transformers",
    "huggingface_hub",
    # Markdown parsing libraries
    "markdown_it",
    "markdown_it.rules_block",
    "markdown_it.rules_inline",
    "markdown_it.rules_core",
    # Data loading
    "datasets",
]


def configure_logging_levels(log_level: str = "INFO", file_logging_enabled: bool = True) -> None:
    """Configure logging levels, silencing noisy third-party loggers.

    Args:
        log_level: Desired log level for Victor loggers (console).
            Supported: TRACE (5), DEBUG, INFO, WARNING, ERROR, CRITICAL
        file_logging_enabled: If True, keeps victor logger at INFO minimum
            to allow file handler to capture INFO+ messages.
    """
    # Handle custom TRACE level
    level_upper = log_level.upper()
    if level_upper == "TRACE":
        level = TRACE
    else:
        level = getattr(logging, level_upper, logging.INFO)

    # Set Victor loggers to desired level
    # When file logging is enabled, ensure we don't filter out INFO messages
    # at the logger level (let handlers do the filtering)
    if file_logging_enabled:
        # Use the lower of requested level or INFO to ensure file logging works
        effective_level = min(level, logging.INFO)
    else:
        effective_level = level
    logging.getLogger("victor").setLevel(effective_level)

    # Silence noisy third-party loggers (always WARNING or above)
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


@dataclass
class ConversationStats:
    """Statistics about conversation state."""

    total_messages: int = 0
    total_chars: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_messages: int = 0
    tool_calls_made: int = 0
    iterations: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since conversation started."""
        return time.time() - self.start_time

    def summary(self) -> str:
        """One-line summary of stats."""
        return (
            f"msgs={self.total_messages} ({self.total_chars:,} chars) | "
            f"tools={self.tool_calls_made} | "
            f"iter={self.iterations} | "
            f"{self.elapsed_seconds:.1f}s"
        )


class DebugLogger:
    """Clean debug logger focused on meaningful events.

    Design principles:
    - One-line log entries where possible
    - Key events at INFO level (visible by default)
    - Detailed traces only at DEBUG level
    - No verbose content dumps
    """

    def __init__(
        self,
        name: str = "victor.debug",
        max_preview: int = 80,
        enabled: bool = True,
        presentation: Optional["PresentationProtocol"] = None,
    ):
        """Initialize debug logger.

        Args:
            name: Logger name
            max_preview: Max characters for inline previews
            enabled: Whether logging is enabled
            presentation: Optional presentation adapter for icons (creates default if None)
        """
        self.logger = logging.getLogger(name)
        self.max_preview = max_preview
        self.enabled = enabled
        self._last_iteration = 0
        self.stats = ConversationStats()
        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

    def reset(self) -> None:
        """Reset state for new conversation."""
        self._last_iteration = 0
        self.stats = ConversationStats()

    def _truncate(self, text: str, max_len: Optional[int] = None) -> str:
        """Truncate text with indicator."""
        max_len = max_len or self.max_preview
        text = text.replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}..."

    def log_iteration_start(self, iteration: int, **context: Any) -> None:
        """Log iteration start (one line)."""
        if not self.enabled or iteration <= self._last_iteration:
            return

        self._last_iteration = iteration
        self.stats.iterations = iteration

        # One-line format
        self.logger.info(
            f"── ITER {iteration} ─────────────────────────────────────────────────────"
        )

    def log_iteration_end(
        self, iteration: int, has_tool_calls: bool = False, **context: Any
    ) -> None:
        """Log iteration end summary."""
        if not self.enabled:
            return

        arrow = self._presentation.icon("arrow_right", with_color=False)
        status = f"{arrow} tools" if has_tool_calls else f"{arrow} done"
        self.logger.info(f"   {self.stats.summary()} {status}")

    def log_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        iteration: int,
    ) -> None:
        """Log tool call (one line)."""
        if not self.enabled:
            return

        self.stats.tool_calls_made += 1

        # Compact args preview
        args_str = ", ".join(f"{k}={self._truncate(str(v), 30)}" for k, v in list(args.items())[:3])
        if len(args) > 3:
            args_str += f", +{len(args)-3} more"

        running_icon = self._presentation.icon("running", with_color=False)
        self.logger.info(f"   {running_icon} {tool_name}({args_str})")

    def log_tool_result(
        self,
        tool_name: str,
        success: bool,
        output: Any,
        elapsed_ms: float,
    ) -> None:
        """Log tool result (one line)."""
        if not self.enabled:
            return

        icon = self._presentation.icon("success" if success else "error", with_color=False)
        output_str = str(output) if output else ""
        size = f"{len(output_str):,} chars" if output_str else "empty"

        self.logger.info(f"   {icon} {tool_name}: {size} ({elapsed_ms:.0f}ms)")

    def log_model_response(
        self,
        content: str,
        has_tool_calls: bool,
        iteration: int,
    ) -> None:
        """Log model response summary (one line)."""
        if not self.enabled:
            return

        tc_str = " +tools" if has_tool_calls else ""
        preview = self._truncate(content, 60)
        self.logger.debug(f"   🤖 {len(content)} chars{tc_str}: {preview}")

    def log_new_messages(self, messages: List[Message]) -> None:
        """Update stats from messages (no logging)."""
        if not self.enabled:
            return

        self.stats.total_messages = len(messages)
        self.stats.total_chars = sum(len(m.content) for m in messages)
        self.stats.user_messages = sum(1 for m in messages if m.role == "user")
        self.stats.assistant_messages = sum(1 for m in messages if m.role == "assistant")
        self.stats.tool_messages = sum(1 for m in messages if m.role == "tool")

    def log_limits(
        self,
        tool_budget: int,
        tool_calls_used: int,
        max_iterations: int,
        current_iteration: int,
        is_analysis_task: bool,
    ) -> None:
        """Log limits (debug level only)."""
        if not self.enabled:
            return

        budget_pct = (tool_calls_used / tool_budget * 100) if tool_budget > 0 else 0
        iter_pct = (current_iteration / max_iterations * 100) if max_iterations > 0 else 0
        mode = "analysis" if is_analysis_task else "standard"

        self.logger.debug(
            f"   limits: tools {tool_calls_used}/{tool_budget} ({budget_pct:.0f}%), "
            f"iter {current_iteration}/{max_iterations} ({iter_pct:.0f}%), {mode}"
        )

    def log_context_size(self, char_count: int, estimated_tokens: int) -> None:
        """Log context size warning if large."""
        if not self.enabled:
            return

        if char_count > 100000:
            warning_icon = self._presentation.icon("warning", with_color=False)
            self.logger.warning(
                f"   {warning_icon} Large context: {char_count:,} chars (~{estimated_tokens:,} tokens)"
            )
        elif char_count > 50000:
            chart_icon = self._presentation.icon("chart", with_color=False)
            self.logger.info(f"   {chart_icon} Context: {char_count:,} chars (~{estimated_tokens:,} tokens)")

    def log_conversation_summary(self, messages: List[Message]) -> None:
        """Log final conversation summary."""
        if not self.enabled:
            return

        self.log_new_messages(messages)

        self.logger.info(
            f"\n═══ SUMMARY ═══════════════════════════════════════════════════════\n"
            f"  Messages: {self.stats.total_messages} "
            f"(user={self.stats.user_messages}, assistant={self.stats.assistant_messages}, "
            f"tool={self.stats.tool_messages})\n"
            f"  Content: {self.stats.total_chars:,} chars\n"
            f"  Tool calls: {self.stats.tool_calls_made}\n"
            f"  Elapsed: {self.stats.elapsed_seconds:.1f}s\n"
            f"═══════════════════════════════════════════════════════════════════"
        )


# Global instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> DebugLogger:
    """Get the global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger()
    return _debug_logger


def configure_debug_logging(
    enabled: bool = True,
    max_preview: int = 80,
    log_level: str = "INFO",
) -> DebugLogger:
    """Configure debug logging with clean output.

    Args:
        enabled: Whether debug logging is enabled
        max_preview: Max chars for previews
        log_level: Log level (INFO recommended for readable output)

    Returns:
        Configured DebugLogger instance
    """
    global _debug_logger

    # Configure logging levels (silence noisy libraries)
    configure_logging_levels(log_level)

    _debug_logger = DebugLogger(
        enabled=enabled,
        max_preview=max_preview,
    )

    return _debug_logger
