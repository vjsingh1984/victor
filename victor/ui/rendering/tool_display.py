"""Tool display manager for interactive streaming.

Handles tool start/progress/result display in the Live rendering context
with enhanced visual hierarchy, structured progress, and rich metadata.
Extracted from LiveDisplayRenderer for focused responsibility.

Design:
- **Visual hierarchy**: Section headers, metadata badges, inline progress
- **Structured progress**: Live terminal blocks for long-running tools
- **Smart previews**: Context-aware summarization per tool category
- **Performance**: Adaptive rendering based on output size and tool type
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from typing import Any

from rich import box
from rich.console import Group
from rich.markup import escape as _markup_escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from victor.ui.rendering.live_manager import LiveManager
from victor.ui.rendering.utils import (
    expand_tool_output,
    format_bash_command_invocation,
    format_duration,
    format_tool_display_name,
    format_tool_metadata_badges,
    get_tool_metadata_for_display,
    render_tool_preview,
)

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

_SLOW_TOOL_PREFIXES = (
    "shell",
    "bash",
    "code_exec",
    "code_search",
    "web_",
    "browser",
    "docker",
    "graph_",
    "embedding_",
    "vector_",
)

_TOOL_CATEGORIES: dict[str, list[str]] = {
    "📁 File System": ["read", "write", "ls", "grep", "file_info", "edit"],
    "🔍 Search": ["code_search", "semantic_code_search", "search", "grep"],
    "🔧 Git": ["git_status", "git_diff", "git_log", "git_blame", "git_commit", "git_push"],
    "📊 Analysis": ["overview", "analyze", "inspect", "metrics"],
    "⚙️ Execution": ["bash", "shell", "run", "code_exec"],
    "🌐 Web": ["web_search", "fetch", "http", "web_fetch"],
    "🗄️ Database": ["db_query", "db_execute", "sql"],
    "🧪 Testing": ["test", "pytest", "run_tests"],
    "🔨 Build": ["build", "compile", "make"],
}

# Tools whose output is typically concise — show full output inline
_CONCISE_TOOLS = {"read", "ls", "file_info", "git_status", "git_blame"}


# ── Tool Status Icons ────────────────────────────────────────────────────────


class _ToolStatusIcon:
    """Icons for tool lifecycle states."""

    PENDING = "⏳"
    RUNNING = "🔄"
    SUCCESS = "✅"
    FAILURE = "❌"
    WARNING = "⚠️"
    PRUNED = "📎"
    CACHED = "⚡"


# ── ToolDisplayManager ───────────────────────────────────────────────────────


class ToolDisplayManager:
    """Manages tool execution display in the Live streaming context.

    Responsibilities:
    - Tool start: Structured invocation line with metadata badges
    - Tool progress: Live terminal block with progress bar
    - Tool result: Status icon + duration + smart preview
    - Tool categorization and visual grouping
    - Output expansion (show/hide)

    All display output goes through the LiveManager's pause/resume cycle.
    """

    # Re-export for backward compatibility
    _SLOW_TOOL_PREFIXES = _SLOW_TOOL_PREFIXES

    def __init__(self, live_manager: LiveManager):
        """Initialize ToolDisplayManager.

        Args:
            live_manager: The LiveManager for pause/resume/lifecycle
        """
        self._live_manager = live_manager
        self._pending_tool: dict | None = None
        self._last_tool_result: dict | None = None
        self._tool_section_shown = False
        self._current_tool_start_time: float | None = None
        self._current_tool_category: str | None = None
        self._tool_progress_lines: deque[str] = deque(maxlen=12)
        self._tool_progress_active = False
        self._tool_progress_name = ""
        self._last_progress_render_ms = 0.0
        self._tool_call_count = 0
        self._tool_success_count = 0
        self._tool_failure_count = 0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def tool_section_shown(self) -> bool:
        return self._tool_section_shown

    @tool_section_shown.setter
    def tool_section_shown(self, value: bool) -> None:
        self._tool_section_shown = value

    @property
    def last_tool_result(self) -> dict | None:
        return self._last_tool_result

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    @property
    def tool_success_count(self) -> int:
        return self._tool_success_count

    @property
    def tool_failure_count(self) -> int:
        return self._tool_failure_count

    # ── Tool Start ──────────────────────────────────────────────────────────

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Handle tool execution start — structured invocation display.

        Shows a categorized section header on first tool call, then renders
        a bash-style invocation line with metadata badges.

        Args:
            name: Tool name
            arguments: Tool arguments
        """
        self._pending_tool = {"name": name, "arguments": arguments}
        self._current_tool_start_time = time.monotonic()
        self._tool_call_count += 1

        if self._live_manager.live:
            self._live_manager.pause()

        # Show section header on first tool call
        if not self._tool_section_shown:
            self._live_manager.print_section_separator("Tools")
            self._tool_section_shown = True

        # Categorize and display tool
        self._categorize_tool(name)
        display_name = format_tool_display_name(name)
        metadata = get_tool_metadata_for_display(name)
        badges = format_tool_metadata_badges(**metadata)

        # Build the invocation display
        icon = _ToolStatusIcon.RUNNING
        format_bash_command_invocation(name, arguments)
        invocation = f"  {icon} [{display_name}]{badges}"

        self._live_manager.console.print(invocation)

        # For long-running tools, show a live progress indicator
        if self._is_slow_tool(name):
            self._tool_progress_active = True
            self._tool_progress_name = name
            self._tool_progress_lines.clear()

        self._live_manager.resume()

    # ── Tool Progress ──────────────────────────────────────────────────────

    def on_tool_progress(
        self,
        name: str,
        stdout: str = "",
        stderr: str = "",
        progress: float = 0.0,
        is_final: bool = False,
    ) -> None:
        """Render live, updating progress from streaming tool output.

        For long-running tools, shows a live terminal block with:
        - Progress bar (when progress estimate available)
        - Last N lines of stdout/stderr
        - Elapsed time

        Args:
            name: Tool name
            stdout: Partial stdout chunk
            stderr: Partial stderr chunk
            progress: Progress estimate (0.0–1.0)
            is_final: Whether this is the last chunk
        """
        if not self._tool_progress_active:
            return

        self._live_manager.pause()

        # Collect progress lines
        if stdout:
            for line in stdout.splitlines():
                self._tool_progress_lines.append(f"[dim]{line}[/]")
        if stderr:
            for line in stderr.splitlines():
                self._tool_progress_lines.append(f"[red]{line}[/]")

        # Build progress display
        lines = list(self._tool_progress_lines)
        elapsed = time.monotonic() - (self._current_tool_start_time or time.monotonic())

        if progress > 0:
            # Show progress bar
            progress_bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self._live_manager.console,
            )
            task = progress_bar.add_task(f"  {format_tool_display_name(name)}", total=100)
            progress_bar.update(task, completed=int(progress * 100))

            renderable = Group(
                Text(f"  ┌ {'─' * 50}"),
                Text(f"  │ elapsed: {elapsed:.1f}s"),
                *[Text(line) for line in lines[-6:]],
                progress_bar,
                Text(f"  └ {'─' * 50}"),
            )
        else:
            # Simple status line with elapsed time
            dots = "." * (min(int(elapsed), 10))
            status = Text(f"  {_ToolStatusIcon.RUNNING} running{dots} ({elapsed:.1f}s)")
            renderable = Group(
                status,
                *[Text(line) for line in lines[-4:]],
            )

        self._live_manager.update_live_with_renderable(renderable)

        if is_final:
            self._tool_progress_active = False

        self._live_manager.resume()

    # ── Tool Result ─────────────────────────────────────────────────────────

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
        follow_up_suggestions: list[dict[str, Any]] | None = None,
        was_pruned: bool = False,
        original_result: Any = None,
        result: Any = None,
    ) -> None:
        """Handle tool execution result — status icon + smart preview.

        Displays:
        - Status icon (✅ success / ❌ failure / ⚡ cached / 📎 pruned)
        - Duration in milliseconds or seconds
        - Smart preview of output (context-aware)
        - Error details if failed
        - Follow-up suggestions if applicable

        Args:
            name: Tool name
            success: Whether the tool succeeded
            elapsed: Execution duration in seconds
            arguments: Original tool arguments
            error: Error message if failed
            follow_up_suggestions: Follow-up suggestions
            was_pruned: Whether output was truncated
            original_result: Full original result
            result: Result (possibly truncated)
        """
        self._pending_tool = None
        self._last_tool_result = {
            "name": name,
            "success": success,
            "result": result,
            "arguments": arguments,
        }

        if success:
            self._tool_success_count += 1
        else:
            self._tool_failure_count += 1

        if self._live_manager.live:
            self._live_manager.pause()

        display_name = format_tool_display_name(name)
        duration = format_duration(elapsed)

        # Determine status icon
        if was_pruned:
            icon = _ToolStatusIcon.PRUNED
        elif not success:
            icon = _ToolStatusIcon.FAILURE
        elif self._was_cached(name, elapsed):
            icon = _ToolStatusIcon.CACHED
        else:
            icon = _ToolStatusIcon.SUCCESS

        # Build status line
        status_parts = [f"  {icon} [{display_name}]"]
        status_parts.append(f"[dim]{duration}[/]")

        if was_pruned:
            status_parts.append("[yellow](truncated)[/]")

        status_line = " ".join(status_parts)
        self._live_manager.console.print(status_line)

        # Show error details
        if error:
            self._live_manager.console.print(f"    [red]Error: {error[:200]}[/]")

        # Show smart preview
        if success and result:
            preview = self._extract_result_summary(name, str(result))
            if preview:
                self._live_manager.console.print(f"    [dim]{preview}[/]")

        # Show follow-up suggestions
        if follow_up_suggestions:
            for suggestion in follow_up_suggestions[:2]:
                label = suggestion.get("label", suggestion.get("command", ""))
                if label:
                    self._live_manager.console.print(f"    [dim]💡 {label}[/]")

        self._tool_progress_active = False
        self._current_tool_start_time = None
        self._live_manager.resume()

    # ── Internal Helpers ────────────────────────────────────────────────────

    def _is_slow_tool(self, tool_name: str) -> bool:
        """Check if a tool is expected to run for a noticeable duration."""
        return any(tool_name.startswith(prefix) for prefix in _SLOW_TOOL_PREFIXES)

    def _was_cached(self, tool_name: str, elapsed: float) -> bool:
        """Heuristic: extremely fast execution likely means cached result."""
        return elapsed < 0.05 and tool_name not in _CONCISE_TOOLS

    def _extract_result_summary(self, tool_name: str, output: str | None) -> str | None:
        """Extract a meaningful summary from tool output for display.

        Uses category-specific heuristics to generate concise summaries.

        Args:
            tool_name: Name of the tool
            output: Raw tool output string

        Returns:
            Concise summary string, or None if no summary available
        """
        if not output:
            return None

        output_lower = output.lower()

        # Search tools — show match count
        if tool_name in ("code_search", "grep", "search", "semantic_code_search"):
            counts = re.findall(r"(\d+)\s+(matches?|results?|files?)", output_lower)
            if counts:
                return f"{counts[0][0]} {counts[0][1]}"
            found = re.search(r"found\s+(\d+)", output_lower)
            if found:
                return f"{found.group(1)} items"
            return "search completed"

        # File write/edit tools
        if tool_name in ("write", "edit", "create_file"):
            if "written" in output_lower or "saved" in output_lower:
                return "file saved"
            if "modified" in output_lower or "updated" in output_lower:
                return "file modified"
            return "file operation complete"

        # Read tool — show line count
        if tool_name in ("read", "file_read"):
            lines = len(output.splitlines())
            return f"{lines} lines"

        # Shell/bash — show exit code
        if tool_name in ("bash", "shell", "code_exec"):
            exit_match = re.search(r"exit code:?\s*(\d+)", output_lower)
            if exit_match:
                code = exit_match.group(1)
                return f"exit {code}" if code != "0" else "completed"
            # Show last non-empty line as summary
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if lines:
                last = lines[-1][:80]
                return f"→ {last}"
            return "completed"

        # Git operations
        if tool_name in ("git_commit", "git_push"):
            if "committed" in output_lower:
                return "committed"
            if "pushed" in output_lower:
                return "pushed"
            return "git operation complete"

        # Test tools
        if tool_name in ("test", "pytest", "run_tests"):
            passed = re.search(r"(\d+)\s+passed", output_lower)
            failed = re.search(r"(\d+)\s+failed", output_lower)
            parts = []
            if passed:
                parts.append(f"{passed.group(1)} passed")
            if failed:
                parts.append(f"{failed.group(1)} failed")
            if parts:
                return ", ".join(parts)
            return "tests completed"

        # Web tools
        if tool_name in ("web_search", "web_fetch", "fetch"):
            result_count = re.search(r"(\d+)\s+results?", output_lower)
            if result_count:
                return f"{result_count.group(1)} results"
            return "web request completed"

        return None

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize a tool name into a logical group.

        Args:
            tool_name: Tool name to categorize

        Returns:
            Category name with icon
        """
        tool_lower = tool_name.lower()
        for category, patterns in _TOOL_CATEGORIES.items():
            if any(pattern in tool_lower for pattern in patterns):
                return category
        return "🔧 Other"

    def expand_last_output(self) -> None:
        """Expand the last tool output to show full content.

        Re-renders the last tool result with full output visible.
        """
        if not self._last_tool_result:
            self._live_manager.console.print("[dim]No tool output to expand[/]")
            return

        data = self._last_tool_result
        if not data["success"] or not data["result"]:
            return

        expand_tool_output(
            self._live_manager.console,
            data["name"],
            data["result"],
            pause_fn=self._live_manager.pause,
            resume_fn=self._live_manager.resume,
        )
