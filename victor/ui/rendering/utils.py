"""Shared rendering utilities for stream renderers.

This module provides common rendering functions used across different
renderer implementations to ensure consistent visual output.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

logger = logging.getLogger(__name__)

_REASONING_PREFIX_MARKERS = (
    "💭 Thinking...",
    "💭",
    "🤔 Thinking...",
    "🤔",
    "Thinking...",
)

_THINKING_STATUS_PREFIXES = ("💭", "🤔", "◌")


class StreamDeltaNormalizer:
    """Normalize provider stream chunks to append-only deltas.

    Some providers stream true deltas while others resend cumulative snapshots.
    Renderers should receive append-only text so UI surfaces do not duplicate
    already-rendered content.
    """

    _TAIL_WINDOW = 4096
    _MIN_OVERLAP_CHARS = 8

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_raw = ""
        self._emitted_tail = ""

    def consume(self, text: str) -> str:
        """Return only the unseen portion of ``text``."""
        if not text:
            return ""

        if text == self._last_raw:
            logger.debug("StreamDeltaNormalizer: exact duplicate, returning empty")
            return ""

        if self._last_raw and text.startswith(self._last_raw):
            delta = text[len(self._last_raw) :]
            self._last_raw = text
            self._extend_tail(delta)
            logger.debug(
                "StreamDeltaNormalizer: accumulated content, "
                "last_raw_len=%d, new_text_len=%d, delta_len=%d",
                len(self._last_raw) - len(delta),
                len(text),
                len(delta),
            )
            return delta

        overlap = self._find_overlap(text)
        self._last_raw = text
        if overlap and (overlap >= self._MIN_OVERLAP_CHARS or overlap * 2 >= len(text)):
            delta = text[overlap:]
            self._extend_tail(delta)
            logger.debug(
                "StreamDeltaNormalizer: found overlap=%d, text_len=%d, delta_len=%d",
                overlap,
                len(text),
                len(delta),
            )
            return delta

        self._extend_tail(text)
        logger.debug("StreamDeltaNormalizer: no overlap, returning full text (%d chars)", len(text))
        return text

    def _extend_tail(self, text: str) -> None:
        if not text:
            return
        self._emitted_tail = (self._emitted_tail + text)[-self._TAIL_WINDOW :]

    # Cap the overlap search to avoid O(n×m) scans on large payloads.
    _OVERLAP_SEARCH_CAP = 256

    def _find_overlap(self, text: str) -> int:
        max_len = min(len(self._emitted_tail), len(text), self._OVERLAP_SEARCH_CAP)
        for size in range(max_len, 0, -1):
            if self._emitted_tail[-size:] == text[:size]:
                return size
        return 0


def normalize_reasoning_content(text: str) -> str:
    """Strip provider-added reasoning banners from reasoning stream content."""
    normalized = text.lstrip()
    for marker in _REASONING_PREFIX_MARKERS:
        if normalized.startswith(marker):
            return normalized[len(marker) :].lstrip()
    return normalized


def is_thinking_status_message(message: str) -> bool:
    """Return True for generic thinking-status messages that should use thinking UI."""
    normalized = message.strip()
    for prefix in _THINKING_STATUS_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break

    lowered = normalized.lower()
    return lowered in {"thinking", "thinking...", "thinking…"}


def format_tool_args(arguments: dict[str, Any], max_width: int = 80) -> str:
    """Format tool arguments for compact display.

    Shared utility used by all renderers for consistent arg formatting.

    Args:
        arguments: Tool arguments dict
        max_width: Maximum total width for args string

    Returns:
        Formatted args string like "path='/file.py', limit=100"
    """
    if not arguments:
        return ""
    parts = []
    total_len = 0
    for k, v in arguments.items():
        if isinstance(v, str):
            display = v if len(v) <= 60 else v[:57] + "..."
            part = f"{k}='{display}'"
        elif isinstance(v, (int, float, bool)):
            part = f"{k}={v}"
        elif v is None:
            continue
        else:
            part = f"{k}=..."
        if total_len + len(part) > max_width and parts:
            parts.append("...")
            break
        parts.append(part)
        total_len += len(part) + 2
    return ", ".join(parts)


def format_duration(elapsed: float) -> str:
    """Format elapsed seconds for compact CLI display."""
    if elapsed <= 0:
        return "0ms"
    if elapsed < 1:
        return f"{elapsed * 1000:.0f}ms"
    if elapsed < 10:
        return f"{elapsed:.1f}s"
    return f"{elapsed:.0f}s"


def render_file_preview(console: Console, path: str, content: str) -> None:
    """Render a file content preview panel.

    Shared utility for displaying file contents with syntax highlighting.

    Args:
        console: Rich Console to render to
        path: File path for title and syntax detection
        content: File content to display
    """
    ext = path.split(".")[-1] if "." in path else "txt"
    syntax = Syntax(content, ext, theme="monokai", line_numbers=False)
    console.print(Panel(syntax, title=f"[dim]{path}[/]", border_style="dim"))


def render_edit_preview(console: Console, path: str, diff: str) -> None:
    """Render an edit diff preview.

    Shared utility for displaying diffs with colored additions/deletions.

    Args:
        console: Rich Console to render to
        path: File path being edited
        diff: Diff content to display
    """
    console.print(f"[dim]{path}:[/]")
    for line in diff.split("\n"):
        if line.startswith("-"):
            console.print(f"[red]{line}[/]")
        elif line.startswith("+"):
            console.print(f"[green]{line}[/]")
        else:
            console.print(f"[dim]{line}[/]")


def render_thinking_indicator(console: Console) -> None:
    """Render the thinking start indicator with enhanced visual emphasis.

    Args:
        console: Rich Console to render to
    """
    from victor.config.theme_settings import get_theme_settings

    theme_settings = get_theme_settings()

    # Use high contrast colors if enabled
    if theme_settings.high_contrast:
        console.rule("[bold cyan]Reasoning[/]", style="bold", align="left")
        console.print("[bold cyan]◌[/] [bold]Thinking[/]")
    else:
        console.rule("[dim cyan]Reasoning[/]", style="dim", align="left")
        console.print("[cyan]◌[/] [dim italic]Thinking[/]")


def render_thinking_text(console: Console, text: str) -> None:
    """Render thinking content as dimmed/italic text.

    Args:
        console: Rich Console to render to
        text: Thinking text to display
    """
    styled = Text(text, style="italic color(246)")
    console.print(styled, end="")


def render_content_badge(console: Console, content_type: str, icon: str = "") -> None:
    """Render a small badge for content type identification.

    Args:
        console: Rich Console to render to
        content_type: Type of content (thinking, tool, response, error)
        icon: Optional icon to display
    """
    from victor.config.theme_settings import get_theme_settings

    theme_settings = get_theme_settings()

    # Use brighter colors for high contrast mode
    if theme_settings.high_contrast:
        badges = {
            "thinking": ("[bold cyan]◌[/]", "[bold cyan]THINKING[/]"),
            "tool": ("[bold blue]⟳[/]", "[bold blue]TOOL[/]"),
            "response": ("[bold green]→[/]", "[bold green]RESPONSE[/]"),
            "error": ("[bold red]✗[/]", "[bold red]ERROR[/]"),
        }
    else:
        badges = {
            "thinking": ("[cyan]◌[/]", "[dim]THINKING[/]"),
            "tool": ("[blue]⟳[/]", "[dim]TOOL[/]"),
            "response": ("[green]→[/]", "[dim]RESPONSE[/]"),
            "error": ("[red]✗[/]", "[dim]ERROR[/]"),
        }

    prefix, label = badges.get(content_type, ("•", content_type.upper()))
    console.print(f"{prefix} {label}")


def render_status_message(console: Console, message: str) -> None:
    """Render a compact status line with lightweight semantic styling."""
    from victor.config.theme_settings import get_theme_settings

    theme_settings = get_theme_settings()

    normalized = message.strip()
    lowered = normalized.lower()

    # Use high contrast styling if enabled
    if theme_settings.high_contrast:
        if normalized.startswith("⚠") or "warning" in lowered or "failed" in lowered:
            prefix = "[bold yellow]![/]"
            body_style = "bold yellow"
            normalized = normalized.removeprefix("⚠️").removeprefix("⚠").strip()
        elif "thinking" in lowered:
            prefix = "[bold cyan]◌[/]"
            body_style = "bold cyan"
        else:
            prefix = "[bold blue]•[/]"
            body_style = "bold"
    else:
        if normalized.startswith("⚠") or "warning" in lowered or "failed" in lowered:
            prefix = "[yellow]![/]"
            body_style = "yellow"
            normalized = normalized.removeprefix("⚠️").removeprefix("⚠").strip()
        elif "thinking" in lowered:
            prefix = "[cyan]◌[/]"
            body_style = "dim italic"
        else:
            prefix = "[blue]•[/]"
            body_style = "dim"

    console.print(f"{prefix} [{body_style}]{normalized}[/]")


def render_tool_preview(
    console: Console,
    preview_text: str,
    *,
    total_lines: int,
    preview_lines: int,
    hotkey: str,
) -> None:
    """Render compact preview lines with an expandable gutter."""
    for line in preview_text.split("\n"):
        if line:
            console.print(f"[dim]│ {line}[/]")

    if total_lines > preview_lines:
        remaining = total_lines - preview_lines
        suffix = "s" if remaining != 1 else ""
        console.print(f"[dim italic]└ {remaining} more line{suffix} • press {hotkey} to expand[/]")


# Tools whose names map to valid Pygments lexer identifiers.
_SYNTAX_TOOL_WHITELIST: frozenset[str] = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "json",
        "yaml",
        "bash",
        "shell",
        "sql",
        "html",
        "css",
        "xml",
        "markdown",
        "toml",
        "go",
        "rust",
        "java",
        "cpp",
        "c",
    }
)


def expand_tool_output(
    console: Console,
    tool_name: str,
    content: str,
    *,
    pause_fn=None,
    resume_fn=None,
    max_chars: int = 10000,
) -> None:
    """Render full tool output in a syntax-highlighted panel.

    Shared implementation used by both FormatterRenderer and LiveDisplayRenderer.

    Args:
        console: Rich Console to render to
        tool_name: Name of the tool (used for panel title)
        content: Full tool output text
        pause_fn: Optional callable to pause the renderer before printing
        resume_fn: Optional callable to resume the renderer after printing
        max_chars: Truncate content beyond this many characters
    """
    if pause_fn is not None:
        pause_fn()

    if len(content) > max_chars:
        console.print(f"[dim yellow]⚠ Output is {len(content)} chars, showing first {max_chars}[/]")
        content = content[:max_chars]

    # Derive a lexer only from the whitelisted set; fall back to plain text.
    last_segment = tool_name.split("_")[-1] if "_" in tool_name else tool_name
    lexer = last_segment if last_segment in _SYNTAX_TOOL_WHITELIST else "text"

    try:
        syntax = Syntax(content, lexer, theme="monokai", line_numbers=True, word_wrap=True)
        console.print(
            Panel(syntax, title=f"[bold]{tool_name}[/] - Full Output", border_style="blue")
        )
    except Exception:
        console.print(
            Panel(content, title=f"[bold]{tool_name}[/] - Full Output", border_style="blue")
        )

    if resume_fn is not None:
        resume_fn()
