"""Shared rendering utilities for stream renderers.

This module provides common rendering functions used across different
renderer implementations to ensure consistent visual output.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


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
    """Render the thinking start indicator.

    Args:
        console: Rich Console to render to
    """
    console.print("[dim italic]💭 Thinking...[/]")


def render_thinking_text(console: Console, text: str) -> None:
    """Render thinking content as dimmed/italic text.

    Args:
        console: Rich Console to render to
        text: Thinking text to display
    """
    styled = Text(text, style="dim italic")
    console.print(styled, end="")


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
