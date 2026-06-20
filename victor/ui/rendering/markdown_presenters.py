# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Markdown presenters — the shared seam between the CLI (Rich) and web (Markdown) surfaces.

The terminal renderer turns a tool result into Rich markup; the Chainlit web app needs
**markdown** (code fences, bold) instead. Both, however, can share the format-agnostic
``RenderedPreview`` produced by :class:`~victor.ui.rendering.tool_preview.ToolPreviewRenderer`
and the canonical formatters in :mod:`victor.ui.rendering.utils`. This module converts that
shared data into markdown, so the web UI matches CLI fidelity (syntax-highlighted diffs, file
reads, search results, …) without re-implementing any tool-specific preview logic.

Pure and dependency-light (no Chainlit, no Rich Console), so it is unit-testable directly.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from victor.ui.rendering.tool_preview import RenderedPreview, ToolPreviewRenderer
from victor.ui.rendering.utils import format_tool_args, format_tool_display_name

# Strips Rich markup tags like ``[green]`` / ``[/]`` so preview lines render cleanly in
# markdown (mirrors the inline patterns already used in tool_preview / formatter_aware_preview).
_RICH_TAG = re.compile(r"\[/?[^\]]*\]")

# One shared renderer instance (registry of per-tool preview strategies).
_RENDERER = ToolPreviewRenderer()


def strip_rich_markup(text: str) -> str:
    """Remove Rich markup tags from *text* for safe markdown rendering."""
    return _RICH_TAG.sub("", text)


def tool_call_summary(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    """A compact ``displayName(arg=val, …)`` label for a tool step or approval card."""
    name = format_tool_display_name(tool_name or "tool")
    args = format_tool_args(arguments or {}, max_width=160)
    return f"{name}({args})" if args else f"{name}()"


def preview_to_markdown(preview: RenderedPreview) -> str:
    """Render a format-agnostic ``RenderedPreview`` as markdown (header + fenced body + footer)."""
    parts: List[str] = []
    if preview.header:
        parts.append(f"**{strip_rich_markup(preview.header)}**")
    if preview.lines:
        clean = preview.contains_rich_markup
        body = "\n".join(strip_rich_markup(line) if clean else line for line in preview.lines)
        lang = "diff" if preview.syntax_hint in ("diff", "udiff") else (preview.syntax_hint or "")
        parts.append(f"```{lang}\n{body}\n```")
    remaining = preview.total_line_count - len(preview.lines)
    if remaining > 0:
        parts.append(f"_… {remaining} more line{'s' if remaining != 1 else ''}_")
    return "\n\n".join(parts)


def tool_result_markdown(
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    result: Any,
    *,
    success: bool = True,
    max_lines: int = 12,
) -> str:
    """Full markdown body for a completed tool call — a syntax-highlighted result preview.

    Delegates to the shared ``ToolPreviewRenderer`` so the web UI gets the same per-tool
    preview (diffs, file reads, search results, shell output) the CLI shows. Never raises:
    rendering failures fall back to a plain fenced block so a malformed result can't break chat.
    """
    try:
        preview = _RENDERER.render(tool_name or "tool", arguments or {}, result, max_lines)
        body = preview_to_markdown(preview)
    except Exception:
        text = result if isinstance(result, str) else str(result)
        body = f"```\n{text[:4000]}\n```"
    if not success:
        return f"⚠️ **failed**\n\n{body}" if body else "⚠️ **failed**"
    return body
