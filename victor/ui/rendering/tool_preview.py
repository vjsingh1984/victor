"""Per-tool preview strategies for CLI tool output.

Each tool category gets a strategy that understands its output format and
renders a meaningful summary — diffs for edits, exit-code + stdout for shell,
match counts for searches — instead of a generic first-N-lines dump.

Usage:
    from victor.ui.rendering.tool_preview import renderer

    preview = renderer.render(tool_name, arguments, raw_result, max_lines=3)
    for line in preview.lines:
        console.print(f"[dim]│ {line}[/]")
"""

from __future__ import annotations

import ast
import difflib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RenderedPreview:
    """Result of rendering a tool output preview."""

    lines: List[str] = field(default_factory=list)
    header: Optional[str] = None
    total_line_count: int = 0
    syntax_hint: str = "text"
    contains_rich_markup: bool = False  # True if lines contain Rich markup tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARSE_SIZE_LIMIT = 8_000  # skip ast.literal_eval on huge strings (perf guard)


def _try_parse(raw: str) -> Optional[Any]:
    """Try JSON then ast.literal_eval; return None if both fail.

    ast.literal_eval is skipped for strings longer than _PARSE_SIZE_LIMIT to
    avoid O(n) parse cost on large tool outputs that are unlikely to be dicts.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    if len(raw) <= _PARSE_SIZE_LIMIT:
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return None


def _first_lines(text: str, n: int, max_width: int = 120) -> List[str]:
    lines = text.splitlines()
    out = []
    for line in lines[:n]:
        out.append(line[:max_width] + "…" if len(line) > max_width else line)
    return out


# ---------------------------------------------------------------------------
# Strategy base and concrete implementations
# ---------------------------------------------------------------------------


class _ToolPreviewStrategy(ABC):
    @abstractmethod
    def render(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: str,
        max_lines: int,
    ) -> RenderedPreview: ...


class _DiffPreviewStrategy(_ToolPreviewStrategy):
    """Edit / patch / replace operations — show a unified diff."""

    def _extract_replace_pairs(self, arguments: Dict[str, Any]) -> List[tuple]:
        """Extract (old, new, path) tuples from edit tool's ops list or legacy flat args."""
        ops = arguments.get("ops")
        if isinstance(ops, list):
            pairs = []
            for op in ops:
                if not isinstance(op, dict):
                    continue
                if op.get("type") == "replace":
                    pairs.append((op.get("old_str", ""), op.get("new_str", ""), op.get("path", "")))
            return pairs
        # Legacy / other tools: top-level old_str/new_str or old_string/new_string
        old = arguments.get("old_str") or arguments.get("old_string", "")
        new = arguments.get("new_str") or arguments.get("new_string", "")
        path = arguments.get("path") or arguments.get("file_path", "")
        if old or new:
            return [(old, new, path)]
        return []

    def render(self, tool_name, arguments, raw_result, max_lines) -> RenderedPreview:
        # First, try to use the actual diff from the result (most accurate)
        parsed = _try_parse(raw_result)
        if isinstance(parsed, dict):
            # Check for formatted diff (Rich-formatted with color codes)
            formatted_diff = parsed.get("diff_formatted")
            if formatted_diff:
                # Parse the formatted diff to extract content lines
                # Strip Rich markup tags for line counting
                import re

                clean_lines = []
                for line in formatted_diff.splitlines():
                    # Remove Rich markup tags like [green], [red], [dim], [cyan], [/]
                    clean_line = re.sub(r"\[[a-z0-9_/]+\]", "", line)
                    clean_lines.append(clean_line)

                # Count additions and removals from clean lines
                added = sum(
                    1 for line in clean_lines if line.startswith("+") and not line.startswith("+++")
                )
                removed = sum(
                    1 for line in clean_lines if line.startswith("-") and not line.startswith("---")
                )

                # Extract file paths from the formatted diff
                file_labels = []
                for line in clean_lines:
                    if line.startswith("---") or line.startswith("+++"):
                        # Extract file path (after the marker)
                        parts = line.split(None, 1)
                        if len(parts) > 1:
                            file_labels.append(parts[1])

                file_part = f" {', '.join(set(file_labels))}" if file_labels else ""
                header = f"+{added} -{removed}{file_part}"

                # Return formatted lines (with Rich markup) for console rendering
                visible = formatted_diff.splitlines()[:max_lines]
                return RenderedPreview(
                    lines=visible,
                    header=header,
                    total_line_count=len(clean_lines),
                    syntax_hint="diff",
                    contains_rich_markup=True,  # Lines contain Rich markup
                )

            # Check for raw diff (unified diff format)
            raw_diff = parsed.get("diff")
            if raw_diff:
                diff_lines = raw_diff.splitlines()

                # Count additions and removals
                added = sum(
                    1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")
                )
                removed = sum(
                    1 for line in diff_lines if line.startswith("-") and not line.startswith("---")
                )

                # Extract file paths
                file_labels = []
                for line in diff_lines:
                    if line.startswith("---") or line.startswith("+++"):
                        parts = line.split(None, 1)
                        if len(parts) > 1:
                            file_labels.append(parts[1])

                file_part = f" {', '.join(set(file_labels))}" if file_labels else ""
                header = f"+{added} -{removed}{file_part}"

                # Return content lines (excluding file headers)
                content_lines = [
                    line
                    for line in diff_lines
                    if not line.startswith("---") and not line.startswith("+++")
                ]
                visible = content_lines[:max_lines]
                return RenderedPreview(
                    lines=visible,
                    header=header,
                    total_line_count=len(content_lines),
                    syntax_hint="diff",
                )

        # Fallback: extract from arguments (old behavior)
        pairs = self._extract_replace_pairs(arguments)

        if pairs:
            all_content_lines: List[str] = []
            file_labels: List[str] = []
            for old, new, path in pairs:
                diff_lines = list(
                    difflib.unified_diff(
                        old.splitlines(),
                        new.splitlines(),
                        fromfile=path,
                        tofile=path,
                        lineterm="",
                        n=1,
                    )
                )
                content_lines = [
                    line
                    for line in diff_lines
                    if not line.startswith("---") and not line.startswith("+++")
                ]
                all_content_lines.extend(content_lines)
                if path:
                    file_labels.append(path)

            added = sum(1 for line in all_content_lines if line.startswith("+"))
            removed = sum(1 for line in all_content_lines if line.startswith("-"))
            file_part = f" {', '.join(file_labels)}" if file_labels else ""
            header = f"+{added} -{removed}{file_part}"
            visible = all_content_lines[:max_lines]
            return RenderedPreview(
                lines=visible,
                header=header,
                total_line_count=len(all_content_lines),
                syntax_hint="diff",
            )

        # Final fallback: parse result for success info
        if isinstance(parsed, dict):
            applied = parsed.get("operations_applied", parsed.get("ops_applied", "?"))
            file_path = (
                arguments.get("file_path") or arguments.get("path") or parsed.get("file_path", "")
            )
            summary = f"{applied} operation(s) applied"
            if file_path:
                summary += f" → {file_path}"
            return RenderedPreview(lines=[summary], header=None, total_line_count=1)

        return _GenericPreviewStrategy().render(tool_name, arguments, raw_result, max_lines)


class _WritePreviewStrategy(_ToolPreviewStrategy):
    """Write / create file — show line count + first few lines of written content."""

    def render(self, tool_name, arguments, raw_result, max_lines) -> RenderedPreview:
        content = arguments.get("content", "")
        file_path = arguments.get("file_path") or arguments.get("path", "")
        if content:
            line_count = content.count("\n") + 1
            header = (
                f"{line_count} lines → {file_path}" if file_path else f"{line_count} lines written"
            )
            preview = _first_lines(content, max_lines)
            return RenderedPreview(
                lines=preview,
                header=header,
                total_line_count=line_count,
                syntax_hint=_ext_hint(file_path),
            )

        # No content in arguments — fall back to result summary
        parsed = _try_parse(raw_result)
        if isinstance(parsed, dict) and parsed.get("success"):
            summary = f"Written: {file_path}" if file_path else "Write succeeded"
            return RenderedPreview(lines=[summary], total_line_count=1)

        return _GenericPreviewStrategy().render(tool_name, arguments, raw_result, max_lines)


class _ReadPreviewStrategy(_ToolPreviewStrategy):
    """Read / view file — show line range header + first N lines."""

    def render(self, tool_name, arguments, raw_result, max_lines) -> RenderedPreview:
        if not raw_result:
            return RenderedPreview()

        lines = raw_result.splitlines()

        # Victor's read tool prefixes with [File: ...] and [Lines X-Y of Z] metadata.
        # Preserve those as the header, show content lines in the gutter.
        meta_lines = []
        content_start = 0
        for i, line in enumerate(lines[:5]):
            if line.startswith("[") and (
                "File:" in line or "Lines" in line or "Size:" in line or "TRUNCATED" in line
            ):
                meta_lines.append(line)
                content_start = i + 1
            else:
                break

        header = " | ".join(meta_lines) if meta_lines else None
        content_lines = lines[content_start:]
        visible = _first_lines("\n".join(content_lines), max_lines)

        file_path = arguments.get("path") or arguments.get("file_path", "")
        return RenderedPreview(
            lines=visible,
            header=header,
            total_line_count=len(content_lines),
            syntax_hint=_ext_hint(file_path),
        )


class _ShellPreviewStrategy(_ToolPreviewStrategy):
    """Shell / exec — show exit code badge + stdout; stderr if nonzero."""

    def render(self, tool_name, arguments, raw_result, max_lines) -> RenderedPreview:
        parsed = _try_parse(raw_result)
        if not isinstance(parsed, dict):
            return _GenericPreviewStrategy().render(tool_name, arguments, raw_result, max_lines)

        rc = parsed.get("return_code", parsed.get("returncode", 0))
        stdout = str(parsed.get("stdout", "")).strip()
        stderr = str(parsed.get("stderr", "")).strip()
        success = parsed.get("success", rc == 0)

        badge = f"exit {rc}" if rc else "exit 0"
        header = f"[{badge}]"

        lines: List[str] = []
        if stdout:
            lines = _first_lines(stdout, max_lines)
        elif stderr and not success:
            lines = _first_lines(stderr, max_lines)

        total = stdout.count("\n") + 1 if stdout else (stderr.count("\n") + 1 if stderr else 0)
        return RenderedPreview(lines=lines, header=header, total_line_count=total)


class _SearchPreviewStrategy(_ToolPreviewStrategy):
    """Grep / glob / find / code_search — show match count + first few hits."""

    def render(self, tool_name, arguments, raw_result, max_lines) -> RenderedPreview:
        if not raw_result:
            return RenderedPreview(header="0 matches", total_line_count=0)

        parsed = _try_parse(raw_result)

        # Check for Rich-formatted results (from code_search tool)
        if isinstance(parsed, dict) and "formatted_results" in parsed:
            lines = parsed["formatted_results"].splitlines()
            count = parsed.get("count", 0)
            mode = parsed.get("mode", "unknown")

            return RenderedPreview(
                lines=lines[:max_lines],
                header=f"{count} match{'es' if count != 1 else ''} ({mode})",
                total_line_count=len(lines),
                contains_rich_markup=True,  # Lines contain Rich markup
            )

        # Structured result (glob/code_search style)
        if isinstance(parsed, dict):
            matches = parsed.get("matches") or parsed.get("results") or parsed.get("files") or []
            if matches:
                count = len(matches)
                header = f"{count} match{'es' if count != 1 else ''}"
                visible_items = [str(m) for m in matches[:max_lines]]
                return RenderedPreview(lines=visible_items, header=header, total_line_count=count)

        # Plain text result (grep-style: file:line:content)
        lines = [line for line in raw_result.splitlines() if line.strip()]
        count = len(lines)
        header = f"{count} match{'es' if count != 1 else ''}"
        return RenderedPreview(
            lines=_first_lines("\n".join(lines), max_lines),
            header=header,
            total_line_count=count,
        )


class _DirectoryPreviewStrategy(_ToolPreviewStrategy):
    """ls / list_dir — show item count + first few names."""

    def render(self, tool_name, arguments, raw_result, max_lines) -> RenderedPreview:
        parsed = _try_parse(raw_result)
        if isinstance(parsed, dict):
            items = parsed.get("items") or parsed.get("files") or parsed.get("entries") or []
            count = len(items)
            header = f"{count} item{'s' if count != 1 else ''}"
            names: List[str] = []
            for item in items[:max_lines]:
                if isinstance(item, dict):
                    names.append(item.get("path") or item.get("name") or str(item))
                else:
                    names.append(str(item))
            return RenderedPreview(lines=names, header=header, total_line_count=count)

        return _GenericPreviewStrategy().render(tool_name, arguments, raw_result, max_lines)


class _TestPreviewStrategy(_ToolPreviewStrategy):
    """Testing tool — show Rich-formatted test results with color-coded status."""

    def render(self, tool_name, arguments, raw_result, max_lines) -> RenderedPreview:
        parsed = _try_parse(raw_result)
        if isinstance(parsed, dict) and "formatted_summary" in parsed:
            # Use pre-formatted Rich output from testing tool
            lines = parsed["formatted_summary"].splitlines()
            summary = parsed.get("summary", {})
            total = summary.get("total_tests", 0)

            return RenderedPreview(
                lines=lines[:max_lines],
                header=f"{total} test{'s' if total != 1 else ''}",
                total_line_count=len(lines),
                contains_rich_markup=True,  # Lines contain Rich markup
            )

        # Fallback: try to show basic summary from parsed data
        if isinstance(parsed, dict) and "summary" in parsed:
            summary = parsed["summary"]
            total = summary.get("total_tests", 0)
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            skipped = summary.get("skipped", 0)

            parts = []
            if passed:
                parts.append(f"✓ {passed} passed")
            if failed:
                parts.append(f"✗ {failed} failed")
            if skipped:
                parts.append(f"○ {skipped} skipped")

            return RenderedPreview(
                lines=[" | ".join(parts)] if parts else [],
                header=f"{total} test{'s' if total != 1 else ''}",
                total_line_count=1,
            )

        return _GenericPreviewStrategy().render(tool_name, arguments, raw_result, max_lines)


class _GenericPreviewStrategy(_ToolPreviewStrategy):
    """Fallback: first N lines of raw result, max-width clamped."""

    def render(self, tool_name, arguments, raw_result, max_lines) -> RenderedPreview:
        if not raw_result:
            return RenderedPreview()
        lines = raw_result.splitlines()
        total = len(lines)
        return RenderedPreview(
            lines=_first_lines(raw_result, max_lines),
            total_line_count=total,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps tool name (exact or prefix) → strategy instance.
# Order matters for prefix matching: more specific entries come first.
_STRATEGY_MAP: Dict[str, _ToolPreviewStrategy] = {
    # Filesystem writes
    "edit": _DiffPreviewStrategy(),
    "write": _WritePreviewStrategy(),
    "patch": _DiffPreviewStrategy(),
    "replace_in_file": _DiffPreviewStrategy(),
    "create_file": _WritePreviewStrategy(),
    "overwrite": _WritePreviewStrategy(),
    # Filesystem reads
    "read": _ReadPreviewStrategy(),
    "view": _ReadPreviewStrategy(),
    "cat": _ReadPreviewStrategy(),
    # Shell / execution
    "shell": _ShellPreviewStrategy(),
    "bash": _ShellPreviewStrategy(),
    "exec": _ShellPreviewStrategy(),
    "run": _ShellPreviewStrategy(),
    "run_command": _ShellPreviewStrategy(),
    "execute_command": _ShellPreviewStrategy(),
    # Search
    "grep": _SearchPreviewStrategy(),
    "glob": _SearchPreviewStrategy(),
    "find": _SearchPreviewStrategy(),
    "code_search": _SearchPreviewStrategy(),
    "search": _SearchPreviewStrategy(),
    "semantic_search": _SearchPreviewStrategy(),
    # Directory listing
    "ls": _DirectoryPreviewStrategy(),
    "list_dir": _DirectoryPreviewStrategy(),
    "tree": _DirectoryPreviewStrategy(),
    "list_files": _DirectoryPreviewStrategy(),
    # Testing
    "test": _TestPreviewStrategy(),
    "pytest": _TestPreviewStrategy(),
    "run_tests": _TestPreviewStrategy(),
}

_GENERIC = _GenericPreviewStrategy()


def _ext_hint(path: str) -> str:
    """Return a Pygments lexer hint from a file path."""
    if not path or "." not in path:
        return "text"
    ext = path.rsplit(".", 1)[-1].lower()
    _EXT_TO_LEXER = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        "sh": "bash",
        "bash": "bash",
        "md": "markdown",
        "toml": "toml",
        "sql": "sql",
        "html": "html",
        "css": "css",
        "rs": "rust",
        "go": "go",
        "java": "java",
        "cpp": "cpp",
        "c": "c",
    }
    return _EXT_TO_LEXER.get(ext, "text")


class ToolPreviewRenderer:
    """Registry that dispatches to the right preview strategy per tool.

    Falls back to _GenericPreviewStrategy for unregistered tools.

    Register custom strategies at startup:
        renderer.register("my_tool", MyStrategy())
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, _ToolPreviewStrategy] = dict(_STRATEGY_MAP)
        # Register enhanced formatter-aware strategies
        self._register_enhanced_strategies()

    def register(self, tool_name: str, strategy: _ToolPreviewStrategy) -> None:
        self._strategies[tool_name] = strategy

    def _register_enhanced_strategies(self) -> None:
        """Register enhanced formatter-aware strategies.

        This replaces the default strategies with enhanced versions that can
        detect and use pre-formatted Rich markup from tools.
        """
        try:
            from victor.ui.rendering.formatter_aware_preview import (
                _TestPreviewStrategyEnhanced,
                _SearchPreviewStrategyEnhanced,
                _GitPreviewStrategyEnhanced,
            )

            # Register enhanced strategies (they override defaults)
            self._strategies["test"] = _TestPreviewStrategyEnhanced()
            self._strategies["pytest"] = _TestPreviewStrategyEnhanced()
            self._strategies["run_tests"] = _TestPreviewStrategyEnhanced()
            self._strategies["code_search"] = _SearchPreviewStrategyEnhanced()
            self._strategies["semantic_code_search"] = _SearchPreviewStrategyEnhanced()
            self._strategies["git"] = _GitPreviewStrategyEnhanced()

            logger.debug("Enhanced formatter-aware preview strategies registered")
        except Exception as exc:
            logger.debug("Could not register enhanced preview strategies: %s", exc)

    def render(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: str,
        max_lines: int = 3,
    ) -> RenderedPreview:
        strategy = self._strategies.get(tool_name, _GENERIC)
        try:
            return strategy.render(tool_name, arguments, raw_result, max_lines)
        except Exception as exc:
            logger.debug(
                "ToolPreviewRenderer: strategy %s failed: %s", type(strategy).__name__, exc
            )
            return _GENERIC.render(tool_name, arguments, raw_result, max_lines)


# Module-level singleton — import this directly.
renderer = ToolPreviewRenderer()
