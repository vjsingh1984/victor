"""Enhanced preview strategies that integrate with Rich formatter system.

These strategies check for formatted_output in tool results and use pre-formatted
Rich markup instead of re-formatting the output.
"""

from typing import Any, Dict, List

from victor.ui.rendering.tool_preview import (
    _ToolPreviewStrategy,
    RenderedPreview,
    _try_parse,
)


class _FormatterAwarePreviewStrategy(_ToolPreviewStrategy):
    """Base class for formatter-aware preview strategies.

    This strategy checks if the tool has already formatted its output using
    the Rich formatter system and uses that pre-formatted output directly.
    """

    def render(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: str,
        max_lines: int,
    ) -> RenderedPreview:
        """Render preview, checking for pre-formatted output first."""
        # Try to parse the result
        parsed = _try_parse(raw_result)

        if isinstance(parsed, dict):
            # Check for pre-formatted output from formatter system
            if self._has_formatted_output(parsed):
                return self._render_formatted(parsed, max_lines)

            # Check for formatted_summary (used by testing tool)
            if "formatted_summary" in parsed:
                return self._render_formatted_summary(parsed, max_lines)

            # Check for formatted_results (used by search tools)
            if "formatted_results" in parsed:
                return self._render_formatted_results(parsed, max_lines)

        # Fall back to base strategy behavior
        return self._render_base(tool_name, arguments, raw_result, max_lines)

    def _has_formatted_output(self, parsed: Dict) -> bool:
        """Check if result has formatted_output field."""
        return "formatted_output" in parsed and parsed["formatted_output"] is not None

    def _render_formatted(self, parsed: Dict, max_lines: int) -> RenderedPreview:
        """Render pre-formatted output."""
        formatted_content = parsed["formatted_output"]
        lines = formatted_content.splitlines()

        # Extract summary if available
        summary = parsed.get("summary") or self._extract_summary_from_content(formatted_content)

        return RenderedPreview(
            lines=lines[:max_lines],
            header=summary,
            total_line_count=len(lines),
            contains_rich_markup=parsed.get("contains_markup", True),
        )

    def _render_formatted_summary(self, parsed: Dict, max_lines: int) -> RenderedPreview:
        """Render formatted_summary (used by testing tool)."""
        formatted_summary = parsed["formatted_summary"]
        lines = formatted_summary.splitlines()

        # Extract summary from the summary field
        summary_data = parsed.get("summary", {})
        if isinstance(summary_data, dict):
            total = summary_data.get("total_tests", 0)
            summary = f"{total} tests"
        else:
            summary = "Test results"

        return RenderedPreview(
            lines=lines[:max_lines],
            header=summary,
            total_line_count=len(lines),
            contains_rich_markup=True,  # formatted_summary always has markup
        )

    def _render_formatted_results(self, parsed: Dict, max_lines: int) -> RenderedPreview:
        """Render formatted_results (used by search tools)."""
        formatted_results = parsed["formatted_results"]
        lines = formatted_results.splitlines()

        # Extract summary from result
        count = parsed.get("count", 0)
        mode = parsed.get("mode", "search")
        summary = f"{count} matches ({mode})" if count else f"0 matches ({mode})"

        return RenderedPreview(
            lines=lines[:max_lines],
            header=summary,
            total_line_count=len(lines),
            contains_rich_markup=True,  # formatted_results always has markup
        )

    def _extract_summary_from_content(self, content: str) -> str:
        """Extract summary from formatted content."""
        # Look for common patterns in formatted output
        lines = content.splitlines()
        if lines:
            first_line = lines[0]
            # Remove Rich markup tags for summary
            import re
            clean_line = re.sub(r'\[/?[^\]]+\]', '', first_line)
            return clean_line.strip()[:100]
        return "Formatted output"

    def _render_base(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: str,
        max_lines: int,
    ) -> RenderedPreview:
        """Base render method - to be overridden by subclasses."""
        # Default generic behavior
        lines = raw_result.splitlines()
        return RenderedPreview(
            lines=lines[:max_lines],
            header=None,
            total_line_count=len(lines),
            contains_rich_markup=False,
        )


class _TestPreviewStrategyEnhanced(_FormatterAwarePreviewStrategy):
    """Enhanced test preview strategy that uses Rich-formatted output."""

    def _render_base(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: str,
        max_lines: int,
    ) -> RenderedPreview:
        """Fallback to original test preview logic."""
        parsed = _try_parse(raw_result)

        if isinstance(parsed, dict):
            summary = parsed.get("summary", {})
            if isinstance(summary, dict):
                total = summary.get("total_tests", 0)
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)

                header = f"Tests: {passed}/{total} passed"
                if failed > 0:
                    header += f", {failed} failed"

                lines = []
                failures = parsed.get("failures", [])
                for i, failure in enumerate(failures[:3]):
                    test_name = failure.get("test_name", "unknown")
                    error_msg = failure.get("error_message", "No error message")

                    # Extract short test name
                    if "::" in test_name:
                        parts = test_name.split("::")
                        test_name = parts[-1]

                    lines.append(f"  ✗ {test_name}: {error_msg[:50]}")

                if len(failures) > 3:
                    lines.append(f"  ... and {len(failures) - 3} more")

                return RenderedPreview(
                    lines=lines[:max_lines],
                    header=header,
                    total_line_count=len(lines),
                    contains_rich_markup=False,
                )

        # Generic fallback
        lines = raw_result.splitlines()
        return RenderedPreview(
            lines=lines[:max_lines],
            header="Test results",
            total_line_count=len(lines),
            contains_rich_markup=False,
        )


class _SearchPreviewStrategyEnhanced(_FormatterAwarePreviewStrategy):
    """Enhanced search preview strategy that uses Rich-formatted output."""

    def _render_base(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: str,
        max_lines: int,
    ) -> RenderedPreview:
        """Fallback to original search preview logic."""
        parsed = _try_parse(raw_result)

        if isinstance(parsed, dict):
            results = parsed.get("results") or parsed.get("matches", [])
            count = len(results)
            mode = parsed.get("mode", "search")

            header = f"{count} matches ({mode})"

            lines = []
            for result in results[:5]:
                path = result.get("path", "unknown")
                line = result.get("line", "?")
                snippet = result.get("snippet", "")

                # Truncate snippet
                if len(snippet) > 60:
                    snippet = snippet[:57] + "..."

                lines.append(f"  {path}:{line} - {snippet}")

            if count > 5:
                lines.append(f"  ... and {count - 5} more")

            return RenderedPreview(
                lines=lines[:max_lines],
                header=header,
                total_line_count=len(lines),
                contains_rich_markup=False,
            )

        # Generic fallback
        lines = raw_result.splitlines()
        return RenderedPreview(
            lines=lines[:max_lines],
            header="Search results",
            total_line_count=len(lines),
            contains_rich_markup=False,
        )


class _GitPreviewStrategyEnhanced(_FormatterAwarePreviewStrategy):
    """Enhanced git preview strategy that uses Rich-formatted output."""

    def _render_base(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: str,
        max_lines: int,
    ) -> RenderedPreview:
        """Fallback to original git preview logic."""
        parsed = _try_parse(raw_result)

        if isinstance(parsed, dict):
            operation = arguments.get("operation", "status")
            output = parsed.get("output", "")

            header = f"git {operation}"

            # Show first few lines
            lines = output.splitlines()[:max_lines]

            return RenderedPreview(
                lines=lines,
                header=header,
                total_line_count=len(output.splitlines()),
                contains_rich_markup=False,
            )

        # Generic fallback
        lines = raw_result.splitlines()
        return RenderedPreview(
            lines=lines[:max_lines],
            header="Git output",
            total_line_count=len(lines),
            contains_rich_markup=False,
        )


# Export the enhanced strategies
_ENHANCED_STRATEGY_MAP = {
    "test": _TestPreviewStrategyEnhanced(),
    "pytest": _TestPreviewStrategyEnhanced(),
    "run_tests": _TestPreviewStrategyEnhanced(),
    "code_search": _SearchPreviewStrategyEnhanced(),
    "semantic_code_search": _SearchPreviewStrategyEnhanced(),
    "search": _SearchPreviewStrategyEnhanced(),
    "git": _GitPreviewStrategyEnhanced(),
}


def get_enhanced_strategy(tool_name: str) -> Any:
    """Get enhanced strategy for a tool if available.

    Args:
        tool_name: Name of the tool

    Returns:
        Enhanced strategy instance or None if not available
    """
    return _ENHANCED_STRATEGY_MAP.get(tool_name)


def register_enhanced_strategies(renderer) -> None:
    """Register enhanced strategies with the ToolPreviewRenderer.

    Args:
        renderer: ToolPreviewRenderer instance
    """
    for tool_name, strategy in _ENHANCED_STRATEGY_MAP.items():
        renderer.register(tool_name, strategy)
