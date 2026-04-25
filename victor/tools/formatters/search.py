"""Code search results formatter with Rich markup."""

from typing import Dict, Any, List

from .base import ToolFormatter, FormattedOutput


class SearchResultsFormatter(ToolFormatter):
    """Format code search results with Rich markup.

    Produces organized output with:
    - Bold cyan match count and file count
    - File grouping with average scores
    - Green highlighted code snippets
    - Line numbers in dim formatting
    """

    def validate_input(self, data: Dict) -> bool:
        """Validate search result has required fields."""
        return isinstance(data, dict) and ("results" in data or "matches" in data)

    def format(
        self, data: Dict[str, Any], max_files: int = 10, max_matches_per_file: int = 3, **kwargs
    ) -> FormattedOutput:
        """Format search results with Rich markup.

        Args:
            data: Search result dict with 'results' or 'matches' key
            max_files: Maximum number of files to show (default: 10)
            max_matches_per_file: Maximum matches per file (default: 3)

        Returns:
            FormattedOutput with Rich markup
        """
        # Support both 'results' and 'matches' key for compatibility
        results = data.get("results") or data.get("matches") or []
        mode = data.get("mode", "search")

        lines = []

        if not results:
            lines.append("[dim]No matches found[/]")
            return FormattedOutput(
                content="\n".join(lines),
                format_type="rich",
                summary="0 matches",
                line_count=1,
                contains_markup=True,
            )

        # Group results by file for better readability
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for result in results:
            path = result.get("path", "unknown")
            if path not in by_file:
                by_file[path] = []
            by_file[path].append(result)

        # Header with match count and file count
        file_count = len(by_file)
        match_count = len(results)
        lines.append(
            f"[bold cyan]{match_count} match{'es' if match_count != 1 else ''}[/] [dim]in {file_count} file{'s' if file_count != 1 else ''}[/]"
        )
        lines.append("")  # Blank line

        # Display results grouped by file (max max_files files for preview)
        for i, (path, matches) in enumerate(list(by_file.items())[:max_files]):
            # Calculate average score for this file
            avg_score = sum(m.get("score", 0) for m in matches) / len(matches) if matches else 0

            # File header with path and score
            lines.append(f"  [bold]{path}[/] [dim]• score: {avg_score:.0f}[/]")

            # Show matches in this file (max max_matches_per_file per file)
            for match in matches[:max_matches_per_file]:
                line_num = match.get("line", "?")
                snippet = match.get("snippet", "")

                # Truncate snippet if too long (max 100 chars)
                if len(snippet) > 100:
                    snippet = snippet[:97] + "..."

                # Display match with line number
                lines.append(f"  [dim]{line_num}:[/][green]{snippet}[/]")

            # Indicator if more matches in this file
            if len(matches) > max_matches_per_file:
                lines.append(
                    f"  [dim]... and {len(matches) - max_matches_per_file} more match{'es' if len(matches) - max_matches_per_file > 1 else ''} in this file[/]"
                )

            # Blank line between files (except after last one)
            if i < min(max_files, len(by_file)) - 1:
                lines.append("")

        # Indicator if more files
        if len(by_file) > max_files:
            lines.append(
                f"[dim]... and {len(by_file) - max_files} more file{'s' if len(by_file) - max_files > 1 else ''}[/]"
            )

        content = "\n".join(lines)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=f"{match_count} matches ({mode})",
            line_count=len(lines),
            contains_markup=True,
        )
