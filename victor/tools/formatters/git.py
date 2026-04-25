"""Git output formatter with Rich markup."""

from typing import Dict, Any

from .base import ToolFormatter, FormattedOutput


class GitFormatter(ToolFormatter):
    """Format git command output with Rich markup.

    Produces color-coded output for:
    - Branch status (green for current, dim for others)
    - File status (yellow=modified, green=added, red=deleted, cyan=renamed)
    - Commit logs (cyan hashes, dim messages)
    - Diffs (reusing edit tool's diff formatter)
    """

    def validate_input(self, data: Dict) -> bool:
        """Validate git result has required fields."""
        return isinstance(data, dict) and ("output" in data or "formatted_output" in data)

    def format(
        self,
        data: Dict[str, Any],
        operation: str = "status",
        **kwargs
    ) -> FormattedOutput:
        """Format git output with Rich markup.

        Args:
            data: Git result dict with 'output' or 'formatted_output' key
            operation: Git operation type (status, log, diff, branch, etc.)
            **kwargs: Ignored (no additional options for git formatter)

        Returns:
            FormattedOutput with Rich markup
        """
        # If already formatted, return as-is
        if "formatted_output" in data:
            return FormattedOutput(
                content=data["formatted_output"],
                format_type="rich",
                summary=f"git {operation}",
                line_count=len(data["formatted_output"].splitlines()),
                contains_markup=True,
            )

        output = data.get("output", "")
        if not output:
            return FormattedOutput(
                content="[dim]No output[/]",
                format_type="rich",
                summary=f"git {operation}",
                line_count=1,
                contains_markup=True,
            )

        # Route to appropriate formatter based on operation
        if operation == "status":
            formatted = self._format_status(output)
        elif operation == "log":
            formatted = self._format_log(output)
        elif operation == "diff":
            formatted = self._format_diff(output)
        elif operation == "branch":
            formatted = self._format_branch(output)
        else:
            # Generic formatting for other operations
            formatted = output

        return FormattedOutput(
            content=formatted,
            format_type="rich",
            summary=f"git {operation}",
            line_count=len(formatted.splitlines()),
            contains_markup=True,
        )

    def _format_status(self, output: str) -> str:
        """Format git status with color-coded file statuses."""
        lines = []

        for line in output.splitlines():
            if not line.strip():
                continue

            # Current branch (starts with *)
            if line.startswith("*"):
                parts = line[1:].strip().split()
                if parts:
                    branch_name = parts[0]
                    lines.append(f"[bold green]*[/] [bold]{branch_name}[/]")

            # Other branches
            elif line.startswith(" "):
                parts = line.strip().split()
                if parts:
                    branch_name = parts[0]
                    lines.append(f"  [dim]{branch_name}[/]")

            # File status (XY filename format)
            elif len(line) > 2 and line[2] == " ":
                status_code = line[:2]
                filename = line[3:]

                # Color-code status
                if "M" in status_code:  # Modified
                    color = "yellow"
                elif "A" in status_code:  # Added
                    color = "green"
                elif "D" in status_code:  # Deleted
                    color = "red"
                elif "R" in status_code:  # Renamed
                    color = "cyan"
                else:
                    color = "white"

                lines.append(f"  [{color}]{status_code}[/] {filename}")

            else:
                lines.append(line)

        return "\n".join(lines)

    def _format_log(self, output: str) -> str:
        """Format git log with colored commit hashes."""
        lines = []

        for line in output.splitlines():
            if not line.strip():
                continue

            # Commit hash (first word) - color it cyan
            parts = line.split(None, 1)
            if parts:
                commit_hash = parts[0]
                rest = parts[1] if len(parts) > 1 else ""
                lines.append(f"[cyan]{commit_hash}[/] [dim]{rest}[/]")
            else:
                lines.append(line)

        return "\n".join(lines)

    def _format_diff(self, output: str) -> str:
        """Format git diff using edit tool's diff formatter."""
        try:
            # Import the edit tool's diff formatter
            from victor.tools.file_editor_tool import _format_diff_for_console
            return _format_diff_for_console(output)
        except Exception:
            # Fallback to basic diff formatting if import fails
            return self._format_diff_basic(output)

    def _format_diff_basic(self, output: str) -> str:
        """Basic diff formatter as fallback."""
        lines = []

        for line in output.splitlines():
            if not line:
                lines.append(line)
            elif line.startswith("+") and not line.startswith("+++"):
                lines.append(f"[green]{line}[/]")
            elif line.startswith("-") and not line.startswith("---"):
                lines.append(f"[red]{line}[/]")
            elif line.startswith("@@"):
                lines.append(f"[cyan]{line}[/]")
            else:
                lines.append(line)

        return "\n".join(lines)

    def _format_branch(self, output: str) -> str:
        """Format git branch output with current branch highlighted."""
        lines = []

        for line in output.splitlines():
            if not line.strip():
                continue

            # Current branch (starts with *)
            if line.startswith("*"):
                parts = line[1:].strip().split()
                if parts:
                    branch_name = parts[0]
                    lines.append(f"[bold green]*[/] [bold]{branch_name}[/]")
            else:
                lines.append(f"  [dim]{line.strip()}[/]")

        return "\n".join(lines)
