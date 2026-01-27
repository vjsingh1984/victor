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

"""Error management commands for Victor CLI."""

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from victor.observability.error_tracker import get_error_tracker

errors_app = typer.Typer(name="errors", help="Manage and view error information.")
console = Console()


ERROR_CATALOG = {
    "PROV-001": {
        "code": "PROV-001",
        "name": "ProviderNotFoundError",
        "category": "Provider",
        "message": "Provider not found: {provider_name}",
        "severity": "ERROR",
        "recovery_hint": "Check provider name spelling. Use 'victor providers list' to list available providers.",
    },
    "PROV-002": {
        "code": "PROV-002",
        "name": "ProviderInitializationError",
        "category": "Provider",
        "message": "Provider '{provider}' failed to initialize",
        "severity": "ERROR",
        "recovery_hint": "Set {PROVIDER}_API_KEY environment variable or check your configuration.",
    },
    "PROV-003": {
        "code": "PROV-003",
        "name": "ProviderConnectionError",
        "category": "Provider",
        "message": "Failed to connect to provider '{provider}'",
        "severity": "ERROR",
        "recovery_hint": "Check network connection and provider URL. Verify the provider service is running.",
    },
    "PROV-004": {
        "code": "PROV-004",
        "name": "ProviderAuthError",
        "category": "Provider",
        "message": "Authentication failed for provider '{provider}'",
        "severity": "ERROR",
        "recovery_hint": "Check your API key or credentials. Ensure they are correctly set in environment variables.",
    },
    "PROV-005": {
        "code": "PROV-005",
        "name": "ProviderRateLimitError",
        "category": "Provider",
        "message": "Rate limit exceeded for provider '{provider}'",
        "severity": "WARNING",
        "recovery_hint": "Wait before retrying. Consider using a different model or provider.",
    },
    "PROV-006": {
        "code": "PROV-006",
        "name": "ProviderTimeoutError",
        "category": "Provider",
        "message": "Request to provider '{provider}' timed out",
        "severity": "ERROR",
        "recovery_hint": "Try increasing timeout or check provider status.",
    },
    "PROV-007": {
        "code": "PROV-007",
        "name": "ProviderInvalidResponseError",
        "category": "Provider",
        "message": "Provider '{provider}' returned invalid response",
        "severity": "ERROR",
        "recovery_hint": "The provider returned an unexpected response format. Try again or use a different provider.",
    },
    "TOOL-001": {
        "code": "TOOL-001",
        "name": "ToolNotFoundError",
        "category": "Tool",
        "message": "Tool not found: {tool_name}",
        "severity": "ERROR",
        "recovery_hint": "Check tool name spelling. Use 'victor tools list' to see available tools.",
    },
    "TOOL-002": {
        "code": "TOOL-002",
        "name": "ToolExecutionError",
        "category": "Tool",
        "message": "Tool '{tool_name}' execution failed",
        "severity": "ERROR",
        "recovery_hint": "Check tool arguments or try with different parameters.",
    },
    "TOOL-003": {
        "code": "TOOL-003",
        "name": "ToolValidationError",
        "category": "Tool",
        "message": "Tool '{tool_name}' validation failed",
        "severity": "ERROR",
        "recovery_hint": "Check the required arguments for this tool.",
    },
    "TOOL-004": {
        "code": "TOOL-004",
        "name": "ToolTimeoutError",
        "category": "Tool",
        "message": "Tool '{tool_name}' timed out",
        "severity": "ERROR",
        "recovery_hint": "Try with a longer timeout or simplify the operation.",
    },
    "CFG-001": {
        "code": "CFG-001",
        "name": "ConfigurationError",
        "category": "Configuration",
        "message": "Configuration error: {message}",
        "severity": "ERROR",
        "recovery_hint": "Fix validation errors in configuration file.",
    },
    "CFG-002": {
        "code": "CFG-002",
        "name": "ValidationError",
        "category": "Configuration",
        "message": "Validation error for field '{field}'",
        "severity": "ERROR",
        "recovery_hint": "Check input values and types.",
    },
    "SRCH-001": {
        "code": "SRCH-001",
        "name": "SearchError",
        "category": "Search",
        "message": "All {count} search backends failed for '{search_type}'",
        "severity": "ERROR",
        "recovery_hint": "Check backend configuration and connectivity. Try alternative search type.",
    },
    "WRK-001": {
        "code": "WRK-001",
        "name": "WorkflowExecutionError",
        "category": "Workflow",
        "message": "Workflow execution failed at node '{node_id}'",
        "severity": "ERROR",
        "recovery_hint": "Fix node '{node_id}' and retry workflow execution. Use checkpoint to resume.",
    },
    "FILE-001": {
        "code": "FILE-001",
        "name": "FileNotFoundError",
        "category": "File",
        "message": "File not found: {path}",
        "severity": "ERROR",
        "recovery_hint": "Check the file path. The file may have been moved or deleted.",
    },
    "FILE-002": {
        "code": "FILE-002",
        "name": "FileError",
        "category": "File",
        "message": "File operation failed",
        "severity": "ERROR",
        "recovery_hint": "Check file permissions and disk space.",
    },
    "NET-001": {
        "code": "NET-001",
        "name": "NetworkError",
        "category": "Network",
        "message": "Network error: {reason}",
        "severity": "ERROR",
        "recovery_hint": "Check your network connection. The service may be temporarily unavailable.",
    },
    "EXT-001": {
        "code": "EXT-001",
        "name": "ExtensionLoadError",
        "category": "Extension",
        "message": "Failed to load '{extension_type}' extension for vertical '{vertical}'",
        "severity": "WARNING",
        "recovery_hint": "Fix the underlying error or mark the extension as optional.",
    },
}


@errors_app.command("list")
def list_errors(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by error category"
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", "-s", help="Filter by severity level"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", help="Search in error names and messages"
    ),
) -> None:
    """List all error codes with their descriptions.

    Examples:
        victor errors list
        victor errors list --category Provider
        victor errors list --severity ERROR
        victor errors list --search "timeout"
    """
    # Build table
    table = Table(title="Victor Error Reference", show_header=True)
    table.add_column("Code", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Category", style="yellow")
    table.add_column("Severity", style="bold")
    table.add_column("Recovery Hint", style="dim", max_width=50)

    # Filter errors
    filtered_errors = {}
    for code, error_info in ERROR_CATALOG.items():
        # Apply filters
        if category and error_info["category"] != category:
            continue
        if severity and error_info["severity"] != severity.upper():
            continue
        if search:
            search_lower = search.lower()
            if (
                search_lower not in error_info["name"].lower()
                and search_lower not in error_info["message"].lower()
            ):
                continue

        filtered_errors[code] = error_info

    if not filtered_errors:
        console.print("[yellow]No errors found matching filters.[/]")
        return

    # Add rows
    for code in sorted(filtered_errors.keys()):
        error_info = filtered_errors[code]
        table.add_row(
            error_info["code"],
            error_info["name"],
            error_info["category"],
            error_info["severity"],
            (
                error_info["recovery_hint"][:50] + "..."
                if len(error_info["recovery_hint"]) > 50
                else error_info["recovery_hint"]
            ),
        )

    console.print(table)

    # Show filters if applied
    if category or severity or search:
        filters = []
        if category:
            filters.append(f"category={category}")
        if severity:
            filters.append(f"severity={severity.upper()}")
        if search:
            filters.append(f"search={search}")

        console.print(f"\n[dim]Filters: {', '.join(filters)}[/]")
        console.print(f"[dim]Showing {len(filtered_errors)} of {len(ERROR_CATALOG)} errors[/]")


@errors_app.command("show")
def show_error(
    code: str = typer.Argument(..., help="Error code (e.g., PROV-001)"),
) -> None:
    """Show detailed information about a specific error.

    Examples:
        victor errors show PROV-001
        victor errors show TOOL-002
    """
    code = code.upper()

    if code not in ERROR_CATALOG:
        console.print(f"[red]Error code '{code}' not found in catalog.[/]")
        console.print("\n[dim]Use 'victor errors list' to see all error codes.[/]")
        raise typer.Exit(1)

    error_info = ERROR_CATALOG[code]

    # Create detailed panel
    content = f"""
[bold cyan]Error Code:[/] {error_info['code']}
[bold green]Name:[/] {error_info['name']}
[bold yellow]Category:[/] {error_info['category']}
[bold]Severity:[/] {error_info['severity']}

[bold]Message:[/]
{error_info['message']}

[bold]Recovery Hint:[/]
{error_info['recovery_hint']}

[dim]For more information, see: docs/errors.md[/]
"""

    panel = Panel(
        content.strip(),
        title=f"Error Details: {code}",
        border_style="cyan",
    )

    console.print(panel)

    # Show related errors
    related = [
        c
        for c, info in ERROR_CATALOG.items()
        if info["category"] == error_info["category"] and c != code
    ][:3]

    if related:
        console.print("\n[bold]Related Errors:[/]")
        for rel_code in related:
            console.print(f"  • {rel_code}: {ERROR_CATALOG[rel_code]['name']}")


@errors_app.command("stats")
def show_stats(
    timeframe: str = typer.Option("24h", "--timeframe", "-t", help="Timeframe: 1h, 24h, 7d"),
) -> None:
    """Show error statistics and trends.

    Examples:
        victor errors stats
        victor errors stats --timeframe 1h
        victor errors stats --timeframe 7d
    """
    tracker = get_error_tracker()
    summary = tracker.get_error_summary()

    console.print(f"\n[bold cyan]Error Statistics[/] [dim]({timeframe})[/]\n")

    # Total errors
    console.print(f"[bold]Total Errors:[/] {summary['total_errors']}")

    # Error counts by type
    if summary["error_counts"]:
        console.print(f"\n[bold]Top Error Types:[/]")
        for error_type, count in summary["most_common"][:10]:
            percentage = (count / summary["total_errors"]) * 100
            console.print(f"  {error_type}: {count} ({percentage:.1f}%)")

    # Error rates
    console.print(f"\n[bold]Error Rates (per hour):[/]")
    for error_type, count in summary["most_common"][:5]:
        rate = tracker.get_error_rate(error_type)
        console.print(f"  {error_type}: {rate:.1f}")

    # Recent errors
    if summary["recent_errors"]:
        console.print(f"\n[bold]Recent Errors:[/]")
        for error in summary["recent_errors"][-5:]:
            timestamp = error["timestamp"][:19]  # Truncate microseconds
            console.print(
                f"  [{error['correlation_id']}] {error['error_type']} - {error['error_message'][:60]}... ({timestamp})"
            )

    console.print(f"\n[dim]Export detailed metrics: victor errors export metrics.json[/]")


@errors_app.command("export")
def export_metrics(
    output: str = typer.Argument("metrics.json", help="Output file path for metrics"),
    format: str = typer.Option("json", "--format", "-f", help="Export format: json, csv"),
) -> None:
    """Export error metrics to a file.

    Examples:
        victor errors export metrics.json
        victor errors export metrics.csv --format csv
        victor errors export /path/to/output.json
    """
    tracker = get_error_tracker()

    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        tracker.export_metrics(str(output_path))
        console.print(f"[green]✓[/] Metrics exported to [cyan]{output}[/]")
    elif format.lower() == "csv":
        # Export as CSV
        import csv

        summary = tracker.get_error_summary()

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Error Type", "Count", "Rate (per hour)"])

            for error_type, count in summary["error_counts"].items():
                rate = tracker.get_error_rate(error_type)
                writer.writerow([error_type, count, f"{rate:.2f}"])

        console.print(f"[green]✓[/] Metrics exported to [cyan]{output}[/]")
    else:
        console.print(f"[red]Unsupported format: {format}[/]")
        console.print("[dim]Supported formats: json, csv[/]")
        raise typer.Exit(1)


@errors_app.command("categories")
def list_categories() -> None:
    """List all error categories.

    Examples:
        victor errors categories
    """
    categories: dict[str, list[str]] = {}
    for error_info in ERROR_CATALOG.values():
        cat = error_info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(error_info["code"])

    table = Table(title="Error Categories", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Error Codes", style="yellow")

    for cat in sorted(categories.keys()):
        codes = ", ".join(sorted(categories[cat]))
        table.add_row(cat, str(len(categories[cat])), codes)

    console.print(table)


@errors_app.command("docs")
def open_docs() -> None:
    """Open error documentation in browser or show path.

    Examples:
        victor errors docs
    """
    docs_path = Path(__file__).parent.parent.parent / "docs" / "errors.md"

    if docs_path.exists():
        console.print(f"[green]Error documentation:[/] [cyan]{docs_path}[/]")

        # Try to open in browser (works on macOS and some Linux systems)
        try:
            import webbrowser

            webbrowser.open(f"file://{docs_path.absolute()}")
            console.print("[dim]Opening in browser...[/]")
        except Exception:
            console.print(
                "[dim]Could not open browser. Use the path above to access documentation.[/]"
            )
    else:
        console.print("[yellow]Error documentation not found.[/]")
        console.print("[dim]Expected location: docs/errors.md[/]")


@errors_app.command("search")
def search_errors(
    query: str = typer.Argument(..., help="Search query"),
) -> None:
    """Search for errors by name, message, or recovery hint.

    Examples:
        victor errors search timeout
        victor errors search "provider not found"
        victor errors search api key
    """
    query_lower = query.lower()

    # Search in all fields
    results = []
    for code, error_info in ERROR_CATALOG.items():
        # Search in name
        if query_lower in error_info["name"].lower():
            results.append((code, error_info, "name"))
            continue

        # Search in message
        if query_lower in error_info["message"].lower():
            results.append((code, error_info, "message"))
            continue

        # Search in recovery hint
        if query_lower in error_info["recovery_hint"].lower():
            results.append((code, error_info, "recovery_hint"))
            continue

    if not results:
        console.print(f"[yellow]No errors found matching: {query}[/]")
        return

    console.print(f"\n[bold cyan]Search Results:[/] [dim]'{query}'[/]\n")
    console.print(f"[dim]Found {len(results)} matching errors[/]\n")

    table = Table(show_header=True)
    table.add_column("Code", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Matched In", style="yellow")

    for code, error_info, matched_in in results:
        table.add_row(code, error_info["name"], matched_in)

    console.print(table)
    console.print(f"\n[dim]Use 'victor errors show <code>' for details[/]")
