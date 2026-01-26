#!/usr/bin/env python3
"""
Victor AI Code Review Assistant - Main CLI Entry Point

This application demonstrates comprehensive code review capabilities using Victor AI:
- AST-based code analysis
- Security vulnerability scanning
- Quality metrics and complexity analysis
- Style checking
- Multi-agent team reviews
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.agent.orchestrator_factory import create_orchestrator
from victor.config.settings import Settings
from victor.core.events import create_event_backend, BackendConfig


console = Console()


@click.group()
@click.version_option(version="0.5.1")
def cli():
    """Victor AI Code Review Assistant - Comprehensive code analysis and review."""
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Review directories recursively")
@click.option(
    "--severity", "-s", multiple=True, help="Filter by severity (low, medium, high, critical)"
)
@click.option(
    "--checks",
    "-c",
    multiple=True,
    help="Specific checks to run (security, style, complexity, quality)",
)
@click.option("--max-complexity", type=int, default=10, help="Maximum cyclomatic complexity")
@click.option("--ignore", "-i", multiple=True, help="Patterns to ignore")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def review(
    path: str,
    recursive: bool,
    severity: tuple,
    checks: tuple,
    max_complexity: int,
    ignore: tuple,
    verbose: bool,
):
    """Review code files or directories.

    Examples:
        victor-review review file.py
        victor-review review src/ --recursive
        victor-review review src/ --severity high --severity critical
    """
    asyncio.run(_review(path, recursive, severity, checks, max_complexity, ignore, verbose))


async def _review(
    path: str,
    recursive: bool,
    severity: tuple,
    checks: tuple,
    max_complexity: int,
    ignore: tuple,
    verbose: bool,
):
    """Execute code review."""
    from src.review_engine import ReviewEngine
    from src.config import load_config

    # Display welcome banner
    console.print(
        Panel.fit(
            "[bold blue]Victor AI Code Review Assistant[/bold blue]\n"
            "Comprehensive code analysis powered by Victor AI",
            border_style="blue",
        )
    )

    # Load configuration
    config = load_config()

    # Create orchestrator
    console.print("[dim]Initializing Victor AI orchestrator...[/dim]")
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    # Create review engine
    engine = ReviewEngine(orchestrator, config)

    # Build options
    options = {
        "recursive": recursive,
        "severity": list(severity) if severity else None,
        "checks": list(checks) if checks else None,
        "max_complexity": max_complexity,
        "ignore_patterns": list(ignore) if ignore else None,
        "verbose": verbose,
    }

    # Run review
    target_path = Path(path)
    console.print(f"[cyan]Reviewing: {target_path}[/cyan]\n")

    results = await engine.review(target_path, **options)

    # Display results
    _display_results(results)


def _display_results(results: dict):
    """Display review results in a formatted table."""
    # Summary table
    summary_table = Table(title="Review Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan", width=30)
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Files Analyzed", str(results.get("files_analyzed", 0)))
    summary_table.add_row("Total Issues", str(results.get("total_issues", 0)))
    summary_table.add_row("Critical", f"[red]{results.get('critical', 0)}[/red]")
    summary_table.add_row("High", f"[orange1]{results.get('high', 0)}[/orange1]")
    summary_table.add_row("Medium", f"[yellow]{results.get('medium', 0)}[/yellow]")
    summary_table.add_row("Low", f"[blue]{results.get('low', 0)}[/blue]")

    console.print("\n")
    console.print(summary_table)

    # Issues by type
    if results.get("issues_by_type"):
        type_table = Table(title="Issues by Type", show_header=True, header_style="bold cyan")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green")

        for issue_type, count in results["issues_by_type"].items():
            type_table.add_row(issue_type, str(count))

        console.print("\n")
        console.print(type_table)

    # Top issues
    if results.get("top_issues"):
        console.print("\n[bold yellow]Top Issues:[/bold yellow]\n")
        for issue in results["top_issues"][:10]:
            severity_color = {
                "critical": "red",
                "high": "orange1",
                "medium": "yellow",
                "low": "blue",
            }.get(issue["severity"], "white")

            console.print(
                f"[{severity_color}]●[/] [bold]{issue['severity'].upper()}[/] "
                f"{issue['message']} ({issue['file']}:{issue['line']})"
            )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "json", "markdown"]),
    default="html",
    help="Report format",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--recursive", "-r", is_flag=True, help="Review directories recursively")
def report(path: str, format: str, output: Optional[str], recursive: bool):
    """Generate a code review report.

    Examples:
        victor-review report src/ --format html --output report.html
        victor-review report src/ --format json --output report.json
    """
    asyncio.run(_report(path, format, output, recursive))


async def _report(path: str, format: str, output: Optional[str], recursive: bool):
    """Generate review report."""
    from src.review_engine import ReviewEngine
    from src.report_generator import ReportGenerator
    from src.config import load_config

    console.print(
        Panel.fit("[bold blue]Victor AI Report Generator[/bold blue]", border_style="blue")
    )

    # Load configuration
    config = load_config()

    # Create orchestrator and engine
    settings = Settings()
    orchestrator = create_orchestrator(settings)
    engine = ReviewEngine(orchestrator, config)

    # Run review
    target_path = Path(path)
    console.print(f"[cyan]Analyzing: {target_path}[/cyan]\n")

    results = await engine.review(target_path, recursive=recursive)

    # Generate report
    generator = ReportGenerator(format)

    if not output:
        output = f"review_report.{format}"

    console.print(f"[cyan]Generating {format.upper()} report: {output}[/cyan]")
    generator.generate(results, output)

    console.print(f"[green]✓ Report generated successfully![/green]")


@cli.command()
def interactive():
    """Start interactive review session.

    Example:
        victor-review interactive
    """
    asyncio.run(_interactive())


async def _interactive():
    """Run interactive review session."""
    from src.interactive_session import InteractiveSession
    from src.config import load_config

    console.print(
        Panel.fit(
            "[bold blue]Victor AI Interactive Review[/bold blue]\n"
            "Type your code or file paths for real-time analysis",
            border_style="blue",
        )
    )

    # Load configuration
    config = load_config()

    # Create orchestrator
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    # Start interactive session
    session = InteractiveSession(orchestrator, config)
    await session.run()


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--formation",
    "-f",
    type=click.Choice(["parallel", "sequential", "hierarchical"]),
    default="parallel",
    help="Team formation type",
)
@click.option("--roles", "-r", multiple=True, help="Specific roles to include")
def team_review(path: str, formation: str, roles: tuple):
    """Run multi-agent team review.

    This demonstrates Victor AI's multi-agent coordination capabilities.

    Examples:
        victor-review team-review src/ --formation parallel
        victor-review team-review src/ --roles security_reviewer --roles quality_reviewer
    """
    asyncio.run(_team_review(path, formation, roles))


async def _team_review(path: str, formation: str, roles: tuple):
    """Run team-based review."""
    from src.team_reviewer import TeamReviewer
    from src.config import load_config

    console.print(
        Panel.fit(
            "[bold blue]Victor AI Multi-Agent Team Review[/bold blue]\n" f"Formation: {formation}",
            border_style="blue",
        )
    )

    # Load configuration
    config = load_config()

    # Create orchestrator
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    # Create team reviewer
    reviewer = TeamReviewer(orchestrator, config, formation=formation)

    # Run team review
    target_path = Path(path)
    console.print(f"[cyan]Team reviewing: {target_path}[/cyan]\n")

    roles_list = list(roles) if roles else None
    results = await reviewer.review(target_path, roles=roles_list)

    # Display team results
    _display_team_results(results)


def _display_team_results(results: dict):
    """Display team review results."""
    console.print("\n[bold cyan]Team Review Results:[/bold cyan]\n")

    # Agent results
    for agent_name, agent_results in results.get("agents", {}).items():
        console.print(
            Panel(
                f"[bold]{agent_name}[/bold]\n"
                f"Issues Found: {agent_results.get('issue_count', 0)}\n"
                f"Confidence: {agent_results.get('confidence', 0):.1%}",
                title=agent_name,
                border_style="cyan",
            )
        )

    # Aggregated findings
    console.print("\n[bold yellow]Aggregated Findings:[/bold yellow]")
    for finding in results.get("aggregated_findings", [])[:10]:
        console.print(f"  • {finding['message']}")


if __name__ == "__main__":
    cli()
