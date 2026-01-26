#!/usr/bin/env python3
"""
Victor AI Multi-Agent Research Team - CLI Interface

Command-line interface for multi-agent research teams.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.agent.orchestrator_factory import create_orchestrator
from victor.config.settings import Settings
from victor.teams import TeamFormation, create_coordinator

from src.team import ResearchTeam
from src.agents import get_available_agents


console = Console()


@click.group()
@click.version_option(version="0.5.1")
def cli():
    """Victor AI Multi-Agent Research Team - Advanced multi-agent coordination."""
    pass


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--formation",
    "-f",
    type=click.Choice(["pipeline", "parallel", "hierarchical", "sequential", "consensus"]),
    default="parallel",
    help="Team formation type",
)
@click.option("--roles", "-r", multiple=True, help="Agent roles to include")
@click.option("--max-iterations", type=int, default=3, help="Maximum team iterations")
@click.option("--output", "-o", type=click.Path(), help="Output file for report")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def research(
    query: str,
    formation: str,
    roles: tuple,
    max_iterations: int,
    output: Optional[str],
    verbose: bool,
):
    """Execute research task with multi-agent team.

    Examples:
        victor-team research "Analyze AI trends in 2024"
        victor-team research "Compare cloud providers" --formation parallel
        victor-team research "Review renewable energy" --roles researcher,analyst,writer
    """
    asyncio.run(_research(query, formation, roles, max_iterations, output, verbose))


async def _research(
    query: str,
    formation: str,
    roles: tuple,
    max_iterations: int,
    output: Optional[str],
    verbose: bool,
):
    """Execute research task."""
    # Display banner
    console.print(
        Panel.fit(
            "[bold blue]Victor AI Multi-Agent Research Team[/bold blue]\n"
            f"Formation: {formation}\n"
            f"Query: {query}",
            border_style="blue",
        )
    )

    # Create orchestrator
    console.print("[dim]Initializing Victor AI orchestrator...[/dim]")
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    # Create research team
    if roles:
        agent_list = [get_available_agents()[r] for r in roles if r in get_available_agents()]
    else:
        # Use default team
        agent_list = [
            get_available_agents()["researcher"],
            get_available_agents()["analyst"],
            get_available_agents()["writer"],
        ]

    console.print(f"[cyan]Creating {formation} team with {len(agent_list)} agents...[/cyan]\n")

    team = ResearchTeam(
        orchestrator=orchestrator,
        formation=TeamFormation(formation),
        agents=agent_list,
        max_iterations=max_iterations,
    )

    # Execute research with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Researching...", total=None)

        result = await team.research(query, verbose=verbose)

        progress.update(task, completed=True)

    # Display results
    _display_results(result, verbose)

    # Save to file if requested
    if output:
        _save_report(result, output)
        console.print(f"\n[green]Report saved to: {output}[/green]")


def _display_results(result, verbose: bool):
    """Display research results."""
    console.print("\n[bold cyan]Research Results:[/bold cyan]\n")

    # Summary table
    summary_table = Table(title="Research Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row(
        "Query", result.query[:50] + "..." if len(result.query) > 50 else result.query
    )
    summary_table.add_row("Formation", result.formation)
    summary_table.add_row("Agents Involved", str(len(result.agent_insights)))
    summary_table.add_row("Execution Time", f"{result.execution_time:.2f}s")
    summary_table.add_row("Iterations", str(result.iterations))

    console.print(summary_table)
    console.print("")

    # Agent insights
    if result.agent_insights:
        console.print("[bold yellow]Agent Insights:[/bold yellow]\n")

        for agent_name, insight in result.agent_insights.items():
            confidence_color = (
                "green"
                if insight["confidence"] > 0.8
                else "yellow" if insight["confidence"] > 0.6 else "red"
            )

            console.print(
                Panel(
                    f"[bold]{agent_name}[/bold]\n"
                    f"Confidence: [{confidence_color}]{insight['confidence']:.1%}[/{confidence_color}]\n"
                    f"Findings: {len(insight.get('findings', []))}\n"
                    f"Time: {insight.get('execution_time', 0):.2f}s",
                    title=agent_name,
                    border_style="cyan",
                )
            )

            if verbose and insight.get("findings"):
                for finding in insight["findings"][:3]:
                    console.print(f"  • {finding}")

    # Final report
    if result.final_report:
        console.print("\n[bold green]Final Report:[/bold green]\n")
        console.print(result.final_report)


def _save_report(result, output_path: str):
    """Save report to file."""
    from datetime import datetime

    report_content = f"""# Research Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Formation:** {result.formation}
**Query:** {result.query}

## Agent Insights

"""

    for agent_name, insight in result.agent_insights.items():
        report_content += f"### {agent_name}\n\n"
        report_content += f"Confidence: {insight['confidence']:.1%}\n\n"

        if insight.get("findings"):
            for finding in insight["findings"]:
                report_content += f"- {finding}\n"
        report_content += "\n"

    report_content += f"""## Final Report

{result.final_report}

## Execution Details

- Execution Time: {result.execution_time:.2f}s
- Iterations: {result.iterations}
- Agents: {len(result.agent_insights)}
"""

    with open(output_path, "w") as f:
        f.write(report_content)


@cli.command()
def list_agents():
    """List available agent roles and their capabilities."""
    agents = get_available_agents()

    table = Table(title="Available Agents", show_header=True)
    table.add_column("Agent", style="cyan", width=20)
    table.add_column("Description", style="white", width=40)
    table.add_column("Capabilities", style="green", width=30)

    for name, agent in agents.items():
        table.add_row(name, agent.get("description", ""), ", ".join(agent.get("capabilities", [])))

    console.print(table)


@cli.command()
def interactive():
    """Start interactive research session.

    Example:
        victor-team interactive
    """
    from src.interactive_session import InteractiveSession

    console.print(
        Panel.fit(
            "[bold blue]Victor AI Interactive Research Team[/bold blue]\n"
            "Type your research queries for real-time analysis",
            border_style="blue",
        )
    )

    # Create orchestrator
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    # Start interactive session
    session = InteractiveSession(orchestrator)
    asyncio.run(session.run())


if __name__ == "__main__":
    cli()
