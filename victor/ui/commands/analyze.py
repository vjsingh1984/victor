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

"""Analyze CLI commands for Victor.

Provides codebase structure analysis and TDD prioritization:

Usage:
    victor analyze --tdd-priority --top 10
    victor analyze --hotspots --top 20
    victor analyze --coupling
    victor analyze --tdd-priority --refresh --format json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(name="analyze", help="Analyze codebase structure and identify priorities.")


@app.callback(invoke_without_command=True)
def analyze(
    tdd_priority: bool = typer.Option(
        False, "--tdd-priority", help="Rank modules by test-writing priority"
    ),
    hotspots: bool = typer.Option(False, "--hotspots", help="Show hotspot modules"),
    coupling: bool = typer.Option(False, "--coupling", help="Show module coupling metrics"),
    cohesion: bool = typer.Option(False, "--cohesion", help="Show module cohesion metrics"),
    top: int = typer.Option(20, "--top", "-n", help="Number of results to show"),
    refresh: bool = typer.Option(False, "--refresh", help="Recompute metrics (ignore cache)"),
    output_format: str = typer.Option("table", "--format", help="Output format: table or json"),
    project_path: str = typer.Option(".", "--project", "-p", help="Project path"),
) -> None:
    """Analyze codebase structure and identify priorities."""
    from victor.analysis.module_analyzer import ModuleAnalyzer

    project = Path(project_path).resolve()
    analyzer = ModuleAnalyzer(project_path=project)

    # Determine which metric to show
    if not any([tdd_priority, hotspots, coupling, cohesion]):
        # Default to hotspots
        hotspots = True

    if tdd_priority:
        order_by = "tdd_priority"
        title = "TDD Priority Ranking"
    elif coupling:
        order_by = "instability"
        title = "Module Coupling Analysis"
    elif cohesion:
        order_by = "cohesion_lcom4"
        title = "Module Cohesion Analysis"
    else:
        order_by = "hotspot_score"
        title = "Codebase Hotspots"

    # Compute or use cache
    if refresh or not analyzer.has_cached_metrics():
        typer.echo("Computing module metrics...")
        metrics = analyzer.compute_all()
        if metrics:
            analyzer.persist(metrics)
            typer.echo(f"Computed metrics for {len(metrics)} modules.")
        else:
            typer.echo("No modules found. Run 'victor index' first.", err=True)
            raise typer.Exit(1)

    results = analyzer.get_cached(order_by=order_by, limit=top)

    if not results:
        typer.echo("No metrics available. Run 'victor index' first.", err=True)
        raise typer.Exit(1)

    if output_format == "json":
        typer.echo(json.dumps(results, indent=2))
        return

    # Rich table output
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title, show_lines=False)

        table.add_column("#", style="dim", width=4)
        table.add_column("Module", style="cyan", max_width=50)

        if tdd_priority:
            table.add_column("Priority", style="bold red", justify="right")
            table.add_column("PageRank", justify="right")
            table.add_column("Coupling", justify="right")
            table.add_column("Changes", justify="right")
        elif coupling:
            table.add_column("Ca", justify="right", help="Afferent coupling")
            table.add_column("Ce", justify="right", help="Efferent coupling")
            table.add_column("Instability", style="bold", justify="right")
            table.add_column("Distance", justify="right")
        elif cohesion:
            table.add_column("LCOM4", style="bold", justify="right")
            table.add_column("Symbols", justify="right")
        else:
            table.add_column("Hotspot", style="bold red", justify="right")
            table.add_column("PageRank", justify="right")
            table.add_column("Coupling", justify="right")
            table.add_column("Changes", justify="right")

        for i, r in enumerate(results, 1):
            module = r.get("module_path", "?")
            # Truncate long paths
            if len(module) > 48:
                module = "..." + module[-45:]

            if tdd_priority:
                table.add_row(
                    str(i),
                    module,
                    f"{r.get('tdd_priority', 0):.4f}",
                    f"{r.get('pagerank_score', 0):.4f}",
                    f"{r.get('instability', 0):.2f}",
                    str(r.get("change_frequency", 0)),
                )
            elif coupling:
                table.add_row(
                    str(i),
                    module,
                    str(r.get("afferent_coupling", 0)),
                    str(r.get("efferent_coupling", 0)),
                    f"{r.get('instability', 0):.3f}",
                    f"{r.get('distance_main_seq', 0):.3f}",
                )
            elif cohesion:
                table.add_row(
                    str(i),
                    module,
                    f"{r.get('cohesion_lcom4', 0):.3f}",
                    str(r.get("symbol_count", 0)),
                )
            else:
                table.add_row(
                    str(i),
                    module,
                    f"{r.get('hotspot_score', 0):.4f}",
                    f"{r.get('pagerank_score', 0):.4f}",
                    f"{r.get('instability', 0):.2f}",
                    str(r.get("change_frequency", 0)),
                )

        console.print(table)

    except ImportError:
        # Fallback to plain text if rich not available
        typer.echo(f"\n{title}\n{'=' * len(title)}")
        for i, r in enumerate(results, 1):
            module = r.get("module_path", "?")
            score = r.get(order_by, 0)
            typer.echo(f"  {i:3d}. {module:<50s} {score:.4f}")
