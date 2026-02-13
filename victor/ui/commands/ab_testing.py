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

"""A/B testing CLI commands for Victor.

Provides commands for creating, managing, and analyzing A/B experiments.

Commands:
    create   - Create a new A/B experiment
    start    - Start an experiment
    stop     - Stop an experiment
    results  - Show experiment results
    list     - List all experiments
    status   - Show experiment status

Example:
    victor ab create experiment_config.yaml
    victor ab start exp_123
    victor ab status exp_123
    victor ab results exp_123
    victor ab stop exp_123
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ab_app = typer.Typer(
    name="ab",
    help="Manage A/B tests for workflow optimization.",
)

console = Console()


@ab_app.command("create")
def create_experiment(
    config_path: Path = typer.Argument(
        ...,
        help="Path to experiment configuration YAML file",
        exists=True,
    ),
):
    """Create a new A/B experiment.

    Args:
        config_path: Path to YAML configuration file

    Example:
        victor ab create experiment_config.yaml
    """

    async def _create():
        from victor.experiments.ab_testing import (
            ABTestManager,
            ExperimentConfig,
            ExperimentMetric,
            ExperimentVariant,
        )
        import yaml

        # Load configuration
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Parse variants
        variants = []
        for v_dict in config_dict.get("variants", []):
            variants.append(
                ExperimentVariant(
                    variant_id=v_dict["variant_id"],
                    name=v_dict["name"],
                    description=v_dict.get("description", ""),
                    workflow_type=v_dict.get("workflow_type", "yaml"),
                    workflow_config=v_dict.get("workflow_config", {}),
                    parameter_overrides=v_dict.get("parameter_overrides", {}),
                    traffic_weight=v_dict.get("traffic_weight", 0.5),
                    is_control=v_dict.get("is_control", False),
                    tags=v_dict.get("tags", {}),
                )
            )

        # Parse primary metric
        primary_dict = config_dict.get("primary_metric", {})
        primary_metric = ExperimentMetric(
            metric_id=primary_dict.get("metric_id", "primary"),
            name=primary_dict.get("name", "Primary Metric"),
            description=primary_dict.get("description", ""),
            metric_type=primary_dict.get("metric_type", "execution_time"),
            optimization_goal=primary_dict.get("optimization_goal", "maximize"),
            success_threshold=primary_dict.get("success_threshold"),
            relative_improvement=primary_dict.get("relative_improvement"),
            aggregation_method=primary_dict.get("aggregation_method", "mean"),
        )

        # Parse secondary metrics
        secondary_metrics = []
        for m_dict in config_dict.get("secondary_metrics", []):
            secondary_metrics.append(
                ExperimentMetric(
                    metric_id=m_dict["metric_id"],
                    name=m_dict["name"],
                    description=m_dict.get("description", ""),
                    metric_type=m_dict.get("metric_type", "custom"),
                    optimization_goal=m_dict.get("optimization_goal", "maximize"),
                    aggregation_method=m_dict.get("aggregation_method", "mean"),
                )
            )

        # Create experiment config
        config = ExperimentConfig(
            name=config_dict.get("name", ""),
            description=config_dict.get("description", ""),
            hypothesis=config_dict.get("hypothesis", ""),
            variants=variants,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            min_sample_size=config_dict.get("min_sample_size", 100),
            max_duration_seconds=config_dict.get("max_duration_seconds"),
            max_iterations=config_dict.get("max_iterations"),
            significance_level=config_dict.get("significance_level", 0.05),
            statistical_power=config_dict.get("statistical_power", 0.8),
            enable_early_stopping=config_dict.get("enable_early_stopping", False),
            early_stopping_threshold=config_dict.get("early_stopping_threshold", 0.99),
            targeting_rules=config_dict.get("targeting_rules", {}),
            tags=config_dict.get("tags", {}),
        )

        # Create experiment
        manager = ABTestManager()
        experiment_id = await manager.create_experiment(config)

        console.print(
            Panel(
                f"[bold green]Experiment Created![/]\n\n"
                f"ID: {experiment_id}\n"
                f"Name: {config.name}\n"
                f"Variants: {len(variants)}\n"
                f"Status: draft",
                title="A/B Test",
            )
        )

        console.print(f"\n[yellow]Next steps:[/]")
        console.print(f"  victor ab start {experiment_id}")

    asyncio.run(_create())


@ab_app.command("start")
def start_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
):
    """Start an A/B experiment.

    Args:
        experiment_id: Experiment identifier

    Example:
        victor ab start exp_123
    """

    async def _start():
        from victor.experiments.ab_testing import ABTestManager

        manager = ABTestManager()

        # Check experiment exists
        experiment = await manager.get_experiment(experiment_id)
        if not experiment:
            console.print(f"[bold red]Error:[/] Experiment {experiment_id} not found")
            raise typer.Exit(1)

        # Start experiment
        await manager.start_experiment(experiment_id)

        console.print(
            Panel(
                f"[bold green]Experiment Started![/]\n\n"
                f"ID: {experiment_id}\n"
                f"Name: {experiment.name}\n"
                f"Variants: {len(experiment.variants)}\n"
                f"Status: running",
                title="A/B Test",
            )
        )

    asyncio.run(_start())


@ab_app.command("stop")
def stop_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
):
    """Stop an A/B experiment.

    Args:
        experiment_id: Experiment identifier

    Example:
        victor ab stop exp_123
    """

    async def _stop():
        from victor.experiments.ab_testing import ABTestManager

        manager = ABTestManager()

        # Stop experiment
        await manager.stop_experiment(experiment_id)

        console.print(
            Panel(
                f"[bold yellow]Experiment Stopped[/]\n\n"
                f"ID: {experiment_id}\n"
                f"Status: completed",
                title="A/B Test",
            )
        )

    asyncio.run(_stop())


@ab_app.command("status")
def show_status(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
):
    """Show experiment status.

    Args:
        experiment_id: Experiment identifier

    Example:
        victor ab status exp_123
    """

    async def _status():
        from victor.experiments.ab_testing import ABTestManager

        manager = ABTestManager()

        # Get experiment
        experiment = await manager.get_experiment(experiment_id)
        if not experiment:
            console.print(f"[bold red]Error:[/] Experiment {experiment_id} not found")
            raise typer.Exit(1)

        # Get status
        status = await manager.get_status(experiment_id)

        # Create table
        table = Table(title=f"Experiment Status: {experiment.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("ID", experiment_id)
        table.add_row("Name", experiment.name)
        table.add_row("Description", experiment.description or "N/A")
        table.add_row("Status", f"[bold]{status.status}[/]")
        table.add_row("Total Samples", str(status.total_samples))

        if status.started_at:
            from datetime import datetime

            started = datetime.fromtimestamp(status.started_at).strftime("%Y-%m-%d %H:%M:%S")
            table.add_row("Started", started)

        # Variant samples
        console.print(table)

        if status.variant_samples:
            variant_table = Table(title="Variant Distribution")
            variant_table.add_column("Variant", style="cyan")
            variant_table.add_column("Samples", style="green")
            variant_table.add_column("Percentage", style="yellow")

            for variant_id, count in status.variant_samples.items():
                percentage = (count / status.total_samples * 100) if status.total_samples > 0 else 0
                variant_table.add_row(variant_id, str(count), f"{percentage:.1f}%")

            console.print(variant_table)

    asyncio.run(_status())


@ab_app.command("results")
def show_results(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics"),
):
    """Show experiment results.

    Args:
        experiment_id: Experiment identifier
        detailed: Show detailed statistical analysis

    Example:
        victor ab results exp_123
        victor ab results exp_123 --detailed
    """

    async def _results():
        from victor.experiments.ab_testing import ABTestManager, MetricsCollector

        manager = ABTestManager()
        collector = MetricsCollector()

        # Get experiment
        experiment = await manager.get_experiment(experiment_id)
        if not experiment:
            console.print(f"[bold red]Error:[/] Experiment {experiment_id} not found")
            raise typer.Exit(1)

        # Get metrics for all variants
        all_metrics = await collector.get_all_variant_metrics(experiment_id)

        if not all_metrics:
            console.print("[yellow]No metrics collected yet[/]")
            raise typer.Exit(0)

        # Create table
        table = Table(title=f"Results: {experiment.name}")
        table.add_column("Variant", style="cyan")
        table.add_column("Samples", style="green")
        table.add_column("Mean", style="yellow")
        table.add_column("Std Dev", style="blue")
        table.add_column("Success Rate", style="magenta")

        for variant_id, metrics in all_metrics.items():
            table.add_row(
                variant_id,
                str(metrics.sample_count),
                f"{metrics.execution_time_mean:.2f}s",
                f"{metrics.execution_time_std:.2f}s",
                f"{metrics.success_rate:.1%}",
            )

        console.print(table)

        if detailed:
            # Show detailed statistics
            console.print("\n[bold]Detailed Statistics:[/]")

            for variant_id, metrics in all_metrics.items():
                console.print(f"\n[cyan]{variant_id}[/]")
                console.print(f"  Execution Time:")
                console.print(f"    Mean:   {metrics.execution_time_mean:.2f}s")
                console.print(f"    Median: {metrics.execution_time_median:.2f}s")
                console.print(f"    P95:    {metrics.execution_time_p95:.2f}s")
                console.print(
                    f"    95% CI: ({metrics.execution_time_ci[0]:.2f}s, {metrics.execution_time_ci[1]:.2f}s)"
                )

                console.print(f"  Token Usage:")
                console.print(f"    Mean:   {metrics.total_tokens_mean:.0f}")
                console.print(f"    Median: {metrics.total_tokens_median:.0f}")
                console.print(f"    Total:  {metrics.total_tokens_sum}")

                console.print(f"  Tool Calls:")
                console.print(f"    Mean:   {metrics.tool_calls_mean:.1f}")

                console.print(f"  Success Rate:")
                console.print(f"    Rate:   {metrics.success_rate:.1%}")
                console.print(
                    f"    95% CI: ({metrics.success_rate_ci[0]:.1%}, {metrics.success_rate_ci[1]:.1%})"
                )

                console.print(f"  Cost:")
                console.print(f"    Total:     ${metrics.total_cost:.4f}")
                console.print(f"    Per Exec:  ${metrics.cost_per_execution_mean:.4f}")

    asyncio.run(_results())


@ab_app.command("list")
def list_experiments(
    status_filter: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (draft, running, paused, completed, analyzed)",
    ),
):
    """List all experiments.

    Args:
        status_filter: Optional status filter

    Example:
        victor ab list
        victor ab list --status running
    """

    async def _list():
        import sqlite3
        from pathlib import Path

        storage_path = Path("~/.victor/ab_tests.db").expanduser()

        if not storage_path.exists():
            console.print("[yellow]No experiments found[/]")
            raise typer.Exit(0)

        conn = sqlite3.connect(str(storage_path))
        cursor = conn.cursor()

        if status_filter:
            cursor.execute(
                """
                SELECT experiment_id, name, status, created_at, started_at, completed_at
                FROM experiments
                WHERE status = ?
                ORDER BY created_at DESC
            """,
                (status_filter,),
            )
        else:
            cursor.execute("""
                SELECT experiment_id, name, status, created_at, started_at, completed_at
                FROM experiments
                ORDER BY created_at DESC
            """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            console.print("[yellow]No experiments found[/]")
            raise typer.Exit(0)

        # Create table
        table = Table(title="A/B Experiments")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="blue")

        from datetime import datetime

        for row in rows:
            experiment_id, name, status, created_at, started_at, completed_at = row
            created = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")

            # Color code status
            status_colored = status
            if status == "running":
                status_colored = f"[bold green]{status}[/]"
            elif status == "completed":
                status_colored = f"[bold yellow]{status}[/]"
            elif status == "draft":
                status_colored = f"[bold blue]{status}[/]"

            table.add_row(experiment_id, name, status_colored, created)

        console.print(table)

    asyncio.run(_list())
