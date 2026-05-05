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

"""Bayesian orchestration monitoring CLI command."""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

import typer

from victor.core.database import get_database
from victor.framework.rl.monitoring.bayesian_monitor import (
    ASCIIChartRenderer,
    BayesianMetricsMonitor,
    MetricsExporter,
)

app = typer.Typer(help="Bayesian orchestration monitoring and metrics")


@app.command()
def summary(
    days: int = typer.Option(7, help="Number of days to look back"),
    output: Optional[str] = typer.Option(None, help="Export summary to JSON file"),
):
    """Show system summary of Bayesian orchestration metrics."""
    conn = get_database()
    monitor = BayesianMetricsMonitor(conn)

    # Get summary
    system_summary = monitor.get_system_summary(days)

    # Display summary
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Bayesian Orchestration System Summary (Last {days} days)")
    typer.echo(f"{'='*60}\n")

    typer.echo("Belief States:")
    typer.echo(f"  Unique belief states: {system_summary['belief_states']['unique_belief_states']}")

    typer.echo("\nAgent Reliability:")
    typer.echo(f"  Tracked agents: {system_summary['agent_reliability']['tracked_agents']}")

    typer.echo("\nObservation Models:")
    typer.echo(f"  Observation records: {system_summary['observation_models']['observation_records']}")

    typer.echo("\nValue of Information:")
    typer.echo(f"  VoI queries: {system_summary['voi_queries']['voi_queries']}")

    typer.echo("\nConsensus Decisions:")
    typer.echo(f"  Consensus decisions: {system_summary['consensus_decisions']['consensus_decisions']}")

    typer.echo("\nCorrelations:")
    typer.echo(f"  Correlation pairs: {system_summary['correlations']['correlation_pairs']}")

    # Export if requested
    if output:
        exporter = MetricsExporter(monitor)
        exporter.export_summary_json(output, days)
        typer.echo(f"\n✓ Summary exported to {output}")


@app.command()
def reliability(
    agent_ids: Optional[str] = typer.Option(None, help="Comma-separated list of agent IDs"),
    days: int = typer.Option(7, help="Number of days to look back"),
    export: Optional[str] = typer.Option(None, help="Export to CSV file"),
):
    """Show agent reliability trends."""
    conn = get_database()
    monitor = BayesianMetricsMonitor(conn)

    agent_list = agent_ids.split(",") if agent_ids else None

    # Get trends
    trends = monitor.get_reliability_trends(agent_list, days)

    if not trends:
        typer.echo("No reliability data found.")
        return

    # Display latest reliability for each agent
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Agent Reliability (Last {days} days)")
    typer.echo(f"{'='*60}\n")

    for agent_id, snapshots in sorted(trends.items()):
        if snapshots:
            latest = snapshots[-1]
            typer.echo(f"{agent_id}:")
            typer.echo(f"  Expected reliability: {latest['expected_reliability']:.2%}")
            typer.echo(f"  Sample count: {latest['sample_count']}")
            typer.echo(f"  Alpha: {latest['alpha']:.2f}, Beta: {latest['beta']:.2f}")

    # Render chart
    chart_data = {}
    for agent_id, snapshots in trends.items():
        if snapshots:
            chart_data[agent_id] = snapshots[-1]["expected_reliability"]

    renderer = ASCIIChartRenderer()
    chart = renderer.render_bar_chart(chart_data, "Reliability Comparison")
    typer.echo(chart)

    # Export if requested
    if export:
        exporter = MetricsExporter(monitor)
        exporter.export_reliability_trends_csv(export, agent_list, days)
        typer.echo(f"\n✓ Reliability trends exported to {export}")


@app.command()
def consensus(
    days: int = typer.Option(7, help="Number of days to look back"),
):
    """Show consensus performance statistics."""
    conn = get_database()
    monitor = BayesianMetricsMonitor(conn)

    # Get performance stats
    stats = monitor.get_consensus_performance(days)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Consensus Performance (Last {days} days)")
    typer.echo(f"{'='*60}\n")

    typer.echo(f"Total consensus decisions: {stats['total_consensus']}")
    typer.echo(f"Correct decisions: {stats['correct_count']}")
    typer.echo(f"Accuracy: {stats['accuracy']:.2%}")
    typer.echo(f"Mean confidence: {stats['mean_confidence']:.2%}")

    typer.echo("\nAgreement Distribution:")
    for level, count in stats["agreement_distribution"].items():
        typer.echo(f"  {level.capitalize()}: {count}")

    typer.echo("\nAccuracy by Agreement Level:")
    for level, level_stats in stats["by_agreement"].items():
        if level_stats["total"] > 0:
            accuracy = level_stats["correct"] / level_stats["total"]
            typer.echo(
                f"  {level.capitalize()}: {level_stats['correct']}/{level_stats['total']} ({accuracy:.2%})"
            )


@app.command()
def voi(
    agent_id: Optional[str] = typer.Option(None, help="Specific agent ID"),
    days: int = typer.Option(7, help="Number of days to look back"),
):
    """Show Value of Information statistics."""
    conn = get_database()
    monitor = BayesianMetricsMonitor(conn)

    # Get VoI stats
    stats = monitor.get_voi_statistics(agent_id, days)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Value of Information Statistics (Last {days} days)")
    typer.echo(f"{'='*60}\n")

    typer.echo(f"Total queries: {stats['total_queries']}")
    typer.echo(f"Beneficial queries: {stats['beneficial_queries']}")
    typer.echo(f"Beneficial rate: {stats['beneficial_rate']:.2%}")
    typer.echo(f"Mean predicted VoI: {stats['mean_predicted_voi']:.4f}")
    typer.echo(f"Mean actual gain: {stats['mean_actual_gain']:.4f}")
    typer.echo(f"Mean query cost: {stats['mean_query_cost']:.4f}")

    if stats["per_agent"]:
        typer.echo("\nPer-Agent Statistics:")
        for agent, agent_stats in sorted(stats["per_agent"].items()):
            typer.echo(f"\n  {agent}:")
            typer.echo(f"    Total queries: {agent_stats['total_queries']}")
            typer.echo(f"    Beneficial rate: {agent_stats['beneficial_rate']:.2%}")
            typer.echo(f"    Mean predicted VoI: {agent_stats['mean_predicted_voi']:.4f}")
            typer.echo(f"    Mean actual gain: {agent_stats['mean_actual_gain']:.4f}")


@app.command()
def correlations(
    agents: str = typer.Argument(..., help="Comma-separated list of agent IDs"),
    days: int = typer.Option(7, help="Number of days to look back"),
):
    """Show correlation matrix heatmap."""
    conn = get_database()
    monitor = BayesianMetricsMonitor(conn)

    agent_list = agents.split(",")

    # Get correlation matrix
    matrix = monitor.get_correlation_matrix(agent_list, days)

    # Render heatmap
    renderer = ASCIIChartRenderer()
    heatmap = renderer.render_heatmap(matrix, f"Agent Correlations (Last {days} days)")
    typer.echo(heatmap)

    # Show highly correlated pairs
    typer.echo("\nHighly Correlated Pairs (|correlation| > 0.7):")

    for i, agent_1 in enumerate(agent_list):
        for agent_2 in agent_list[i + 1 :]:
            corr = matrix[agent_1].get(agent_2, 0.0)
            if abs(corr) > 0.7:
                typer.echo(f"  {agent_1} ↔ {agent_2}: {corr:.3f}")


@app.command()
def belief(
    belief_id: str = typer.Argument(..., help="Belief state ID"),
    export: Optional[str] = typer.Option(None, help="Export evolution to CSV file"),
):
    """Show belief state evolution."""
    conn = get_database()
    monitor = BayesianMetricsMonitor(conn)

    # Get evolution
    evolution = monitor.get_belief_evolution(belief_id)

    if not evolution:
        typer.echo(f"No evolution data found for belief_id: {belief_id}")
        return

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Belief State Evolution: {belief_id}")
    typer.echo(f"{'='*60}\n")

    # Show latest state
    latest = evolution[-1]
    typer.echo("Current State:")
    typer.echo(f"  Success probability: {latest['success_prob']:.2%}")
    typer.echo(f"  Failure probability: {latest['failure_prob']:.2%}")
    typer.echo(f"  Entropy: {latest['entropy']:.4f} nats")
    typer.echo(f"  Updates: {len(evolution)}")

    # Show evolution over time
    if len(evolution) > 1:
        typer.echo("\nEvolution (last 10 updates):")
        for snapshot in evolution[-10:]:
            timestamp = snapshot["timestamp"][:19]  # Truncate milliseconds
            agent = snapshot["agent_id"] or "System"
            typer.echo(
                f"  {timestamp} | {agent:10} | P(success)={snapshot['success_prob']:.2f}, "
                f"H={snapshot['entropy']:.3f}"
            )

    # Export if requested
    if export:
        exporter = MetricsExporter(monitor)
        exporter.export_belief_evolution_csv(belief_id, export)
        typer.echo(f"\n✓ Belief evolution exported to {export}")


if __name__ == "__main__":
    app()
