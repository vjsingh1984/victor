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

"""Team monitoring example demonstrating observability features.

This example shows how to:
- Enable team metrics collection
- Subscribe to team events
- Query metrics and traces
- Visualize team execution

Usage:
    python examples/observability/team_monitoring.py
"""

import asyncio
import json
import logging
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def subscribe_to_team_events() -> None:
    """Subscribe to and display team execution events."""
    from victor.core.events import ObservabilityBus

    bus = ObservabilityBus.get_instance()

    # Subscribe to team execution started
    async def on_team_started(event: Any):
        data = event.data
        logger.info(
            f"🚀 Team started: {data.get('team_id')} "
            f"(formation={data.get('formation')}, members={data.get('member_count')})"
        )

    # Subscribe to team execution completed
    async def on_team_completed(event: Any):
        data = event.data
        success_icon = "✅" if data.get("success") else "❌"
        logger.info(
            f"{success_icon} Team completed: {data.get('team_id')} "
            f"in {data.get('duration_seconds', 0):.2f}s"
        )

    # Subscribe to member completion
    async def on_member_completed(event: Any):
        data = event.data
        success_icon = "✅" if data.get("success") else "❌"
        logger.info(
            f"{success_icon} Member {data.get('member_id')} completed: "
            f"{data.get('duration_seconds', 0):.2f}s, "
            f"{data.get('tool_calls_used', 0)} tools"
        )

    # Subscribe to recursion depth exceeded
    async def on_depth_exceeded(event: Any):
        data = event.data
        logger.warning(
            f"⚠️ Recursion depth exceeded: {data.get('current_depth')}/{data.get('max_depth')}"
        )
        logger.warning(f"Execution stack: {' -> '.join(data.get('execution_stack', []))}")

    # Subscribe to all team events
    bus.subscribe("team.execution.started", on_team_started)
    bus.subscribe("team.execution.completed", on_team_completed)
    bus.subscribe("team.member.completed", on_member_completed)
    bus.subscribe("team.recursion.depth_exceeded", on_depth_exceeded)

    logger.info("✓ Subscribed to team events")


def display_metrics_summary() -> None:
    """Display summary of team metrics."""
    from victor.workflows.team_metrics import get_team_metrics_collector

    collector = get_team_metrics_collector()
    summary = collector.get_summary()

    print("\n" + "=" * 70)
    print("TEAM METRICS SUMMARY")
    print("=" * 70)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total teams executed: {summary['total_teams_executed']}")
    print(f"  Successful teams: {summary['successful_teams']}")
    print(f"  Failed teams: {summary['failed_teams']}")
    print(f"  Active teams: {summary['active_teams']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")

    print(f"\n⏱️ Performance:")
    print(f"  Average duration: {summary['average_duration_seconds']:.2f}s")
    print(f"  Average member count: {summary['average_member_count']:.1f}")
    print(f"  Total tool calls: {summary['total_tool_calls']}")

    if summary["formation_distribution"]:
        print(f"\n🎯 Formation Distribution:")
        for formation, count in summary["formation_distribution"].items():
            print(f"  {formation}: {count} executions")

    print("=" * 70)


def display_formation_stats() -> None:
    """Display formation-specific statistics."""
    from victor.workflows.team_metrics import get_team_metrics_collector

    collector = get_team_metrics_collector()

    formations = ["sequential", "parallel", "pipeline", "hierarchical", "consensus"]

    print("\n" + "=" * 70)
    print("FORMATION-SPECIFIC STATISTICS")
    print("=" * 70)

    for formation in formations:
        stats = collector.get_formation_stats(formation)
        if stats["total_executions"] > 0:
            success_rate = stats["successful_executions"] / stats["total_executions"]
            print(f"\n📈 {formation.upper()}:")
            print(f"  Executions: {stats['total_executions']}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Average duration: {stats['average_duration_seconds']:.2f}s")
            print(f"  Average member count: {stats['average_member_count']:.1f}")

    print("=" * 70)


def display_recursion_depth_stats() -> None:
    """Display recursion depth statistics."""
    from victor.workflows.team_metrics import get_team_metrics_collector

    collector = get_team_metrics_collector()
    stats = collector.get_recursion_depth_stats()

    print("\n" + "=" * 70)
    print("RECURSION DEPTH ANALYSIS")
    print("=" * 70)

    print(f"\n📏 Depth Statistics:")
    print(f"  Max depth observed: {stats['max_depth_observed']}")
    print(f"  Average depth: {stats['average_depth']:.2f}")

    if stats["depth_distribution"]:
        print(f"\n📊 Depth Distribution:")
        for depth, count in sorted(stats["depth_distribution"].items()):
            bar = "█" * count
            print(f"  Depth {depth}: {bar} ({count})")

    print("=" * 70)


def display_team_details(team_id: str) -> None:
    """Display detailed metrics for a specific team."""
    from victor.workflows.team_metrics import get_team_metrics_collector

    collector = get_team_metrics_collector()
    team_metrics = collector.get_team_metrics(team_id)

    if not team_metrics:
        logger.warning(f"Team '{team_id}' not found")
        return

    print("\n" + "=" * 70)
    print(f"TEAM DETAILS: {team_id}")
    print("=" * 70)

    print(f"\n📋 Team Information:")
    print(f"  Formation: {team_metrics.formation}")
    print(f"  Member count: {team_metrics.member_count}")
    print(f"  Recursion depth: {team_metrics.recursion_depth}")
    print(f"  Success: {team_metrics.success}")
    print(f"  Duration: {team_metrics.duration_seconds:.2f}s")
    print(f"  Total tool calls: {team_metrics.total_tool_calls}")

    if team_metrics.unique_tools_used:
        print(f"  Unique tools: {', '.join(sorted(team_metrics.unique_tools_used))}")

    if team_metrics.consensus_achieved is not None:
        print(f"\n🤝 Consensus:")
        print(f"  Achieved: {team_metrics.consensus_achieved}")
        if team_metrics.consensus_rounds:
            print(f"  Rounds: {team_metrics.consensus_rounds}")

    print(f"\n👥 Member Performance:")
    for member_id, member_metrics in team_metrics.member_metrics.items():
        status_icon = "✅" if member_metrics.success else "❌"
        print(f"\n  {status_icon} {member_id}:")
        print(f"    Role: {member_metrics.role}")
        print(f"    Duration: {member_metrics.duration_seconds:.2f}s")
        print(f"    Tool calls: {member_metrics.tool_calls_used}")
        if member_metrics.tools_used:
            print(f"    Tools: {', '.join(sorted(member_metrics.tools_used))}")
        if member_metrics.error_message:
            print(f"    Error: {member_metrics.error_message}")

    print("=" * 70)


def display_traces() -> None:
    """Display distributed traces."""
    from victor.workflows.team_tracing import get_all_traces

    traces = get_all_traces()

    if not traces:
        print("\nNo traces available")
        return

    print("\n" + "=" * 70)
    print("DISTRIBUTED TRACES")
    print("=" * 70)

    for trace in traces[:5]:  # Show first 5 traces
        print(f"\n🔍 Trace ID: {trace['trace_id']}")
        print(f"  Span count: {trace['span_count']}")

        for span in trace["spans"]:
            indent = "  " if span["parent_span_id"] else ""
            status_icon = "✅" if span["status"] == 0 else "❌"
            print(f"\n{indent}{status_icon} {span['name']}")
            print(f"{indent}  Duration: {span['duration_seconds']:.3f}s")
            print(f"{indent}  Kind: {span['kind']}")

            # Display key attributes
            attrs = span.get("attributes", {})
            if attrs:
                if "team_id" in attrs:
                    print(f"{indent}  Team: {attrs['team_id']}")
                if "member_id" in attrs:
                    print(f"{indent}  Member: {attrs['member_id']}")
                if "formation" in attrs:
                    print(f"{indent}  Formation: {attrs['formation']}")

    print("=" * 70)


async def simulate_team_execution() -> None:
    """Simulate team execution for demonstration."""
    from victor.workflows.team_metrics import get_team_metrics_collector
    from victor.workflows.team_tracing import trace_team_execution, trace_member_execution

    collector = get_team_metrics_collector()

    print("\n" + "=" * 70)
    print("SIMULATING TEAM EXECUTION")
    print("=" * 70)

    # Simulate parallel team execution
    team_id = "code_review_team"

    with trace_team_execution(team_id, "parallel", 3) as span:
        span.set_attribute("task", "Review PR #123")
        span.set_attribute("repository", "victor")

        # Record team start
        collector.record_team_start(
            team_id=team_id,
            formation="parallel",
            member_count=3,
            recursion_depth=1,
        )

        # Simulate member executions
        members = [
            ("security_reviewer", "reviewer", True, 5.2, 8),
            ("quality_reviewer", "reviewer", True, 6.1, 12),
            ("performance_reviewer", "reviewer", True, 4.8, 6),
        ]

        for member_id, role, success, duration, tool_calls in members:
            with trace_member_execution(team_id, member_id, role) as member_span:
                member_span.set_attribute("tool_calls", tool_calls)
                member_span.set_attribute("duration_seconds", duration)

                # Simulate work
                await asyncio.sleep(0.1)

                # Record member completion
                collector.record_member_complete(
                    team_id=team_id,
                    member_id=member_id,
                    success=success,
                    duration_seconds=duration,
                    tool_calls_used=tool_calls,
                    tools_used={"read", "search", "edit"},
                    role=role,
                )

        # Record team completion
        collector.record_team_complete(
            team_id=team_id,
            success=True,
            duration_seconds=6.5,
        )

    print(f"\n✓ Simulated execution for team '{team_id}'")

    # Simulate consensus team execution
    team_id_2 = "decision_team"

    with trace_team_execution(team_id_2, "consensus", 5) as span:
        span.set_attribute("task", "Decide on implementation approach")

        collector.record_team_start(
            team_id=team_id_2,
            formation="consensus",
            member_count=5,
            recursion_depth=0,
        )

        # Simulate members
        for i in range(5):
            member_id = f"voter_{i}"
            with trace_member_execution(team_id_2, member_id, "voter") as member_span:
                await asyncio.sleep(0.05)

                collector.record_member_complete(
                    team_id=team_id_2,
                    member_id=member_id,
                    success=True,
                    duration_seconds=2.0,
                    tool_calls_used=3,
                    tools_used={"read"},
                    role="voter",
                )

        # Record team completion with consensus
        collector.record_team_complete(
            team_id=team_id_2,
            success=True,
            duration_seconds=8.0,
            consensus_achieved=True,
            consensus_rounds=3,
        )

    print(f"✓ Simulated execution for team '{team_id_2}'")
    print("=" * 70)


async def export_metrics_to_json() -> None:
    """Export metrics to JSON file."""
    from victor.workflows.team_metrics import get_team_metrics_collector

    collector = get_team_metrics_collector()

    # Collect all data
    data = {
        "summary": collector.get_summary(),
        "formation_stats": {},
        "recursion_stats": collector.get_recursion_depth_stats(),
        "team_details": {},
    }

    # Add formation stats
    for formation in ["sequential", "parallel", "pipeline", "hierarchical", "consensus"]:
        data["formation_stats"][formation] = collector.get_formation_stats(formation)

    # Add team details
    # Note: In production, you'd want to add a method to get all team IDs
    summary = collector.get_summary()
    if summary["total_teams_executed"] > 0:
        # Export a few example teams
        data["team_details"]["code_review_team"] = collector.get_team_metrics(
            "code_review_team"
        )
        data["team_details"]["decision_team"] = collector.get_team_metrics("decision_team")

    # Write to file
    output_file = "/tmp/team_metrics.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n✓ Metrics exported to {output_file}")


async def main() -> None:
    """Main function."""
    print("\n" + "=" * 70)
    print("VICTOR TEAM MONITORING EXAMPLE")
    print("=" * 70)

    # Subscribe to events
    await subscribe_to_team_events()

    # Simulate team execution
    await simulate_team_execution()

    # Wait a bit for events to be processed
    await asyncio.sleep(0.5)

    # Display metrics
    display_metrics_summary()
    display_formation_stats()
    display_recursion_depth_stats()

    # Display team details
    display_team_details("code_review_team")
    display_team_details("decision_team")

    # Display traces
    display_traces()

    # Export metrics
    await export_metrics_to_json()

    print("\n✓ Monitoring example complete!")
    print("\nNext steps:")
    print("  - Query specific metrics using get_team_metrics_collector()")
    print("  - Subscribe to additional events for real-time monitoring")
    print("  - Integrate with Prometheus/Grafana for dashboards")
    print("  - Enable OpenTelemetry for distributed tracing")


if __name__ == "__main__":
    asyncio.run(main())
