#!/usr/bin/env python3
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

"""Generate coordinator performance metrics for the dashboard.

This script runs benchmarks and collects metrics from the coordinator-based
orchestrator, exporting them in JSON format for the dashboard.

Usage:
    python scripts/generate_coordinator_metrics.py
    python scripts/generate_coordinator_metrics.py --output metrics.json
    python scripts/generate_coordinator_metrics.py --format prometheus
    python scripts/generate_coordinator_metrics.py --benchmark --iterations 100

Options:
    --output PATH     Output file path (default: docs/dashboard/metrics.json)
    --format FORMAT   Output format: json or prometheus (default: json)
    --benchmark       Run benchmarks before collecting metrics
    --iterations N    Number of iterations for benchmarks (default: 50)
    --verbose         Show detailed output
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.observability.coordinator_metrics import CoordinatorMetricsCollector


# =============================================================================
# Mock Benchmark (for demonstration)
# =============================================================================


class MockCoordinators:
    """Mock coordinators for benchmarking."""

    @staticmethod
    async def mock_config_coordinator():
        """Mock ConfigCoordinator work."""
        await asyncio.sleep(0.002)  # 2ms
        return {"config": "loaded"}

    @staticmethod
    async def mock_prompt_coordinator():
        """Mock PromptCoordinator work."""
        await asyncio.sleep(0.005)  # 5ms
        return {"prompt": "built"}

    @staticmethod
    async def mock_context_coordinator():
        """Mock ContextCoordinator work."""
        await asyncio.sleep(0.008)  # 8ms
        return {"context": "managed"}

    @staticmethod
    async def mock_chat_coordinator():
        """Mock ChatCoordinator work."""
        await asyncio.sleep(0.015)  # 15ms
        return {"chat": "completed"}

    @staticmethod
    async def mock_tool_coordinator():
        """Mock ToolCoordinator work."""
        await asyncio.sleep(0.010)  # 10ms
        return {"tools": "executed"}

    @staticmethod
    async def mock_session_coordinator():
        """Mock SessionCoordinator work."""
        await asyncio.sleep(0.003)  # 3ms
        return {"session": "managed"}

    @staticmethod
    async def mock_analytics_coordinator():
        """Mock AnalyticsCoordinator work."""
        await asyncio.sleep(0.004)  # 4ms
        return {"analytics": "tracked"}

    @staticmethod
    async def mock_provider_coordinator():
        """Mock ProviderCoordinator work."""
        await asyncio.sleep(0.006)  # 6ms
        return {"provider": "switched"}

    @staticmethod
    async def mock_mode_coordinator():
        """Mock ModeCoordinator work."""
        await asyncio.sleep(0.002)  # 2ms
        return {"mode": "checked"}

    @staticmethod
    async def mock_evaluation_coordinator():
        """Mock EvaluationCoordinator work."""
        await asyncio.sleep(0.007)  # 7ms
        return {"evaluation": "completed"}

    @staticmethod
    async def mock_workflow_coordinator():
        """Mock WorkflowCoordinator work."""
        await asyncio.sleep(0.004)  # 4ms
        return {"workflow": "executed"}

    @staticmethod
    async def mock_checkpoint_coordinator():
        """Mock CheckpointCoordinator work."""
        await asyncio.sleep(0.003)  # 3ms
        return {"checkpoint": "saved"}

    @staticmethod
    async def mock_tool_selection_coordinator():
        """Mock ToolSelectionCoordinator work."""
        await asyncio.sleep(0.005)  # 5ms
        return {"tools": "selected"}


async def run_benchmark(
    collector: CoordinatorMetricsCollector,
    iterations: int = 50,
    verbose: bool = False,
) -> None:
    """Run benchmark against mock coordinators.

    Args:
        collector: Metrics collector instance
        iterations: Number of iterations
        verbose: Show detailed output
    """
    print(f"Running benchmarks with {iterations} iterations...")

    coordinators = [
        ("ConfigCoordinator", MockCoordinators.mock_config_coordinator),
        ("PromptCoordinator", MockCoordinators.mock_prompt_coordinator),
        ("ContextCoordinator", MockCoordinators.mock_context_coordinator),
        ("ChatCoordinator", MockCoordinators.mock_chat_coordinator),
        ("ToolCoordinator", MockCoordinators.mock_tool_coordinator),
        ("SessionCoordinator", MockCoordinators.mock_session_coordinator),
        ("AnalyticsCoordinator", MockCoordinators.mock_analytics_coordinator),
        ("ProviderCoordinator", MockCoordinators.mock_provider_coordinator),
        ("ModeCoordinator", MockCoordinators.mock_mode_coordinator),
        ("EvaluationCoordinator", MockCoordinators.mock_evaluation_coordinator),
        ("WorkflowCoordinator", MockCoordinators.mock_workflow_coordinator),
        ("CheckpointCoordinator", MockCoordinators.mock_checkpoint_coordinator),
        ("ToolSelectionCoordinator", MockCoordinators.mock_tool_selection_coordinator),
    ]

    # Simulate cache hits/misses
    cache_hit_rate = 0.75

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(f"  Iteration {i + 1}/{iterations}...")

        for coordinator_name, coordinator_func in coordinators:
            # Track execution
            with collector.track_coordinator(coordinator_name):
                result = await coordinator_func()

            # Simulate cache activity
            if i % 2 == 0:  # Every other operation involves cache
                if (i * len(coordinators) + hash(coordinator_name)) % 100 < (cache_hit_rate * 100):
                    collector.record_cache_hit(coordinator_name)
                else:
                    collector.record_cache_miss(coordinator_name)

        # Simulate analytics events
        if i % 5 == 0:
            collector.record_analytics_event("tool_call")
        if i % 10 == 0:
            collector.record_analytics_event("model_request")
        if i % 20 == 0:
            collector.record_analytics_event("state_transition")

    print(
        f"✓ Benchmarks completed: {iterations} iterations across {len(coordinators)} coordinators"
    )


async def generate_sample_data(collector: CoordinatorMetricsCollector) -> None:
    """Generate sample data for dashboard demonstration.

    Args:
        collector: Metrics collector instance
    """
    print("Generating sample data...")

    # Generate historical data with variations
    coordinators = [
        "ConfigCoordinator",
        "PromptCoordinator",
        "ContextCoordinator",
        "ChatCoordinator",
        "ToolCoordinator",
        "SessionCoordinator",
        "AnalyticsCoordinator",
        "ProviderCoordinator",
        "ModeCoordinator",
        "EvaluationCoordinator",
        "WorkflowCoordinator",
        "CheckpointCoordinator",
        "ToolSelectionCoordinator",
    ]

    # Simulate past executions
    base_time = time.time() - 3600  # 1 hour ago

    for i in range(1000):  # 1000 historical executions
        coordinator = coordinators[i % len(coordinators)]

        # Vary duration based on coordinator and randomness
        base_durations = {
            "ConfigCoordinator": 2.0,
            "PromptCoordinator": 5.0,
            "ContextCoordinator": 8.0,
            "ChatCoordinator": 15.0,
            "ToolCoordinator": 10.0,
            "SessionCoordinator": 3.0,
            "AnalyticsCoordinator": 4.0,
            "ProviderCoordinator": 6.0,
            "ModeCoordinator": 2.0,
            "EvaluationCoordinator": 7.0,
            "WorkflowCoordinator": 4.0,
            "CheckpointCoordinator": 3.0,
            "ToolSelectionCoordinator": 5.0,
        }

        base_duration = base_durations[coordinator]
        variation = (i % 10) / 10.0  # 0-1 variation
        duration = base_duration * (0.8 + variation * 0.4)  # ±20% variation

        # Simulate occasional errors
        success = (i % 50) != 0  # 2% error rate

        # Simulate cache activity
        if i % 3 == 0:
            if (i % 10) < 7:  # 70% hit rate
                collector.record_cache_hit(coordinator)
            else:
                collector.record_cache_miss(coordinator)

        # Record execution with timestamp
        execution_time = base_time + (i * 3.6)  # Spread over 1 hour
        collector.track_execution(
            coordinator_name=coordinator,
            duration_ms=duration,
            success=success,
            error_message=None if success else "Simulated error",
        )

    print("✓ Sample data generated")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate coordinator performance metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        "-o",
        default="docs/dashboard/metrics.json",
        help="Output file path (default: docs/dashboard/metrics.json)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "prometheus"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run benchmarks before collecting metrics",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )

    parser.add_argument(
        "--sample-data",
        "-s",
        action="store_true",
        help="Generate sample historical data",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Create metrics collector
    collector = CoordinatorMetricsCollector()

    # Generate sample data if requested
    if args.sample_data:
        await generate_sample_data(collector)

    # Run benchmarks if requested
    if args.benchmark:
        await run_benchmark(collector, args.iterations, args.verbose)

    # Export metrics
    print(f"\nExporting metrics to {args.output}...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "prometheus":
        metrics_data = collector.export_prometheus()
        with open(output_path, "w") as f:
            f.write(metrics_data)
    else:
        metrics_data = collector.export_json(include_history=True)
        with open(output_path, "w") as f:
            f.write(metrics_data)

    print(f"✓ Metrics exported successfully")

    # Print summary
    print("\n=== Metrics Summary ===")
    overall = collector.get_overall_stats()
    print(f"Total executions: {overall['total_executions']}")
    print(f"Total coordinators: {overall['total_coordinators']}")
    print(f"Total errors: {overall['total_errors']}")
    print(f"Overall error rate: {overall['overall_error_rate']:.2%}")
    print(f"Uptime: {overall['uptime_seconds']:.2f}s")

    print("\n=== Top Coordinators by Execution Count ===")
    snapshots = collector.get_all_snapshots()
    sorted_snapshots = sorted(snapshots, key=lambda s: s.execution_count, reverse=True)
    for snapshot in sorted_snapshots[:5]:
        print(f"  {snapshot.coordinator_name}: {snapshot.execution_count} executions")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
