#!/usr/bin/env python3
"""Load testing script for coordinators.

This script simulates concurrent users and measures:
- Throughput (operations per second)
- Latency under load
- Resource utilization
- Bottleneck identification

Usage:
    python scripts/load_test_coordinators.py [--concurrent-users 10] [--duration 60]

Exit codes:
    0: Load tests completed successfully
    1: Performance thresholds exceeded
    2: Errors encountered during testing
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import psutil
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from victor.config.settings import Settings
from victor.teams import create_coordinator, TeamFormation
from victor.agent.coordinators.checkpoint_coordinator import CheckpointCoordinator
from victor.agent.coordinators.evaluation_coordinator import EvaluationCoordinator
from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""

    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    operations_per_user: int = 100
    think_time_ms: int = 100


@dataclass
class LoadTestResult:
    """Result of a single load test operation."""

    operation_type: str
    user_id: int
    start_time: float
    end_time: float
    success: bool
    error: Optional[str] = None
    memory_mb: float = 0.0

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000


@dataclass
class LoadTestReport:
    """Comprehensive load test report."""

    config: LoadTestConfig
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    results: List[LoadTestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: LoadTestResult) -> None:
        """Add a test result."""
        self.results.append(result)

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.results:
            return {}

        # Group by operation type
        by_operation: Dict[str, List[LoadTestResult]] = {}
        for result in self.results:
            if result.operation_type not in by_operation:
                by_operation[result.operation_type] = []
            by_operation[result.operation_type].append(result)

        metrics = {
            "total_operations": len(self.results),
            "successful_operations": sum(1 for r in self.results if r.success),
            "failed_operations": sum(1 for r in self.results if not r.success),
            "overall_success_rate": sum(1 for r in self.results if r.success) / len(self.results) * 100,
            "test_duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            "throughput_ops_per_second": 0,
            "by_operation": {},
        }

        # Calculate throughput
        if metrics["test_duration_seconds"] > 0:
            metrics["throughput_ops_per_second"] = len(self.results) / metrics["test_duration_seconds"]

        # Calculate per-operation metrics
        for op_type, results in by_operation.items():
            durations = [r.duration_ms for r in results if r.success]
            successful = [r for r in results if r.success]

            op_metrics = {
                "count": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "avg_latency_ms": sum(durations) / len(durations) if durations else 0,
                "min_latency_ms": min(durations) if durations else 0,
                "max_latency_ms": max(durations) if durations else 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
            }

            # Calculate percentiles
            if durations:
                sorted_durations = sorted(durations)
                op_metrics["p50_latency_ms"] = sorted_durations[int(len(sorted_durations) * 0.5)]
                op_metrics["p95_latency_ms"] = sorted_durations[int(len(sorted_durations) * 0.95)]
                op_metrics["p99_latency_ms"] = sorted_durations[int(len(sorted_durations) * 0.99)]

            metrics["by_operation"][op_type] = op_metrics

        return metrics

    def generate_report(self, output_path: Path) -> None:
        """Generate detailed report."""
        metrics = self.calculate_metrics()

        report = {
            "config": {
                "concurrent_users": self.config.concurrent_users,
                "duration_seconds": self.config.duration_seconds,
                "ramp_up_seconds": self.config.ramp_up_seconds,
                "operations_per_user": self.config.operations_per_user,
            },
            "test_start": self.start_time.isoformat(),
            "test_end": self.end_time.isoformat() if self.end_time else None,
            "metrics": metrics,
            "errors": self.errors[:20],  # First 20 errors
        }

        output_path.write_text(json.dumps(report, indent=2))
        console.print(f"[green]Report generated: {output_path}[/green]")


class CoordinatorLoadTester:
    """Load tester for coordinators."""

    def __init__(self, config: LoadTestConfig):
        """Initialize load tester."""
        self.config = config
        self.report = LoadTestReport(config=config)
        self.process = psutil.Process(os.getpid())

    async def run_all_tests(self) -> LoadTestReport:
        """Run all load tests."""
        console.print(Panel.fit("[bold cyan]Starting Coordinator Load Tests[/bold cyan]"))

        # Start memory tracking
        tracemalloc.start()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # Test 1: Coordinator Creation
            task = progress.add_task("Coordinator Creation Load Test", total=1)
            await self._test_coordinator_creation_load(progress, task)

            # Test 2: Team Formation Switching
            task = progress.add_task("Team Formation Load Test", total=1)
            await self._test_team_formation_load(progress, task)

            # Test 3: Checkpoint Operations
            task = progress.add_task("Checkpoint Load Test", total=1)
            await self._test_checkpoint_load(progress, task)

            # Test 4: Evaluation Recording
            task = progress.add_task("Evaluation Load Test", total=1)
            await self._test_evaluation_load(progress, task)

            # Test 5: Metrics Recording
            task = progress.add_task("Metrics Load Test", total=1)
            await self._test_metrics_load(progress, task)

            # Test 6: Concurrent Users Simulation
            task = progress.add_task("Concurrent Users Test", total=1)
            await self._test_concurrent_users(progress, task)

        # Final metrics
        self.report.end_time = datetime.now()
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.report.performance_metrics = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": final_memory - initial_memory,
            "tracemalloc_current_mb": current / 1024 / 1024,
            "tracemalloc_peak_mb": peak / 1024 / 1024,
        }

        return self.report

    async def _test_coordinator_creation_load(self, progress, task) -> None:
        """Test coordinator creation under load."""
        console.print("\n[bold]Test: Coordinator Creation[/bold]")

        async def create_coordinators(user_id: int, count: int):
            """Create multiple coordinators."""
            for i in range(count):
                start = time.time()
                try:
                    coordinator = create_coordinator(lightweight=True)
                    end = time.time()

                    self.report.add_result(
                        LoadTestResult(
                            operation_type="coordinator_creation",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=True,
                            memory_mb=self.process.memory_info().rss / 1024 / 1024,
                        )
                    )

                    # Think time
                    await asyncio.sleep(self.config.think_time_ms / 1000)

                except Exception as e:
                    end = time.time()
                    self.report.add_result(
                        LoadTestResult(
                            operation_type="coordinator_creation",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=False,
                            error=str(e),
                        )
                    )
                    self.report.errors.append(f"User {user_id}: {e}")

        # Launch concurrent users
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task_count = self.config.operations_per_user // self.config.concurrent_users
            tasks.append(create_coordinators(user_id, task_count))

        await asyncio.gather(*tasks)
        progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Created {len(self.report.results)} coordinators")

    async def _test_team_formation_load(self, progress, task) -> None:
        """Test team formation switching under load."""
        console.print("\n[bold]Test: Team Formation Switching[/bold]")

        formations = [
            TeamFormation.SEQUENTIAL,
            TeamFormation.PARALLEL,
            TeamFormation.HIERARCHICAL,
            TeamFormation.PIPELINE,
            TeamFormation.CONSENSUS,
        ]

        async def switch_formations(user_id: int, count: int):
            """Switch team formations."""
            coordinator = create_coordinator(lightweight=True)

            for i in range(count):
                start = time.time()
                try:
                    formation = formations[i % len(formations)]
                    coordinator.set_formation(formation)
                    end = time.time()

                    self.report.add_result(
                        LoadTestResult(
                            operation_type="formation_switch",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=True,
                            memory_mb=self.process.memory_info().rss / 1024 / 1024,
                        )
                    )

                    # Think time
                    await asyncio.sleep(self.config.think_time_ms / 1000)

                except Exception as e:
                    end = time.time()
                    self.report.add_result(
                        LoadTestResult(
                            operation_type="formation_switch",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=False,
                            error=str(e),
                        )
                    )
                    self.report.errors.append(f"User {user_id}: {e}")

        # Launch concurrent users
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task_count = self.config.operations_per_user // self.config.concurrent_users
            tasks.append(switch_formations(user_id, task_count))

        await asyncio.gather(*tasks)
        progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Switched formations {self.config.operations_per_user} times")

    async def _test_checkpoint_load(self, progress, task) -> None:
        """Test checkpoint operations under load."""
        console.print("\n[bold]Test: Checkpoint Operations[/bold]")

        async def checkpoint_operations(user_id: int, count: int):
            """Perform checkpoint operations."""
            coordinator = CheckpointCoordinator()

            for i in range(count):
                # Test checkpoint
                start = time.time()
                try:
                    session_id = f"user_{user_id}_session_{i}"
                    await coordinator.checkpoint(session_id, {"data": f"value_{i}"})

                    # Test restore
                    await coordinator.restore(session_id)

                    end = time.time()

                    self.report.add_result(
                        LoadTestResult(
                            operation_type="checkpoint_operation",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=True,
                            memory_mb=self.process.memory_info().rss / 1024 / 1024,
                        )
                    )

                    # Think time
                    await asyncio.sleep(self.config.think_time_ms / 1000)

                except Exception as e:
                    end = time.time()
                    self.report.add_result(
                        LoadTestResult(
                            operation_type="checkpoint_operation",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=False,
                            error=str(e),
                        )
                    )
                    self.report.errors.append(f"User {user_id}: {e}")

        # Launch concurrent users
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task_count = self.config.operations_per_user // self.config.concurrent_users
            tasks.append(checkpoint_operations(user_id, task_count))

        await asyncio.gather(*tasks)
        progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Performed {self.config.operations_per_user} checkpoint operations")

    async def _test_evaluation_load(self, progress, task) -> None:
        """Test evaluation recording under load."""
        console.print("\n[bold]Test: Evaluation Recording[/bold]")

        async def evaluation_operations(user_id: int, count: int):
            """Record evaluations."""
            coordinator = EvaluationCoordinator()

            for i in range(count):
                start = time.time()
                try:
                    await coordinator.record_evaluation(
                        task_id=f"user_{user_id}_task_{i}",
                        score=0.9,
                        metrics={"metric": i},
                    )
                    end = time.time()

                    self.report.add_result(
                        LoadTestResult(
                            operation_type="evaluation_recording",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=True,
                            memory_mb=self.process.memory_info().rss / 1024 / 1024,
                        )
                    )

                    # Think time
                    await asyncio.sleep(self.config.think_time_ms / 1000)

                except Exception as e:
                    end = time.time()
                    self.report.add_result(
                        LoadTestResult(
                            operation_type="evaluation_recording",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=False,
                            error=str(e),
                        )
                    )
                    self.report.errors.append(f"User {user_id}: {e}")

        # Launch concurrent users
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task_count = self.config.operations_per_user // self.config.concurrent_users
            tasks.append(evaluation_operations(user_id, task_count))

        await asyncio.gather(*tasks)
        progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Recorded {self.config.operations_per_user} evaluations")

    async def _test_metrics_load(self, progress, task) -> None:
        """Test metrics recording under load."""
        console.print("\n[bold]Test: Metrics Recording[/bold]")

        async def metrics_operations(user_id: int, count: int):
            """Record metrics."""
            coordinator = MetricsCoordinator()

            for i in range(count):
                start = time.time()
                try:
                    await coordinator.record_metric(
                        name=f"user_{user_id}_metric",
                        value=float(i),
                        tags={"user": str(user_id)},
                    )
                    end = time.time()

                    self.report.add_result(
                        LoadTestResult(
                            operation_type="metric_recording",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=True,
                            memory_mb=self.process.memory_info().rss / 1024 / 1024,
                        )
                    )

                    # Think time
                    await asyncio.sleep(self.config.think_time_ms / 1000)

                except Exception as e:
                    end = time.time()
                    self.report.add_result(
                        LoadTestResult(
                            operation_type="metric_recording",
                            user_id=user_id,
                            start_time=start,
                            end_time=end,
                            success=False,
                            error=str(e),
                        )
                    )
                    self.report.errors.append(f"User {user_id}: {e}")

        # Launch concurrent users
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task_count = self.config.operations_per_user // self.config.concurrent_users
            tasks.append(metrics_operations(user_id, task_count))

        await asyncio.gather(*tasks)
        progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Recorded {self.config.operations_per_user} metrics")

    async def _test_concurrent_users(self, progress, task) -> None:
        """Test concurrent users simulation."""
        console.print("\n[bold]Test: Concurrent Users Simulation[/bold]")

        async def user_session(user_id: int, duration: int):
            """Simulate a user session."""
            start_time = time.time()
            coordinator = create_coordinator(lightweight=True)

            while (time.time() - start_time) < duration:
                # Perform various operations
                operations = [
                    self._op_formation_switch(coordinator),
                    self._op_checkpoint(),
                    self._op_evaluation(),
                ]

                for op in operations:
                    op_start = time.time()
                    try:
                        await op
                        op_end = time.time()

                        self.report.add_result(
                            LoadTestResult(
                                operation_type="user_session",
                                user_id=user_id,
                                start_time=op_start,
                                end_time=op_end,
                                success=True,
                                memory_mb=self.process.memory_info().rss / 1024 / 1024,
                            )
                        )

                    except Exception as e:
                        op_end = time.time()
                        self.report.add_result(
                            LoadTestResult(
                                operation_type="user_session",
                                user_id=user_id,
                                start_time=op_start,
                                end_time=op_end,
                                success=False,
                                error=str(e),
                            )
                        )

                # Think time
                await asyncio.sleep(self.config.think_time_ms / 1000)

        # Launch concurrent users with ramp-up
        tasks = []
        for user_id in range(self.config.concurrent_users):
            # Ramp-up delay
            await asyncio.sleep(self.config.ramp_up_seconds / self.config.concurrent_users)
            tasks.append(user_session(user_id, self.config.duration_seconds))

        await asyncio.gather(*tasks)
        progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Completed {self.config.concurrent_users} concurrent user sessions")

    async def _op_formation_switch(self, coordinator):
        """Operation: Switch formation."""
        formations = [
            TeamFormation.SEQUENTIAL,
            TeamFormation.PARALLEL,
            TeamFormation.HIERARCHICAL,
        ]
        coordinator.set_formation(formations[len(coordinator.set_formation.__self__.__dict__) % len(formations)])

    async def _op_checkpoint(self):
        """Operation: Checkpoint."""
        coordinator = CheckpointCoordinator()
        await coordinator.checkpoint(f"session_{time.time()}", {"data": "value"})

    async def _op_evaluation(self):
        """Operation: Record evaluation."""
        coordinator = EvaluationCoordinator()
        await coordinator.record_evaluation(f"task_{time.time()}", score=0.9)

    def print_summary(self) -> None:
        """Print load test summary."""
        metrics = self.report.calculate_metrics()

        # Overall metrics table
        table = Table(title="Load Test Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Operations", f"{metrics['total_operations']:,}")
        table.add_row("Successful", f"[green]{metrics['successful_operations']:,}[/green]")
        table.add_row("Failed", f"[{'red' if metrics['failed_operations'] > 0 else 'green'}]{metrics['failed_operations']:,}[/]")
        table.add_row("Success Rate", f"{metrics['overall_success_rate']:.2f}%")
        table.add_row("Test Duration", f"{metrics['test_duration_seconds']:.1f}s")
        table.add_row("Throughput", f"{metrics['throughput_ops_per_second']:.2f} ops/sec")

        console.print(table)

        # Per-operation metrics
        for op_type, op_metrics in metrics["by_operation"].items():
            console.print(f"\n[bold cyan]{op_type.replace('_', ' ').title()}[/bold cyan]")

            op_table = Table(show_header=False)
            op_table.add_column("Metric", style="cyan")
            op_table.add_column("Value", justify="right")

            op_table.add_row("Count", f"{op_metrics['count']:,}")
            op_table.add_row("Avg Latency", f"{op_metrics['avg_latency_ms']:.2f}ms")
            op_table.add_row("Min Latency", f"{op_metrics['min_latency_ms']:.2f}ms")
            op_table.add_row("Max Latency", f"{op_metrics['max_latency_ms']:.2f}ms")
            op_table.add_row("P50 Latency", f"{op_metrics['p50_latency_ms']:.2f}ms")
            op_table.add_row("P95 Latency", f"{op_metrics['p95_latency_ms']:.2f}ms")
            op_table.add_row("P99 Latency", f"{op_metrics['p99_latency_ms']:.2f}ms")

            console.print(op_table)

        # Memory metrics
        console.print("\n[bold cyan]Memory Usage[/bold cyan]")
        mem_table = Table(show_header=False)
        mem_table.add_column("Metric", style="cyan")
        mem_table.add_column("Value", justify="right")

        mem_metrics = self.report.performance_metrics
        mem_table.add_row("Initial Memory", f"{mem_metrics['initial_memory_mb']:.2f} MB")
        mem_table.add_row("Final Memory", f"{mem_metrics['final_memory_mb']:.2f} MB")
        mem_table.add_row("Memory Increase", f"{mem_metrics['memory_increase_mb']:.2f} MB")
        mem_table.add_row("Peak Memory (tracemalloc)", f"{mem_metrics['tracemalloc_peak_mb']:.2f} MB")

        console.print(mem_table)

        # Bottleneck analysis
        console.print("\n[bold yellow]Bottleneck Analysis[/bold yellow]")

        # Find slowest operation
        slowest_op = max(
            metrics["by_operation"].items(),
            key=lambda x: x[1]["avg_latency_ms"],
            default=(None, {"avg_latency_ms": 0}),
        )

        if slowest_op[0]:
            console.print(f"  • Slowest operation: {slowest_op[0]} ({slowest_op[1]['avg_latency_ms']:.2f}ms avg)")

        # Find highest failure rate
        failure_rates = {
            op: metrics["failed"] / metrics["count"] * 100
            for op, metrics in metrics["by_operation"].items()
            if metrics["count"] > 0
        }

        if failure_rates:
            highest_failure = max(failure_rates.items(), key=lambda x: x[1])
            if highest_failure[1] > 0:
                console.print(f"  • Highest failure rate: {highest_failure[0]} ({highest_failure[1]:.2f}%)")

        # Performance assessment
        console.print("\n[bold cyan]Performance Assessment[/bold cyan]")

        issues = []
        if metrics["overall_success_rate"] < 99.0:
            issues.append(f"Success rate below 99%: {metrics['overall_success_rate']:.2f}%")

        if metrics["throughput_ops_per_second"] < 100:
            issues.append(f"Throughput below 100 ops/sec: {metrics['throughput_ops_per_second']:.2f}")

        for op_type, op_metrics in metrics["by_operation"].items():
            if op_metrics["p95_latency_ms"] > 100:
                issues.append(f"High P95 latency for {op_type}: {op_metrics['p95_latency_ms']:.2f}ms")

        if mem_metrics["memory_increase_mb"] > 500:
            issues.append(f"High memory increase: {mem_metrics['memory_increase_mb']:.2f} MB")

        if issues:
            console.print("[bold red]Issues detected:[/bold red]")
            for issue in issues:
                console.print(f"  • {issue}")
        else:
            console.print("[bold green]No performance issues detected[/bold green]")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Load testing for coordinators")
    parser.add_argument("--concurrent-users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--ramp-up", type=int, default=10, help="Ramp-up time in seconds")
    parser.add_argument("--operations", type=int, default=100, help="Operations per user")
    parser.add_argument("--think-time", type=int, default=100, help="Think time between operations (ms)")
    parser.add_argument("--output", type=str, default="/tmp/load_test_report.json", help="Output report path")
    args = parser.parse_args()

    try:
        # Create config
        config = LoadTestConfig(
            concurrent_users=args.concurrent_users,
            duration_seconds=args.duration,
            ramp_up_seconds=args.ramp_up,
            operations_per_user=args.operations,
            think_time_ms=args.think_time,
        )

        # Create tester
        tester = CoordinatorLoadTester(config)

        # Run tests
        await tester.run_all_tests()

        # Print summary
        tester.print_summary()

        # Generate report
        tester.report.generate_report(Path(args.output))
        console.print(f"\n[bold]Report saved:[/bold] {args.output}")

        # Exit with appropriate code
        metrics = tester.report.calculate_metrics()

        # Check if performance thresholds exceeded
        if metrics["overall_success_rate"] < 95.0:
            console.print("\n[bold red]FAIL: Success rate below 95%[/bold red]")
            sys.exit(1)

        if metrics["throughput_ops_per_second"] < 50:
            console.print("\n[bold red]FAIL: Throughput below 50 ops/sec[/bold red]")
            sys.exit(1)

        console.print("\n[bold green]PASS: All performance thresholds met[/bold green]")
        sys.exit(0)

    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        logger.exception("Fatal error during load testing")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
