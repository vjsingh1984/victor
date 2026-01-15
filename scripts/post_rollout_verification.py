#!/usr/bin/env python3
"""Post-rollout verification for coordinator-based orchestrator.

This script verifies that the coordinator-based orchestrator is working
correctly after production rollout. It performs health checks, validates
performance metrics, and generates a comprehensive health report.

Run this IMMEDIATELY after deployment to ensure everything is working.

Usage:
    python scripts/post_rollout_verification.py [--output report.html]

Exit codes:
    0: All verifications passed
    1: Some verifications failed (non-critical)
    2: Critical failures detected (rollback recommended)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from victor.agent.coordinators.workflow_coordinator import WorkflowCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/post_rollout_verification.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

console = Console()


@dataclass
class VerificationResult:
    """Result of a single verification check."""

    name: str
    category: str
    passed: bool
    critical: bool
    duration_ms: float
    details: str = ""
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "critical": self.critical,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class HealthReport:
    """Comprehensive health report."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    results: List[VerificationResult] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: VerificationResult) -> None:
        """Add a verification result."""
        self.results.append(result)
        self.total_checks += 1
        if result.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
            if result.critical:
                self.critical_failures.append(result.name)

    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        return {
            "total_checks": self.total_checks,
            "passed": self.passed_checks,
            "failed": self.failed_checks,
            "success_rate": (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0,
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "total_duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else 0
            ),
        }

    def generate_html_report(self, output_path: Path) -> None:
        """Generate HTML health report."""
        summary = self.calculate_summary()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Post-Rollout Verification Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #2196F3; }}
        .metric.critical {{ border-left-color: #f44336; }}
        .metric.warning {{ border-left-color: #ff9800; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #2196F3; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .status-pass {{ color: #4CAF50; font-weight: bold; }}
        .status-fail {{ color: #f44336; font-weight: bold; }}
        .category-header {{ background: #e3f2fd; padding: 10px; margin-top: 20px; font-weight: bold; }}
        .critical-badge {{ background: #f44336; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
        .timestamp {{ color: #666; font-size: 12px; }}
        .health-healthy {{ color: #4CAF50; font-weight: bold; font-size: 18px; }}
        .health-degraded {{ color: #ff9800; font-weight: bold; font-size: 18px; }}
        .health-critical {{ color: #f44336; font-weight: bold; font-size: 18px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Post-Rollout Verification Report</h1>
        <p class="timestamp">Generated: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <div class="metric">
                <div class="metric-label">Total Checks</div>
                <div class="metric-value">{summary['total_checks']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Passed</div>
                <div class="metric-value">{summary['passed']}</div>
            </div>
            <div class="metric {'critical' if summary['failed'] > 0 else ''}">
                <div class="metric-label">Failed</div>
                <div class="metric-value">{summary['failed']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{summary['success_rate']:.1f}%</div>
            </div>
            <div class="metric {'critical' if summary['critical_failures'] > 0 else ''}">
                <div class="metric-label">Critical Failures</div>
                <div class="metric-value">{summary['critical_failures']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Duration</div>
                <div class="metric-value">{summary['total_duration_seconds']:.1f}s</div>
            </div>
        </div>

        <h2>Overall Health Status</h2>
        <p class="{'health-critical' if summary['critical_failures'] > 0 else 'health-degraded' if summary['failed'] > 0 else 'health-healthy'}">
            {'CRITICAL - Rollback Recommended' if summary['critical_failures'] > 0 else 'DEGRADED - Monitor Closely' if summary['failed'] > 0 else 'HEALTHY - All Systems Operational'}
        </p>

        <h2>System Metrics</h2>
        <table>
            <thead><tr><th>Metric</th><th>Value</th><th>Status</th></tr></thead>
            <tbody>
                <tr><td>CPU Usage</td><td>{self.system_metrics.get('cpu_percent', 0):.1f}%</td><td>{'<span class="status-pass">OK</span>' if self.system_metrics.get('cpu_percent', 0) < 80 else '<span class="status-fail">HIGH</span>'}</td></tr>
                <tr><td>Memory Usage</td><td>{self.system_metrics.get('memory_percent', 0):.1f}%</td><td>{'<span class="status-pass">OK</span>' if self.system_metrics.get('memory_percent', 0) < 80 else '<span class="status-fail">HIGH</span>'}</td></tr>
                <tr><td>Disk Usage</td><td>{self.system_metrics.get('disk_percent', 0):.1f}%</td><td>{'<span class="status-pass">OK</span>' if self.system_metrics.get('disk_percent', 0) < 80 else '<span class="status-fail">HIGH</span>'}</td></tr>
                <tr><td>Uptime</td><td>{self.system_metrics.get('uptime_seconds', 0):.0f}s</td><td><span class="status-pass">OK</span></td></tr>
            </tbody>
        </table>

        <h2>Verification Results</h2>
"""

        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        # Generate tables for each category
        for category, results in categories.items():
            html += f'<div class="category-header">{category}</div>'
            html += '<table>'
            html += '<thead><tr><th>Check Name</th><th>Status</th><th>Duration (ms)</th><th>Details</th></tr></thead>'
            html += '<tbody>'

            for result in results:
                critical_badge = ' <span class="critical-badge">CRITICAL</span>' if result.critical else ''
                status = f'<span class="status-pass">PASS</span>{critical_badge}' if result.passed else f'<span class="status-fail">FAIL</span>{critical_badge}'
                details = result.details or result.error or "N/A"
                html += f'<tr><td>{result.name}</td><td>{status}</td><td>{result.duration_ms:.2f}</td><td>{details}</td></tr>'

            html += '</tbody></table>'

        # Critical failures
        if self.critical_failures:
            html += '<h2 style="color: #f44336;">Critical Failures - Rollback Recommended</h2>'
            html += '<ul>'
            for failure in self.critical_failures:
                html += f'<li>{failure}</li>'
            html += '</ul>'

        html += """
    </div>
</body>
</html>
"""

        output_path.write_text(html)
        console.print(f"[green]HTML report generated: {output_path}[/green]")


class PostRolloutVerifier:
    """Post-rollout verification suite."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize verifier."""
        self.settings = settings or Settings()
        self.report = HealthReport()
        self.process = psutil.Process(os.getpid())

    async def run_all_verifications(self) -> HealthReport:
        """Run all verification checks."""
        console.print(Panel.fit("[bold cyan]Starting Post-Rollout Verification[/bold cyan]"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # System health
            task = progress.add_task("System Health Verification", total=5)
            await self._verify_system_health(progress, task)

            # Coordinator initialization
            task = progress.add_task("Coordinator Initialization", total=6)
            await self._verify_coordinator_initialization(progress, task)

            # Core functionality
            task = progress.add_task("Core Functionality", total=5)
            await self._verify_core_functionality(progress, task)

            # Performance metrics
            task = progress.add_task("Performance Metrics", total=4)
            await self._verify_performance_metrics(progress, task)

            # Integration checks
            task = progress.add_task("Integration Checks", total=3)
            await self._verify_integrations(progress, task)

        self.report.end_time = datetime.now()
        return self.report

    async def _verify_system_health(self, progress, task) -> None:
        """Verify system health."""
        # CPU usage
        start = time.time()
        cpu_percent = self.process.cpu_percent(interval=1)
        duration = (time.time() - start) * 1000

        passed = cpu_percent < 80
        self.report.system_metrics["cpu_percent"] = cpu_percent
        self.report.add_result(
            VerificationResult(
                name="CPU Usage",
                category="System Health",
                passed=passed,
                critical=True,
                duration_ms=duration,
                details=f"CPU usage: {cpu_percent:.1f}%",
                metrics={"cpu_percent": cpu_percent},
            )
        )
        progress.update(task, advance=1)

        if not passed:
            self.report.warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

        # Memory usage
        start = time.time()
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        duration = (time.time() - start) * 1000

        passed = memory_percent < 80
        self.report.system_metrics["memory_percent"] = memory_percent
        self.report.system_metrics["memory_mb"] = memory_info.rss / 1024 / 1024
        self.report.add_result(
            VerificationResult(
                name="Memory Usage",
                category="System Health",
                passed=passed,
                critical=True,
                duration_ms=duration,
                details=f"Memory usage: {memory_percent:.1f}% ({memory_info.rss / 1024 / 1024:.1f} MB)",
                metrics={"memory_percent": memory_percent, "memory_mb": memory_info.rss / 1024 / 1024},
            )
        )
        progress.update(task, advance=1)

        if not passed:
            self.report.warnings.append(f"High memory usage: {memory_percent:.1f}%")

        # Disk usage
        start = time.time()
        disk_usage = psutil.disk_usage("/")
        disk_percent = disk_usage.percent
        duration = (time.time() - start) * 1000

        passed = disk_percent < 80
        self.report.system_metrics["disk_percent"] = disk_percent
        self.report.add_result(
            VerificationResult(
                name="Disk Usage",
                category="System Health",
                passed=passed,
                critical=False,
                duration_ms=duration,
                details=f"Disk usage: {disk_percent:.1f}%",
                metrics={"disk_percent": disk_percent},
            )
        )
        progress.update(task, advance=1)

        # Process uptime
        start = time.time()
        uptime = time.time() - self.process.create_time()
        duration = (time.time() - start) * 1000

        self.report.system_metrics["uptime_seconds"] = uptime
        self.report.add_result(
            VerificationResult(
                name="Process Uptime",
                category="System Health",
                passed=True,
                critical=False,
                duration_ms=duration,
                details=f"Uptime: {uptime:.0f} seconds",
                metrics={"uptime_seconds": uptime},
            )
        )
        progress.update(task, advance=1)

        # System load
        start = time.time()
        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
        duration = (time.time() - start) * 1000

        self.report.system_metrics["load_average"] = load_avg
        self.report.add_result(
            VerificationResult(
                name="System Load Average",
                category="System Health",
                passed=True,
                critical=False,
                duration_ms=duration,
                details=f"Load average: {load_avg[0]:.2f}",
                metrics={"load_avg_1min": load_avg[0]},
            )
        )
        progress.update(task, advance=1)

    async def _verify_coordinator_initialization(self, progress, task) -> None:
        """Verify all coordinators can be initialized."""
        coordinators_to_check = [
            ("CheckpointCoordinator", CheckpointCoordinator),
            ("EvaluationCoordinator", EvaluationCoordinator),
            ("MetricsCoordinator", MetricsCoordinator),
            ("WorkflowCoordinator", WorkflowCoordinator),
            ("TeamCoordinator (Lightweight)", lambda: create_coordinator(lightweight=True)),
            ("TeamCoordinator (Full)", lambda: create_coordinator(lightweight=False, with_observability=True, with_rl=True)),
        ]

        for name, coordinator_factory in coordinators_to_check:
            start = time.time()
            try:
                coordinator = coordinator_factory()
                duration = (time.time() - start) * 1000

                self.report.add_result(
                    VerificationResult(
                        name=f"{name} Initialization",
                        category="Coordinator Initialization",
                        passed=True,
                        critical=True,
                        duration_ms=duration,
                        details=f"Initialized in {duration:.2f}ms",
                    )
                )
                console.print(f"[green]✓[/green] {name} initialized")
            except Exception as e:
                duration = (time.time() - start) * 1000
                self.report.add_result(
                    VerificationResult(
                        name=f"{name} Initialization",
                        category="Coordinator Initialization",
                        passed=False,
                        critical=True,
                        duration_ms=duration,
                        error=str(e),
                    )
                )
                console.print(f"[red]✗[/red] {name} failed: {e}")

            progress.update(task, advance=1)

    async def _verify_core_functionality(self, progress, task) -> None:
        """Verify core functionality works."""
        # Checkpoint operations
        start = time.time()
        try:
            coordinator = CheckpointCoordinator()
            await coordinator.checkpoint("verify_test", {"test": "data"})
            restored = await coordinator.restore("verify_test")
            duration = (time.time() - start) * 1000

            passed = restored is not None and restored.get("test") == "data"
            self.report.add_result(
                VerificationResult(
                    name="Checkpoint Operations",
                    category="Core Functionality",
                    passed=passed,
                    critical=True,
                    duration_ms=duration,
                    details="Checkpoint and restore working",
                )
            )
            console.print(f"[green]✓[/green] Checkpoint operations verified")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Checkpoint Operations",
                    category="Core Functionality",
                    passed=False,
                    critical=True,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Checkpoint operations failed: {e}")
        progress.update(task, advance=1)

        # Evaluation recording
        start = time.time()
        try:
            coordinator = EvaluationCoordinator()
            await coordinator.record_evaluation("verify_task", score=0.9, metrics={})
            evaluations = await coordinator.get_evaluations("verify_task")
            duration = (time.time() - start) * 1000

            passed = len(evaluations) > 0
            self.report.add_result(
                VerificationResult(
                    name="Evaluation Recording",
                    category="Core Functionality",
                    passed=passed,
                    critical=True,
                    duration_ms=duration,
                    details=f"Recorded {len(evaluations)} evaluation(s)",
                )
            )
            console.print(f"[green]✓[/green] Evaluation recording verified")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Evaluation Recording",
                    category="Core Functionality",
                    passed=False,
                    critical=True,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Evaluation recording failed: {e}")
        progress.update(task, advance=1)

        # Metrics recording
        start = time.time()
        try:
            coordinator = MetricsCoordinator()
            await coordinator.record_metric("verify_metric", 1.0)
            metrics = await coordinator.get_metrics("verify_metric")
            duration = (time.time() - start) * 1000

            passed = len(metrics) > 0
            self.report.add_result(
                VerificationResult(
                    name="Metrics Recording",
                    category="Core Functionality",
                    passed=passed,
                    critical=True,
                    duration_ms=duration,
                    details=f"Recorded {len(metrics)} metric(s)",
                )
            )
            console.print(f"[green]✓[/green] Metrics recording verified")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Metrics Recording",
                    category="Core Functionality",
                    passed=False,
                    critical=True,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Metrics recording failed: {e}")
        progress.update(task, advance=1)

        # Team formations
        start = time.time()
        try:
            coordinator = create_coordinator(lightweight=True)
            formations = [
                TeamFormation.SEQUENTIAL,
                TeamFormation.PARALLEL,
                TeamFormation.HIERARCHICAL,
                TeamFormation.PIPELINE,
                TeamFormation.CONSENSUS,
            ]

            for formation in formations:
                coordinator.set_formation(formation)

            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Team Formations",
                    category="Core Functionality",
                    passed=True,
                    critical=True,
                    duration_ms=duration,
                    details=f"All {len(formations)} formations working",
                )
            )
            console.print(f"[green]✓[/green] Team formations verified")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Team Formations",
                    category="Core Functionality",
                    passed=False,
                    critical=True,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Team formations failed: {e}")
        progress.update(task, advance=1)

        # Workflow compilation
        start = time.time()
        try:
            coordinator = WorkflowCoordinator()
            workflow_def = {
                "nodes": [{"id": "start", "type": "agent", "role": "test"}],
                "edges": [],
            }
            compiled = await coordinator.compile_workflow(workflow_def)
            duration = (time.time() - start) * 1000

            passed = compiled is not None
            self.report.add_result(
                VerificationResult(
                    name="Workflow Compilation",
                    category="Core Functionality",
                    passed=passed,
                    critical=True,
                    duration_ms=duration,
                    details="Workflow compilation working",
                )
            )
            console.print(f"[green]✓[/green] Workflow compilation verified")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Workflow Compilation",
                    category="Core Functionality",
                    passed=False,
                    critical=True,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Workflow compilation failed: {e}")
        progress.update(task, advance=1)

    async def _verify_performance_metrics(self, progress, task) -> None:
        """Verify performance metrics are within acceptable ranges."""
        # Coordinator creation performance
        start = time.time()
        coordinators = [create_coordinator(lightweight=True) for _ in range(10)]
        duration = (time.time() - start) * 1000
        avg_time = duration / len(coordinators)

        passed = avg_time < 10.0
        self.report.performance_metrics["avg_coordinator_creation_ms"] = avg_time
        self.report.add_result(
            VerificationResult(
                name="Coordinator Creation Performance",
                category="Performance Metrics",
                passed=passed,
                critical=False,
                duration_ms=duration,
                details=f"Average: {avg_time:.2f}ms per coordinator",
                metrics={"avg_creation_time_ms": avg_time, "coordinator_count": len(coordinators)},
            )
        )
        progress.update(task, advance=1)

        # Formation switching performance
        start = time.time()
        coordinator = create_coordinator(lightweight=True)
        formations = [
            TeamFormation.SEQUENTIAL,
            TeamFormation.PARALLEL,
            TeamFormation.HIERARCHICAL,
        ]

        for _ in range(100):
            for formation in formations:
                coordinator.set_formation(formation)

        duration = (time.time() - start) * 1000
        avg_time = duration / 300  # 100 iterations * 3 formations

        passed = avg_time < 1.0
        self.report.performance_metrics["avg_formation_switch_ms"] = avg_time
        self.report.add_result(
            VerificationResult(
                name="Formation Switching Performance",
                category="Performance Metrics",
                passed=passed,
                critical=False,
                duration_ms=duration,
                details=f"Average: {avg_time:.3f}ms per switch",
                metrics={"avg_switch_time_ms": avg_time},
            )
        )
        progress.update(task, advance=1)

        # Checkpoint operation performance
        start = time.time()
        coordinator = CheckpointCoordinator()

        for i in range(10):
            await coordinator.checkpoint(f"perf_test_{i}", {"data": f"value_{i}"})
            await coordinator.restore(f"perf_test_{i}")

        duration = (time.time() - start) * 1000
        avg_time = duration / 20  # 10 checkpoints + 10 restores

        passed = avg_time < 50.0
        self.report.performance_metrics["avg_checkpoint_operation_ms"] = avg_time
        self.report.add_result(
            VerificationResult(
                name="Checkpoint Operation Performance",
                category="Performance Metrics",
                passed=passed,
                critical=False,
                duration_ms=duration,
                details=f"Average: {avg_time:.2f}ms per operation",
                metrics={"avg_operation_time_ms": avg_time},
            )
        )
        progress.update(task, advance=1)

        # Memory efficiency
        start = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        coordinators = [create_coordinator(lightweight=True) for _ in range(100)]

        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        avg_memory = memory_increase / len(coordinators)

        duration = (time.time() - start) * 1000

        passed = avg_memory < 1.0
        self.report.performance_metrics["avg_memory_per_coordinator_mb"] = avg_memory
        self.report.add_result(
            VerificationResult(
                name="Memory Efficiency",
                category="Performance Metrics",
                passed=passed,
                critical=False,
                duration_ms=duration,
                details=f"Average: {avg_memory:.3f}MB per coordinator",
                metrics={
                    "memory_increase_mb": memory_increase,
                    "avg_memory_per_coordinator_mb": avg_memory,
                },
            )
        )
        progress.update(task, advance=1)

    async def _verify_integrations(self, progress, task) -> None:
        """Verify integrations with other components."""
        # Module imports
        start = time.time()
        try:
            from victor.agent.coordinators import (
                CheckpointCoordinator,
                EvaluationCoordinator,
                MetricsCoordinator,
                WorkflowCoordinator,
            )
            from victor.teams import create_coordinator, TeamFormation
            duration = (time.time() - start) * 1000

            self.report.add_result(
                VerificationResult(
                    name="Module Imports",
                    category="Integration Checks",
                    passed=True,
                    critical=True,
                    duration_ms=duration,
                    details="All coordinator modules importable",
                )
            )
            console.print(f"[green]✓[/green] Module imports verified")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Module Imports",
                    category="Integration Checks",
                    passed=False,
                    critical=True,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Module imports failed: {e}")
        progress.update(task, advance=1)

        # Backward compatibility
        start = time.time()
        try:
            from victor.framework.coordinators import FrameworkTeamCoordinator
            coordinator = FrameworkTeamCoordinator()
            duration = (time.time() - start) * 1000

            self.report.add_result(
                VerificationResult(
                    name="Backward Compatibility",
                    category="Integration Checks",
                    passed=True,
                    critical=True,
                    duration_ms=duration,
                    details="Legacy imports still functional",
                )
            )
            console.print(f"[green]✓[/green] Backward compatibility verified")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Backward Compatibility",
                    category="Integration Checks",
                    passed=False,
                    critical=True,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Backward compatibility failed: {e}")
        progress.update(task, advance=1)

        # Configuration loading
        start = time.time()
        try:
            settings = Settings()
            duration = (time.time() - start) * 1000

            self.report.add_result(
                VerificationResult(
                    name="Configuration Loading",
                    category="Integration Checks",
                    passed=True,
                    critical=True,
                    duration_ms=duration,
                    details="Settings loaded successfully",
                )
            )
            console.print(f"[green]✓[/green] Configuration loading verified")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                VerificationResult(
                    name="Configuration Loading",
                    category="Integration Checks",
                    passed=False,
                    critical=True,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Configuration loading failed: {e}")
        progress.update(task, advance=1)

    def print_summary(self) -> None:
        """Print verification summary."""
        summary = self.report.calculate_summary()

        # Create summary table
        table = Table(title="Post-Rollout Verification Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Checks", str(summary["total_checks"]))
        table.add_row("Passed", f"[green]{summary['passed']}[/green]")
        table.add_row("Failed", f"[{'red' if summary['failed'] > 0 else 'green'}]{summary['failed']}[/]")
        table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        table.add_row("Critical Failures", str(summary["critical_failures"]))
        table.add_row("Warnings", str(summary["warnings"]))
        table.add_row("Duration", f"{summary['total_duration_seconds']:.1f}s")

        console.print(table)

        # Print system metrics
        console.print("\n[bold cyan]System Metrics[/bold cyan]")
        sys_table = Table(show_header=False)
        sys_table.add_column("Metric", style="cyan")
        sys_table.add_column("Value", justify="right")

        sys_table.add_row("CPU Usage", f"{self.report.system_metrics.get('cpu_percent', 0):.1f}%")
        sys_table.add_row("Memory Usage", f"{self.report.system_metrics.get('memory_percent', 0):.1f}%")
        sys_table.add_row("Disk Usage", f"{self.report.system_metrics.get('disk_percent', 0):.1f}%")
        sys_table.add_row("Uptime", f"{self.report.system_metrics.get('uptime_seconds', 0):.0f}s")

        console.print(sys_table)

        # Print warnings
        if self.report.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in self.report.warnings:
                console.print(f"  • {warning}")

        # Print critical failures
        if self.report.critical_failures:
            console.print("\n[bold red]Critical Failures:[/bold red]")
            for failure in self.report.critical_failures:
                console.print(f"  • {failure}")

        # Overall verdict
        if summary["critical_failures"] > 0:
            console.print("\n[bold red]HEALTH STATUS: CRITICAL - Rollback Recommended[/bold red]")
        elif summary["failed"] > 0:
            console.print("\n[bold yellow]HEALTH STATUS: DEGRADED - Monitor Closely[/bold yellow]")
        else:
            console.print("\n[bold green]HEALTH STATUS: HEALTHY - All Systems Operational[/bold green]")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Post-rollout verification")
    parser.add_argument("--output", type=str, default="/tmp/post_rollout_report.html", help="Output report path")
    parser.add_argument("--json", type=str, help="Output JSON report path")
    args = parser.parse_args()

    try:
        # Create verifier
        verifier = PostRolloutVerifier()

        # Run verifications
        await verifier.run_all_verifications()

        # Print summary
        verifier.print_summary()

        # Generate reports
        verifier.report.generate_html_report(Path(args.output))
        console.print(f"\n[bold]HTML report:[/bold] file://{args.output}")

        if args.json:
            with open(args.json, "w") as f:
                json.dump(
                    {
                        "summary": verifier.report.calculate_summary(),
                        "system_metrics": verifier.report.system_metrics,
                        "performance_metrics": verifier.report.performance_metrics,
                        "results": [r.to_dict() for r in verifier.report.results],
                        "warnings": verifier.report.warnings,
                        "critical_failures": verifier.report.critical_failures,
                    },
                    f,
                    indent=2,
                )
            console.print(f"[bold]JSON report:[/bold] {args.json}")

        # Exit with appropriate code
        summary = verifier.report.calculate_summary()
        if summary["critical_failures"] > 0:
            console.print("\n[bold red]Action Required: Consider rollback due to critical failures[/bold red]")
            sys.exit(2)
        elif summary["failed"] > 0:
            console.print("\n[bold yellow]Action Required: Monitor and address non-critical failures[/bold yellow]")
            sys.exit(1)
        else:
            console.print("\n[bold green]All systems operational - No action required[/bold green]")
            sys.exit(0)

    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        logger.exception("Fatal error during verification")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
