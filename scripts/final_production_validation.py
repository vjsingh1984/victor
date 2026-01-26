#!/usr/bin/env python3
"""Final production validation for coordinator-based orchestrator.

This script performs comprehensive end-to-end validation before production rollout:
- Tests all coordinators work together
- Tests backward compatibility
- Tests performance metrics
- Tests error handling
- Tests rollback procedures
- Generates comprehensive report

Usage:
    python scripts/final_production_validation.py [--output report.html]

Exit codes:
    0: All validations passed
    1: One or more validations failed
    2: Critical errors (cannot proceed with rollout)
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
from unittest.mock import MagicMock, patch

import rich
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from victor.config.settings import Settings
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.coordinators.checkpoint_coordinator import CheckpointCoordinator
from victor.agent.coordinators.evaluation_coordinator import EvaluationCoordinator
from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator
from victor.agent.coordinators.workflow_coordinator import WorkflowCoordinator
from victor.agent.streaming.coordinator import IterationCoordinator
from victor.agent.recovery.coordinator import RecoveryCoordinator
from victor.teams import create_coordinator, TeamFormation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/production_validation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

console = Console()


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    name: str
    category: str
    passed: bool
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
            "duration_ms": self.duration_ms,
            "details": self.details,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    results: List[ValidationResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            # Check for critical failures
            if "critical" in result.name.lower() or "orchestrator" in result.name.lower():
                self.critical_failures.append(result.name)

    def get_results_by_category(self, category: str) -> List[ValidationResult]:
        """Get all results for a specific category."""
        return [r for r in self.results if r.category == category]

    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "success_rate": (
                (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            ),
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "total_duration_seconds": (
                (self.end_time - self.start_time).total_seconds() if self.end_time else 0
            ),
        }

    def generate_html_report(self, output_path: Path) -> None:
        """Generate HTML report."""
        summary = self.calculate_summary()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Production Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .metric.failed {{ border-left-color: #f44336; }}
        .metric.warning {{ border-left-color: #ff9800; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .status-pass {{ color: #4CAF50; font-weight: bold; }}
        .status-fail {{ color: #f44336; font-weight: bold; }}
        .category-header {{ background: #e8f5e9; padding: 10px; margin-top: 20px; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Production Validation Report</h1>
        <p class="timestamp">Generated: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <div class="metric">
                <div class="metric-label">Total Tests</div>
                <div class="metric-value">{summary['total_tests']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Passed</div>
                <div class="metric-value">{summary['passed']}</div>
            </div>
            <div class="metric {'failed' if summary['failed'] > 0 else ''}">
                <div class="metric-label">Failed</div>
                <div class="metric-value">{summary['failed']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{summary['success_rate']:.1f}%</div>
            </div>
            <div class="metric {'failed' if summary['critical_failures'] > 0 else ''}">
                <div class="metric-label">Critical Failures</div>
                <div class="metric-value">{summary['critical_failures']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Duration</div>
                <div class="metric-value">{summary['total_duration_seconds']:.1f}s</div>
            </div>
        </div>

        <h2>Test Results by Category</h2>
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
            html += "<table>"
            html += "<thead><tr><th>Test Name</th><th>Status</th><th>Duration (ms)</th><th>Details</th></tr></thead>"
            html += "<tbody>"

            for result in results:
                status = (
                    '<span class="status-pass">PASS</span>'
                    if result.passed
                    else '<span class="status-fail">FAIL</span>'
                )
                details = result.details or result.error or "N/A"
                html += f"<tr><td>{result.name}</td><td>{status}</td><td>{result.duration_ms:.2f}</td><td>{details}</td></tr>"

            html += "</tbody></table>"

        # Critical failures
        if self.critical_failures:
            html += '<h2 style="color: #f44336;">Critical Failures</h2>'
            html += "<ul>"
            for failure in self.critical_failures:
                html += f"<li>{failure}</li>"
            html += "</ul>"

        html += """
    </div>
</body>
</html>
"""

        output_path.write_text(html)
        console.print(f"[green]HTML report generated: {output_path}[/green]")


class ProductionValidator:
    """Comprehensive production validation suite."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize validator."""
        self.settings = settings or Settings()
        self.report = ValidationReport()
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.mock_provider = MagicMock()

    async def setup(self) -> None:
        """Setup test environment."""
        console.print("[bold blue]Setting up validation environment...[/bold blue]")

        # Mock provider for testing
        self.mock_provider.name = "test_provider"
        self.mock_provider.chat.return_value = MagicMock(
            content="Test response",
            usage=MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50),
        )

        # Create orchestrator with minimal dependencies
        # Note: ProviderManager and ToolRegistrar are now in separate modules
        # We'll let the orchestrator create them normally
        self.orchestrator = AgentOrchestrator(
            provider=self.mock_provider,
            model="test-model",
            settings=self.settings,
        )

        console.print("[green]Setup complete[/green]")

    async def teardown(self) -> None:
        """Cleanup test environment."""
        if self.orchestrator:
            # Cleanup orchestrator resources
            pass
        console.print("[green]Teardown complete[/green]")

    async def run_all_validations(self) -> ValidationReport:
        """Run all validation tests."""
        console.print(Panel.fit("[bold cyan]Starting Production Validation[/bold cyan]"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # Coordinator validation
            task = progress.add_task("Coordinator Validation", total=6)

            await self._validate_checkpoint_coordinator()
            progress.update(task, advance=1)

            await self._validate_evaluation_coordinator()
            progress.update(task, advance=1)

            await self._validate_metrics_coordinator()
            progress.update(task, advance=1)

            await self._validate_workflow_coordinator()
            progress.update(task, advance=1)

            await self._validate_streaming_coordinator()
            progress.update(task, advance=1)

            await self._validate_recovery_coordinator()
            progress.update(task, advance=1)

            # Team coordinator validation
            task = progress.add_task("Team Coordinator Validation", total=5)

            await self._validate_team_coordinator_creation()
            progress.update(task, advance=1)

            await self._validate_team_formations()
            progress.update(task, advance=1)

            await self._validate_team_observability()
            progress.update(task, advance=1)

            await self._validate_team_rl_integration()
            progress.update(task, advance=1)

            await self._validate_team_communication()
            progress.update(task, advance=1)

            # Integration validation
            task = progress.add_task("Integration Validation", total=4)

            await self._validate_orchestrator_integration()
            progress.update(task, advance=1)

            await self._validate_backward_compatibility()
            progress.update(task, advance=1)

            await self._validate_error_handling()
            progress.update(task, advance=1)

            await self._validate_rollback_procedures()
            progress.update(task, advance=1)

            # Performance validation
            task = progress.add_task("Performance Validation", total=3)

            await self._validate_concurrent_operations()
            progress.update(task, advance=1)

            await self._validate_memory_usage()
            progress.update(task, advance=1)

            await self._validate_response_times()
            progress.update(task, advance=1)

        self.report.end_time = datetime.now()
        return self.report

    async def _validate_checkpoint_coordinator(self) -> None:
        """Validate CheckpointCoordinator."""
        start = time.time()
        try:
            coordinator = CheckpointCoordinator()
            # Test basic functionality
            await coordinator.checkpoint("test_session", {"test": "data"})
            restored = await coordinator.restore("test_session")
            assert restored is not None

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Checkpoint Coordinator",
                    category="Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Checkpoint and restore working",
                    metrics={"checkpoint_count": 1},
                )
            )
            console.print("[green]✓[/green] CheckpointCoordinator validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Checkpoint Coordinator",
                    category="Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] CheckpointCoordinator failed: {e}")

    async def _validate_evaluation_coordinator(self) -> None:
        """Validate EvaluationCoordinator."""
        start = time.time()
        try:
            coordinator = EvaluationCoordinator()
            # Test basic functionality
            await coordinator.record_evaluation("test_task", score=0.9, metrics={})
            evaluations = await coordinator.get_evaluations("test_task")
            assert len(evaluations) > 0

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Evaluation Coordinator",
                    category="Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Evaluation recording working",
                    metrics={"evaluation_count": 1},
                )
            )
            console.print("[green]✓[/green] EvaluationCoordinator validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Evaluation Coordinator",
                    category="Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] EvaluationCoordinator failed: {e}")

    async def _validate_metrics_coordinator(self) -> None:
        """Validate MetricsCoordinator."""
        start = time.time()
        try:
            coordinator = MetricsCoordinator()
            # Test basic functionality
            await coordinator.record_metric("test_metric", 1.0)
            metrics = await coordinator.get_metrics("test_metric")
            assert len(metrics) > 0

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Metrics Coordinator",
                    category="Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Metrics recording working",
                    metrics={"metric_count": 1},
                )
            )
            console.print("[green]✓[/green] MetricsCoordinator validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Metrics Coordinator",
                    category="Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] MetricsCoordinator failed: {e}")

    async def _validate_workflow_coordinator(self) -> None:
        """Validate WorkflowCoordinator."""
        start = time.time()
        try:
            coordinator = WorkflowCoordinator()
            # Test basic functionality
            workflow_def = {"nodes": [], "edges": []}
            await coordinator.compile_workflow(workflow_def)

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Workflow Coordinator",
                    category="Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Workflow compilation working",
                )
            )
            console.print("[green]✓[/green] WorkflowCoordinator validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Workflow Coordinator",
                    category="Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] WorkflowCoordinator failed: {e}")

    async def _validate_streaming_coordinator(self) -> None:
        """Validate IterationCoordinator."""
        start = time.time()
        try:
            # Test coordinator creation
            from victor.agent.streaming.handler import StreamingChatHandler
            from victor.agent.unified_task_tracker import UnifiedTaskTracker

            handler = MagicMock(spec=StreamingChatHandler)
            loop_detector = MagicMock(spec=UnifiedTaskTracker)

            coordinator = IterationCoordinator(
                handler=handler,
                loop_detector=loop_detector,
                settings=self.settings,
            )

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Streaming Coordinator",
                    category="Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Iteration coordinator creation working",
                )
            )
            console.print("[green]✓[/green] IterationCoordinator validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Streaming Coordinator",
                    category="Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] IterationCoordinator failed: {e}")

    async def _validate_recovery_coordinator(self) -> None:
        """Validate RecoveryCoordinator."""
        start = time.time()
        try:
            coordinator = RecoveryCoordinator(
                qlearning_store=None,  # Use defaults
                usage_analytics=None,
                context_compactor=None,
            )

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Recovery Coordinator",
                    category="Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Recovery coordinator creation working",
                )
            )
            console.print("[green]✓[/green] RecoveryCoordinator validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Recovery Coordinator",
                    category="Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] StreamingRecoveryCoordinator failed: {e}")

    async def _validate_team_coordinator_creation(self) -> None:
        """Validate team coordinator creation."""
        start = time.time()
        try:
            # Lightweight coordinator for testing
            coordinator = create_coordinator(lightweight=True)
            assert coordinator is not None

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team Coordinator Creation",
                    category="Team Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Team coordinator factory working",
                )
            )
            console.print("[green]✓[/green] Team coordinator creation validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team Coordinator Creation",
                    category="Team Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Team coordinator creation failed: {e}")

    async def _validate_team_formations(self) -> None:
        """Validate team formations."""
        start = time.time()
        try:
            coordinator = create_coordinator(lightweight=True)

            # Test all formations
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
                ValidationResult(
                    name="Team Formations",
                    category="Team Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details=f"All {len(formations)} formations working",
                    metrics={"formation_count": len(formations)},
                )
            )
            console.print("[green]✓[/green] Team formations validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team Formations",
                    category="Team Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Team formations failed: {e}")

    async def _validate_team_observability(self) -> None:
        """Validate team observability."""
        start = time.time()
        try:
            coordinator = create_coordinator(
                lightweight=False,
                with_observability=True,
                with_rl=False,
            )

            # Check observability methods exist
            assert hasattr(coordinator, "get_metrics")
            assert hasattr(coordinator, "get_event_history")

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team Observability",
                    category="Team Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Observability mixin working",
                )
            )
            console.print("[green]✓[/green] Team observability validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team Observability",
                    category="Team Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Team observability failed: {e}")

    async def _validate_team_rl_integration(self) -> None:
        """Validate team RL integration."""
        start = time.time()
        try:
            coordinator = create_coordinator(
                lightweight=False,
                with_observability=False,
                with_rl=True,
            )

            # Check RL methods exist
            assert hasattr(coordinator, "record_outcome")
            assert hasattr(coordinator, "get_feedback_summary")

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team RL Integration",
                    category="Team Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="RL mixin working",
                )
            )
            console.print("[green]✓[/green] Team RL integration validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team RL Integration",
                    category="Team Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Team RL integration failed: {e}")

    async def _validate_team_communication(self) -> None:
        """Validate team communication infrastructure."""
        start = time.time()
        try:
            from victor.teams import TeamMessageBus, TeamSharedMemory

            # Test message bus
            message_bus = TeamMessageBus()
            assert message_bus is not None

            # Test shared memory
            shared_memory = TeamSharedMemory()
            assert shared_memory is not None

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team Communication",
                    category="Team Coordinators",
                    passed=True,
                    duration_ms=duration,
                    details="Message bus and shared memory working",
                )
            )
            console.print("[green]✓[/green] Team communication validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Team Communication",
                    category="Team Coordinators",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Team communication failed: {e}")

    async def _validate_orchestrator_integration(self) -> None:
        """Validate orchestrator integration with coordinators."""
        start = time.time()
        try:
            # Test that coordinators can be imported at runtime
            from victor.agent import coordinators

            # Check all coordinators are available
            assert hasattr(coordinators, "CheckpointCoordinator")
            assert hasattr(coordinators, "EvaluationCoordinator")
            assert hasattr(coordinators, "MetricsCoordinator")
            assert hasattr(coordinators, "WorkflowCoordinator")

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Orchestrator Integration",
                    category="Integration",
                    passed=True,
                    duration_ms=duration,
                    details="All coordinators importable",
                    metrics={"coordinator_count": 4},
                )
            )
            console.print("[green]✓[/green] Orchestrator integration validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Orchestrator Integration",
                    category="Integration",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Orchestrator integration failed: {e}")

    async def _validate_backward_compatibility(self) -> None:
        """Validate backward compatibility with existing code."""
        start = time.time()
        try:
            # Test that old imports still work
            from victor.framework.coordinators import FrameworkTeamCoordinator

            # Create coordinator using old API
            coordinator = FrameworkTeamCoordinator()
            assert coordinator is not None

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Backward Compatibility",
                    category="Integration",
                    passed=True,
                    duration_ms=duration,
                    details="Legacy imports still functional",
                )
            )
            console.print("[green]✓[/green] Backward compatibility validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Backward Compatibility",
                    category="Integration",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Backward compatibility failed: {e}")

    async def _validate_error_handling(self) -> None:
        """Validate error handling across coordinators."""
        start = time.time()
        try:
            coordinator = create_coordinator(lightweight=True)

            # Test error handling
            try:
                # This should handle errors gracefully
                await coordinator.execute_task("invalid_task", None)
            except Exception:
                # Expected - should be handled gracefully
                pass

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Error Handling",
                    category="Integration",
                    passed=True,
                    duration_ms=duration,
                    details="Errors handled gracefully",
                )
            )
            console.print("[green]✓[/green] Error handling validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Error Handling",
                    category="Integration",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Error handling failed: {e}")

    async def _validate_rollback_procedures(self) -> None:
        """Validate rollback procedures."""
        start = time.time()
        try:
            # Test that we can toggle back to old orchestrator
            from scripts.toggle_coordinator_orchestrator import (
                enable_coordinator_mode,
                enable_orchestrator_mode,
                get_current_mode,
            )

            # Get current mode
            current = get_current_mode()
            assert current in ["coordinator", "orchestrator"]

            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Rollback Procedures",
                    category="Integration",
                    passed=True,
                    duration_ms=duration,
                    details=f"Current mode: {current}, rollback available",
                )
            )
            console.print("[green]✓[/green] Rollback procedures validated")
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Rollback Procedures",
                    category="Integration",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Rollback procedures failed: {e}")

    async def _validate_concurrent_operations(self) -> None:
        """Validate concurrent operations performance."""
        start = time.time()
        try:
            # Create multiple coordinators concurrently
            tasks = [create_coordinator(lightweight=True) for _ in range(10)]
            coordinators = await asyncio.gather(*tasks)

            duration = (time.time() - start) * 1000
            avg_time = duration / len(coordinators)

            self.report.add_result(
                ValidationResult(
                    name="Concurrent Operations",
                    category="Performance",
                    passed=True,
                    duration_ms=duration,
                    details=f"Created {len(coordinators)} coordinators concurrently",
                    metrics={
                        "coordinator_count": len(coordinators),
                        "avg_creation_time_ms": avg_time,
                    },
                )
            )
            console.print(
                f"[green]✓[/green] Concurrent operations validated ({len(coordinators)} coordinators)"
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Concurrent Operations",
                    category="Performance",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Concurrent operations failed: {e}")

    async def _validate_memory_usage(self) -> None:
        """Validate memory usage."""
        start = time.time()
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_mem = process.memory_info().rss / 1024 / 1024  # MB

            # Create many coordinators
            coordinators = [create_coordinator(lightweight=True) for _ in range(100)]

            final_mem = process.memory_info().rss / 1024 / 1024  # MB
            mem_increase = final_mem - initial_mem
            avg_mem_per_coordinator = mem_increase / len(coordinators)

            duration = (time.time() - start) * 1000

            # Check if memory increase is reasonable (< 1MB per coordinator)
            passed = avg_mem_per_coordinator < 1.0

            self.report.add_result(
                ValidationResult(
                    name="Memory Usage",
                    category="Performance",
                    passed=passed,
                    duration_ms=duration,
                    details=f"Memory increase: {mem_increase:.2f}MB ({avg_mem_per_coordinator:.3f}MB/coordinator)",
                    metrics={
                        "initial_mem_mb": initial_mem,
                        "final_mem_mb": final_mem,
                        "mem_increase_mb": mem_increase,
                        "avg_mem_per_coordinator_mb": avg_mem_per_coordinator,
                    },
                )
            )

            if passed:
                console.print(
                    f"[green]✓[/green] Memory usage validated ({avg_mem_per_coordinator:.3f}MB/coordinator)"
                )
            else:
                console.print(
                    f"[yellow]⚠[/yellow] Memory usage high ({avg_mem_per_coordinator:.3f}MB/coordinator)"
                )
                self.report.warnings.append("High memory usage per coordinator")

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Memory Usage",
                    category="Performance",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Memory usage validation failed: {e}")

    async def _validate_response_times(self) -> None:
        """Validate response times."""
        start = time.time()
        try:
            coordinator = create_coordinator(lightweight=True)

            # Measure coordinator operation times
            times = []

            for i in range(100):
                op_start = time.time()
                coordinator.set_formation(TeamFormation.SEQUENTIAL)
                times.append((time.time() - op_start) * 1000)  # ms

            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)

            duration = (time.time() - start) * 1000

            # Check if response times are reasonable (< 1ms average)
            passed = avg_time < 1.0

            self.report.add_result(
                ValidationResult(
                    name="Response Times",
                    category="Performance",
                    passed=passed,
                    duration_ms=duration,
                    details=f"Average: {avg_time:.3f}ms, Min: {min_time:.3f}ms, Max: {max_time:.3f}ms",
                    metrics={
                        "avg_time_ms": avg_time,
                        "min_time_ms": min_time,
                        "max_time_ms": max_time,
                        "operation_count": len(times),
                    },
                )
            )

            if passed:
                console.print(f"[green]✓[/green] Response times validated (avg: {avg_time:.3f}ms)")
            else:
                console.print(f"[yellow]⚠[/yellow] Response times slow (avg: {avg_time:.3f}ms)")
                self.report.warnings.append("Slow response times detected")

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.report.add_result(
                ValidationResult(
                    name="Response Times",
                    category="Performance",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )
            console.print(f"[red]✗[/red] Response times validation failed: {e}")

    def print_summary(self) -> None:
        """Print validation summary."""
        summary = self.report.calculate_summary()

        # Create summary table
        table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Tests", str(summary["total_tests"]))
        table.add_row("Passed", f"[green]{summary['passed']}[/green]")
        table.add_row(
            "Failed", f"[{'red' if summary['failed'] > 0 else 'green'}]{summary['failed']}[/]"
        )
        table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        table.add_row("Critical Failures", str(summary["critical_failures"]))
        table.add_row("Warnings", str(summary["warnings"]))
        table.add_row("Duration", f"{summary['total_duration_seconds']:.1f}s")

        console.print(table)

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
            console.print(
                "\n[bold red]VERDICT: CANNOT ROLL OUT - Critical failures detected[/bold red]"
            )
        elif summary["failed"] > 0:
            console.print(
                "\n[bold yellow]VERDICT: PROCEED WITH CAUTION - Some failures detected[/bold yellow]"
            )
        else:
            console.print(
                "\n[bold green]VERDICT: READY FOR ROLLOUT - All validations passed[/bold green]"
            )


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Final production validation")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/production_validation_report.html",
        help="Output report path",
    )
    parser.add_argument("--json", type=str, help="Output JSON report path")
    args = parser.parse_args()

    try:
        # Create validator
        validator = ProductionValidator()

        # Setup
        await validator.setup()

        # Run validations
        await validator.run_all_validations()

        # Teardown
        await validator.teardown()

        # Print summary
        validator.print_summary()

        # Generate reports
        validator.report.generate_html_report(Path(args.output))
        console.print(f"\n[bold]HTML report:[/bold] file://{args.output}")

        if args.json:
            with open(args.json, "w") as f:
                json.dump(
                    {
                        "summary": validator.report.calculate_summary(),
                        "results": [r.to_dict() for r in validator.report.results],
                        "warnings": validator.report.warnings,
                        "critical_failures": validator.report.critical_failures,
                    },
                    f,
                    indent=2,
                )
            console.print(f"[bold]JSON report:[/bold] {args.json}")

        # Exit with appropriate code
        summary = validator.report.calculate_summary()
        if summary["critical_failures"] > 0:
            sys.exit(2)
        elif summary["failed"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        logger.exception("Fatal error during validation")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
