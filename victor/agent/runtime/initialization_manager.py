"""Initialization phase manager for orchestrator runtime boundaries.

Coordinates the 8 runtime initialization phases in the correct order,
providing structured results and error reporting for each phase.
Supports fail-fast for critical phases and dependency-based skipping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class InitializationError(Exception):
    """Raised when a critical initialization phase fails."""

    def __init__(self, phase: str, error: str, completed_phases: List[str]):
        self.phase = phase
        self.error = error
        self.completed_phases = completed_phases
        super().__init__(f"Critical initialization phase '{phase}' failed: {error}")


@dataclass
class PhaseResult:
    """Result of a single initialization phase."""

    name: str
    success: bool = True
    duration_ms: float = 0.0
    components_created: List[str] = field(default_factory=list)
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class InitializationResult:
    """Aggregate result of all initialization phases."""

    phases: List[PhaseResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return all(p.success for p in self.phases)

    @property
    def total_duration_ms(self) -> float:
        return sum(p.duration_ms for p in self.phases)

    @property
    def failed_phases(self) -> List[PhaseResult]:
        return [p for p in self.phases if not p.success]

    @property
    def skipped_phases(self) -> List[PhaseResult]:
        return [p for p in self.phases if p.skipped]


class InitializationPhaseManager:
    """Manages orchestrator runtime initialization phases.

    Wraps the 9 _initialize_* methods on the orchestrator to provide
    structured timing, error reporting, component tracking, fail-fast
    for critical phases, and dependency-based skipping.

    Usage:
        manager = InitializationPhaseManager()
        result = manager.run_all_phases(orchestrator)
    """

    def run_all_phases(self, orchestrator: Any) -> InitializationResult:
        """Run all initialization phases in dependency order.

        Phase order:
        1. provider_runtime — lazy provider coordinator loading (CRITICAL)
        2. metrics_runtime — metrics collectors and coordinators
        3. workflow_runtime — lazy workflow registry
        4. memory_runtime — memory manager and embedding store
        5. resilience_runtime — recovery handler and integration
        6. coordination_runtime — recovery/chunk/planner/task coordinators
        7. interaction_runtime — chat/tool/session coordinators (CRITICAL)
        8. services — DI service layer delegation (Strangler Fig)
        9. credit_runtime — credit assignment tracking (opt-in)

        Raises:
            InitializationError: If a critical phase fails.
        """
        result = InitializationResult()
        succeeded_phases: Set[str] = set()

        phases: List[tuple[str, Callable[[], None], List[str], bool, List[str]]] = [
            (
                "provider_runtime",
                orchestrator._initialize_provider_runtime,
                ["provider_coordinator", "provider_switch_coordinator"],
                True,  # critical
                [],  # dependencies
            ),
            (
                "metrics_runtime",
                orchestrator._initialize_metrics_runtime,
                [
                    "usage_logger",
                    "streaming_metrics_collector",
                    "metrics_coordinator",
                ],
                False,
                [],
            ),
            (
                "workflow_runtime",
                orchestrator._initialize_workflow_runtime,
                ["workflow_registry"],
                False,
                [],
            ),
            (
                "memory_runtime",
                orchestrator._initialize_memory_runtime,
                ["memory_manager"],
                False,
                [],
            ),
            (
                "resilience_runtime",
                lambda: orchestrator._initialize_resilience_runtime(
                    context_compactor=orchestrator._context_compactor,
                ),
                ["recovery_handler", "recovery_integration"],
                False,
                ["provider_runtime"],
            ),
            (
                "coordination_runtime",
                orchestrator._initialize_coordination_runtime,
                [
                    "recovery_coordinator",
                    "chunk_generator",
                    "tool_planner",
                    "task_coordinator",
                ],
                False,
                ["provider_runtime"],
            ),
            (
                "interaction_runtime",
                orchestrator._initialize_interaction_runtime,
                [
                    "chat_coordinator",
                    "session_coordinator",
                ],
                True,  # critical
                ["provider_runtime", "coordination_runtime"],
            ),
            (
                "services",
                orchestrator._initialize_services,
                [
                    "chat_service",
                    "tool_service",
                    "session_service",
                    "context_service",
                ],
                False,
                ["interaction_runtime"],
            ),
            (
                "credit_runtime",
                orchestrator._initialize_credit_runtime,
                ["credit_tracking_service"],
                False,  # non-critical
                ["interaction_runtime"],
            ),
        ]

        for name, initializer, components, critical, dependencies in phases:
            # Check if all dependencies succeeded
            missing_deps = [dep for dep in dependencies if dep not in succeeded_phases]
            if missing_deps:
                reason = (
                    f"skipped due to failed/skipped dependencies: " f"{', '.join(missing_deps)}"
                )
                phase_result = PhaseResult(
                    name=name,
                    success=False,
                    skipped=True,
                    skip_reason=reason,
                    error=reason,
                )
                result.phases.append(phase_result)
                logger.warning(
                    "Phase '%s' %s",
                    name,
                    reason,
                )
                if critical:
                    raise InitializationError(
                        phase=name,
                        error=reason,
                        completed_phases=sorted(succeeded_phases),
                    )
                continue

            phase_result = self._run_phase(name, initializer, components)
            result.phases.append(phase_result)

            if phase_result.success:
                succeeded_phases.add(name)
            elif critical:
                raise InitializationError(
                    phase=name,
                    error=phase_result.error or "unknown error",
                    completed_phases=sorted(succeeded_phases),
                )
            else:
                logger.warning(
                    "Non-critical phase '%s' failed: %s — "
                    "system will continue with degraded %s functionality",
                    name,
                    phase_result.error,
                    name.replace("_runtime", "").replace("_", " "),
                )

        if result.all_succeeded:
            logger.debug(
                "All %d initialization phases completed in %.1fms",
                len(result.phases),
                result.total_duration_ms,
            )
        else:
            failed = ", ".join(p.name for p in result.failed_phases)
            logger.warning("Initialization failures in phases: %s", failed)

        return result

    def _run_phase(
        self,
        name: str,
        initializer: Callable[[], None],
        expected_components: List[str],
    ) -> PhaseResult:
        """Run a single initialization phase with timing and error handling."""
        start = time.monotonic()
        phase = PhaseResult(name=name)

        try:
            initializer()
            phase.components_created = expected_components
        except Exception as exc:
            phase.success = False
            phase.error = str(exc)
            logger.error("Initialization phase '%s' failed: %s", name, exc)

        phase.duration_ms = (time.monotonic() - start) * 1000
        return phase
