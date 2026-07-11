"""Tests for InitializationPhaseManager."""

from unittest.mock import MagicMock

from victor.agent.runtime.initialization_manager import (
    InitializationPhaseManager,
    InitializationResult,
    PhaseGroup,
    PhaseResult,
)


def _mock_orchestrator_all_phases():
    """A MagicMock orchestrator whose 9 _initialize_* phases are plain callables."""
    orch = MagicMock()
    for name in (
        "_initialize_provider_runtime",
        "_initialize_metrics_runtime",
        "_initialize_workflow_runtime",
        "_initialize_memory_runtime",
        "_initialize_resilience_runtime",
        "_initialize_coordination_runtime",
        "_initialize_interaction_runtime",
        "_initialize_services",
        "_initialize_credit_runtime",
    ):
        setattr(orch, name, MagicMock())
    orch._context_compactor = MagicMock()
    return orch


class TestPhaseResult:
    def test_defaults(self):
        r = PhaseResult(name="test")
        assert r.success is True
        assert r.duration_ms == 0.0
        assert r.components_created == []
        assert r.error is None


class TestInitializationResult:
    def test_all_succeeded(self):
        result = InitializationResult(
            phases=[
                PhaseResult(name="a", success=True),
                PhaseResult(name="b", success=True),
            ]
        )
        assert result.all_succeeded is True

    def test_not_all_succeeded(self):
        result = InitializationResult(
            phases=[
                PhaseResult(name="a", success=True),
                PhaseResult(name="b", success=False, error="boom"),
            ]
        )
        assert result.all_succeeded is False
        assert len(result.failed_phases) == 1

    def test_total_duration(self):
        result = InitializationResult(
            phases=[
                PhaseResult(name="a", duration_ms=10.0),
                PhaseResult(name="b", duration_ms=20.0),
            ]
        )
        assert result.total_duration_ms == 30.0


class TestInitializationPhaseManager:
    def test_run_phase_success(self):
        manager = InitializationPhaseManager()
        result = manager._run_phase("test", lambda: None, ["comp1"])
        assert result.success is True
        assert result.components_created == ["comp1"]
        assert result.duration_ms >= 0

    def test_run_phase_failure(self):
        manager = InitializationPhaseManager()

        def failing():
            raise RuntimeError("init failed")

        result = manager._run_phase("test", failing, ["comp1"])
        assert result.success is False
        assert "init failed" in result.error

    def test_run_all_phases(self):
        manager = InitializationPhaseManager()
        orchestrator = MagicMock()

        # All init methods should be plain callables
        orchestrator._initialize_provider_runtime = MagicMock()
        orchestrator._initialize_metrics_runtime = MagicMock()
        orchestrator._initialize_workflow_runtime = MagicMock()
        orchestrator._initialize_memory_runtime = MagicMock()
        orchestrator._initialize_resilience_runtime = MagicMock()
        orchestrator._initialize_coordination_runtime = MagicMock()
        orchestrator._initialize_interaction_runtime = MagicMock()
        orchestrator._initialize_services = MagicMock()
        orchestrator._context_compactor = MagicMock()

        result = manager.run_all_phases(orchestrator)
        assert result.all_succeeded is True
        assert len(result.phases) == 9
        provider_runtime_phase = next(p for p in result.phases if p.name == "provider_runtime")
        assert provider_runtime_phase.components_created == ["provider_runtime"]

        # Verify all init methods were called
        orchestrator._initialize_provider_runtime.assert_called_once()
        orchestrator._initialize_metrics_runtime.assert_called_once()
        orchestrator._initialize_workflow_runtime.assert_called_once()
        orchestrator._initialize_memory_runtime.assert_called_once()

    def test_run_all_phases_partial_failure(self):
        manager = InitializationPhaseManager()
        orchestrator = MagicMock()
        orchestrator._context_compactor = MagicMock()

        # Make metrics_runtime fail
        orchestrator._initialize_provider_runtime = MagicMock()
        orchestrator._initialize_metrics_runtime = MagicMock(side_effect=RuntimeError("boom"))
        orchestrator._initialize_workflow_runtime = MagicMock()
        orchestrator._initialize_memory_runtime = MagicMock()
        orchestrator._initialize_resilience_runtime = MagicMock()
        orchestrator._initialize_coordination_runtime = MagicMock()
        orchestrator._initialize_interaction_runtime = MagicMock()
        orchestrator._initialize_services = MagicMock()

        result = manager.run_all_phases(orchestrator)
        assert not result.all_succeeded
        assert len(result.failed_phases) == 1
        assert result.failed_phases[0].name == "metrics_runtime"


class TestRunGroup:
    """FEP-0016: phases run in three groups at construction boundaries."""

    def test_group_membership(self):
        """EARLY=4, ASSEMBLY=2, SERVICE=3, covering all 9 phases exactly once."""
        manager = InitializationPhaseManager()
        specs = manager._phase_specs(_mock_orchestrator_all_phases())
        by_group = {g: [s.name for s in specs if s.group is g] for g in PhaseGroup}
        assert by_group[PhaseGroup.EARLY] == [
            "provider_runtime",
            "metrics_runtime",
            "workflow_runtime",
            "memory_runtime",
        ]
        assert by_group[PhaseGroup.ASSEMBLY] == ["resilience_runtime", "coordination_runtime"]
        assert by_group[PhaseGroup.SERVICE] == [
            "interaction_runtime",
            "services",
            "credit_runtime",
        ]

    def test_run_group_runs_only_that_group(self):
        manager = InitializationPhaseManager()
        orch = _mock_orchestrator_all_phases()

        result = manager.run_group(orch, PhaseGroup.EARLY)

        assert [p.name for p in result.phases] == [
            "provider_runtime",
            "metrics_runtime",
            "workflow_runtime",
            "memory_runtime",
        ]
        orch._initialize_provider_runtime.assert_called_once()
        orch._initialize_memory_runtime.assert_called_once()
        # A later-group phase must NOT have run yet.
        orch._initialize_interaction_runtime.assert_not_called()
        orch._initialize_credit_runtime.assert_not_called()

    def test_groups_accumulate_and_cross_group_deps_satisfied(self):
        """Running the 3 groups in order = all 9 phases; cross-group deps (e.g.
        resilience_runtime -> provider_runtime) are NOT skipped."""
        manager = InitializationPhaseManager()
        orch = _mock_orchestrator_all_phases()

        manager.run_group(orch, PhaseGroup.EARLY)
        manager.run_group(orch, PhaseGroup.ASSEMBLY)
        result = manager.run_group(orch, PhaseGroup.SERVICE)

        assert result.all_succeeded is True
        assert len(result.phases) == 9
        assert result.skipped_phases == []  # provider succeeded in EARLY, so resilience runs
        orch._initialize_credit_runtime.assert_called_once()  # the phase that was lost before

    def test_critical_phase_failure_in_group_fails_fast(self):
        """A failed critical phase raises InitializationError from its group."""
        import pytest

        from victor.agent.runtime.initialization_manager import InitializationError

        manager = InitializationPhaseManager()
        orch = _mock_orchestrator_all_phases()
        # provider_runtime (critical, EARLY) fails -> EARLY raises immediately.
        orch._initialize_provider_runtime = MagicMock(side_effect=RuntimeError("no provider"))

        with pytest.raises(InitializationError) as exc_info:
            manager.run_group(orch, PhaseGroup.EARLY)
        assert exc_info.value.phase == "provider_runtime"


class TestNoRawInitCallSites:
    """FEP-0016 guard: every init phase is invoked via the manager's run_phase;
    no raw ``orchestrator._initialize_*()`` call sites remain in the construction
    path. This is the regression guard that prevents a phase from being silently
    lost (the class of bug that hid credit_runtime until #464)."""

    def test_no_raw_initialize_calls_in_construction_path(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[3]
        files = [
            "victor/agent/orchestrator.py",
            "victor/agent/runtime/component_assembler.py",
            "victor/agent/runtime/bootstrapper.py",
        ]
        # A raw call = something like `x._initialize_<phase>(` (a dot before the
        # name, so `def _initialize_x(` definitions are not matched).
        pattern = re.compile(
            r"\w+\._initialize_"
            r"(provider|metrics|workflow|memory|resilience|coordination|interaction|credit)"
            r"_runtime\s*\(|\w+\._initialize_services\s*\("
        )
        offenders = []
        for rel in files:
            for i, line in enumerate((root / rel).read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        assert not offenders, (
            "raw _initialize_* call sites must go through _init_manager.run_phase "
            "(FEP-0016):\n" + "\n".join(offenders)
        )

    def test_manager_covers_all_nine_phases(self):
        from unittest.mock import MagicMock

        manager = InitializationPhaseManager()
        names = [s.name for s in manager._phase_specs(MagicMock())]
        assert names == [
            "provider_runtime",
            "metrics_runtime",
            "workflow_runtime",
            "memory_runtime",
            "resilience_runtime",
            "coordination_runtime",
            "interaction_runtime",
            "services",
            "credit_runtime",
        ]
