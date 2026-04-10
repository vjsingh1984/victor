"""Tests for InitializationPhaseManager."""

from unittest.mock import MagicMock

from victor.agent.runtime.initialization_manager import (
    InitializationPhaseManager,
    InitializationResult,
    PhaseResult,
)


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
        assert len(result.phases) == 8

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
