"""Tests for victor.framework.agentic_loop module.

Tests the AgenticLoop integration that wires together existing Victor
components for PERCEIVE → PLAN → ACT → EVALUATE → DECIDE loops.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.action_authorizer import ActionIntent
from victor.agent.services.turn_execution_runtime import TurnResult
from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyDecisionInput,
    TopologyGroundingRequirements,
    TopologyKind,
)
from victor.agent.topology_grounder import GroundedTopologyPlan
from victor.framework.agentic_loop import (
    AgenticLoop,
    LoopIteration,
    LoopResult,
    LoopStage,
)
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.fulfillment import FulfillmentResult, FulfillmentStatus, TaskType
from victor.framework.perception_integration import Perception
from victor.framework.team_runtime import ResolvedTeamExecutionPlan
from victor.framework.task.protocols import TaskComplexity
from victor.providers.base import CompletionResponse
from victor.providers.performance_tracker import ProviderPerformanceTracker, RequestMetric
from victor.teams.types import TeamFormation, TeamResult

# ============================================================================
# LoopStage enum tests
# ============================================================================


class TestLoopStage:
    """Tests for LoopStage enum."""

    def test_values(self):
        assert LoopStage.PERCEIVE.value == "perceive"
        assert LoopStage.PLAN.value == "plan"
        assert LoopStage.ACT.value == "act"
        assert LoopStage.EVALUATE.value == "evaluate"
        assert LoopStage.DECIDE.value == "decide"
        assert LoopStage.COMPLETE.value == "complete"


# ============================================================================
# LoopIteration tests
# ============================================================================


class TestLoopIteration:
    """Tests for LoopIteration dataclass."""

    def test_to_dict_minimal(self):
        iteration = LoopIteration(iteration=1, stage=LoopStage.PERCEIVE)
        d = iteration.to_dict()
        assert d["iteration"] == 1
        assert d["stage"] == "perceive"
        assert d["perception"] is None
        assert d["evaluation"] is None

    def test_to_dict_with_perception(self):
        perception = MagicMock(spec=Perception)
        perception.to_dict.return_value = {"intent": "write_allowed"}

        iteration = LoopIteration(
            iteration=2,
            stage=LoopStage.EVALUATE,
            perception=perception,
        )
        d = iteration.to_dict()
        assert d["perception"] == {"intent": "write_allowed"}

    def test_to_dict_with_evaluation(self):
        evaluation = EvaluationResult(
            decision=EvaluationDecision.CONTINUE,
            score=0.7,
            reason="Making progress",
        )
        iteration = LoopIteration(
            iteration=1,
            stage=LoopStage.EVALUATE,
            evaluation=evaluation,
        )
        d = iteration.to_dict()
        assert d["evaluation"]["score"] == 0.7
        assert d["evaluation"]["reason"] == "Making progress"


# ============================================================================
# LoopResult tests
# ============================================================================


class TestLoopResult:
    """Tests for LoopResult dataclass."""

    def test_to_dict(self):
        result = LoopResult(
            success=True,
            iterations=[LoopIteration(iteration=1, stage=LoopStage.COMPLETE)],
            final_state={"done": True},
            total_duration=1.5,
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["iterations_count"] == 1
        assert d["total_duration"] == 1.5
        assert d["final_state"] == {"done": True}
        assert d["metadata"] == {"key": "value"}


# ============================================================================
# AgenticLoop tests
# ============================================================================


def _make_perception():
    """Create a mock Perception for testing."""
    return Perception(
        intent=ActionIntent.WRITE_ALLOWED,
        complexity=TaskComplexity.MEDIUM,
        task_analysis=MagicMock(task_type="code_generation"),
        confidence=0.8,
    )


class TestAgenticLoop:
    """Tests for AgenticLoop."""

    def _make_loop(self, orchestrator=None, **kwargs):
        """Create AgenticLoop with mock orchestrator."""
        if orchestrator is None:
            orchestrator = MagicMock()
        return AgenticLoop(
            orchestrator=orchestrator,
            enable_fulfillment_check=kwargs.pop("enable_fulfillment_check", False),
            **kwargs,
        )

    async def test_run_completes_on_high_confidence(self):
        """Loop should complete when perception confidence is high."""
        # Use spec=[] to prevent MagicMock from auto-generating attributes
        # like planning_coordinator/execute/run
        loop = self._make_loop(orchestrator=MagicMock(spec=[]), max_iterations=3)

        # Patch perception to return high confidence
        perception = _make_perception()
        perception.confidence = 0.9
        loop.perception.perceive = AsyncMock(return_value=perception)

        result = await loop.run("Fix the bug")
        assert result.success is True
        assert len(result.iterations) >= 1
        assert result.total_duration > 0

    async def test_run_stops_at_max_iterations(self):
        """Loop should stop after max_iterations."""
        loop = self._make_loop(max_iterations=2)

        # Patch perception to return low confidence (force continue)
        perception = _make_perception()
        perception.confidence = 0.3
        loop.perception.perceive = AsyncMock(return_value=perception)

        result = await loop.run("Fix the bug")
        assert len(result.iterations) <= 2

    async def test_run_handles_exception(self):
        """Loop should handle exceptions gracefully."""
        loop = self._make_loop(max_iterations=1)
        loop.perception.perceive = AsyncMock(side_effect=RuntimeError("test error"))

        result = await loop.run("Fix the bug")
        assert result.success is False
        assert "error" in result.metadata

    async def test_run_with_context(self):
        """Loop should pass context through."""
        loop = self._make_loop(max_iterations=1)

        perception = _make_perception()
        perception.confidence = 0.9
        loop.perception.perceive = AsyncMock(return_value=perception)

        result = await loop.run(
            "Fix the bug",
            context={"project": "myapp"},
        )
        assert result.final_state.get("project") == "myapp"

    async def test_run_uses_runtime_intelligence_snapshot(self):
        """Loop should consume the canonical runtime intelligence service when provided."""
        perception = _make_perception()
        perception.confidence = 0.9
        runtime_intelligence = MagicMock()
        runtime_intelligence.analyze_turn = AsyncMock(
            return_value=MagicMock(perception=perception, task_analysis=perception.task_analysis)
        )
        loop = self._make_loop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=1,
            runtime_intelligence=runtime_intelligence,
        )

        result = await loop.run("Fix the bug")

        assert result.success is True
        runtime_intelligence.analyze_turn.assert_awaited_once()

    async def test_run_applies_topology_overrides_and_records_event(self, monkeypatch):
        perception = _make_perception()
        perception.confidence = 0.9
        turn_executor = MagicMock()
        turn_executor.execute_turn = AsyncMock(
            return_value=TurnResult(
                response=CompletionResponse(content="done", role="assistant"),
                tool_results=[],
                has_tool_calls=False,
                tool_calls_count=0,
                all_tools_blocked=False,
                is_qa_response=False,
            )
        )
        loop = self._make_loop(
            orchestrator=MagicMock(spec=[]),
            turn_executor=turn_executor,
            max_iterations=1,
            config={"disable_enhanced_completion": True},
        )
        agentic_loop_module = __import__(
            "victor.framework.agentic_loop",
            fromlist=["emit_topology_telemetry_event"],
        )
        emit_mock = AsyncMock(return_value=True)
        monkeypatch.setattr(agentic_loop_module, "emit_topology_telemetry_event", emit_mock)
        loop._analyze_turn = AsyncMock(return_value=perception)
        loop._plan = AsyncMock(return_value={"steps": ["inspect", "execute"]})
        loop._evaluate = AsyncMock(
            return_value=EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=0.92,
                reason="Topology-guided execution completed",
            )
        )
        loop._get_topology_provider_hints = AsyncMock(return_value={})
        loop.paradigm_router = MagicMock()
        loop.paradigm_router.route.return_value = MagicMock(
            skip_planning=False,
            paradigm=SimpleNamespace(value="deep"),
            model_tier=SimpleNamespace(value="large"),
            max_tokens=4096,
            confidence=0.88,
            to_dict=MagicMock(return_value={"paradigm": "deep", "model_tier": "large"}),
        )
        loop.paradigm_router.build_topology_input = MagicMock(
            return_value=TopologyDecisionInput(
                query="Fix the bug",
                task_type="code_generation",
                task_complexity="high",
                tool_budget=6,
                iteration_budget=1,
                available_team_formations=["parallel", "hierarchical"],
            )
        )
        topology_decision = TopologyDecision(
            action=TopologyAction.TEAM_PLAN,
            topology=TopologyKind.TEAM,
            confidence=0.82,
            rationale="Task benefits from coordinated parallel execution",
            grounding_requirements=TopologyGroundingRequirements(
                provider="smart-router",
                formation="parallel",
                max_workers=3,
                tool_budget=4,
                iteration_budget=1,
            ),
            provider="smart-router",
            formation="parallel",
        )
        topology_plan = GroundedTopologyPlan(
            action=TopologyAction.TEAM_PLAN,
            topology=TopologyKind.TEAM,
            execution_mode="team_execution",
            provider="smart-router",
            formation="parallel",
            max_workers=3,
            tool_budget=4,
            iteration_budget=1,
            metadata={"source": "test"},
        )
        loop._topology_selector.select = MagicMock(return_value=topology_decision)
        loop._topology_grounder.ground = MagicMock(return_value=topology_plan)
        loop.runtime_intelligence.get_topology_routing_context = MagicMock(
            return_value={
                "learned_topology_action": "team_plan",
                "learned_provider_hint": "anthropic",
                "learned_topology_support": 0.7,
            }
        )
        loop.runtime_intelligence.record_topology_outcome = MagicMock()

        result = await loop.run("Fix the bug")

        assert result.success is True
        turn_kwargs = turn_executor.execute_turn.await_args.kwargs
        assert turn_kwargs["runtime_context_overrides"]["formation_hint"] == "parallel"
        assert turn_kwargs["runtime_context_overrides"]["max_workers"] == 3
        assert turn_kwargs["runtime_context_overrides"]["provider_hint"] == "smart-router"
        assert result.metadata["topology_events"][0]["action"] == "team_plan"
        assert result.metadata["topology_events"][0]["formation"] == "parallel"
        assert result.final_state["topology_plan"]["execution_mode"] == "team_execution"
        assert result.final_state["tool_budget"] == 4
        topology_context = loop.paradigm_router.build_topology_input.call_args.kwargs["context"]
        assert topology_context["learned_topology_action"] == "team_plan"
        assert topology_context["learned_provider_hint"] == "anthropic"
        learned_scope_context = (
            loop.runtime_intelligence.get_topology_routing_context.call_args.kwargs["scope_context"]
        )
        assert loop.runtime_intelligence.get_topology_routing_context.call_args.kwargs["query"] == (
            "Fix the bug"
        )
        assert learned_scope_context["task_type"] == "code_generation"
        assert "provider_hint" not in learned_scope_context
        loop.runtime_intelligence.record_topology_outcome.assert_called_once()
        feedback_payload = loop.runtime_intelligence.record_topology_outcome.call_args.args[0]
        assert feedback_payload["status"] == "completed"
        assert feedback_payload["completion_score"] == pytest.approx(0.92)
        emit_mock.assert_awaited_once()

    async def test_run_parallel_topology_prepares_runtime_once_selected(self, monkeypatch):
        perception = _make_perception()
        perception.confidence = 0.9
        task_classification = SimpleNamespace(tool_budget=6, complexity=TaskComplexity.COMPLEX)
        turn_executor = MagicMock()
        turn_executor.prepare_runtime_topology = AsyncMock(
            return_value={
                "action": "parallel_exploration",
                "prepared": True,
                "execution_mode": "parallel_exploration",
            }
        )
        turn_executor.execute_turn = AsyncMock(
            return_value=TurnResult(
                response=CompletionResponse(content="done", role="assistant"),
                tool_results=[],
                has_tool_calls=False,
                tool_calls_count=0,
                all_tools_blocked=False,
                is_qa_response=False,
            )
        )
        loop = self._make_loop(
            orchestrator=MagicMock(spec=[]),
            turn_executor=turn_executor,
            max_iterations=1,
        )
        agentic_loop_module = __import__(
            "victor.framework.agentic_loop",
            fromlist=["emit_topology_telemetry_event"],
        )
        emit_mock = AsyncMock(return_value=True)
        monkeypatch.setattr(agentic_loop_module, "emit_topology_telemetry_event", emit_mock)
        loop._analyze_turn = AsyncMock(return_value=perception)
        loop._plan = AsyncMock(return_value={"steps": ["inspect", "summarize"]})
        loop._evaluate = AsyncMock(
            return_value=EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=0.9,
                reason="Parallel exploration prepared the runtime",
            )
        )
        loop._get_topology_provider_hints = AsyncMock(return_value={})
        loop.paradigm_router = MagicMock()
        loop.paradigm_router.route.return_value = MagicMock(
            skip_planning=False,
            paradigm=SimpleNamespace(value="deep"),
            model_tier=SimpleNamespace(value="large"),
            max_tokens=4096,
            confidence=0.88,
            to_dict=MagicMock(return_value={"paradigm": "deep", "model_tier": "large"}),
        )
        loop.paradigm_router.build_topology_input = MagicMock(
            return_value=TopologyDecisionInput(
                query="Fix the bug",
                task_type="code_generation",
                task_complexity="high",
                expected_breadth="high",
                tool_budget=6,
                iteration_budget=1,
                available_team_formations=["parallel", "hierarchical"],
            )
        )
        topology_decision = TopologyDecision(
            action=TopologyAction.PARALLEL_EXPLORATION,
            topology=TopologyKind.PARALLEL_EXPLORATION,
            confidence=0.84,
            rationale="Breadth-heavy task benefits from exploration before execution",
            grounding_requirements=TopologyGroundingRequirements(
                max_workers=3,
                tool_budget=4,
                iteration_budget=1,
            ),
        )
        topology_plan = GroundedTopologyPlan(
            action=TopologyAction.PARALLEL_EXPLORATION,
            topology=TopologyKind.PARALLEL_EXPLORATION,
            execution_mode="parallel_exploration",
            max_workers=3,
            tool_budget=4,
            iteration_budget=1,
            metadata={"source": "test"},
        )
        loop._topology_selector.select = MagicMock(return_value=topology_decision)
        loop._topology_grounder.ground = MagicMock(return_value=topology_plan)

        result = await loop.run(
            "Fix the bug", context={"_task_classification": task_classification}
        )

        assert result.success is True
        turn_executor.prepare_runtime_topology.assert_awaited_once_with(
            topology_plan,
            user_message="Fix the bug",
            task_classification=task_classification,
        )
        assert result.final_state["topology_preparation"]["action"] == "parallel_exploration"
        assert result.final_state["topology_preparation"]["prepared"] is True
        assert (
            result.final_state["topology_preparation"]["execution_mode"] == "parallel_exploration"
        )

    async def test_run_team_topology_executes_framework_team_runtime(self, monkeypatch):
        perception = _make_perception()
        perception.confidence = 0.9
        turn_executor = MagicMock()
        turn_executor.prepare_runtime_topology = AsyncMock(
            return_value={
                "action": "team_plan",
                "prepared": True,
                "execution_mode": "team_execution",
                "team_name": "feature_team",
                "display_name": "Feature Team",
                "formation": "parallel",
                "member_count": 2,
                "runtime_context_overrides": {
                    "team_name": "feature_team",
                    "team_display_name": "Feature Team",
                    "formation_hint": "parallel",
                    "execution_mode": "team_execution",
                    "max_workers": 2,
                    "worktree_isolation": True,
                    "dry_run_worktrees": True,
                    "cleanup_worktrees": False,
                },
            }
        )
        turn_executor.execute_turn = AsyncMock()
        loop = self._make_loop(
            orchestrator=MagicMock(spec=[]),
            turn_executor=turn_executor,
            max_iterations=1,
        )
        loop._analyze_turn = AsyncMock(return_value=perception)
        loop._plan = AsyncMock(return_value={"steps": ["research", "implement"]})
        loop._get_topology_provider_hints = AsyncMock(return_value={})
        loop.paradigm_router = MagicMock()
        loop.paradigm_router.route.return_value = MagicMock(
            skip_planning=False,
            paradigm=SimpleNamespace(value="deep"),
            model_tier=SimpleNamespace(value="large"),
            max_tokens=4096,
            confidence=0.88,
            to_dict=MagicMock(return_value={"paradigm": "deep", "model_tier": "large"}),
        )
        loop.paradigm_router.build_topology_input = MagicMock(
            return_value=TopologyDecisionInput(
                query="Build the feature",
                task_type="feature",
                task_complexity="high",
                tool_budget=6,
                iteration_budget=1,
                available_team_formations=["parallel", "hierarchical"],
            )
        )
        topology_decision = TopologyDecision(
            action=TopologyAction.TEAM_PLAN,
            topology=TopologyKind.TEAM,
            confidence=0.84,
            rationale="Complex feature work benefits from coordinated execution",
            grounding_requirements=TopologyGroundingRequirements(
                formation="parallel",
                max_workers=2,
                tool_budget=4,
                iteration_budget=1,
            ),
            formation="parallel",
        )
        topology_plan = GroundedTopologyPlan(
            action=TopologyAction.TEAM_PLAN,
            topology=TopologyKind.TEAM,
            execution_mode="team_execution",
            formation="parallel",
            max_workers=2,
            tool_budget=4,
            iteration_budget=1,
        )
        loop._topology_selector.select = MagicMock(return_value=topology_decision)
        loop._topology_grounder.ground = MagicMock(return_value=topology_plan)

        with patch(
            "victor.framework.team_runtime.run_configured_team",
            new=AsyncMock(
                return_value=(
                    ResolvedTeamExecutionPlan(
                        team_name="feature_team",
                        display_name="Feature Team",
                        formation=TeamFormation.PARALLEL,
                        member_count=2,
                        total_tool_budget=4,
                        max_iterations=20,
                        max_workers=2,
                    ),
                    TeamResult(
                        success=True,
                        final_output=(
                            "Team synthesis: the feature plan is complete, code paths were "
                            "reviewed, and the final implementation guidance is ready."
                        ),
                        member_results={},
                        formation=TeamFormation.PARALLEL,
                        total_tool_calls=3,
                    ),
                )
            ),
        ) as run_team:
            result = await loop.run("Build the feature")

        run_team.assert_awaited_once()
        team_context = run_team.await_args.kwargs["context"]
        turn_executor.execute_turn.assert_not_called()
        assert result.success is True
        assert result.final_state["topology_preparation"]["team_name"] == "feature_team"
        assert result.final_state["topology_overrides"]["team_name"] == "feature_team"
        assert team_context["worktree_isolation"] is True
        assert team_context["dry_run_worktrees"] is True
        assert team_context["cleanup_worktrees"] is False

    async def test_evaluate_framework_team_execution_turn_completes(self):
        loop = self._make_loop(max_iterations=1, config={"disable_enhanced_completion": True})
        perception = _make_perception()
        action_result = TurnResult(
            response=CompletionResponse(
                content="Team execution completed with a final synthesized answer.",
                role="assistant",
                metadata={"execution_mode": "team_execution", "team_success": True},
            ),
            tool_results=[],
            has_tool_calls=False,
            tool_calls_count=0,
            all_tools_blocked=False,
            is_qa_response=False,
        )

        result = await loop._evaluate(perception, action_result, {"query": "Build the feature"})

        assert result.decision == EvaluationDecision.COMPLETE
        assert result.score == pytest.approx(0.92)

    async def test_experiment_memory_planning_hints_can_override_fast_path(self):
        perception = _make_perception()
        perception.confidence = 0.9
        loop = self._make_loop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=1,
            config={"enable_topology_routing": False},
        )
        loop._analyze_turn = AsyncMock(return_value=perception)
        loop._plan = AsyncMock(return_value={"steps": ["verify tests", "execute"]})
        loop._evaluate = AsyncMock(
            return_value=EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=0.89,
                reason="Forced planning handled experiment constraints",
            )
        )
        loop.paradigm_router = MagicMock()
        loop.paradigm_router.route.return_value = MagicMock(
            skip_planning=False,
            paradigm=SimpleNamespace(value="fast"),
            model_tier=SimpleNamespace(value="small"),
            max_tokens=1024,
            confidence=0.73,
            to_dict=MagicMock(return_value={"paradigm": "fast", "model_tier": "small"}),
        )
        loop.runtime_intelligence.get_planning_routing_context = MagicMock(
            return_value={
                "planning_force_llm": True,
                "planning_force_reason": "experiment_constraints: tests_pass",
                "planning_constraint_tags": ["tests_pass"],
                "planning_experiment_support": 0.3333,
            }
        )

        result = await loop.run(
            "run the tests",
            context={"task_type": "action", "tool_budget": 1},
        )

        assert result.success is True
        assert loop._plan.await_count >= 1
        assert result.final_state["plan"]["steps"] == ["verify tests", "execute"]
        assert result.metadata["planning_events"][0]["selection_policy"] == (
            "experiment_forced_slow_path"
        )
        assert result.metadata["planning_events"][0]["used_llm_planning"] is True
        assert result.final_state["planning_routing_hints"]["planning_force_llm"] is True
        assert result.metadata["planning_routing_hints"]["planning_force_reason"] == (
            "experiment_constraints: tests_pass"
        )
        planning_scope_context = (
            loop.runtime_intelligence.get_planning_routing_context.call_args.kwargs["scope_context"]
        )
        assert planning_scope_context["task_type"] == "action"

    async def test_run_records_provider_degradation_recovery_event(self):
        perception = _make_perception()
        perception.confidence = 0.9
        tracker = ProviderPerformanceTracker(db=None)
        provider = SimpleNamespace(name="ollama", tracker=tracker)
        provider_context = SimpleNamespace(provider=provider, model="test-model")
        now = datetime.now()
        for offset in range(2):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test-model",
                    success=False,
                    latency_ms=2000.0,
                    timestamp=now,
                    error_type="ProviderError",
                )
            )

        turn_executor = MagicMock()
        turn_executor._provider_context = provider_context
        turn_executor.execute_turn = AsyncMock(
            return_value=TurnResult(
                response=CompletionResponse(content="done", role="assistant"),
                tool_results=[],
                has_tool_calls=False,
                tool_calls_count=0,
                all_tools_blocked=False,
                is_qa_response=False,
            )
        )
        loop = self._make_loop(
            orchestrator=MagicMock(spec=[]),
            turn_executor=turn_executor,
            max_iterations=1,
            config={"enable_topology_routing": False},
        )
        loop._analyze_turn = AsyncMock(return_value=perception)
        loop._plan = AsyncMock(return_value={"steps": ["recover", "execute"]})
        loop._evaluate = AsyncMock(
            return_value=EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=0.88,
                reason="Provider recovered",
            )
        )
        loop.paradigm_router = MagicMock()
        loop.paradigm_router.route.return_value = MagicMock(
            skip_planning=False,
            paradigm=SimpleNamespace(value="deep"),
            model_tier=SimpleNamespace(value="small"),
            max_tokens=2048,
            confidence=0.72,
            to_dict=MagicMock(return_value={"paradigm": "deep", "model_tier": "small"}),
        )

        async def _act_with_recovery(*_args, **_kwargs):
            for step in range(3):
                tracker.record_request(
                    RequestMetric(
                        provider="ollama",
                        model="test-model",
                        success=True,
                        latency_ms=700.0,
                        timestamp=now + timedelta(seconds=step + 2),
                    )
                )
            return TurnResult(
                response=CompletionResponse(content="done", role="assistant"),
                tool_results=[],
                has_tool_calls=False,
                tool_calls_count=0,
                all_tools_blocked=False,
                is_qa_response=False,
            )

        loop._act = AsyncMock(side_effect=_act_with_recovery)

        result = await loop.run("Fix the provider issue", context={"task_type": "action"})

        assert result.success is True
        assert result.metadata["degradation_events"][0]["source"] == "provider_performance"
        assert result.metadata["degradation_events"][0]["recovered"] is True
        assert result.metadata["degradation_events"][0]["provider"] == "ollama"
        assert result.metadata["degradation_events"][0]["failure_type"] == "PROVIDER_ERROR"

    async def test_fast_path_skips_planning_and_uses_direct_execution_plan(self):
        perception = _make_perception()
        perception.confidence = 0.9
        loop = self._make_loop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=1,
            config={"enable_topology_routing": False},
        )
        loop._analyze_turn = AsyncMock(return_value=perception)
        loop._plan = AsyncMock(return_value={"steps": ["should-not-run"]})
        loop._evaluate = AsyncMock(
            return_value=EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=0.81,
                reason="Fast-path execution completed",
            )
        )
        loop.paradigm_router = MagicMock()
        loop.paradigm_router.route.return_value = MagicMock(
            skip_planning=False,
            paradigm=SimpleNamespace(value="fast"),
            model_tier=SimpleNamespace(value="small"),
            max_tokens=1024,
            confidence=0.71,
            to_dict=MagicMock(return_value={"paradigm": "fast", "model_tier": "small"}),
        )
        loop.runtime_intelligence.get_planning_routing_context = MagicMock(return_value={})

        result = await loop.run(
            "run the tests",
            context={"task_type": "action", "tool_budget": 1},
        )

        assert result.success is True
        loop._plan.assert_not_awaited()
        assert result.final_state["plan"]["planning_skipped"] is True
        assert result.metadata["planning_events"][0]["selection_policy"] == "heuristic_fast_path"
        assert result.metadata["planning_events"][0]["used_llm_planning"] is False

    def test_loop_uses_policy_completion_threshold_from_config(self):
        loop = self._make_loop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=1,
            config={"completion_threshold": 0.93},
        )

        assert loop._evaluation_policy.completion_threshold == 0.93
        assert loop.enhanced_completion_evaluator.completion_threshold == 0.93
        assert loop.enhanced_completion_evaluator.completion_scorer.default_threshold == 0.93

    async def test_determine_success_complete(self):
        """Success determined by last evaluation."""
        loop = self._make_loop()
        iterations = [
            LoopIteration(
                iteration=1,
                stage=LoopStage.EVALUATE,
                evaluation=EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0),
            )
        ]
        assert loop._determine_success(iterations) is True

    async def test_determine_success_fail(self):
        loop = self._make_loop()
        iterations = [
            LoopIteration(
                iteration=1,
                stage=LoopStage.EVALUATE,
                evaluation=EvaluationResult(decision=EvaluationDecision.FAIL, score=0.0),
            )
        ]
        assert loop._determine_success(iterations) is False

    async def test_determine_success_empty(self):
        loop = self._make_loop()
        assert loop._determine_success([]) is False


# ============================================================================
# _plan fallback tests
# ============================================================================


class TestPlanFallbacks:
    """Tests for _plan method fallback chain."""

    async def test_plan_with_planning_coordinator(self):
        """Uses PlanningCoordinator when available."""
        mock_coordinator = AsyncMock()
        mock_coordinator.chat_with_planning = AsyncMock(return_value={"steps": ["step1"]})
        orchestrator = MagicMock()
        orchestrator.planning_coordinator = mock_coordinator

        loop = AgenticLoop(orchestrator=orchestrator, enable_fulfillment_check=False)
        perception = _make_perception()
        result = await loop._plan(perception, {"query": "test"})

        mock_coordinator.chat_with_planning.assert_called_once_with("test")
        assert result == {"steps": ["step1"]}

    async def test_plan_with_orchestrator_plan_method(self):
        """Falls back to orchestrator.plan()."""
        orchestrator = MagicMock(spec=[])
        orchestrator.plan = AsyncMock(return_value={"plan": "direct"})

        loop = AgenticLoop(orchestrator=orchestrator, enable_fulfillment_check=False)
        perception = _make_perception()
        result = await loop._plan(perception, {"query": "test"})

        assert result == {"plan": "direct"}

    async def test_plan_fallback_to_perception(self):
        """Falls back to perception dict when no plan methods available."""
        orchestrator = MagicMock(spec=[])

        loop = AgenticLoop(orchestrator=orchestrator, enable_fulfillment_check=False)
        perception = _make_perception()
        result = await loop._plan(perception, {"query": "test"})

        assert "perception" in result


# ============================================================================
# _act fallback tests
# ============================================================================


class TestActFallbacks:
    """Tests for _act method fallback chain."""

    async def test_act_with_execute(self):
        orchestrator = MagicMock(spec=[])
        orchestrator.execute = AsyncMock(return_value={"result": "done"})

        loop = AgenticLoop(orchestrator=orchestrator, enable_fulfillment_check=False)
        result = await loop._act({"plan": "test"}, {"query": "test"})
        assert result == {"result": "done"}

    async def test_act_with_run(self):
        orchestrator = MagicMock(spec=[])
        orchestrator.run = AsyncMock(return_value="response")

        loop = AgenticLoop(orchestrator=orchestrator, enable_fulfillment_check=False)
        result = await loop._act({"plan": "test"}, {"query": "test"})
        assert result == "response"

    async def test_act_fallback(self):
        orchestrator = MagicMock(spec=[])

        loop = AgenticLoop(orchestrator=orchestrator, enable_fulfillment_check=False)
        result = await loop._act({"plan": "test"}, {"query": "test"})
        assert result["plan_executed"] is True


# ============================================================================
# _evaluate tests
# ============================================================================


class TestEvaluate:
    """Tests for _evaluate method."""

    async def test_evaluate_high_confidence(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.confidence = 0.9

        result = await loop._evaluate(perception, {}, {})
        assert result.decision == EvaluationDecision.COMPLETE

    async def test_evaluate_medium_confidence(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.confidence = 0.6

        result = await loop._evaluate(perception, {}, {})
        assert result.decision == EvaluationDecision.CONTINUE

    async def test_evaluate_uses_configured_medium_confidence_threshold(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
            config={"medium_confidence_threshold": 0.7, "low_confidence_retry_limit": 2},
        )
        perception = _make_perception()
        perception.confidence = 0.6

        state = {}
        result = await loop._evaluate(perception, {}, state)
        assert result.decision == EvaluationDecision.RETRY
        assert state["low_confidence_retries"] == 1

    async def test_evaluate_low_confidence(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.confidence = 0.3

        state = {}
        result = await loop._evaluate(perception, {}, state)
        assert result.decision == EvaluationDecision.RETRY
        assert state["low_confidence_retries"] == 1
        assert result.metadata["low_confidence_retries"] == 1

    async def test_evaluate_requires_clarification_before_retry(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.confidence = 0.32
        perception.needs_clarification = True
        perception.clarification_reason = "target artifact or scope is underspecified"
        perception.clarification_prompt = "Which file, component, or bug should I target first?"

        result = await loop._evaluate(perception, {}, {})

        assert result.decision == EvaluationDecision.FAIL
        assert result.score == 0.32
        assert "Clarification required" in result.reason
        assert result.metadata["requires_clarification"] is True
        assert (
            result.metadata["clarification_prompt"]
            == "Which file, component, or bug should I target first?"
        )

    async def test_evaluate_requires_clarification_uses_default_prompt_when_missing(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.confidence = 0.28
        perception.needs_clarification = True
        perception.clarification_reason = "target artifact or scope is underspecified"
        perception.clarification_prompt = None

        result = await loop._evaluate(perception, {}, {})

        assert result.decision == EvaluationDecision.FAIL
        assert result.metadata["requires_clarification"] is True
        assert (
            result.metadata["clarification_prompt"]
            == "Please clarify the target file, component, or bug before I continue."
        )

    async def test_evaluate_fails_after_low_confidence_retry_budget_exhausted(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
            config={"low_confidence_retry_limit": 2},
        )
        perception = _make_perception()
        perception.confidence = 0.2
        state = {"low_confidence_retries": 2}

        result = await loop._evaluate(perception, {}, state)

        assert result.decision == EvaluationDecision.FAIL
        assert result.metadata["low_confidence_retry_exhausted"] is True
        assert result.metadata["low_confidence_retries"] == 2

    async def test_evaluate_resets_low_confidence_retry_budget_on_progress(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
            config={"low_confidence_retry_limit": 2},
        )
        perception = _make_perception()
        perception.confidence = 0.6
        state = {"low_confidence_retries": 1}

        result = await loop._evaluate(perception, {}, state)

        assert result.decision == EvaluationDecision.CONTINUE
        assert state["low_confidence_retries"] == 0

    async def test_evaluate_applies_retry_budget_to_enhanced_low_confidence_result(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
            config={"low_confidence_retry_limit": 2},
        )
        loop.enhanced_completion_evaluator = MagicMock()
        loop.enhanced_completion_evaluator.evaluate = AsyncMock(
            return_value=EvaluationResult(
                decision=EvaluationDecision.RETRY,
                score=0.2,
                reason="Low confidence - retry",
                metadata={"source": "enhanced"},
            )
        )
        loop._should_use_enhanced_evaluation = MagicMock(return_value=True)
        perception = _make_perception()
        perception.confidence = 0.2
        state = {"low_confidence_retries": 2}

        result = await loop._evaluate(perception, MagicMock(), state)

        assert result.decision == EvaluationDecision.FAIL
        assert result.metadata["low_confidence_retry_exhausted"] is True
        assert result.metadata["source"] == "enhanced"


# ============================================================================
# _map_to_task_type tests
# ============================================================================


class TestMapToTaskType:
    """Tests for _map_to_task_type method."""

    def test_write_allowed_maps_to_code_generation(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.intent = ActionIntent.WRITE_ALLOWED
        perception.task_analysis = MagicMock(task_type=None)
        result = loop._map_to_task_type(perception)
        assert result == TaskType.CODE_GENERATION

    def test_display_only_maps_to_search(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.intent = ActionIntent.DISPLAY_ONLY
        perception.task_analysis = MagicMock(task_type=None)
        result = loop._map_to_task_type(perception)
        assert result == TaskType.SEARCH

    def test_read_only_maps_to_analysis(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.intent = ActionIntent.READ_ONLY
        perception.task_analysis = MagicMock(task_type=None)
        result = loop._map_to_task_type(perception)
        assert result == TaskType.ANALYSIS

    def test_unknown_intent_maps_to_unknown(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.intent = ActionIntent.AMBIGUOUS
        perception.task_analysis = MagicMock(task_type=None)
        result = loop._map_to_task_type(perception)
        assert result == TaskType.UNKNOWN


# ============================================================================
# stream() tests
# ============================================================================


class TestStream:
    """Tests for AgenticLoop.stream()."""

    async def test_stream_yields_iterations(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=2,
            enable_fulfillment_check=False,
        )

        perception = _make_perception()
        perception.confidence = 0.9
        loop.perception.perceive = AsyncMock(return_value=perception)

        iterations = []
        async for iteration in loop.stream("Fix the bug"):
            iterations.append(iteration)
            # Break early after first complete evaluation
            if (
                iteration.evaluation
                and iteration.evaluation.decision == EvaluationDecision.COMPLETE
            ):
                break

        assert len(iterations) >= 1


# ============================================================================
# Adaptive iteration tests
# ============================================================================


class TestAdaptiveIterations:
    """Tests for adaptive iteration features."""

    def test_check_adaptive_plateau(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
            plateau_window=3,
            plateau_tolerance=0.02,
        )
        loop._progress_scores = [0.5, 0.505, 0.51]
        evaluation = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.51)
        result = loop._check_adaptive_termination(3, evaluation)
        assert result == "plateau"

    def test_check_adaptive_extend(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        loop._progress_scores = [0.5, 0.75]
        evaluation = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.75)
        result = loop._check_adaptive_termination(2, evaluation)
        assert result == "extend"

    def test_check_adaptive_none(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        loop._progress_scores = [0.3, 0.5]
        evaluation = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.5)
        result = loop._check_adaptive_termination(2, evaluation)
        assert result is None

    def test_no_adaptive_when_disabled(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
            max_iterations=1,
        )
        perception = _make_perception()
        perception.confidence = 0.6
        loop.perception.perceive = AsyncMock(return_value=perception)
        # Just verify it doesn't crash with adaptive disabled

    async def test_progress_scores_tracked(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=2,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )
        perception = _make_perception()
        perception.confidence = 0.6
        loop.perception.perceive = AsyncMock(return_value=perception)

        result = await loop.run("test query")
        assert "progress_scores" in result.metadata
        assert len(result.metadata["progress_scores"]) > 0


# ============================================================================
# Enhanced _map_to_task_type tests
# ============================================================================


class TestEnhancedTaskTypeMapping:
    """Tests for enhanced _map_to_task_type with TaskAnalysis fields."""

    def test_maps_from_task_analysis_type(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.task_analysis = MagicMock(task_type="debugging")
        result = loop._map_to_task_type(perception)
        assert result == TaskType.DEBUGGING

    def test_maps_from_task_analysis_testing(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.task_analysis = MagicMock(task_type="testing")
        result = loop._map_to_task_type(perception)
        assert result == TaskType.TESTING

    def test_falls_back_to_intent_when_no_task_type(self):
        loop = AgenticLoop(
            orchestrator=MagicMock(),
            enable_fulfillment_check=False,
        )
        perception = _make_perception()
        perception.task_analysis = MagicMock(task_type=None)
        perception.intent = ActionIntent.READ_ONLY
        result = loop._map_to_task_type(perception)
        assert result == TaskType.ANALYSIS
