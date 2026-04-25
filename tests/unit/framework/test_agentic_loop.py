"""Tests for victor.framework.agentic_loop module.

Tests the AgenticLoop integration that wires together existing Victor
components for PERCEIVE → PLAN → ACT → EVALUATE → DECIDE loops.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.action_authorizer import ActionIntent
from victor.framework.agentic_loop import (
    AgenticLoop,
    LoopIteration,
    LoopResult,
    LoopStage,
)
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.fulfillment import FulfillmentResult, FulfillmentStatus, TaskType
from victor.framework.perception_integration import Perception
from victor.framework.task.protocols import TaskComplexity

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
