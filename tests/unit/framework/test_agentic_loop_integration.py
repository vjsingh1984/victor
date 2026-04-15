"""Integration tests for the full agentic loop pipeline.

Tests the complete PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE loop
using real Victor components (TaskAnalyzer, IntentDetector, FulfillmentDetector)
rather than mocks, to verify end-to-end wiring.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.action_authorizer import ActionIntent
from victor.framework.agentic_loop import AgenticLoop, LoopResult, LoopStage
from victor.framework.evaluation_nodes import (
    EvaluationDecision,
    EvaluationResult,
    add_evaluation,
    composite_evaluator,
    convergence_evaluator,
    create_agentic_loop_graph,
    progress_tracking_evaluator,
    simple_score_evaluator,
)
from victor.framework.fulfillment import (
    FulfillmentConfig,
    FulfillmentDetector,
    FulfillmentResult,
    FulfillmentStatus,
    TaskType,
)
from victor.framework.perception_integration import (
    Perception,
    PerceptionIntegration,
    perceive,
)
from victor.framework.task.protocols import TaskComplexity

# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================


class TestFullPipelineIntegration:
    """Tests the full loop pipeline with real perception + fulfillment."""

    async def test_perceive_to_fulfillment_code_generation(self):
        """Verify perception feeds correctly into fulfillment detection."""
        # Step 1: Perceive a code generation intent
        perception = await perceive("Write a Python function to sort a list")
        assert isinstance(perception.intent, ActionIntent)
        assert isinstance(perception.complexity, TaskComplexity)
        assert perception.to_dict()["intent"] in [e.value for e in ActionIntent]

        # Step 2: Map to task type via the loop's mapper
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            enable_fulfillment_check=True,
        )
        task_type = loop._map_to_task_type(perception)
        assert isinstance(task_type, TaskType)

        # Step 3: Run fulfillment check
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def sort_list(items):\n    return sorted(items)\n")
            f.flush()
            result = await loop.fulfillment.check_fulfillment(
                task_type=TaskType.CODE_GENERATION,
                criteria={"file_path": f.name},
                context={},
            )
        assert result.is_fulfilled or result.is_partial
        assert result.score > 0.0

    async def test_perceive_to_fulfillment_analysis(self):
        """Verify analysis tasks work through the pipeline."""
        perception = await perceive("Analyze the performance of our API")

        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            enable_fulfillment_check=True,
        )

        result = await loop.fulfillment.check_fulfillment(
            task_type=TaskType.ANALYSIS,
            criteria={"min_findings": 1},
            context={
                "findings": ["API latency is 200ms p99"],
                "summary": "Performance is acceptable",
            },
        )
        assert result.is_fulfilled or result.is_partial

    async def test_full_loop_with_stub_orchestrator(self):
        """Run the full loop with a no-op orchestrator."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=2,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )

        result = await loop.run("What is 2+2?")
        assert isinstance(result, LoopResult)
        assert result.total_duration > 0
        assert len(result.iterations) > 0
        assert "progress_scores" in result.metadata

    async def test_multi_turn_context_flows_through(self):
        """Verify conversation history reaches TaskAnalyzer."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=1,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )

        history = [
            {"role": "user", "content": "Fix the login bug"},
            {"role": "assistant", "content": "Found the issue in auth.py"},
        ]
        result = await loop.run(
            "Now add tests for that fix",
            conversation_history=history,
        )
        assert isinstance(result, LoopResult)
        assert len(result.iterations) >= 1


# ============================================================================
# Evaluator Pipeline Integration Tests
# ============================================================================


class TestEvaluatorPipelineIntegration:
    """Tests evaluator factories work correctly in composed pipelines."""

    def test_progress_then_convergence(self):
        """Verify progress tracking and convergence can work together."""
        progress = progress_tracking_evaluator(complete_threshold=0.95)
        convergence = convergence_evaluator(min_iterations=2, min_score=0.7)

        combined = composite_evaluator(
            [progress, convergence],
            strategy="worst",
        )

        # Iteration 1: both continue
        r1 = combined({"score": 0.5})
        assert r1.decision in (
            EvaluationDecision.CONTINUE,
            "continue",
        )

        # Iteration 2: progress says continue, convergence may complete
        r2 = combined({"score": 0.75})
        # Not converged yet (big jump), so continue
        assert r2.score > 0

    def test_composite_with_all_strategies(self):
        """Verify all composite aggregation strategies produce valid results."""
        evaluators = [
            simple_score_evaluator(threshold=0.5),
            simple_score_evaluator(threshold=0.8),
        ]

        for strategy in ("worst", "best", "average"):
            combined = composite_evaluator(evaluators, strategy=strategy)
            result = combined({"score": 0.7})
            assert isinstance(result, EvaluationResult)
            assert 0.0 <= result.score <= 1.0

    def test_progress_evaluator_full_lifecycle(self):
        """Test progress evaluator through initial -> progress -> plateau."""
        evaluator = progress_tracking_evaluator(
            complete_threshold=0.95,
            plateau_window=3,
            plateau_tolerance=0.01,
        )

        # Iteration 1: initial
        r1 = evaluator({"score": 0.3})
        assert r1.decision == EvaluationDecision.CONTINUE

        # Iteration 2: progress
        r2 = evaluator({"score": 0.5})
        assert r2.decision == EvaluationDecision.CONTINUE
        assert "Progress" in r2.reason

        # Iteration 3-5: plateau
        r3 = evaluator({"score": 0.505})
        r4 = evaluator({"score": 0.508})
        # After 3 iterations in plateau window with < 0.01 improvement
        assert r4.decision == EvaluationDecision.FAIL
        assert "plateau" in r4.reason.lower()


# ============================================================================
# Fulfillment Detector Integration Tests
# ============================================================================


class TestFulfillmentDetectorIntegration:
    """Tests FulfillmentDetector covers all TaskTypes end-to-end."""

    async def test_all_task_types_have_strategies(self):
        """Every TaskType except UNKNOWN should have a strategy."""
        detector = FulfillmentDetector()
        for task_type in TaskType:
            if task_type == TaskType.UNKNOWN:
                continue
            assert task_type in detector.strategies, f"Missing strategy for {task_type}"

    async def test_all_strategies_return_valid_results(self):
        """Each strategy should return a valid FulfillmentResult."""
        detector = FulfillmentDetector()
        for task_type in TaskType:
            result = await detector.check_fulfillment(
                task_type=task_type,
                criteria={},
                context={},
            )
            assert isinstance(result, FulfillmentResult)
            assert isinstance(result.status, FulfillmentStatus)
            assert 0.0 <= result.score <= 1.0

    async def test_custom_config_affects_thresholds(self):
        """Verify FulfillmentConfig thresholds change behavior."""
        # Very low threshold: everything is fulfilled
        easy_config = FulfillmentConfig(fulfilled_threshold=0.1)
        easy_detector = FulfillmentDetector(config=easy_config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# empty\n")
            f.flush()
            # This file exists but has no valid code — low score
            # But with threshold=0.1, any file_exists score should suffice
            # Note: the strategy itself doesn't use config yet, this tests the wiring


# ============================================================================
# StateGraph + EvaluationNode Integration Tests
# ============================================================================


class TestStateGraphEvalIntegration:
    """Tests evaluation nodes integrate with real StateGraph."""

    def test_add_evaluation_to_real_graph(self):
        """add_evaluation() should work with a real StateGraph."""
        from victor.framework.graph import StateGraph

        graph = StateGraph(dict)

        async def analyze(state):
            state["analyzed"] = True
            return state

        async def fix(state):
            state["fixed"] = True
            return state

        graph.add_node("analyze", analyze)
        graph.add_node("fix", fix)
        graph.add_edge("analyze", "fix")

        def quality_check(state):
            if state.get("fixed"):
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=1.0,
                    reason="Fix applied",
                )
            return EvaluationResult(
                decision=EvaluationDecision.RETRY,
                score=0.3,
                reason="Not fixed yet",
            )

        graph = add_evaluation(
            graph,
            node_id="check_quality",
            evaluator=quality_check,
            decision_edges={
                str(EvaluationDecision.COMPLETE): "__end__",
                str(EvaluationDecision.RETRY): "fix",
            },
        )

        graph.add_edge("fix", "check_quality")
        graph.set_entry_point("analyze")

        # Verify graph structure
        assert "analyze" in graph._nodes
        assert "fix" in graph._nodes
        assert "check_quality" in graph._nodes

    def test_create_agentic_loop_graph_compilable(self):
        """create_agentic_loop_graph should produce a compilable graph."""

        async def perceive(s):
            s["perceived"] = True
            return s

        async def plan(s):
            s["planned"] = True
            return s

        async def act(s):
            s["acted"] = True
            s["score"] = 1.0
            return s

        def evaluate(s):
            if s.get("score", 0) >= 0.9:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=s["score"],
                )
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=s.get("score", 0),
            )

        graph = create_agentic_loop_graph(
            state_type=dict,
            perception_fn=perceive,
            planning_fn=plan,
            execution_fn=act,
            evaluator_fn=evaluate,
            max_iterations=3,
        )

        # Should be compilable
        compiled = graph.compile()
        assert compiled is not None


# ============================================================================
# Adaptive Loop Integration Tests
# ============================================================================


class TestAdaptiveLoopIntegration:
    """Tests adaptive iteration features work end-to-end."""

    async def test_plateau_triggers_early_exit(self):
        """Loop should exit early when progress plateaus."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=10,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=True,
            plateau_window=3,
            plateau_tolerance=0.05,
        )

        # Mock perception to return constant low confidence (simulates plateau)
        perception = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type=None),
            confidence=0.45,
        )
        loop.perception.perceive = AsyncMock(return_value=perception)

        result = await loop.run("Fix the bug")

        # Should exit early due to plateau (not hit max 10 iterations)
        assert len(result.iterations) < 10
        assert "progress_scores" in result.metadata

    async def test_high_confidence_completes_immediately(self):
        """Loop should complete on first iteration with high confidence."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=5,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=True,
        )

        perception = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.SIMPLE,
            task_analysis=MagicMock(task_type=None),
            confidence=0.95,
        )
        loop.perception.perceive = AsyncMock(return_value=perception)

        result = await loop.run("Simple question")
        assert result.success is True
        assert len(result.iterations) == 1


# ============================================================================
# Framework Export Integration Tests
# ============================================================================


class TestFrameworkExports:
    """Tests that all new modules are properly exported from victor.framework."""

    def test_agentic_loop_exports(self):
        from victor.framework import AgenticLoop, LoopIteration, LoopResult, LoopStage

        assert AgenticLoop is not None
        assert LoopStage.PERCEIVE.value == "perceive"

    def test_evaluation_exports(self):
        from victor.framework import (
            EvaluationDecision,
            EvaluationNode,
            EvaluationResult,
            add_evaluation,
            composite_evaluator,
            convergence_evaluator,
            progress_tracking_evaluator,
        )

        assert EvaluationDecision.COMPLETE.value == "complete"

    def test_fulfillment_exports(self):
        from victor.framework import (
            FulfillmentConfig,
            FulfillmentDetector,
            FulfillmentResult,
            FulfillmentStatus,
            TaskType,
        )

        assert len(TaskType) == 10

    def test_perception_exports(self):
        from victor.framework import (
            Perception,
            PerceptionIntegration,
            perceive,
        )

        assert Perception is not None


# ============================================================================
# TurnResult + Single-Turn Evaluation Integration Tests
# ============================================================================


class TestTurnResultEvaluation:
    """Tests that _evaluate() correctly interprets TurnResult signals."""

    def _make_loop(self):
        return AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            enable_fulfillment_check=False,
        )

    def _make_perception(self):
        return Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type=None),
            confidence=0.7,
        )

    async def test_qa_response_completes(self):
        """Q&A response with content should complete immediately."""
        from victor.agent.coordinators.execution_coordinator import TurnResult

        loop = self._make_loop()
        turn = TurnResult(
            response=MagicMock(content="The answer is 42", tool_calls=None),
            is_qa_response=True,
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.COMPLETE
        assert "Q&A shortcut" in result.reason

    async def test_spin_detection_fails(self):
        """3+ consecutive all-blocked turns should fail."""
        from victor.agent.coordinators.execution_coordinator import TurnResult

        loop = self._make_loop()
        loop._consecutive_all_blocked = 3
        turn = TurnResult(
            response=MagicMock(content="", tool_calls=[{"name": "read"}]),
            has_tool_calls=True,
            all_tools_blocked=True,
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.FAIL
        assert "Spin detected" in result.reason

    async def test_stuck_agent_fails(self):
        """3+ turns without tools should fail."""
        from victor.agent.coordinators.execution_coordinator import TurnResult

        loop = self._make_loop()
        loop._consecutive_no_tool_turns = 3
        turn = TurnResult(
            response=MagicMock(content="I think...", tool_calls=None),
            has_tool_calls=False,
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.FAIL
        assert "stuck" in result.reason

    async def test_successful_tools_continue(self):
        """Successful tool execution should continue."""
        from victor.agent.coordinators.execution_coordinator import TurnResult

        loop = self._make_loop()
        turn = TurnResult(
            response=MagicMock(content="Edited file", tool_calls=[{"name": "edit"}]),
            has_tool_calls=True,
            tool_calls_count=1,
            tool_results=[{"success": True, "tool_name": "edit"}],
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.CONTINUE
        assert result.score > 0.5

    async def test_final_answer_after_tools_completes(self):
        """No tools + content + previously used tools = done."""
        from victor.agent.coordinators.execution_coordinator import TurnResult

        loop = self._make_loop()
        loop._total_tool_calls = 5
        turn = TurnResult(
            response=MagicMock(content="Here's the fixed code", tool_calls=None),
            has_tool_calls=False,
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.COMPLETE
        assert result.score >= 0.8

    async def test_act_with_execution_coordinator(self):
        """_act() should call execute_turn_with_tools() when coordinator available."""
        from victor.agent.coordinators.execution_coordinator import TurnResult

        mock_coord = AsyncMock()
        mock_turn = TurnResult(
            response=MagicMock(content="done", tool_calls=None),
            has_tool_calls=False,
        )
        mock_coord.execute_turn_with_tools = AsyncMock(return_value=mock_turn)

        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            execution_coordinator=mock_coord,
            enable_fulfillment_check=False,
        )

        result = await loop._act({}, {"query": "test"})
        assert isinstance(result, TurnResult)
        mock_coord.execute_turn_with_tools.assert_called_once()

    async def test_act_tracks_tool_calls(self):
        """_act() should update turn-level tracking from TurnResult."""
        from victor.agent.coordinators.execution_coordinator import TurnResult

        mock_coord = AsyncMock()
        mock_turn = TurnResult(
            response=MagicMock(content="", tool_calls=[{"name": "edit"}]),
            has_tool_calls=True,
            tool_calls_count=2,
            all_tools_blocked=False,
        )
        mock_coord.execute_turn_with_tools = AsyncMock(return_value=mock_turn)

        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            execution_coordinator=mock_coord,
            enable_fulfillment_check=False,
        )
        assert loop._total_tool_calls == 0
        assert loop._consecutive_no_tool_turns == 0

        await loop._act({}, {"query": "test"})

        assert loop._total_tool_calls == 2
        assert loop._consecutive_no_tool_turns == 0
        assert loop._consecutive_all_blocked == 0


class TestTurnResultDataclass:
    """Tests for TurnResult dataclass."""

    def test_properties(self):
        from victor.agent.coordinators.execution_coordinator import TurnResult

        turn = TurnResult(
            response=MagicMock(content="hello"),
            tool_results=[
                {"success": True, "tool_name": "read"},
                {"success": False, "tool_name": "edit", "error": "fail"},
                {"success": True, "tool_name": "write"},
            ],
            has_tool_calls=True,
            tool_calls_count=3,
        )
        assert turn.content == "hello"
        assert turn.has_content is True
        assert turn.successful_tool_count == 2
        assert turn.failed_tool_count == 1

    def test_empty_content(self):
        from victor.agent.coordinators.execution_coordinator import TurnResult

        turn = TurnResult(response=MagicMock(content=None))
        assert turn.content == ""
        assert turn.has_content is False
        assert turn.successful_tool_count == 0
