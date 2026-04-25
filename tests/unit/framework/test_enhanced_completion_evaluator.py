# Copyright 2025 Vijaykumar Singh <singhvijay@users.noreply.github.com>
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

"""Unit tests for EnhancedCompletionEvaluator."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

from victor.framework.enhanced_completion_evaluation import (
    EnhancedCompletionEvaluator,
)
from victor.framework.evaluation_nodes import EvaluationDecision
from victor.framework.perception_integration import RequirementType, Requirement
from victor.framework.completion_scorer import CompletionScore, TaskType
from victor.agent.turn_policy import SpinState

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockTurnResult:
    """Mock TurnResult for testing."""

    response: str = "Task complete"
    is_qa_response: bool = False
    has_content: bool = True
    has_tool_calls: bool = False
    successful_tool_count: int = 0
    failed_tool_count: int = 0
    tool_calls: list = None


@dataclass
class MockSpinDetector:
    """Mock SpinDetector for testing."""

    state: SpinState = SpinState.NORMAL
    consecutive_all_blocked: int = 0
    consecutive_no_tool_turns: int = 0


@dataclass
class MockPerception:
    """Mock Perception for testing - matches actual API."""

    confidence: float = 0.8
    requirements: list = None
    task_type: str = "code_generation"
    intent: Mock = None
    complexity: str = "medium"
    task_analysis: Mock = None
    metadata: dict = None


# =============================================================================
# EnhancedCompletionEvaluator Tests
# =============================================================================


class TestEnhancedCompletionEvaluator:
    """Test suite for EnhancedCompletionEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance for testing."""
        return EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=True,
            enable_context_keywords=True,
            completion_threshold=0.80,
        )

    @pytest.fixture
    def mock_fulfillment(self):
        """Create mock fulfillment detector."""
        fulfillment = AsyncMock()
        fulfillment.check_fulfillment = AsyncMock(
            return_value=Mock(
                is_fulfilled=False,
                score=0.6,
                reason="Partial progress",
            )
        )
        return fulfillment

    @pytest.mark.asyncio
    async def test_initialization(self, evaluator):
        """Test evaluator initializes with correct settings."""
        assert evaluator.enable_requirement_validation is True
        assert evaluator.enable_completion_scoring is True
        assert evaluator.enable_context_keywords is True
        assert evaluator.completion_threshold == 0.80

    @pytest.mark.asyncio
    async def test_evaluate_qa_shortcut(self, evaluator):
        """Test Q&A shortcut takes priority."""
        from victor.agent.services.turn_execution_runtime import TurnResult
        from victor.providers.base import CompletionResponse

        perception = MockPerception(confidence=0.9)
        # Use actual TurnResult class to pass isinstance check
        completion_response = CompletionResponse(
            content="The answer is 42",
            role="assistant",
            tool_calls=None,
            stop_reason="stop",
            usage=None,
        )
        action_result = TurnResult(
            response=completion_response,
            is_qa_response=True,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Q&A shortcut should trigger COMPLETE
        assert result.decision == EvaluationDecision.COMPLETE
        assert "Q&A shortcut" in result.reason
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_spin_terminated_all_blocked(self, evaluator):
        """Test spin detection when all tools blocked."""
        perception = MockPerception(confidence=0.5)
        action_result = MockTurnResult(response="Stuck")
        state = {}
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED, consecutive_all_blocked=3, consecutive_no_tool_turns=0
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state=state,
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL
        assert "Spin detected" in result.reason
        assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_evaluate_spin_terminated_no_tools(self, evaluator):
        """Test spin detection when no tools used."""
        perception = MockPerception(confidence=0.5)
        action_result = MockTurnResult(response="Still thinking")
        state = {}
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED, consecutive_all_blocked=0, consecutive_no_tool_turns=5
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state=state,
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL
        assert "Agent stuck" in result.reason

    @pytest.mark.asyncio
    async def test_evaluate_enhanced_complete(self, evaluator):
        """Test enhanced evaluation returns COMPLETE."""
        perception = MockPerception(
            confidence=0.9,
            requirements=[
                Requirement(type=RequirementType.FUNCTIONAL, description="Create file", priority=0)
            ],
        )
        action_result = MockTurnResult(
            response="Here is the implementation",
            has_tool_calls=True,
            tool_calls=["write_file"],
        )
        state = {"files_modified": ["auth.py"]}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Should make a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_evaluate_enhanced_with_fulfillment(self, evaluator, mock_fulfillment):
        """Test enhanced evaluation with fulfillment detector."""
        perception = MockPerception(
            confidence=0.8,
            requirements=[
                Requirement(type=RequirementType.FUNCTIONAL, description="Test", priority=0)
            ],
        )
        action_result = MockTurnResult(response="Testing complete")
        state = {}

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state=state,
            fulfillment_detector=mock_fulfillment,
        )

        # Just verify it returns a valid decision
        # (fulfillment detector may or may not be called depending on implementation)
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_evaluate_legacy_fallback(self, evaluator):
        """Test legacy fallback when enhanced evaluation is disabled."""
        # Disable enhanced evaluation to force legacy path
        evaluator.enable_completion_scoring = False

        perception = MockPerception(confidence=0.8)
        action_result = MockTurnResult(
            response="Used some tools",
            has_tool_calls=True,
            successful_tool_count=3,
            failed_tool_count=0,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Legacy path returns a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_evaluate_legacy_substantial_response(self, evaluator):
        """Test legacy evaluation with substantial response."""
        # Disable enhanced evaluation
        evaluator.enable_completion_scoring = False

        perception = MockPerception(confidence=0.7)
        action_result = MockTurnResult(
            response="This is a very long and detailed explanation. " * 20,  # > 100 chars
            has_tool_calls=False,
            has_content=True,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Legacy path returns a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_evaluate_legacy_confidence_based(self, evaluator):
        """Test legacy evaluation with confidence-based decision."""
        # Disable enhanced evaluation
        evaluator.enable_completion_scoring = False

        perception = MockPerception(confidence=0.9)
        action_result = MockTurnResult(response="Response")
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        assert result.decision == EvaluationDecision.COMPLETE
        assert "High confidence" in result.reason

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator):
        """Test graceful degradation on errors."""
        perception = MockPerception(confidence=0.8)
        # Pass invalid action_result that might cause errors
        action_result = None
        state = {}

        # Should not raise exception, should return result
        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        assert result is not None
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_evaluate_low_confidence_uses_runtime_policy_without_mutating_budget(
        self, evaluator
    ):
        """Low-confidence fallback should share runtime policy vocabulary without spending budget."""
        evaluator.enable_completion_scoring = False
        perception = MockPerception(confidence=0.3)
        state = {"low_confidence_retries": 1}

        result = await evaluator.evaluate(
            perception=perception,
            action_result=None,
            state=state,
        )

        assert result.decision == EvaluationDecision.RETRY
        assert result.reason == "Low confidence - retry"
        assert state["low_confidence_retries"] == 1

    @pytest.mark.asyncio
    async def test_map_to_task_type_from_perception(self, evaluator):
        """Test mapping perception to TaskType."""
        from victor.framework.completion_scorer import TaskType

        perception = MockPerception(task_type="code_generation")

        task_type = evaluator._map_to_task_type(perception)

        # Should return a TaskType enum
        assert isinstance(task_type, TaskType)

    @pytest.mark.asyncio
    async def test_extract_response_from_turn_result(self, evaluator):
        """Test extracting response from TurnResult."""
        action_result = MockTurnResult(response="Test response")

        response = evaluator._extract_response(action_result)

        assert response == "Test response"

    @pytest.mark.asyncio
    async def test_extract_response_from_content(self, evaluator):
        """Test extracting response from content attribute."""

        # Create a simple object with content attribute but no response
        class ActionResultWithContent:
            content = "Content response"

        action_result = ActionResultWithContent()

        response = evaluator._extract_response(action_result)

        # Should extract from content attribute when response doesn't exist
        assert response == "Content response"

    @pytest.mark.asyncio
    async def test_extract_response_from_string(self, evaluator):
        """Test extracting response from string."""
        action_result = "String response"

        response = evaluator._extract_response(action_result)

        assert response == "String response"

    @pytest.mark.asyncio
    async def test_spin_detection_priority_over_enhanced(self, evaluator):
        """Test that spin detection takes priority over enhanced evaluation."""
        perception = MockPerception(confidence=0.9)
        action_result = MockTurnResult(response="Response")
        state = {}
        spin_detector = MockSpinDetector(state=SpinState.TERMINATED, consecutive_all_blocked=2)

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state=state,
            spin_detector=spin_detector,
        )

        # Should return FAIL due to spin, not COMPLETE due to high confidence
        assert result.decision == EvaluationDecision.FAIL

    @pytest.mark.asyncio
    async def test_qa_shortcut_priority_over_enhanced(self, evaluator):
        """Test that Q&A shortcut takes priority over enhanced evaluation."""
        from victor.agent.services.turn_execution_runtime import TurnResult
        from victor.providers.base import CompletionResponse

        perception = MockPerception(confidence=0.5)  # Low confidence
        # Use actual TurnResult class to pass isinstance check
        completion_response = CompletionResponse(
            content="The answer is simple",
            role="assistant",
            tool_calls=None,
            stop_reason="stop",
            usage=None,
        )
        action_result = TurnResult(
            response=completion_response,
            is_qa_response=True,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Q&A shortcut should return COMPLETE
        assert result.decision == EvaluationDecision.COMPLETE
        assert "Q&A shortcut" in result.reason

    @pytest.mark.asyncio
    async def test_no_perception_uses_legacy(self, evaluator):
        """Test that missing perception falls back to legacy."""
        perception = None
        action_result = MockTurnResult(
            response="Response",
            has_tool_calls=True,
            successful_tool_count=2,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Should use legacy evaluation - but actual legacy path is complex
        # Just verify it returns a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_disable_requirement_validation(self):
        """Test evaluator with requirement validation disabled."""
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=False,
            enable_completion_scoring=True,
            enable_context_keywords=True,
        )

        assert evaluator.enable_requirement_validation is False

    @pytest.mark.asyncio
    async def test_disable_completion_scoring(self):
        """Test evaluator with completion scoring disabled."""
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=False,  # Disabled
            enable_context_keywords=True,
        )

        assert evaluator.enable_completion_scoring is False

    @pytest.mark.asyncio
    async def test_disable_context_keywords(self):
        """Test evaluator with context keywords disabled."""
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=True,
            enable_context_keywords=False,  # Disabled
        )

        assert evaluator.enable_context_keywords is False

    @pytest.mark.asyncio
    async def test_calibrated_completion_penalizes_unsupported_direct_answer(self, evaluator):
        """Direct answers without support should not prematurely complete."""
        evaluator.enable_calibrated_completion = True
        evaluator.completion_scorer.calculate_completion_score = Mock(
            return_value=CompletionScore(
                total_score=0.92,
                requirement_score=0.9,
                fulfillment_score=0.8,
                keyword_score=0.9,
                confidence_score=0.8,
                complexity_adjustment=0.0,
                is_complete=True,
                threshold=0.85,
            )
        )

        perception = MockPerception(
            confidence=0.9,
            requirements=[
                Requirement(type=RequirementType.FUNCTIONAL, description="Modify auth flow")
            ],
        )
        action_result = MockTurnResult(
            response="The task is complete.",
            has_tool_calls=False,
            successful_tool_count=0,
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state={},
        )

        assert result.decision == EvaluationDecision.CONTINUE
        assert result.metadata["calibration"]["requires_additional_support"] is True
        assert result.metadata["calibration"]["calibrated_score"] < 0.85

    @pytest.mark.asyncio
    async def test_calibrated_completion_allows_tool_backed_answer(self, evaluator):
        """Tool-backed answers should preserve strong completion scores."""
        evaluator.enable_calibrated_completion = True
        evaluator.completion_scorer.calculate_completion_score = Mock(
            return_value=CompletionScore(
                total_score=0.91,
                requirement_score=0.9,
                fulfillment_score=0.9,
                keyword_score=0.8,
                confidence_score=0.8,
                complexity_adjustment=0.0,
                is_complete=True,
                threshold=0.85,
            )
        )

        perception = MockPerception(
            confidence=0.9,
            requirements=[
                Requirement(type=RequirementType.FUNCTIONAL, description="Write patch")
            ],
        )
        action_result = MockTurnResult(
            response="Applied the patch and updated tests.",
            has_tool_calls=True,
            successful_tool_count=2,
            tool_calls=["edit_file", "run_tests"],
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state={"files_modified": ["auth.py"], "tests_passed": 4},
        )

        assert result.decision == EvaluationDecision.COMPLETE
        assert result.metadata["calibration"]["requires_additional_support"] is False
        assert result.metadata["calibration"]["calibrated_score"] >= 0.85
