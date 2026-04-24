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

"""Enhanced Completion Evaluation - Requirement-driven completion detection.

This module provides enhanced completion evaluation for the AgenticLoop by
integrating:
1. RequirementValidator - validates against extracted requirements
2. CompletionScorer - combines multiple signals into unified score
3. ContextAwareKeywordDetector - task-type-specific keyword detection

This replaces and enhances the existing _evaluate() method in AgenticLoop
with requirement-driven, multi-signal completion detection.

Design Principles:
1. Requirement-driven: Completion based on what was asked
2. Multi-signal fusion: Combines diverse signals for robustness
3. Backward compatible: Falls back to legacy behavior when components unavailable
4. Explainable: Clear logging of why completion was triggered

Based on research from:
- arXiv:2603.07379 - Agentic RAG Taxonomy (requirement extraction)
- arXiv:2604.07415 - SubSearch intermediate reward design
- arXiv:2601.21268 - Meta-evaluation without ground truth

Example:
    from victor.framework.enhanced_completion_evaluation import (
        EnhancedCompletionEvaluator,
        EvaluationDecision,
        EvaluationResult,
    )

    evaluator = EnhancedCompletionEvaluator()

    result = await evaluator.evaluate(
        perception=perception,
        action_result=turn_result,
        state=state,
        fulfillment_detector=fulfillment,
    )

    if result.decision == EvaluationDecision.COMPLETE:
        print("Task complete!")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.completion_scorer import (
    CompletionScorer,
    CompletionScore,
    CompletionSignal,
    TaskType,
)
from victor.framework.context_aware_keyword_detector import (
    ContextAwareKeywordDetector,
)
from victor.framework.evaluation_nodes import EvaluationDecision
from victor.framework.requirement_validator import (
    RequirementValidator,
    ValidationResult,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.framework.agentic_loop import SpinState
    from victor.framework.fulfillment import FulfillmentDetector
    from victor.framework.perception_integration import Perception


class EnhancedCompletionEvaluator:
    """Enhanced completion evaluator with requirement-driven detection.

    This evaluator provides algorithmic completion detection:
    - Uses PerceptionIntegration to understand task requirements
    - Validates completion against extracted requirements
    - Combines multiple signals (fulfillment, keywords, confidence)
    - Provides unified, explainable completion decisions

    Usage:
        evaluator = EnhancedCompletionEvaluator()

        result = await evaluator.evaluate(
            perception=perception,
            action_result=turn_result,
            state=state,
            fulfillment_detector=fulfillment,
            spin_detector=spin_detector,
        )
    """

    def __init__(
        self,
        enable_requirement_validation: bool = True,
        enable_completion_scoring: bool = True,
        enable_context_keywords: bool = True,
        completion_threshold: float = 0.80,
    ):
        """Initialize enhanced evaluator.

        Args:
            enable_requirement_validation: Enable requirement-driven validation
            enable_completion_scoring: Enable multi-signal scoring
            enable_context_keywords: Enable context-aware keyword detection
            completion_threshold: Threshold for completion decision (0.0-1.0)
        """
        self.enable_requirement_validation = enable_requirement_validation
        self.enable_completion_scoring = enable_completion_scoring
        self.enable_context_keywords = enable_context_keywords
        self.completion_threshold = completion_threshold

        # Initialize components
        self.requirement_validator = RequirementValidator()
        self.completion_scorer = CompletionScorer()
        self.keyword_detector = ContextAwareKeywordDetector()

    async def evaluate(
        self,
        perception: Optional[Perception],
        action_result: Any,
        state: Dict[str, Any],
        fulfillment_detector: Optional[Any] = None,
        spin_detector: Optional[Any] = None,
    ) -> Any:
        """Evaluate completion with enhanced detection.

        Args:
            perception: Perception from PerceptionIntegration
            action_result: TurnResult with response/tool results
            state: Current state dictionary
            fulfillment_detector: Optional FulfillmentDetector instance
            spin_detector: Optional SpinDetector instance

        Returns:
            EvaluationResult with decision and rationale
        """
        from victor.framework.evaluation_nodes import EvaluationResult
        from victor.agent.services.turn_execution_runtime import TurnResult

        # === PRIORITY 1: Spin Detection (always check) ===
        if spin_detector is not None:
            spin_result = self._check_spin_detection(spin_detector)
            if spin_result is not None:
                return spin_result

        # === PRIORITY 2: Q&A Shortcut (fast path for questions) ===
        if isinstance(action_result, TurnResult):
            qa_result = self._check_qa_shortcut(action_result)
            if qa_result is not None:
                return qa_result

        # === PRIORITY 3: Enhanced Completion Detection ===
        if self.enable_completion_scoring and perception is not None:
            enhanced_result = await self._evaluate_enhanced(
                perception=perception,
                action_result=action_result,
                state=state,
                fulfillment_detector=fulfillment_detector,
            )
            if enhanced_result is not None:
                return enhanced_result

        # === PRIORITY 4: Legacy Fallback ===
        return self._evaluate_legacy(
            perception=perception,
            action_result=action_result,
            state=state,
        )

    async def _evaluate_enhanced(
        self,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
        fulfillment_detector: Optional[Any],
    ) -> Optional[Any]:
        """Enhanced evaluation with requirement-driven detection.

        Returns None if enhanced evaluation cannot be applied,
        allowing fallback to legacy logic.
        """
        from victor.framework.evaluation_nodes import EvaluationResult

        try:
            # Step 1: Validate against requirements
            requirement_result = None
            if self.enable_requirement_validation:
                requirements = getattr(perception, "requirements", None)
                if requirements:
                    requirement_result = self.requirement_validator.validate_completion(
                        requirements=requirements,
                        action_result=action_result,
                        context=state,
                    )

                    # Fast fail: Critical requirements not met
                    if (
                        requirement_result is not None
                        and not requirement_result.is_satisfied
                        and len(requirement_result.critical_gaps) > 0
                    ):
                        return EvaluationResult(
                            decision=EvaluationDecision.CONTINUE,
                            score=requirement_result.satisfaction_score,
                            reason=f"Critical requirements not met: {requirement_result.summary}",
                        )

            # Step 2: Check fulfillment (task-specific validation)
            fulfillment_result = None
            if fulfillment_detector is not None and hasattr(perception, "task_type"):
                try:
                    task_type = self._map_to_task_type(perception)
                    fulfillment_result = await fulfillment_detector.check_fulfillment(
                        task_type=task_type,
                        criteria=state.get("criteria", {}),
                        context=state,
                    )

                    # Fast success: Task fulfilled
                    if (
                        fulfillment_result is not None
                        and hasattr(fulfillment_result, "is_fulfilled")
                        and fulfillment_result.is_fulfilled
                    ):
                        return EvaluationResult(
                            decision=EvaluationDecision.COMPLETE,
                            score=fulfillment_result.score,
                            reason=f"Task fulfilled: {fulfillment_result.reason}",
                        )
                except Exception as e:
                    logger.warning(f"Fulfillment check failed: {e}")

            # Step 3: Detect completion signals (context-aware keywords)
            keyword_result = None
            if self.enable_context_keywords:
                response = self._extract_response(action_result)
                if response:
                    task_type = self._map_to_task_type(perception)
                    requirements = getattr(perception, "requirements", None)

                    keyword_result = self.keyword_detector.detect_completion(
                        response=response,
                        task_type=task_type,
                        requirements=requirements,
                    )

                    # Fast fail: Model requesting continuation
                    if keyword_result.is_continuation_request:
                        return EvaluationResult(
                            decision=EvaluationDecision.CONTINUE,
                            score=0.6,
                            reason="Model offered continuation - awaiting user direction",
                        )

            # Step 4: Calculate unified completion score
            completion_score = self.completion_scorer.calculate_completion_score(
                requirement_result=requirement_result,
                fulfillment_result=fulfillment_result,
                keyword_result=keyword_result,
                perception=perception,
                task_type=self._map_to_task_type(perception),
            )

            # Step 5: Make decision based on score
            if completion_score.is_complete:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=completion_score.total_score,
                    reason=f"Requirements satisfied: {completion_score.total_score:.2f} >= {completion_score.threshold:.2f}",
                )
            elif completion_score.total_score >= 0.5:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=completion_score.total_score,
                    reason=f"Progress: {completion_score.total_score:.2f} (threshold: {completion_score.threshold:.2f})",
                )
            else:
                return EvaluationResult(
                    decision=EvaluationDecision.RETRY,
                    score=completion_score.total_score,
                    reason=f"Insufficient progress: {completion_score.total_score:.2f}",
                )

        except Exception as e:
            logger.warning(f"Enhanced evaluation failed: {e}, falling back to legacy")
            return None

    def _check_spin_detection(self, spin_detector: Any) -> Optional[Any]:
        """Check for spin detection (agent stuck).

        Returns EvaluationResult if spin detected, None otherwise.
        """
        from victor.framework.evaluation_nodes import EvaluationResult
        from victor.framework.agentic_loop import SpinState

        if not hasattr(spin_detector, "state"):
            return None

        spin_state = spin_detector.state

        if spin_state == SpinState.TERMINATED:
            if hasattr(spin_detector, "consecutive_all_blocked"):
                if spin_detector.consecutive_all_blocked > 0:
                    return EvaluationResult(
                        decision=EvaluationDecision.FAIL,
                        score=0.1,
                        reason=(
                            f"Spin detected: {spin_detector.consecutive_all_blocked} "
                            "consecutive fully-blocked tool batches"
                        ),
                    )

            if hasattr(spin_detector, "consecutive_no_tool_turns"):
                return EvaluationResult(
                    decision=EvaluationDecision.FAIL,
                    score=0.2,
                    reason=(
                        f"Agent stuck: {spin_detector.consecutive_no_tool_turns} "
                        "turns without tool calls"
                    ),
                )

        return None

    def _check_qa_shortcut(self, action_result: Any) -> Optional[Any]:
        """Check for Q&A shortcut (model answered without tools).

        Returns EvaluationResult if Q&A shortcut applies, None otherwise.
        """
        from victor.framework.evaluation_nodes import EvaluationResult
        from victor.agent.services.turn_execution_runtime import TurnResult

        if not isinstance(action_result, TurnResult):
            return None

        turn: TurnResult = action_result

        # Q&A shortcut: model answered without tools on a question task
        if (
            hasattr(turn, "is_qa_response")
            and turn.is_qa_response
            and hasattr(turn, "has_content")
            and turn.has_content
        ):
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=0.9,
                reason="Q&A shortcut: accepted direct answer",
            )

        return None

    def _evaluate_legacy(
        self,
        perception: Optional[Perception],
        action_result: Any,
        state: Dict[str, Any],
    ) -> Any:
        """Legacy evaluation fallback.

        This preserves the original AgenticLoop._evaluate() behavior
        when enhanced components are not available.
        """
        from victor.framework.evaluation_nodes import EvaluationResult
        from victor.agent.services.turn_execution_runtime import TurnResult

        # Legacy: Tool execution tracking
        if isinstance(action_result, TurnResult):
            turn: TurnResult = action_result

            # Model used tools = normal progress
            if (
                hasattr(turn, "has_tool_calls")
                and turn.has_tool_calls
                and hasattr(turn, "successful_tool_count")
            ):
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.5 + min(turn.successful_tool_count * 0.1, 0.3),
                    reason=f"Tools executed: {turn.successful_tool_count} ok, "
                    f"{turn.failed_tool_count} failed",
                )

            # Detect final answers
            if not turn.has_tool_calls and turn.has_content and len(turn.response.strip()) > 100:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=0.8,
                    reason=f"Model provided substantial response ({len(turn.response)} chars)",
                )

            # No tools, no content = needs retry
            if not turn.has_tool_calls:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.3,
                    reason="No tools used yet — giving another chance",
                )

        # Legacy: Confidence-based fallback
        if perception is not None and hasattr(perception, "confidence"):
            if perception.confidence >= 0.8:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=perception.confidence,
                    reason="High confidence in perception",
                )
            elif perception.confidence >= 0.5:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=perception.confidence,
                    reason="Medium confidence - continue",
                )
            else:
                return EvaluationResult(
                    decision=EvaluationDecision.RETRY,
                    score=perception.confidence,
                    reason="Low confidence - retry",
                )

        # Default: continue with low confidence
        return EvaluationResult(
            decision=EvaluationDecision.CONTINUE,
            score=0.4,
            reason="No clear completion signal - continuing",
        )

    def _map_to_task_type(self, perception: Perception) -> TaskType:
        """Map perception to TaskType for completion detection."""
        # Check if perception has task_type attribute
        if hasattr(perception, "task_type"):
            perception_task_type = str(perception.task_type).lower()

            # Map to TaskType enum
            task_type_map = {
                "code_generation": TaskType.CODE_GENERATION,
                "testing": TaskType.TESTING,
                "debugging": TaskType.DEBUGGING,
                "search": TaskType.SEARCH,
                "analysis": TaskType.ANALYSIS,
                "setup": TaskType.SETUP,
                "documentation": TaskType.DOCUMENTATION,
                "deployment": TaskType.DEPLOYMENT,
            }

            for key, task_type in task_type_map.items():
                if key in perception_task_type:
                    return task_type

        # Map from ActionIntent
        if hasattr(perception, "intent"):
            from victor.agent.action_authorizer import ActionIntent

            intent_to_type = {
                ActionIntent.TOOL_USE: TaskType.CODE_GENERATION,
                ActionIntent.MODIFICATION: TaskType.CODE_GENERATION,
                ActionIntent.DISPLAY_ONLY: TaskType.SEARCH,
                ActionIntent.READ_ONLY: TaskType.ANALYSIS,
            }

            intent = perception.intent
            if intent in intent_to_type:
                return intent_to_type[intent]

        # Default: unknown
        return TaskType.UNKNOWN

    def _extract_response(self, action_result: Any) -> Optional[str]:
        """Extract response text from action_result."""
        if hasattr(action_result, "response"):
            return action_result.response
        elif hasattr(action_result, "content"):
            return action_result.content
        elif isinstance(action_result, str):
            return action_result
        return None
