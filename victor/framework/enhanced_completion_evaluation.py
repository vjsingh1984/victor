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
from dataclasses import dataclass, field
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


@dataclass
class CompletionCalibration:
    """Post-processing view of raw completion score grounded in execution support."""

    raw_score: float
    calibrated_score: float
    evidence_score: float
    threshold: float
    requires_additional_support: bool
    support_penalty: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize for logging and benchmark traces."""
        return {
            "raw_score": round(self.raw_score, 4),
            "calibrated_score": round(self.calibrated_score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "threshold": round(self.threshold, 4),
            "support_penalty": round(self.support_penalty, 4),
            "requires_additional_support": self.requires_additional_support,
            "reasons": list(self.reasons),
        }


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
        enable_calibrated_completion: Optional[bool] = None,
    ):
        """Initialize enhanced evaluator.

        Args:
            enable_requirement_validation: Enable requirement-driven validation
            enable_completion_scoring: Enable multi-signal scoring
            enable_context_keywords: Enable context-aware keyword detection
            completion_threshold: Threshold for completion decision (0.0-1.0)
            enable_calibrated_completion: Whether to require execution support
                before accepting strong completion scores. Defaults to feature flag.
        """
        self.enable_requirement_validation = enable_requirement_validation
        self.enable_completion_scoring = enable_completion_scoring
        self.enable_context_keywords = enable_context_keywords
        self.completion_threshold = completion_threshold
        if enable_calibrated_completion is None:
            try:
                from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

                enable_calibrated_completion = get_feature_flag_manager().is_enabled(
                    FeatureFlag.USE_CALIBRATED_COMPLETION
                )
            except Exception:
                enable_calibrated_completion = False
        self.enable_calibrated_completion = enable_calibrated_completion

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

            threshold = completion_score.threshold
            score = completion_score.total_score
            metadata: Dict[str, Any] = {}

            if self.enable_calibrated_completion:
                calibration = self._calibrate_completion(
                    completion_score=completion_score,
                    perception=perception,
                    action_result=action_result,
                    state=state,
                    requirement_result=requirement_result,
                    keyword_result=keyword_result,
                )
                metadata["calibration"] = calibration.to_metadata()
                threshold = calibration.threshold
                score = calibration.calibrated_score

                if calibration.requires_additional_support:
                    return EvaluationResult(
                        decision=EvaluationDecision.CONTINUE,
                        score=score,
                        reason="Completion score is high but needs stronger execution support",
                        metadata=metadata,
                    )

            # Step 5: Make decision based on score
            if score >= threshold:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=score,
                    reason=f"Requirements satisfied: {score:.2f} >= {threshold:.2f}",
                    metadata=metadata,
                )
            elif score >= 0.5:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=score,
                    reason=f"Progress: {score:.2f} (threshold: {threshold:.2f})",
                    metadata=metadata,
                )
            else:
                return EvaluationResult(
                    decision=EvaluationDecision.RETRY,
                    score=score,
                    reason=f"Insufficient progress: {score:.2f}",
                    metadata=metadata,
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

    def _calibrate_completion(
        self,
        completion_score: CompletionScore,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
        requirement_result: Optional[ValidationResult],
        keyword_result: Optional[CompletionSignal],
    ) -> CompletionCalibration:
        """Adjust raw completion score using support and execution signals."""
        evidence_score, reasons = self._estimate_evidence_support(
            perception=perception,
            action_result=action_result,
            state=state,
        )
        support_penalty = 0.0

        if keyword_result is not None and keyword_result.is_continuation_request:
            support_penalty += 0.10
            reasons.append("continuation_requested")

        if (
            requirement_result is not None
            and not requirement_result.is_satisfied
            and evidence_score < 0.75
        ):
            support_penalty += 0.10
            reasons.append("requirements_not_fully_satisfied")

        raw_score = completion_score.total_score
        threshold = completion_score.threshold
        calibrated_score = max(
            0.0,
            min(1.0, (raw_score * 0.75) + (evidence_score * 0.25) - support_penalty),
        )

        requires_additional_support = raw_score >= threshold and calibrated_score < threshold

        return CompletionCalibration(
            raw_score=raw_score,
            calibrated_score=calibrated_score,
            evidence_score=evidence_score,
            threshold=threshold,
            requires_additional_support=requires_additional_support,
            support_penalty=support_penalty,
            reasons=reasons,
        )

    def _estimate_evidence_support(
        self,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """Estimate how well the current answer is supported by execution evidence."""
        reasons: List[str] = []
        task_type = self._map_to_task_type(perception)

        if self._state_has_artifacts(state):
            reasons.append("state_artifacts_present")
            return 0.90, reasons

        if getattr(action_result, "has_tool_calls", False):
            successful = max(0, int(getattr(action_result, "successful_tool_count", 0) or 0))
            total = int(getattr(action_result, "tool_calls_count", 0) or 0)
            if total <= 0 and hasattr(action_result, "tool_calls"):
                total = len(action_result.tool_calls or [])
            if total <= 0 and hasattr(action_result, "tool_results"):
                total = len(action_result.tool_results or [])
            total = max(total, successful, 1)
            reasons.append("tool_backed_execution")
            return min(1.0, 0.65 + 0.25 * (successful / total)), reasons

        if getattr(action_result, "is_qa_response", False) and task_type in {
            TaskType.SEARCH,
            TaskType.ANALYSIS,
            TaskType.DOCUMENTATION,
        }:
            reasons.append("qa_shortcut_allowed")
            return 0.75, reasons

        reasons.append("direct_answer_without_execution_evidence")
        if getattr(perception, "requirements", None):
            reasons.append("requirements_present")
        return 0.30, reasons

    def _state_has_artifacts(self, state: Dict[str, Any]) -> bool:
        """Detect durable evidence that work was executed."""
        if state.get("files_modified"):
            return True
        if state.get("tests_passed") or state.get("tests_total"):
            return True
        if state.get("source_count", 0) > 0:
            return True
        sources = state.get("sources")
        if isinstance(sources, list) and len(sources) > 0:
            return True
        return False
