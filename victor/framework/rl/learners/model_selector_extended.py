"""Extended ModelSelectorLearner with HybridDecisionService integration.

Priority 1 Feature Integration:
- Uses deterministic decision rules for fast decisions
- Falls back to LLM for complex cases
- Learns from decision outcomes
"""

from typing import Any, Dict, List, Optional

from victor.agent.services.hybrid_decision_service import HybridDecisionService
from victor.framework.rl.base import RLOutcome, RLRecommendation
from victor.framework.rl.learners.model_selector import ModelSelectorLearner


class ExtendedModelSelectorLearner(ModelSelectorLearner):
    """Extend ModelSelectorLearner with HybridDecisionService integration.

    Integrates Priority 1's HybridDecisionService with the existing
    ModelSelectorLearner to provide fast, cheap model selection decisions
    while maintaining LLM fallback quality for complex cases.

    Features:
        - Deterministic fast-path for common decisions
        - Confidence-based fallback to LLM
        - Learning from decision outcomes
        - Adaptive threshold tuning
    """

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize extended learner with hybrid decision service.

        Args:
            name: Learner name
            db_connection: Database connection
            learning_rate: Learning rate for Q-learning
            provider_adapter: Optional provider adapter
            **kwargs: Additional parameters passed to base class
        """
        # Initialize base class
        super().__init__(
            name=name,
            db_connection=db_connection,
            learning_rate=learning_rate,
            provider_adapter=provider_adapter,
            **kwargs,
        )

        # Integrate HybridDecisionService from Priority 1
        self.decision_service = HybridDecisionService()

    @staticmethod
    def _make_recommendation(
        *,
        key: str,
        value: str,
        confidence: float,
        reason: str,
        sample_size: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RLRecommendation:
        payload = dict(metadata or {})
        payload.update(
            {
                "learner_name": "model_selector",
                "recommendation_type": "decision_threshold",
                "key": key,
            }
        )
        return RLRecommendation(
            value=value,
            confidence=confidence,
            reason=reason,
            sample_size=sample_size,
            metadata=payload,
        )

    def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn from model selection outcomes using hybrid decision service.

        Tracks:
        - Fast-path vs LLM usage patterns
        - Decision latency
        - Optimal confidence thresholds

        Args:
            outcomes: List of model selection outcomes

        Returns:
            List of recommendations for threshold tuning
        """
        recommendations = []

        for outcome in outcomes:
            # Get decision metadata
            used_llm = outcome.metadata.get("used_llm", False)
            decision_latency_ms = outcome.metadata.get("decision_latency_ms", 0)
            confidence = outcome.metadata.get("confidence", 0.0)
            success = outcome.success

            # Learn from fast-path decisions
            if not used_llm and success and decision_latency_ms < 100:
                # Fast path worked well - reinforce
                recommendations.append(
                    self._make_recommendation(
                        key="fast_path_confidence",
                        value="increase",
                        confidence=0.8,
                        reason="Fast-path decisions succeeded with low latency",
                        sample_size=1,
                        metadata={
                            "decision_latency_ms": decision_latency_ms,
                            "used_llm": used_llm,
                            "confidence": confidence,
                        },
                    )
                )

            # Learn from LLM fallbacks
            elif used_llm and success:
                # LLM path was necessary - adjust threshold
                recommendations.append(
                    self._make_recommendation(
                        key="llm_fallback_threshold",
                        value="decrease",
                        confidence=0.7,
                        reason="LLM fallback was required for successful completion",
                        sample_size=1,
                        metadata={
                            "reason": "llm_was_required",
                            "confidence": confidence,
                        },
                    )
                )

        return recommendations

    def select_model(self, task_type: str, context: Dict[str, Any]) -> str:
        """Select model using hybrid decision service.

        Uses deterministic rules when confident, falls back to LLM
        decision for complex cases.

        Args:
            task_type: Type of task (e.g., "tool_call", "chat")
            context: Additional context for decision

        Returns:
            Selected model name
        """
        from victor.agent.decisions.schemas import DecisionType

        if hasattr(DecisionType, "MODEL_SELECTION"):
            decision = self.decision_service.decide_sync(
                decision_type=DecisionType.MODEL_SELECTION,
                context={"task_type": task_type, **context},
            )
            model_name = getattr(getattr(decision, "result", None), "model_name", None)
            if isinstance(model_name, str) and model_name:
                return model_name

        for key in ("preferred_model", "model", "default_model"):
            value = context.get(key)
            if isinstance(value, str) and value:
                return value

        complexity = str(context.get("complexity", "")).lower()
        if complexity in {"low", "simple"}:
            return "fast"
        if complexity in {"high", "complex"}:
            return "balanced"
        return "balanced"

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics from hybrid service.

        Returns:
            Statistics about fast-path vs LLM usage
        """
        return self.decision_service.get_statistics()
