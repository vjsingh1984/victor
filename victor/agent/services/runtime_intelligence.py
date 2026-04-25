# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical runtime-intelligence service for live agent execution.

This service consolidates the currently scattered runtime collaborators used for:
- turn perception and task understanding
- prompt optimization bundle retrieval
- decision-service turn budget management

It composes the existing strong building blocks instead of replacing them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.optimization_injector import OptimizationInjector
from victor.agent.task_analyzer import TaskAnalyzer, get_task_analyzer
from victor.framework.perception_integration import PerceptionIntegration

if TYPE_CHECKING:
    from victor.agent.services.protocols.decision_service import LLMDecisionServiceProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptOptimizationBundle:
    """Typed prompt-optimization payload for a single turn."""

    evolved_sections: List[str] = field(default_factory=list)
    few_shots: Optional[str] = None
    failure_hint: Optional[str] = None


@dataclass(frozen=True)
class RuntimeIntelligenceSnapshot:
    """Typed runtime view of a turn analysis."""

    query: str
    perception: Optional[Any] = None
    task_analysis: Optional[Any] = None
    decision_service_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClarificationDecision:
    """Typed clarification policy outcome for a perception result."""

    requires_clarification: bool = False
    reason: Optional[str] = None
    prompt: Optional[str] = None
    confidence: float = 0.0


class RuntimeIntelligenceService:
    """Service-first boundary for runtime task intelligence."""

    def __init__(
        self,
        task_analyzer: Optional[TaskAnalyzer] = None,
        perception_integration: Optional[PerceptionIntegration] = None,
        optimization_injector: Optional[OptimizationInjector] = None,
        decision_service: Optional["LLMDecisionServiceProtocol"] = None,
    ) -> None:
        self._task_analyzer = task_analyzer or get_task_analyzer()
        self._perception_integration = perception_integration or PerceptionIntegration()
        self._optimization_injector = optimization_injector
        self._decision_service = decision_service
        if hasattr(self._task_analyzer, "set_runtime_intelligence"):
            try:
                self._task_analyzer.set_runtime_intelligence(self)
            except Exception as exc:
                logger.debug("Runtime intelligence could not bind task analyzer: %s", exc)

    @classmethod
    def from_orchestrator(
        cls,
        orchestrator: Any,
        *,
        task_analyzer: Optional[TaskAnalyzer] = None,
        perception_integration: Optional[PerceptionIntegration] = None,
        optimization_injector: Optional[OptimizationInjector] = None,
    ) -> "RuntimeIntelligenceService":
        """Create a service instance by resolving the decision service from an orchestrator."""
        container = getattr(orchestrator, "_container", None)
        if container is None:
            return cls(
                task_analyzer=task_analyzer,
                perception_integration=perception_integration,
                optimization_injector=optimization_injector,
            )

        return cls.from_container(
            container,
            task_analyzer=task_analyzer,
            perception_integration=perception_integration,
            optimization_injector=optimization_injector,
        )

    @classmethod
    def from_container(
        cls,
        container: Any,
        *,
        task_analyzer: Optional[TaskAnalyzer] = None,
        perception_integration: Optional[PerceptionIntegration] = None,
        optimization_injector: Optional[OptimizationInjector] = None,
    ) -> "RuntimeIntelligenceService":
        """Create a service instance by resolving the decision service from a container."""
        decision_service = cls._resolve_decision_service(container)
        return cls(
            task_analyzer=task_analyzer,
            perception_integration=perception_integration,
            optimization_injector=optimization_injector,
            decision_service=decision_service,
        )

    @staticmethod
    def _resolve_decision_service(container: Any) -> Optional["LLMDecisionServiceProtocol"]:
        """Resolve the decision service from a DI container when available."""
        try:
            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )

            if hasattr(container, "get_optional"):
                return container.get_optional(LLMDecisionServiceProtocol)
            if hasattr(container, "get"):
                return container.get(LLMDecisionServiceProtocol)
        except Exception as exc:
            logger.debug("Runtime intelligence could not resolve decision service: %s", exc)
        return None

    @property
    def perception_integration(self) -> PerceptionIntegration:
        """Expose the underlying perception integration for compatibility."""
        return self._perception_integration

    @property
    def task_analyzer(self) -> TaskAnalyzer:
        """Expose the task analyzer for compatibility."""
        return self._task_analyzer

    def has_decision_service(self) -> bool:
        """Return whether the runtime has an attached decision service."""
        return self._decision_service is not None

    def clear_session_cache(self) -> None:
        """Clear cached prompt-optimization state when the session changes."""
        if self._optimization_injector is not None:
            self._optimization_injector.clear_session_cache()

    def reset_decision_budget(self) -> None:
        """Reset the decision-service per-turn budget when available."""
        if self._decision_service is not None and hasattr(self._decision_service, "reset_budget"):
            self._decision_service.reset_budget()

    def decide_sync(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> Optional[Any]:
        """Delegate a synchronous decision to the underlying decision service."""
        if self._decision_service is None or not hasattr(self._decision_service, "decide_sync"):
            return None
        kwargs = {"heuristic_confidence": heuristic_confidence}
        if heuristic_result is not None:
            kwargs["heuristic_result"] = heuristic_result
        return self._decision_service.decide_sync(
            decision_type,
            context,
            **kwargs,
        )

    async def analyze_turn(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuntimeIntelligenceSnapshot:
        """Analyze a turn and return a typed snapshot."""
        perception = None
        task_analysis = None

        if self._perception_integration is not None:
            perception = await self._perception_integration.perceive(
                query,
                context,
                conversation_history,
            )
            task_analysis = getattr(perception, "task_analysis", None)

        if task_analysis is None and self._task_analyzer is not None:
            analysis_context = dict(context or {})
            if conversation_history:
                analysis_context["history"] = conversation_history
            task_analysis = self._task_analyzer.analyze(query, context=analysis_context)

        return RuntimeIntelligenceSnapshot(
            query=query,
            perception=perception,
            task_analysis=task_analysis,
            decision_service_available=self.has_decision_service(),
            metadata={
                "has_context": bool(context),
                "history_length": len(conversation_history or []),
            },
        )

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task intent using the underlying task analyzer when available."""
        if self._task_analyzer and hasattr(self._task_analyzer, "classify_keywords"):
            try:
                return self._task_analyzer.classify_keywords(user_message)
            except Exception as exc:
                logger.debug("Runtime intelligence keyword classification failed: %s", exc)
        return {"task_type": "default", "confidence": 0.0}

    def classify_task_with_context(
        self,
        user_message: str,
        history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Classify task intent with history context when supported."""
        if self._task_analyzer and hasattr(self._task_analyzer, "classify_with_context"):
            try:
                return self._task_analyzer.classify_with_context(user_message, history or [])
            except Exception as exc:
                logger.debug(
                    "Runtime intelligence contextual task classification failed: %s",
                    exc,
                )
        return {"task_type": "default", "confidence": 0.0}

    @staticmethod
    def get_clarification_decision(
        perception: Optional[Any],
        *,
        default_prompt: str = (
            "Please clarify the target file, component, or bug before I continue."
        ),
    ) -> ClarificationDecision:
        """Normalize clarification policy into one typed runtime decision."""
        if not getattr(perception, "needs_clarification", False):
            return ClarificationDecision(
                requires_clarification=False,
                confidence=float(getattr(perception, "confidence", 0.0) or 0.0),
            )

        reason = getattr(perception, "clarification_reason", None) or "task details are incomplete"
        prompt = getattr(perception, "clarification_prompt", None) or default_prompt
        confidence = float(getattr(perception, "confidence", 0.0) or 0.0)
        return ClarificationDecision(
            requires_clarification=True,
            reason=reason,
            prompt=prompt,
            confidence=confidence,
        )

    @staticmethod
    def evaluate_confidence_progress(
        confidence: float,
        state: Dict[str, Any],
        *,
        retry_limit: int = 2,
        high_confidence_threshold: float = 0.8,
        medium_confidence_threshold: float = 0.5,
    ) -> Any:
        """Apply the canonical confidence-band policy for live-loop evaluation."""
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        if confidence >= high_confidence_threshold:
            if "low_confidence_retries" in state:
                state["low_confidence_retries"] = 0
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=confidence,
                reason="High confidence in perception",
            )

        if confidence >= medium_confidence_threshold:
            if "low_confidence_retries" in state:
                state["low_confidence_retries"] = 0
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=confidence,
                reason="Medium confidence - continue",
            )

        retry_limit = max(int(retry_limit), 0)
        retry_count = int(state.get("low_confidence_retries", 0))
        if retry_count >= retry_limit:
            return EvaluationResult(
                decision=EvaluationDecision.FAIL,
                score=confidence,
                reason=f"Low confidence retry budget exhausted after {retry_count} retries",
                metadata={
                    "low_confidence_retry_exhausted": True,
                    "low_confidence_retries": retry_count,
                    "low_confidence_retry_limit": retry_limit,
                },
            )

        retry_count += 1
        state["low_confidence_retries"] = retry_count
        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=confidence,
            reason="Low confidence - retry",
            metadata={
                "low_confidence_retries": retry_count,
                "low_confidence_retry_limit": retry_limit,
            },
        )

    @staticmethod
    def apply_low_confidence_retry_budget(
        evaluation: Any,
        state: Dict[str, Any],
        *,
        retry_limit: int = 2,
        low_confidence_threshold: float = 0.5,
    ) -> Any:
        """Apply the canonical retry-budget policy to low-confidence retry results."""
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        if getattr(evaluation, "should_complete", False) or getattr(
            evaluation, "should_continue", False
        ):
            if "low_confidence_retries" in state:
                state["low_confidence_retries"] = 0
            return evaluation

        if not getattr(evaluation, "should_retry", False) or evaluation.score >= low_confidence_threshold:
            return evaluation

        retry_limit = max(int(retry_limit), 0)
        retry_count = int(state.get("low_confidence_retries", 0))
        if retry_count >= retry_limit:
            return EvaluationResult(
                decision=EvaluationDecision.FAIL,
                score=evaluation.score,
                reason=f"Low confidence retry budget exhausted after {retry_count} retries",
                metrics=dict(evaluation.metrics),
                metadata={
                    **dict(evaluation.metadata),
                    "low_confidence_retry_exhausted": True,
                    "low_confidence_retries": retry_count,
                    "low_confidence_retry_limit": retry_limit,
                },
            )

        retry_count += 1
        state["low_confidence_retries"] = retry_count
        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=evaluation.score,
            reason=evaluation.reason,
            metrics=dict(evaluation.metrics),
            metadata={
                **dict(evaluation.metadata),
                "low_confidence_retries": retry_count,
                "low_confidence_retry_limit": retry_limit,
            },
        )

    def get_prompt_optimization_bundle(
        self,
        user_message: str,
        turn_context: Any,
    ) -> PromptOptimizationBundle:
        """Collect prompt optimizations for the current turn."""
        if self._optimization_injector is None:
            return PromptOptimizationBundle()

        evolved_sections = self._optimization_injector.get_evolved_sections(
            provider=getattr(turn_context, "provider_name", ""),
            model=getattr(turn_context, "model", ""),
            task_type=getattr(turn_context, "task_type", "default"),
        )
        few_shots = self._optimization_injector.get_few_shots(user_message)
        failure_hint = None
        if getattr(turn_context, "last_turn_failed", False):
            failure_hint = self._optimization_injector.get_failure_hint(
                getattr(turn_context, "last_failure_category", None),
                getattr(turn_context, "last_failure_error", None),
            )

        return PromptOptimizationBundle(
            evolved_sections=list(evolved_sections or []),
            few_shots=few_shots,
            failure_hint=failure_hint,
        )
