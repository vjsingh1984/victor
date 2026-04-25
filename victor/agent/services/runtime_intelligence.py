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
        decision_service = None
        try:
            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )

            container = getattr(orchestrator, "_container", None)
            if container is not None:
                if hasattr(container, "get_optional"):
                    decision_service = container.get_optional(LLMDecisionServiceProtocol)
                elif hasattr(container, "get"):
                    decision_service = container.get(LLMDecisionServiceProtocol)
        except Exception as exc:
            logger.debug("Runtime intelligence could not resolve decision service: %s", exc)

        return cls(
            task_analyzer=task_analyzer,
            perception_integration=perception_integration,
            optimization_injector=optimization_injector,
            decision_service=decision_service,
        )

    @property
    def perception_integration(self) -> PerceptionIntegration:
        """Expose the underlying perception integration for compatibility."""
        return self._perception_integration

    @property
    def task_analyzer(self) -> TaskAnalyzer:
        """Expose the task analyzer for compatibility."""
        return self._task_analyzer

    def clear_session_cache(self) -> None:
        """Clear cached prompt-optimization state when the session changes."""
        if self._optimization_injector is not None:
            self._optimization_injector.clear_session_cache()

    def reset_decision_budget(self) -> None:
        """Reset the decision-service per-turn budget when available."""
        if self._decision_service is not None and hasattr(self._decision_service, "reset_budget"):
            self._decision_service.reset_budget()

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
            decision_service_available=self._decision_service is not None,
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
