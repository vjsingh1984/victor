"""State-passed coordination recommendation wrapper.

Wraps the service-owned coordination runtime with the state-passed pattern:
- Input: ContextSnapshot + task metadata
- Output: CoordinatorResult with coordination recommendation transitions
- No direct orchestrator mutation
"""

from __future__ import annotations

from typing import Any, Optional

from victor.agent.coordinators.state_context import (
    ContextSnapshot,
    CoordinatorResult,
    TransitionBatch,
)


class CoordinationStatePassedCoordinator:
    """State-passed wrapper for shared coordination recommendations."""

    def __init__(
        self,
        *,
        coordination_runtime: Any,
        coordination_advisor: Optional[Any] = None,
        vertical_context: Optional[Any] = None,
    ) -> None:
        self._coordination_runtime = coordination_runtime
        self._coordination_advisor = coordination_advisor
        self._vertical_context = vertical_context

    async def suggest(
        self,
        context: ContextSnapshot,
        *,
        task_type: str,
        complexity: Optional[str] = None,
        mode: str = "build",
    ) -> CoordinatorResult:
        """Return coordination suggestions as explicit state transitions."""
        resolved_complexity = complexity or self._resolve_complexity(context)
        suggestion = self._coordination_runtime.suggest_for_task(
            task_type=task_type,
            complexity=resolved_complexity,
            mode=mode,
            coordination_advisor=self._coordination_advisor,
            vertical_context=self._vertical_context,
        )
        payload = self._coordination_runtime.serialize_suggestion(suggestion)

        batch = TransitionBatch()
        batch.update_state("coordination_suggestion", payload, scope="conversation")

        if suggestion.primary_team is not None:
            batch.update_state(
                "coordination_primary_team",
                suggestion.primary_team.team_name,
                scope="conversation",
            )
        if suggestion.primary_workflow is not None:
            batch.update_state(
                "coordination_primary_workflow",
                suggestion.primary_workflow.workflow_name,
                scope="conversation",
            )

        confidence = 0.0
        if suggestion.primary_team is not None:
            confidence = float(getattr(suggestion.primary_team, "confidence", 0.0))
        elif suggestion.primary_workflow is not None:
            confidence = float(getattr(suggestion.primary_workflow, "confidence", 0.0))
        elif suggestion.team_recommendations or suggestion.workflow_recommendations:
            confidence = 0.5

        return CoordinatorResult(
            transitions=batch,
            reasoning=(
                f"Built coordination suggestion for {task_type} "
                f"(complexity={resolved_complexity}, mode={mode})"
            ),
            confidence=confidence,
            metadata=payload,
        )

    @staticmethod
    def _resolve_complexity(context: ContextSnapshot) -> str:
        """Resolve task complexity from state-passed context."""
        for key in ("task_complexity", "complexity"):
            value = context.get_state(key)
            if value is None:
                value = context.get_capability_value(key)
            if value is None:
                continue
            resolved = getattr(value, "value", value)
            if isinstance(resolved, str) and resolved:
                return resolved
        return "medium"
