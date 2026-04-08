"""Tests for auto-planning integration — Layer 3 of agentic execution quality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.query_classifier import (
    QueryClassification,
    QueryClassifier,
    QueryType,
)
from victor.framework.task.protocols import TaskComplexity


def _make_classification(
    query_type: QueryType = QueryType.IMPLEMENTATION,
    should_plan: bool = True,
    complexity: TaskComplexity = TaskComplexity.COMPLEX,
) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        complexity=complexity,
        should_plan=should_plan,
        should_use_subagents=False,
        continuation_budget_hint=6,
        confidence=0.9,
    )


def _make_coordinator(query_classifier=None, orchestrator=None):
    from victor.agent.coordinators.sync_chat_coordinator import SyncChatCoordinator

    mock_chat_ctx = MagicMock()
    mock_chat_ctx.settings = MagicMock()
    mock_chat_ctx.settings.enable_planning = True

    mock_tool_ctx = MagicMock()
    mock_provider_ctx = MagicMock()
    mock_provider_ctx.task_classifier = MagicMock()
    mock_provider_ctx.task_classifier.classify.return_value = MagicMock(
        complexity=TaskComplexity.MEDIUM,
    )

    mock_exec = AsyncMock()
    mock_exec.execute_agentic_loop = AsyncMock(
        return_value=MagicMock(content="direct response")
    )

    coordinator = SyncChatCoordinator(
        chat_context=mock_chat_ctx,
        tool_context=mock_tool_ctx,
        provider_context=mock_provider_ctx,
        execution_coordinator=mock_exec,
        orchestrator=orchestrator or MagicMock(),
        query_classifier=query_classifier,
    )
    return coordinator, mock_exec


class TestAutoPlanning:
    @pytest.mark.asyncio
    async def test_auto_none_complex_query_activates_planning(self):
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_classifier.classify.return_value = _make_classification(
            QueryType.IMPLEMENTATION, should_plan=True
        )
        coordinator, mock_exec = _make_coordinator(query_classifier=mock_classifier)

        with patch.object(
            coordinator, "_chat_with_planning", new_callable=AsyncMock
        ) as mock_plan:
            mock_plan.return_value = MagicMock(content="planned response")
            result = await coordinator.chat("Implement JWT auth", use_planning=None)
            mock_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_none_simple_query_no_planning(self):
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_classifier.classify.return_value = _make_classification(
            QueryType.QUICK_QUESTION,
            should_plan=False,
            complexity=TaskComplexity.SIMPLE,
        )
        coordinator, mock_exec = _make_coordinator(query_classifier=mock_classifier)
        result = await coordinator.chat("What is Python?", use_planning=None)
        mock_exec.execute_agentic_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_true_always_plans(self):
        coordinator, _ = _make_coordinator()
        with patch.object(coordinator, "_should_use_planning", return_value=True):
            with patch.object(
                coordinator, "_chat_with_planning", new_callable=AsyncMock
            ) as mock_plan:
                mock_plan.return_value = MagicMock(content="planned")
                await coordinator.chat("anything", use_planning=True)
                mock_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_false_never_plans(self):
        coordinator, mock_exec = _make_coordinator()
        await coordinator.chat("Implement complex auth system", use_planning=False)
        mock_exec.execute_agentic_loop.assert_called_once()

    def test_classifier_injected_via_constructor(self):
        mock_classifier = MagicMock(spec=QueryClassifier)
        coordinator, _ = _make_coordinator(query_classifier=mock_classifier)
        assert coordinator._query_classifier is mock_classifier

    @pytest.mark.asyncio
    async def test_fallback_to_keyword_heuristic(self):
        # No classifier → existing keyword-based _should_use_planning behavior
        coordinator, _ = _make_coordinator(query_classifier=None)
        # Without classifier, use_planning=None falls back to keyword heuristic
        with patch.object(coordinator, "_should_use_planning", return_value=False):
            with patch.object(
                coordinator, "_chat_with_planning", new_callable=AsyncMock
            ) as mock_plan:
                await coordinator.chat("analyze architecture", use_planning=None)
                mock_plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_classification_passed_to_planning_coordinator(self):
        mock_classifier = MagicMock(spec=QueryClassifier)
        classification = _make_classification(
            QueryType.IMPLEMENTATION, should_plan=True
        )
        mock_classifier.classify.return_value = classification
        coordinator, _ = _make_coordinator(query_classifier=mock_classifier)

        with patch.object(
            coordinator, "_chat_with_planning", new_callable=AsyncMock
        ) as mock_plan:
            mock_plan.return_value = MagicMock(content="planned")
            await coordinator.chat("Implement feature X", use_planning=None)
            # Verify the planning method was called
            mock_plan.assert_called_once()
