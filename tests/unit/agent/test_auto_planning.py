"""Tests for deprecated auto-planning shim delegation."""

from unittest.mock import AsyncMock, MagicMock

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
    from victor.agent.services.sync_chat_compat import SyncChatCoordinator

    mock_chat_ctx = MagicMock()
    mock_chat_ctx.settings = MagicMock()
    mock_chat_ctx.settings.enable_planning = True

    mock_tool_ctx = MagicMock()
    mock_provider_ctx = MagicMock()
    mock_provider_ctx.task_classifier = MagicMock()
    mock_provider_ctx.task_classifier.classify.return_value = MagicMock(
        complexity=TaskComplexity.MEDIUM,
    )

    chat_service = AsyncMock()
    chat_service.chat = AsyncMock(return_value=MagicMock(content="direct response"))

    coordinator = SyncChatCoordinator(
        chat_context=mock_chat_ctx,
        tool_context=mock_tool_ctx,
        provider_context=mock_provider_ctx,
        turn_executor=AsyncMock(),
        orchestrator=orchestrator or MagicMock(),
        query_classifier=query_classifier,
        chat_service=chat_service,
    )
    return coordinator, chat_service


class TestAutoPlanning:
    async def _chat(self, coordinator, user_message: str, use_planning):
        with pytest.warns(
            DeprecationWarning,
            match="SyncChatCoordinator.chat\\(\\) is deprecated compatibility surface",
        ):
            return await coordinator.chat(user_message, use_planning=use_planning)

    @pytest.mark.asyncio
    async def test_auto_none_preserves_auto_planning_request_for_chat_service(self):
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_classifier.classify.return_value = _make_classification(
            QueryType.IMPLEMENTATION, should_plan=True
        )
        coordinator, chat_service = _make_coordinator(query_classifier=mock_classifier)

        await self._chat(coordinator, "Implement JWT auth", use_planning=None)

        chat_service.chat.assert_awaited_once_with("Implement JWT auth", use_planning=None)

    @pytest.mark.asyncio
    async def test_auto_none_simple_query_still_delegates_to_chat_service(self):
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_classifier.classify.return_value = _make_classification(
            QueryType.QUICK_QUESTION,
            should_plan=False,
            complexity=TaskComplexity.SIMPLE,
        )
        coordinator, chat_service = _make_coordinator(query_classifier=mock_classifier)

        await self._chat(coordinator, "What is Python?", use_planning=None)

        chat_service.chat.assert_awaited_once_with("What is Python?", use_planning=None)

    @pytest.mark.asyncio
    async def test_explicit_true_passes_through_to_chat_service(self):
        coordinator, chat_service = _make_coordinator()

        await self._chat(coordinator, "anything", use_planning=True)

        chat_service.chat.assert_awaited_once_with("anything", use_planning=True)

    @pytest.mark.asyncio
    async def test_explicit_false_passes_through_to_chat_service(self):
        coordinator, chat_service = _make_coordinator()
        await self._chat(coordinator, "Implement complex auth system", use_planning=False)
        chat_service.chat.assert_awaited_once_with(
            "Implement complex auth system", use_planning=False
        )

    def test_classifier_injected_via_constructor(self):
        mock_classifier = MagicMock(spec=QueryClassifier)
        coordinator, _ = _make_coordinator(query_classifier=mock_classifier)
        assert coordinator._query_classifier is mock_classifier

    @pytest.mark.asyncio
    async def test_no_classifier_still_delegates_to_chat_service(self):
        coordinator, chat_service = _make_coordinator(query_classifier=None)

        await self._chat(coordinator, "analyze architecture", use_planning=None)

        chat_service.chat.assert_awaited_once_with("analyze architecture", use_planning=None)
