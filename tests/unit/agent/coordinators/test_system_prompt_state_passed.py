"""Tests for SystemPromptStatePassedCoordinator (SPA-3).

Validates:
1. Reads from ContextSnapshot (no orchestrator reference)
2. Returns CoordinatorResult with task classification transitions
3. Extracts conversation history from snapshot messages
4. No side effects during classification
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from victor.agent.coordinators.state_context import (
    CoordinatorResult,
    ContextSnapshot,
    TransitionType,
)
from victor.agent.coordinators.system_prompt_state_passed import (
    SystemPromptStatePassedCoordinator,
)


def _make_snapshot(**overrides: Any) -> ContextSnapshot:
    """Create a minimal ContextSnapshot for testing."""
    defaults = {
        "messages": (),
        "session_id": "test-session",
        "conversation_stage": "initial",
        "settings": MagicMock(),
        "model": "test-model",
        "provider": "test-provider",
        "max_tokens": 4096,
        "temperature": 0.7,
        "conversation_state": {},
        "session_state": {},
        "observed_files": (),
        "capabilities": {},
    }
    defaults.update(overrides)
    return ContextSnapshot(**defaults)


def _make_mock_analyzer(classification: dict = None) -> MagicMock:
    """Create a mock TaskAnalyzer."""
    analyzer = MagicMock()
    if classification is None:
        classification = {
            "task_type": "coding",
            "complexity": "medium",
            "keywords": ["fix", "bug"],
            "confidence": 0.85,
        }
    analyzer.classify_task_with_context.return_value = classification
    return analyzer


class TestSystemPromptStatePassedInit:
    """Test coordinator initialization."""

    def test_creates_with_analyzer(self):
        analyzer = _make_mock_analyzer()
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        assert coord._task_analyzer is analyzer

    def test_no_orchestrator_reference(self):
        coord = SystemPromptStatePassedCoordinator(task_analyzer=MagicMock())
        assert not hasattr(coord, "_orchestrator")
        assert not hasattr(coord, "orchestrator")


class TestSystemPromptClassify:
    """Test the classify() method."""

    @pytest.mark.asyncio
    async def test_returns_coordinator_result(self):
        analyzer = _make_mock_analyzer()
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        snapshot = _make_snapshot()

        result = await coord.classify(snapshot, "fix the login bug")
        assert isinstance(result, CoordinatorResult)

    @pytest.mark.asyncio
    async def test_stores_task_type_in_transitions(self):
        analyzer = _make_mock_analyzer(
            {"task_type": "research", "complexity": "high", "confidence": 0.9}
        )
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        snapshot = _make_snapshot()

        result = await coord.classify(snapshot, "analyze the market trends")
        task_type_transitions = [
            t
            for t in result.transitions.transitions
            if t.transition_type == TransitionType.UPDATE_STATE and t.data.get("key") == "task_type"
        ]
        assert len(task_type_transitions) == 1
        assert task_type_transitions[0].data["value"] == "research"

    @pytest.mark.asyncio
    async def test_stores_complexity_in_transitions(self):
        analyzer = _make_mock_analyzer(
            {"task_type": "coding", "complexity": "high", "confidence": 0.8}
        )
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        snapshot = _make_snapshot()

        result = await coord.classify(snapshot, "refactor the entire auth module")
        complexity_transitions = [
            t
            for t in result.transitions.transitions
            if t.transition_type == TransitionType.UPDATE_STATE
            and t.data.get("key") == "task_complexity"
        ]
        assert len(complexity_transitions) == 1
        assert complexity_transitions[0].data["value"] == "high"

    @pytest.mark.asyncio
    async def test_stores_keywords_in_transitions(self):
        analyzer = _make_mock_analyzer(
            {
                "task_type": "coding",
                "complexity": "low",
                "keywords": ["test", "unit"],
                "confidence": 0.7,
            }
        )
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        snapshot = _make_snapshot()

        result = await coord.classify(snapshot, "write unit tests")
        kw_transitions = [
            t
            for t in result.transitions.transitions
            if t.transition_type == TransitionType.UPDATE_STATE
            and t.data.get("key") == "task_keywords"
        ]
        assert len(kw_transitions) == 1
        assert kw_transitions[0].data["value"] == ["test", "unit"]

    @pytest.mark.asyncio
    async def test_no_op_when_no_classification(self):
        analyzer = _make_mock_analyzer({})
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        snapshot = _make_snapshot()

        result = await coord.classify(snapshot, "hello")
        assert result.transitions.is_empty()

    @pytest.mark.asyncio
    async def test_passes_history_from_snapshot(self):
        """Conversation history from snapshot messages should be passed to analyzer."""
        msg1 = MagicMock(role="user", content="first message")
        msg2 = MagicMock(role="assistant", content="response")
        analyzer = _make_mock_analyzer()
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        snapshot = _make_snapshot(messages=(msg1, msg2))

        await coord.classify(snapshot, "follow up question")
        call_args = analyzer.classify_task_with_context.call_args
        history = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("history")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_confidence_from_classification(self):
        analyzer = _make_mock_analyzer(
            {"task_type": "devops", "complexity": "low", "confidence": 0.95}
        )
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        snapshot = _make_snapshot()

        result = await coord.classify(snapshot, "deploy to production")
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_metadata_contains_full_classification(self):
        classification = {
            "task_type": "coding",
            "complexity": "medium",
            "confidence": 0.8,
            "extra": "data",
        }
        analyzer = _make_mock_analyzer(classification)
        coord = SystemPromptStatePassedCoordinator(task_analyzer=analyzer)
        snapshot = _make_snapshot()

        result = await coord.classify(snapshot, "fix bug")
        assert result.metadata == classification


class TestExtractHistory:
    """Test history extraction from snapshot messages."""

    def test_extracts_role_and_content(self):
        msg = MagicMock(role="user", content="hello")
        snapshot = _make_snapshot(messages=(msg,))
        history = SystemPromptStatePassedCoordinator._extract_history(snapshot)
        assert history == [{"role": "user", "content": "hello"}]

    def test_limits_to_last_10_messages(self):
        msgs = tuple(MagicMock(role="user", content=f"msg{i}") for i in range(20))
        snapshot = _make_snapshot(messages=msgs)
        history = SystemPromptStatePassedCoordinator._extract_history(snapshot)
        assert len(history) == 10

    def test_empty_messages(self):
        snapshot = _make_snapshot(messages=())
        history = SystemPromptStatePassedCoordinator._extract_history(snapshot)
        assert history == []
