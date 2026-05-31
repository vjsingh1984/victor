"""Tests for ContextServiceAdapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.adapters.context_adapter import ContextServiceAdapter


@pytest.fixture
def mock_conversation_controller():
    controller = MagicMock()
    controller.get_context_metrics.return_value = MagicMock(
        total_tokens=5000,
        estimated_tokens=50,
        message_count=10,
    )
    controller.config = SimpleNamespace(max_context_chars=400, chars_per_token_estimate=4)
    controller.messages = [
        MagicMock(role="user", content="x" * 100),
        MagicMock(role="assistant", content="y" * 100),
    ]
    controller.add_message = MagicMock()
    return controller


@pytest.fixture
def mock_context_compactor():
    compactor = MagicMock()
    compactor.check_and_compact = AsyncMock(
        return_value=SimpleNamespace(messages_removed=3, tokens_freed=24)
    )
    compactor.get_statistics.return_value = {
        "compaction_count": 2,
        "total_tokens_freed": 36,
    }
    return compactor


@pytest.fixture
def context_adapter(mock_conversation_controller, mock_context_compactor):
    return ContextServiceAdapter(mock_conversation_controller, mock_context_compactor)


def test_get_context_metrics(context_adapter, mock_conversation_controller):
    metrics = context_adapter.get_context_metrics()
    mock_conversation_controller.get_context_metrics.assert_called_once()
    assert metrics.total_tokens == 5000


async def test_compact_context(context_adapter, mock_context_compactor):
    removed = await context_adapter.compact_context()
    mock_context_compactor.check_and_compact.assert_awaited_once_with(force=True)
    assert removed == 3


async def test_compact_context_no_compactor(mock_conversation_controller):
    adapter = ContextServiceAdapter(mock_conversation_controller, None)
    removed = await adapter.compact_context()
    assert removed == 0


def test_add_message(context_adapter, mock_conversation_controller):
    context_adapter.add_message("user", "test message")
    mock_conversation_controller.add_message.assert_called_once_with("user", "test message")


def test_get_messages(context_adapter):
    messages = context_adapter.get_messages()
    assert len(messages) == 2


def test_get_messages_with_limit(context_adapter):
    messages = context_adapter.get_messages(limit=1)
    assert len(messages) == 1


def test_get_messages_with_role_filter(context_adapter):
    messages = context_adapter.get_messages(role="user")
    assert len(messages) == 1
    assert messages[0].role == "user"


@pytest.mark.asyncio
async def test_prepare_for_tool_output_injection_uses_adapter_policy_and_compactor(
    context_adapter,
):
    result = await context_adapter.prepare_for_tool_output_injection(
        80,
        provider_name="deepseek",
        model_name="deepseek-chat",
        task_type="analysis",
    )

    assert result["should_compact"] is True
    assert result["compacted"] is True
    assert result["messages_removed"] == 3
    assert result["saved_tokens"] == 24
    assert result["strategy"] == "hybrid"
    assert result["reason"] == "pre_tool_output"


def test_get_performance_metrics_exposes_compactor_statistics(context_adapter):
    metrics = context_adapter.get_performance_metrics()

    assert metrics["last_compaction_saved_tokens"] == 0
    assert metrics["total_tokens_saved"] == 36
    assert metrics["operation_count"] == 2


def test_is_healthy(context_adapter):
    assert context_adapter.is_healthy() is True


def test_is_healthy_with_none():
    adapter = ContextServiceAdapter(None)
    assert adapter.is_healthy() is False
