"""Tests for ContextService compatibility with dict-based message history."""

import pytest

from victor.agent.services.context_service import ContextService, ContextServiceConfig


@pytest.mark.asyncio
async def test_get_context_metrics_counts_dict_messages_added_via_keyword_fields():
    service = ContextService(ContextServiceConfig(max_tokens=1000))

    service.add_message(role="system", content="system prompt")
    service.add_message(role="user", content="hello")
    service.add_message({"role": "assistant", "content": "done"})

    metrics = await service.get_context_metrics()

    assert metrics.message_count == 3
    assert metrics.user_message_count == 1
    assert metrics.assistant_message_count == 1
    assert metrics.system_prompt_tokens == service.estimate_tokens("system prompt")


def test_get_messages_filters_dict_messages_by_role():
    service = ContextService(ContextServiceConfig())

    service.add_messages(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "done"},
        ]
    )

    assert service.get_messages(role="assistant") == [{"role": "assistant", "content": "done"}]


def test_clear_messages_retains_system_dict_messages():
    service = ContextService(ContextServiceConfig())

    service.add_messages(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ]
    )

    service.clear_messages(retain_system=True)

    assert service.get_messages() == [{"role": "system", "content": "system prompt"}]
