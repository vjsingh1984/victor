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


def test_get_context_size_counts_dict_messages():
    service = ContextService(ContextServiceConfig(max_tokens=1000))

    service.add_messages(
        [
            {"role": "user", "content": "x" * 40},
            {"role": "assistant", "content": "y" * 80},
        ]
    )

    assert service.get_context_size() == 30


@pytest.mark.asyncio
async def test_prepare_for_tool_output_injection_compacts_and_tracks_saved_tokens():
    service = ContextService(ContextServiceConfig(max_tokens=100, default_compaction_strategy="tiered"))
    service.add_messages(
        [
            {"role": "user", "content": "x" * 40},
            {"role": "assistant", "content": "y" * 40},
            {"role": "user", "content": "z" * 40},
            {"role": "assistant", "content": "w" * 40},
            {"role": "user", "content": "q" * 40},
            {"role": "assistant", "content": "r" * 40},
            {"role": "user", "content": "s" * 40},
            {"role": "assistant", "content": "t" * 40},
        ]
    )

    result = await service.prepare_for_tool_output_injection(
        18,
        provider_name="ollama",
        model_name="local-model",
        task_type="analysis",
        min_messages=6,
        default_strategy="tiered",
    )

    assert result["should_compact"] is True
    assert result["compacted"] is True
    assert result["messages_removed"] == 2
    assert result["saved_tokens"] == 20
    assert result["strategy"] == "tiered"
    assert result["policy_reason"] == "tool_output_exceeds_remaining_budget"
    assert service.get_performance_metrics()["last_compaction_saved_tokens"] == 20
