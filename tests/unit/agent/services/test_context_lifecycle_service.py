from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.runtime.context import AgentRuntimeContext
from victor.agent.services.context_lifecycle_service import (
    ContextLifecycleService,
    LifecycleCompactionSummarizerAdapter,
)
from victor.agent.services.context_service import ContextServiceConfig, ContextServiceRegistry


@pytest.mark.asyncio
async def test_before_tool_output_compacts_and_persists_event():
    registry = ContextServiceRegistry(
        ContextServiceConfig(
            max_tokens=20,
            min_messages_to_keep=2,
            overflow_threshold_percent=50.0,
        )
    )
    store = MagicMock()
    store.record_compaction_event.return_value = "compact_tool"
    service = ContextLifecycleService(registry=registry, conversation_store=store)
    runtime_context = AgentRuntimeContext(
        agent_id="root_agent",
        display_name="Root Agent",
        role="manager",
        session_id="session_root",
    )

    result = await service.before_tool_output(
        runtime_context,
        estimated_output_tokens=18,
        messages=[
            {"role": "user", "content": "x" * 20},
            {"role": "assistant", "content": "y" * 20},
            {"role": "tool", "content": "z" * 20},
            {"role": "assistant", "content": "w" * 20},
        ],
        provider_name="deepseek",
        model_name="deepseek-chat",
        task_type="analysis",
        default_strategy="tiered",
        min_messages=2,
    )

    assert result["compacted"] is True
    assert result["messages_removed"] == 2
    assert result["saved_tokens"] == 10
    assert result["tokens_freed"] == 10
    assert result["compaction_event_id"] == "compact_tool"
    assert result["reason"] == "pre_tool_output"
    store.record_compaction_event.assert_called_once()
    assert store.record_compaction_event.call_args.kwargs["session_id"] == "session_root"


@pytest.mark.asyncio
async def test_after_agent_turn_compacts_target_agent_and_persists_event():
    registry = ContextServiceRegistry(
        ContextServiceConfig(
            max_tokens=20,
            min_messages_to_keep=2,
            overflow_threshold_percent=50.0,
        )
    )
    store = MagicMock()
    store.record_compaction_event.return_value = "compact_1"
    service = ContextLifecycleService(registry=registry, conversation_store=store)
    runtime_context = AgentRuntimeContext(
        agent_id="agent_researcher_1",
        display_name="API Researcher",
        role="researcher",
        session_id="session_child",
        parent_session_id="session_root",
        team_id="team_1",
        member_id="api_researcher",
        plan_id="plan_1",
        plan_step_id="1",
    )

    result = await service.after_agent_turn(
        runtime_context,
        messages=[
            {"role": "user", "content": "x" * 20},
            {"role": "assistant", "content": "y" * 20},
            {"role": "tool", "content": "z" * 20},
            {"role": "assistant", "content": "w" * 20},
        ],
    )

    assert result["compacted"] is True
    assert result["messages_removed"] == 2
    assert result["tokens_freed"] == 10
    assert "2 messages compacted" in result["summary"]
    assert result["compaction_event_id"] == "compact_1"
    assert len(registry.get_or_create(runtime_context).get_messages()) == 2
    store.record_compaction_event.assert_called_once()
    call = store.record_compaction_event.call_args.kwargs
    assert call["session_id"] == "session_child"
    assert call["agent_id"] == "agent_researcher_1"
    assert call["messages_removed"] == 2
    assert call["tokens_freed"] == 10
    assert "2 messages compacted" in call["summary"]
    assert call["metadata"]["team_id"] == "team_1"
    assert call["metadata"]["parent_session_id"] == "session_root"


@pytest.mark.asyncio
async def test_after_agent_turn_uses_injected_async_summarizer_for_compaction_event():
    registry = ContextServiceRegistry(
        ContextServiceConfig(
            max_tokens=20,
            min_messages_to_keep=2,
            overflow_threshold_percent=50.0,
        )
    )
    store = MagicMock()
    summarizer = SimpleSummarizer()
    service = ContextLifecycleService(
        registry=registry,
        conversation_store=store,
        compaction_summarizer=summarizer,
    )
    runtime_context = AgentRuntimeContext(
        agent_id="agent_reviewer_1",
        display_name="Reviewer",
        role="reviewer",
        session_id="session_child",
    )

    result = await service.after_agent_turn(
        runtime_context,
        messages=[
            {"role": "user", "content": "review api.py"},
            {"role": "assistant", "content": "found issue in db.py"},
            {"role": "assistant", "content": "kept 1"},
            {"role": "assistant", "content": "kept 2"},
        ],
    )

    assert result["summary"] == "LLM summary: review api.py | found issue in db.py"
    assert store.record_compaction_event.call_args.kwargs["summary"] == result["summary"]
    assert summarizer.summarize.await_count == 1


class SimpleSummarizer:
    def __init__(self):
        self.summarize = AsyncMock(
            side_effect=lambda **kwargs: "LLM summary: "
            + " | ".join(message["content"] for message in kwargs["removed_messages"])
        )


@pytest.mark.asyncio
async def test_lifecycle_compaction_summarizer_adapter_uses_legacy_strategy():
    strategy = LegacySummarizer()
    adapter = LifecycleCompactionSummarizerAdapter(strategy, ledger="ledger")
    runtime_context = AgentRuntimeContext(
        agent_id="agent_reviewer_1",
        display_name="Reviewer",
        role="reviewer",
        session_id="session_child",
    )

    summary = await adapter.summarize(
        runtime_context=runtime_context,
        removed_messages=[{"role": "user", "content": "review api.py"}],
        retained_messages=[],
        reason="after_agent_turn",
    )

    assert summary == "legacy summary"
    args = strategy.summarize.call_args.args
    assert args[1] == "ledger"
    assert args[0][0].role == "user"
    assert args[0][0].content == "review api.py"


@pytest.mark.asyncio
async def test_lifecycle_compaction_summarizer_adapter_prefers_async_strategy():
    strategy = AsyncLegacySummarizer()
    adapter = LifecycleCompactionSummarizerAdapter(strategy, ledger="ledger")
    runtime_context = AgentRuntimeContext(
        agent_id="agent_reviewer_1",
        display_name="Reviewer",
        role="reviewer",
        session_id="session_child",
    )

    summary = await adapter.summarize(
        runtime_context=runtime_context,
        removed_messages=[{"role": "assistant", "content": "found issue in db.py"}],
        retained_messages=[],
        reason="pre_tool_output",
    )

    assert summary == "async legacy summary"
    strategy.summarize_async.assert_awaited_once()


class LegacySummarizer:
    def __init__(self):
        self.summarize = MagicMock(return_value="legacy summary")


class AsyncLegacySummarizer:
    def __init__(self):
        self.summarize_async = AsyncMock(return_value="async legacy summary")


@pytest.mark.asyncio
async def test_after_agent_turn_does_not_mix_sibling_contexts():
    registry = ContextServiceRegistry(
        ContextServiceConfig(max_tokens=20, overflow_threshold_percent=50.0)
    )
    service = ContextLifecycleService(registry=registry)
    root = AgentRuntimeContext(
        agent_id="root",
        display_name="Root",
        role="manager",
        session_id="session_root",
    )
    child = root.derive_child(
        agent_id="child",
        display_name="Child",
        role="researcher",
        member_id="member_1",
        team_id="team_1",
    )

    registry.get_or_create(root).add_message(role="user", content="root")
    await service.after_agent_turn(
        child,
        messages=[
            {"role": "user", "content": "x" * 20},
            {"role": "assistant", "content": "y" * 20},
            {"role": "assistant", "content": "z" * 20},
        ],
        min_messages=1,
    )

    assert registry.get_or_create(root).get_messages() == [{"role": "user", "content": "root"}]
    assert len(registry.get_or_create(child).get_messages()) == 1


def test_build_parent_handoff_returns_bounded_child_summary():
    service = ContextLifecycleService(
        registry=ContextServiceRegistry(ContextServiceConfig(max_tokens=100))
    )
    runtime_context = AgentRuntimeContext(
        agent_id="agent_reviewer_1",
        display_name="Rust Arc Reviewer",
        role="reviewer",
        session_id="session_child",
        parent_session_id="session_root",
        team_id="team_1",
        member_id="reviewer_1",
    )

    handoff = service.build_parent_handoff(
        runtime_context,
        summary="A" * 700,
        status="success",
        metadata={"tool_calls_used": 3},
    )

    assert handoff["agent_id"] == "agent_reviewer_1"
    assert handoff["parent_session_id"] == "session_root"
    assert handoff["status"] == "success"
    assert handoff["tool_calls_used"] == 3
    assert len(handoff["summary"]) <= 520
