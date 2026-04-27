from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.task_completion import CompletionConfidence
from victor.agent.streaming.intent_classification import IntentClassificationResult
from victor.core.completion_markers import SUMMARY_MARKER
from victor.agent.streaming.pipeline import StreamingChatPipeline
from victor.providers.base import StreamChunk
from .streaming_pipeline_stubs import (
    DummyCoordinator,
    StubContinuationHandler,
    StubContinuationResult,
    StubIntentHandler,
    StubToolExecutionHandler,
    StubToolExecutionResult,
)


@pytest.mark.asyncio
async def test_pipeline_forwards_precheck_chunks():
    cancel_chunk = StreamChunk(content="", is_final=True)
    coordinator = DummyCoordinator(pre_chunks=[cancel_chunk])
    pipeline = StreamingChatPipeline(coordinator)

    chunks = []
    async for chunk in pipeline.run("hello"):
        chunks.append(chunk)

    assert chunks == [cancel_chunk]
    assert coordinator._create_stream_calls == ["hello"]


@pytest.mark.asyncio
async def test_pipeline_emits_iteration_limit_chunk():
    limit_chunk = StreamChunk(content="stop", is_final=True)
    coordinator = DummyCoordinator(limit_result=(True, limit_chunk))
    pipeline = StreamingChatPipeline(coordinator)

    chunks = []
    async for chunk in pipeline.run("world"):
        chunks.append(chunk)

    assert chunks == [limit_chunk]
    assert coordinator._create_stream_calls == ["world"]


@pytest.mark.asyncio
async def test_pipeline_invokes_intent_and_continuation_handlers():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = ("assistant notes", None, None, False)

    intent_result = IntentClassificationResult(
        chunks=[StreamChunk(content="intent")],
        action_result={"reason": "finish"},
        action="continue",
    )
    coordinator._intent_classification_handler = StubIntentHandler(intent_result)
    cont_result = StubContinuationResult(
        chunks=[StreamChunk(content="cont")],
        state_updates={"cumulative_prompt_interventions": 7},
        should_return=True,
    )
    coordinator._continuation_handler = StubContinuationHandler(cont_result)

    pipeline = StreamingChatPipeline(coordinator)

    chunks = []
    async for chunk in pipeline.run("task"):
        chunks.append(chunk.content)

    assert chunks == ["intent", "cont"]
    assert len(coordinator._intent_classification_handler.calls) == 1
    assert len(coordinator._continuation_handler.calls) == 1
    assert coordinator._orchestrator._cumulative_prompt_interventions == 7


@pytest.mark.asyncio
async def test_pipeline_executes_tool_calls():
    coordinator = DummyCoordinator(limit_result=(False, None))
    tool_calls = [{"name": "tool.echo", "arguments": {"text": "hi"}}]
    coordinator._provider_response = ("", tool_calls, None, False)

    exec_result = StubToolExecutionResult(
        chunks=[StreamChunk(content="tool-result", is_final=True)],
        tool_calls_executed=1,
        should_return=True,
    )
    coordinator._tool_execution_handler = StubToolExecutionHandler(exec_result)

    pipeline = StreamingChatPipeline(coordinator)

    chunks = []
    async for chunk in pipeline.run("tool task"):
        chunks.append(chunk.content)

    assert chunks == ["tool-result"]
    assert coordinator._tool_execution_handler.updated_files == set()
    assert coordinator._orchestrator.tool_calls_used == 1


@pytest.mark.asyncio
async def test_pipeline_ignores_stale_blocked_state_before_current_tool_execution():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = (
        "",
        [{"name": "read", "arguments": {"path": "victor/agent/streaming/pipeline.py"}}],
        None,
        False,
    )
    coordinator._orchestrator._tool_pipeline = SimpleNamespace(
        last_batch_effectively_blocked=True,
        last_batch_all_skipped=True,
    )

    class _TwoIterationToolHandler:
        def __init__(self) -> None:
            self.calls = []
            self.updated_files = None

        def update_observed_files(self, files):
            self.updated_files = files

        async def execute_tools(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return StubToolExecutionResult(
                    chunks=[StreamChunk(content="blocked-once")],
                    tool_calls_executed=0,
                    should_return=False,
                )
            return StubToolExecutionResult(
                chunks=[StreamChunk(content="recovered", is_final=True)],
                tool_calls_executed=0,
                should_return=True,
            )

    coordinator._tool_execution_handler = _TwoIterationToolHandler()
    pipeline = StreamingChatPipeline(coordinator)

    chunks = []
    async for chunk in pipeline.run("retry the blocked read"):
        chunks.append(chunk.content)

    assert chunks == ["blocked-once", "recovered"]
    assert len(coordinator._tool_execution_handler.calls) == 2


@pytest.mark.asyncio
async def test_pipeline_records_pending_grounding_feedback():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = ("analysis output " * 5, None, None, False)
    validate_result = {
        "should_retry": True,
        "grounding_feedback": "cite sources",
        "quality_score": 0.4,
        "is_grounded": False,
    }

    async def mock_validate(*args, **kwargs):
        return validate_result

    coordinator._orchestrator._validate_intelligent_response = mock_validate
    intent_result = IntentClassificationResult(
        chunks=[],
        action_result={"reason": "finish"},
        action="finish",
    )
    coordinator._intent_classification_handler = StubIntentHandler(intent_result)
    coordinator._continuation_handler = StubContinuationHandler(
        StubContinuationResult(chunks=[], state_updates={}, should_return=True)
    )

    pipeline = StreamingChatPipeline(coordinator)
    async for _ in pipeline.run("needs grounding"):
        pass

    assert coordinator._stream_ctx.pending_grounding_feedback == "cite sources"


@pytest.mark.asyncio
async def test_pipeline_returns_targeted_clarification_before_provider_call():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._stream_provider_response = AsyncMock(
        return_value=("assistant notes", None, None, False)
    )
    mock_perception = SimpleNamespace(
        perceive=AsyncMock(
            return_value=SimpleNamespace(
                needs_clarification=True,
                clarification_prompt="Which file, component, or bug should I target first?",
                confidence=0.3,
                intent="write_allowed",
                complexity="medium",
            )
        )
    )

    pipeline = StreamingChatPipeline(coordinator, perception=mock_perception)

    chunks = []
    async for chunk in pipeline.run("Fix it and add tests."):
        chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == [
        "Which file, component, or bug should I target first?"
    ]
    assert chunks[0].is_final is True
    assert coordinator._stream_ctx.perception.needs_clarification is True
    coordinator._stream_provider_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_uses_runtime_clarification_policy_default_prompt():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._stream_provider_response = AsyncMock(
        return_value=("assistant notes", None, None, False)
    )
    runtime_intelligence = SimpleNamespace(
        reset_decision_budget=MagicMock(),
        analyze_turn=AsyncMock(
            return_value=SimpleNamespace(
                perception=SimpleNamespace(
                    needs_clarification=True,
                    clarification_prompt=None,
                    clarification_reason="target artifact or scope is underspecified",
                    confidence=0.3,
                    intent="write_allowed",
                    complexity="medium",
                )
            )
        ),
    )

    pipeline = StreamingChatPipeline(coordinator, runtime_intelligence=runtime_intelligence)

    chunks = []
    async for chunk in pipeline.run("Fix it and add tests."):
        chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == [
        "Please clarify the target file, component, or bug before I continue."
    ]
    assert chunks[0].is_final is True
    coordinator._stream_provider_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_uses_runtime_intelligence_for_budget_reset_and_analysis():
    cancel_chunk = StreamChunk(content="", is_final=True)
    coordinator = DummyCoordinator(pre_chunks=[cancel_chunk])
    runtime_intelligence = SimpleNamespace(
        reset_decision_budget=MagicMock(),
        analyze_turn=AsyncMock(
            return_value=SimpleNamespace(
                perception=SimpleNamespace(
                    needs_clarification=False,
                    confidence=0.7,
                    intent="write_allowed",
                    complexity="medium",
                )
            )
        ),
    )

    pipeline = StreamingChatPipeline(coordinator, runtime_intelligence=runtime_intelligence)

    chunks = []
    async for chunk in pipeline.run("hello"):
        chunks.append(chunk)

    assert chunks == [cancel_chunk]
    runtime_intelligence.reset_decision_budget.assert_called_once_with()
    runtime_intelligence.analyze_turn.assert_awaited_once()
    assert coordinator._stream_ctx.perception.confidence == 0.7


@pytest.mark.asyncio
async def test_pipeline_merges_stream_context_provider_kwargs():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._stream_ctx.provider_kwargs = {
        "provider_hint": "smart-router",
        "execution_mode": "escalated_single_agent",
    }
    coordinator._stream_provider_response = AsyncMock(
        return_value=("assistant notes", None, None, False)
    )
    coordinator._intent_classification_handler = StubIntentHandler(
        IntentClassificationResult(
            chunks=[],
            action_result={"reason": "finish"},
            action="finish",
        )
    )
    coordinator._continuation_handler = StubContinuationHandler(
        StubContinuationResult(chunks=[], state_updates={}, should_return=True)
    )

    pipeline = StreamingChatPipeline(coordinator)

    async for _ in pipeline.run("hello"):
        pass

    provider_kwargs = coordinator._stream_provider_response.await_args.kwargs["provider_kwargs"]
    assert provider_kwargs["provider_hint"] == "smart-router"
    assert provider_kwargs["execution_mode"] == "escalated_single_agent"
    assert provider_kwargs["thinking"]["type"] == "enabled"


@pytest.mark.asyncio
async def test_pipeline_resets_streaming_turn_state_before_execution():
    cancel_chunk = StreamChunk(content="", is_final=True)
    coordinator = DummyCoordinator(pre_chunks=[cancel_chunk])
    coordinator._orchestrator.tool_calls_used = 9
    coordinator._orchestrator._tool_pipeline = SimpleNamespace(reset=MagicMock())
    coordinator._orchestrator._task_completion_detector = SimpleNamespace(reset=MagicMock())

    pipeline = StreamingChatPipeline(coordinator)

    chunks = []
    async for chunk in pipeline.run("hello"):
        chunks.append(chunk)

    assert chunks == [cancel_chunk]
    assert coordinator._orchestrator.tool_calls_used == 0
    coordinator._orchestrator._tool_pipeline.reset.assert_called_once_with()
    coordinator._orchestrator._task_completion_detector.reset.assert_called_once_with()


@pytest.mark.asyncio
async def test_pipeline_passes_history_to_runtime_intelligence():
    cancel_chunk = StreamChunk(content="", is_final=True)
    coordinator = DummyCoordinator(pre_chunks=[cancel_chunk])
    coordinator._orchestrator.messages = [
        SimpleNamespace(
            role="assistant",
            content="Start with victor/agent/services/tool_compat.py first.",
            model_dump=lambda: {
                "role": "assistant",
                "content": "Start with victor/agent/services/tool_compat.py first.",
            },
        )
    ]
    runtime_intelligence = SimpleNamespace(
        reset_decision_budget=MagicMock(),
        analyze_turn=AsyncMock(
            return_value=SimpleNamespace(
                perception=SimpleNamespace(
                    needs_clarification=False,
                    confidence=0.7,
                    intent="write_allowed",
                    complexity="medium",
                )
            )
        ),
    )

    pipeline = StreamingChatPipeline(coordinator, runtime_intelligence=runtime_intelligence)

    async for _ in pipeline.run("Fix that first."):
        pass

    assert runtime_intelligence.analyze_turn.await_args.kwargs["conversation_history"] == [
        {
            "role": "assistant",
            "content": "Start with victor/agent/services/tool_compat.py first.",
        }
    ]


@pytest.mark.asyncio
async def test_pipeline_prefers_assembled_history_for_runtime_intelligence():
    cancel_chunk = StreamChunk(content="", is_final=True)
    coordinator = DummyCoordinator(pre_chunks=[cancel_chunk])
    coordinator._orchestrator.messages = [
        SimpleNamespace(
            role="assistant",
            content="raw message should not be used",
            model_dump=lambda: {
                "role": "assistant",
                "content": "raw message should not be used",
            },
        )
    ]
    coordinator._orchestrator.get_assembled_messages = lambda current_query=None: [
        SimpleNamespace(
            role="assistant",
            content="assembled history should be used",
            model_dump=lambda: {
                "role": "assistant",
                "content": "assembled history should be used",
            },
        ),
        SimpleNamespace(
            role="user",
            content=current_query,
            model_dump=lambda: {
                "role": "user",
                "content": current_query,
            },
        ),
    ]
    runtime_intelligence = SimpleNamespace(
        reset_decision_budget=MagicMock(),
        analyze_turn=AsyncMock(
            return_value=SimpleNamespace(
                perception=SimpleNamespace(
                    needs_clarification=False,
                    confidence=0.7,
                    intent="write_allowed",
                    complexity="medium",
                )
            )
        ),
    )

    pipeline = StreamingChatPipeline(coordinator, runtime_intelligence=runtime_intelligence)

    async for _ in pipeline.run("Fix that first."):
        pass

    assert runtime_intelligence.analyze_turn.await_args.kwargs["conversation_history"] == [
        {
            "role": "assistant",
            "content": "assembled history should be used",
        }
    ]


@pytest.mark.asyncio
async def test_pipeline_passes_history_to_perception_fallback():
    cancel_chunk = StreamChunk(content="", is_final=True)
    coordinator = DummyCoordinator(pre_chunks=[cancel_chunk])
    coordinator._orchestrator.messages = [
        SimpleNamespace(
            role="assistant",
            content="Start with victor/agent/services/tool_compat.py first.",
            model_dump=lambda: {
                "role": "assistant",
                "content": "Start with victor/agent/services/tool_compat.py first.",
            },
        )
    ]
    mock_perception = SimpleNamespace(
        perceive=AsyncMock(
            return_value=SimpleNamespace(
                needs_clarification=False,
                confidence=0.7,
                intent="write_allowed",
                complexity="medium",
            )
        )
    )

    pipeline = StreamingChatPipeline(coordinator, perception=mock_perception)

    async for _ in pipeline.run("Fix that first."):
        pass

    assert mock_perception.perceive.await_args.kwargs["conversation_history"] == [
        {
            "role": "assistant",
            "content": "Start with victor/agent/services/tool_compat.py first.",
        }
    ]


@pytest.mark.asyncio
async def test_pipeline_yields_recovery_fallback_when_empty():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = ("", None, None, False)
    coordinator._empty_recovery = (False, None, None)

    pipeline = StreamingChatPipeline(coordinator)
    chunks = []
    async for chunk in pipeline.run("empty response"):
        chunks.append(chunk.content)

    assert any("fallback" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_pipeline_prefers_canonical_recovery_context_factory():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = ("", None, None, False)
    coordinator._empty_recovery = (False, None, None)
    coordinator._orchestrator.create_recovery_context = lambda *_: object()

    pipeline = StreamingChatPipeline(coordinator)
    chunks = []
    async for chunk in pipeline.run("empty response"):
        chunks.append(chunk.content)

    assert any("fallback" in chunk for chunk in chunks)


def test_prepare_visible_content_strips_completion_markers():
    pipeline = StreamingChatPipeline(DummyCoordinator())

    prepared = pipeline._prepare_visible_content(f"{SUMMARY_MARKER} Key findings")

    assert prepared == "Key findings"


def test_prepare_visible_content_keeps_new_block_while_suppressing_repeated_block():
    pipeline = StreamingChatPipeline(DummyCoordinator())
    repeated = (
        "Now I have enough data to produce the complete analysis based on the "
        "evidence collected from the repository."
    )
    first = pipeline._prepare_visible_content(f"{repeated}\n\nFirst unique detail.")
    second = pipeline._prepare_visible_content(f"{repeated}\n\nSecond unique detail.")

    assert "First unique detail." in first
    assert second == "Second unique detail."


@pytest.mark.asyncio
async def test_pipeline_persists_normalized_visible_content_but_classifies_raw_content():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = (f"{SUMMARY_MARKER} Key findings", None, None, False)
    added_messages = []

    def add_message(role, content, **kwargs):
        added_messages.append((role, content, kwargs))

    coordinator._orchestrator.add_message = add_message
    intent_result = IntentClassificationResult(
        chunks=[],
        action_result={"reason": "finish"},
        action="finish",
    )
    coordinator._intent_classification_handler = StubIntentHandler(intent_result)
    coordinator._continuation_handler = StubContinuationHandler(
        StubContinuationResult(chunks=[], state_updates={}, should_return=True)
    )

    pipeline = StreamingChatPipeline(coordinator)
    async for _ in pipeline.run("summarize"):
        pass

    assert ("assistant", "Key findings", {"tool_calls": None}) in added_messages
    assert (
        coordinator._intent_classification_handler.calls[0]["full_content"]
        == f"{SUMMARY_MARKER} Key findings"
    )


@pytest.mark.asyncio
async def test_pipeline_does_not_force_completion_when_high_confidence_response_has_tool_calls():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = (
        f"{SUMMARY_MARKER} I need one more read before I can answer.",
        [{"name": "read", "arguments": {"path": "victor/agent/streaming/pipeline.py"}}],
        None,
        False,
    )
    detector = SimpleNamespace(
        _state=SimpleNamespace(last_summary="I need one more read before I can answer."),
        reset=MagicMock(),
        analyze_response=MagicMock(),
        get_completion_confidence=MagicMock(return_value=CompletionConfidence.HIGH),
    )
    coordinator._orchestrator._task_completion_detector = detector
    coordinator._orchestrator._conversation_controller = SimpleNamespace(
        persist_compaction_summary=MagicMock(),
        inject_compaction_context=MagicMock(),
    )
    exec_result = StubToolExecutionResult(
        chunks=[StreamChunk(content="tool-result", is_final=True)],
        tool_calls_executed=1,
        should_return=True,
    )
    coordinator._tool_execution_handler = StubToolExecutionHandler(exec_result)

    pipeline = StreamingChatPipeline(coordinator)

    chunks = []
    async for chunk in pipeline.run("finish the analysis"):
        chunks.append(chunk.content)

    assert chunks == ["I need one more read before I can answer.", "tool-result"]
    assert coordinator._stream_ctx.force_completion is False
    detector.analyze_response.assert_called_once_with(
        f"{SUMMARY_MARKER} I need one more read before I can answer."
    )
    detector.get_completion_confidence.assert_called_once_with()
    coordinator._orchestrator._conversation_controller.persist_compaction_summary.assert_not_called()
    coordinator._orchestrator._conversation_controller.inject_compaction_context.assert_not_called()
    assert len(coordinator._tool_execution_handler.calls) == 1


@pytest.mark.asyncio
async def test_pipeline_forced_completion_bypasses_recovery_and_stale_blocked_state():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = (
        f"{SUMMARY_MARKER} Final findings mention graph and metrics but are complete.",
        None,
        None,
        False,
    )

    class _HighConfidenceDetector:
        def __init__(self) -> None:
            self._state = SimpleNamespace(
                last_summary=f"{SUMMARY_MARKER} Final findings mention graph and metrics but are complete."
            )
            self.analyzed = None

        def analyze_response(self, content: str) -> None:
            self.analyzed = content

        def get_completion_confidence(self):
            return CompletionConfidence.HIGH

    detector = _HighConfidenceDetector()
    recovery_mock = AsyncMock(return_value=SimpleNamespace(action="continue"))
    coordinator._orchestrator._task_completion_detector = detector
    coordinator._orchestrator._tool_pipeline = SimpleNamespace(
        last_batch_effectively_blocked=True,
        last_batch_all_skipped=True,
    )
    coordinator._orchestrator._handle_recovery_with_integration = recovery_mock
    coordinator._orchestrator._conversation_controller = SimpleNamespace(
        persist_compaction_summary=MagicMock(),
        inject_compaction_context=MagicMock(),
    )
    coordinator._intent_classification_handler = StubIntentHandler(
        IntentClassificationResult(chunks=[], action_result={"reason": "finish"}, action="finish")
    )
    coordinator._continuation_handler = StubContinuationHandler(
        StubContinuationResult(chunks=[], state_updates={}, should_return=True)
    )

    pipeline = StreamingChatPipeline(coordinator)
    chunks = []
    async for chunk in pipeline.run("summarize architecture"):
        chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == [
        "Final findings mention graph and metrics but are complete."
    ]
    assert chunks[0].is_final is True
    assert detector.analyzed == (
        f"{SUMMARY_MARKER} Final findings mention graph and metrics but are complete."
    )
    recovery_mock.assert_not_awaited()
    assert coordinator._intent_classification_handler.calls == []
    assert coordinator._continuation_handler.calls == []
    coordinator._orchestrator._conversation_controller.persist_compaction_summary.assert_called_once_with(
        "Final findings mention graph and metrics but are complete.", []
    )
    coordinator._orchestrator._conversation_controller.inject_compaction_context.assert_called_once_with()
