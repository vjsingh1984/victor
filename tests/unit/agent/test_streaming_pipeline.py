from types import SimpleNamespace

import pytest

from victor.agent.streaming.intent_classification import IntentClassificationResult
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
async def test_pipeline_yields_recovery_fallback_when_empty():
    coordinator = DummyCoordinator(limit_result=(False, None))
    coordinator._provider_response = ("", None, None, False)
    coordinator._empty_recovery = (False, None, None)

    pipeline = StreamingChatPipeline(coordinator)
    chunks = []
    async for chunk in pipeline.run("empty response"):
        chunks.append(chunk.content)

    assert any("fallback" in chunk for chunk in chunks)
