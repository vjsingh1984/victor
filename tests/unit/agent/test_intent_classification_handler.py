from unittest.mock import MagicMock

from victor.agent.response_sanitizer import ResponseSanitizer
from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.intent_classification import (
    IntentClassificationHandler,
    TrackingState,
    create_intent_classification_handler,
)
from victor.providers.base import StreamChunk


def test_create_intent_classification_handler_preserves_runtime_intelligence():
    orchestrator = MagicMock()
    orchestrator.intent_classifier = MagicMock()
    orchestrator.unified_tracker = MagicMock()
    orchestrator.sanitizer = MagicMock()
    orchestrator._chunk_generator = MagicMock()
    orchestrator.settings = MagicMock()
    orchestrator._rl_coordinator = MagicMock()
    orchestrator.provider.name = "openai"
    orchestrator.model = "gpt-5"
    orchestrator.tool_budget = 12
    runtime_intelligence = MagicMock()
    orchestrator.__dict__["_runtime_intelligence"] = runtime_intelligence

    handler = create_intent_classification_handler(orchestrator)

    assert handler._runtime_intelligence is runtime_intelligence


def test_intent_handler_finishes_early_on_malformed_tool_style_plaintext():
    intent_classifier = MagicMock()
    unified_tracker = MagicMock()
    unified_tracker.check_response_loop.return_value = False
    settings = MagicMock()
    settings.one_shot_mode = False
    chunk_generator = MagicMock()
    chunk_generator.generate_content_chunk.side_effect = (
        lambda content, is_final=False: StreamChunk(content=content, is_final=is_final)
    )

    handler = IntentClassificationHandler(
        intent_classifier=intent_classifier,
        unified_tracker=unified_tracker,
        sanitizer=ResponseSanitizer(),
        chunk_generator=chunk_generator,
        settings=settings,
        provider_name="zai",
        model="glm-5.1",
    )

    full_content = (
        "I'll analyze the codebase to understand symbol indexing.\n\n"
        "ls rust\n"
        "scripts ls\n"
        "victor/core ls\n"
    )

    result = handler.classify_and_determine_action(
        stream_ctx=StreamingChatContext(user_message="analyze symbol identity"),
        full_content=full_content,
        content_length=len(full_content),
        mentioned_tools=[],
        tracking_state=TrackingState(),
    )

    assert result.action == "finish"
    assert result.action_result["action"] == "finish"
    assert "Malformed tool-style plaintext" in result.action_result["reason"]
    assert any(chunk.is_final for chunk in result.chunks)
    assert any("malformed tool-style text" in chunk.content.lower() for chunk in result.chunks)
    intent_classifier.classify_intent_sync.assert_not_called()


def test_build_task_completion_signals_tracks_explicit_database_requirements():
    handler = IntentClassificationHandler(
        intent_classifier=MagicMock(),
        unified_tracker=MagicMock(),
        sanitizer=ResponseSanitizer(),
        chunk_generator=MagicMock(),
        settings=MagicMock(),
        provider_name="zai",
        model="glm-5.1",
    )

    tracking_state = TrackingState()
    stream_ctx = StreamingChatContext(
        user_message="please review the sqllite db directly and inspect the schema"
    )
    stream_ctx.executed_tool_names.add("shell")

    signals = handler._build_task_completion_signals(tracking_state, stream_ctx)

    assert signals["explicit_database_query_requested"] is True
    assert signals["database_query_satisfied"] is True
    assert signals["original_user_message"] == stream_ctx.user_message
    assert signals["executed_tool_names"] == {"shell"}
