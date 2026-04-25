from unittest.mock import MagicMock

from victor.agent.streaming.intent_classification import (
    create_intent_classification_handler,
)


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
