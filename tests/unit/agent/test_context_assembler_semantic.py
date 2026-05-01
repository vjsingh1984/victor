"""Tests for semantic retrieval augmentation in context assembler."""

from unittest.mock import MagicMock

import pytest

from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.providers.base import Message


@pytest.fixture
def mock_controller():
    controller = MagicMock()
    controller.retrieve_relevant_history.return_value = [
        "Previously discussed JWT authentication approach",
        "Decided to use factory pattern for auth middleware",
    ]
    return controller


@pytest.fixture
def many_messages():
    """Create enough messages to have older + recent sections."""
    messages = [Message(role="system", content="System prompt")]
    for i in range(12):
        messages.append(Message(role="user", content=f"User message {i}"))
        messages.append(Message(role="assistant", content=f"Response {i}"))
    return messages


def make_score_fn():
    """Create a score function that returns tuples."""

    def score_fn(messages, query=None):
        return [(msg, idx * 0.1) for idx, msg in enumerate(messages)]

    return score_fn


class TestContextAssemblerSemantic:
    def test_semantic_retrieval_within_budget(self, mock_controller, many_messages):
        assembler = TurnBoundaryContextAssembler(
            score_fn=make_score_fn(),
            conversation_controller=mock_controller,
        )
        result = assembler.assemble(
            many_messages,
            max_context_chars=50000,
            current_query="How should I implement auth?",
        )

        # Should include historical context messages
        historical = [m for m in result if "Historical context" in m.content]
        assert len(historical) > 0
        mock_controller.retrieve_relevant_history.assert_called_once()

    def test_no_retrieval_without_controller(self, many_messages):
        assembler = TurnBoundaryContextAssembler(
            score_fn=make_score_fn(),
            conversation_controller=None,
        )
        result = assembler.assemble(
            many_messages,
            max_context_chars=50000,
            current_query="test query",
        )

        historical = [m for m in result if "Historical context" in m.content]
        assert len(historical) == 0

    def test_no_retrieval_without_query(self, mock_controller, many_messages):
        assembler = TurnBoundaryContextAssembler(
            score_fn=make_score_fn(),
            conversation_controller=mock_controller,
        )
        result = assembler.assemble(
            many_messages,
            max_context_chars=50000,
            current_query=None,
        )

        historical = [m for m in result if "Historical context" in m.content]
        assert len(historical) == 0
        mock_controller.retrieve_relevant_history.assert_not_called()

    def test_budget_respected_no_overflow(self, mock_controller, many_messages):
        # Return a very long context string
        mock_controller.retrieve_relevant_history.return_value = [
            "x" * 100000,
        ]
        assembler = TurnBoundaryContextAssembler(
            score_fn=make_score_fn(),
            conversation_controller=mock_controller,
        )
        result = assembler.assemble(
            many_messages,
            max_context_chars=5000,
            current_query="test",
        )
        total_chars = sum(len(m.content) for m in result)
        # Should not massively exceed budget
        assert total_chars < 100000

    def test_graceful_degradation_on_error(self, many_messages):
        error_controller = MagicMock()
        error_controller.retrieve_relevant_history.side_effect = RuntimeError("DB error")

        assembler = TurnBoundaryContextAssembler(
            score_fn=make_score_fn(),
            conversation_controller=error_controller,
        )
        # Should not raise
        result = assembler.assemble(
            many_messages,
            max_context_chars=50000,
            current_query="test",
        )
        assert len(result) > 0  # Still returns results despite error

    def test_budget_calculation_uses_older_chars(self, mock_controller):
        """Verify remaining_budget uses older_chars, not a wrong slice."""
        # Create messages with known sizes
        messages = [Message(role="system", content="S" * 100)]
        for i in range(8):
            messages.append(Message(role="user", content=f"U{'x' * 200}"))
            messages.append(Message(role="assistant", content=f"A{'y' * 200}"))

        # Score function that selects first few older messages
        def score_fn(msgs, query=None):
            return [(msg, 1.0 / (idx + 1)) for idx, msg in enumerate(msgs)]

        # Return small context strings that should fit in remaining budget
        mock_controller.retrieve_relevant_history.return_value = [
            "small context A",
            "small context B",
        ]

        assembler = TurnBoundaryContextAssembler(
            score_fn=score_fn,
            conversation_controller=mock_controller,
        )
        result = assembler.assemble(
            messages,
            max_context_chars=50000,
            current_query="test query",
        )

        # The semantic retrieval should work without errors
        historical = [m for m in result if "Historical context" in m.content]
        assert len(historical) > 0
