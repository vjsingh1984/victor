"""Tests for SDK-owned conversation contracts."""

from victor_sdk.conversation import ConversationCoordinator, TurnType


def test_conversation_coordinator_tracks_history_and_stats() -> None:
    coordinator = ConversationCoordinator(enable_deduplication=False)
    coordinator.add_message("user", "hello", TurnType.USER)
    coordinator.add_message("assistant", "world", TurnType.ASSISTANT)

    history = coordinator.get_history()
    stats = coordinator.get_stats()

    assert [item["role"] for item in history] == ["user", "assistant"]
    assert stats.user_turns == 1
    assert stats.assistant_turns == 1
