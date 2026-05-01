from victor.agent.conversation.store import ConversationStore
from victor.agent.conversation.types import MessagePriority, MessageRole


def test_tool_priority_uses_canonical_core_names(tmp_path):
    store = ConversationStore(db_path=tmp_path / "project.db")

    assert (
        store._determine_priority(MessageRole.TOOL, "read_file")  # noqa: SLF001
        == MessagePriority.HIGH
    )
    assert (
        store._determine_priority(MessageRole.TOOL, "list_directory")  # noqa: SLF001
        == MessagePriority.HIGH
    )
    assert (
        store._determine_priority(MessageRole.TOOL, "read") == MessagePriority.HIGH
    )  # noqa: SLF001
    assert (
        store._determine_priority(MessageRole.TOOL, "shell") == MessagePriority.MEDIUM
    )  # noqa: SLF001
