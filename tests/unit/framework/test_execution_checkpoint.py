from __future__ import annotations

from datetime import datetime, timezone

from victor.framework.execution_checkpoint import ApprovalState, ExecutionCheckpoint


def test_execution_checkpoint_round_trips_all_checkpoint_ids() -> None:
    checkpoint = ExecutionCheckpoint.create(
        session_id="session-1",
        graph_checkpoint_id="graph-1",
        conversation_checkpoint_id="conversation-1",
        filesystem_checkpoint_id="git-1",
        triggering_tool_call={
            "id": "tool-call-1",
            "name": "write",
            "arguments": {"path": "victor/app.py"},
        },
        approval_state=ApprovalState.APPROVED,
        metadata={"reason": "before file write"},
    )

    restored = ExecutionCheckpoint.from_dict(checkpoint.to_dict())

    assert restored == checkpoint
    assert restored.id.startswith("exec_ckpt_")
    assert restored.session_id == "session-1"
    assert restored.graph_checkpoint_id == "graph-1"
    assert restored.conversation_checkpoint_id == "conversation-1"
    assert restored.filesystem_checkpoint_id == "git-1"
    assert restored.triggering_tool_call["name"] == "write"
    assert restored.approval_state is ApprovalState.APPROVED
    assert restored.metadata == {"reason": "before file write"}


def test_execution_checkpoint_trace_metadata_is_identifier_focused() -> None:
    checkpoint = ExecutionCheckpoint(
        id="exec_ckpt_fixed",
        session_id="session-1",
        graph_checkpoint_id="graph-1",
        conversation_checkpoint_id=None,
        filesystem_checkpoint_id="git-1",
        triggering_tool_call={
            "id": "tool-call-1",
            "name": "write",
            "arguments": {"path": "victor/app.py", "content": "large"},
        },
        approval_state=ApprovalState.PENDING,
        created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        metadata={"trace": "abc"},
    )

    assert checkpoint.to_trace_metadata() == {
        "execution_checkpoint_id": "exec_ckpt_fixed",
        "session_id": "session-1",
        "graph_checkpoint_id": "graph-1",
        "conversation_checkpoint_id": None,
        "filesystem_checkpoint_id": "git-1",
        "approval_state": "pending",
        "triggering_tool_call_id": "tool-call-1",
        "triggering_tool_name": "write",
        "has_triggering_tool_arguments": True,
        "metadata": {"trace": "abc"},
    }


def test_execution_checkpoint_importable_from_framework_namespace() -> None:
    import victor.framework as framework

    assert framework.ExecutionCheckpoint is ExecutionCheckpoint
    assert framework.ApprovalState is ApprovalState
