# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unified execution checkpoint contract.

This module defines the framework-level envelope that links existing checkpoint
owners together. It does not persist graph, conversation, filesystem, or HITL
state itself; those responsibilities stay with their canonical managers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional
import uuid


class ApprovalState(str, Enum):
    """Approval status associated with an execution checkpoint."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExecutionCheckpoint:
    """Identifier envelope for a resumable execution point.

    Distinct checkpoint managers continue to own their data:
    - ``WorkflowCheckpoint`` owns graph state.
    - ``ConversationCheckpointManager`` owns conversation/session state.
    - ``GitCheckpointManager`` or another filesystem owner owns file snapshots.
    - HITL controllers own approval interruption state.
    """

    id: str
    session_id: str
    graph_checkpoint_id: Optional[str] = None
    conversation_checkpoint_id: Optional[str] = None
    filesystem_checkpoint_id: Optional[str] = None
    triggering_tool_call: Optional[Dict[str, Any]] = None
    approval_state: ApprovalState = ApprovalState.NOT_REQUIRED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        graph_checkpoint_id: Optional[str] = None,
        conversation_checkpoint_id: Optional[str] = None,
        filesystem_checkpoint_id: Optional[str] = None,
        triggering_tool_call: Optional[Mapping[str, Any]] = None,
        approval_state: ApprovalState | str = ApprovalState.NOT_REQUIRED,
        metadata: Optional[Mapping[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> "ExecutionCheckpoint":
        """Create a new execution checkpoint envelope."""
        return cls(
            id=checkpoint_id or f"exec_ckpt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            graph_checkpoint_id=graph_checkpoint_id,
            conversation_checkpoint_id=conversation_checkpoint_id,
            filesystem_checkpoint_id=filesystem_checkpoint_id,
            triggering_tool_call=dict(triggering_tool_call) if triggering_tool_call else None,
            approval_state=ApprovalState(approval_state),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the checkpoint envelope to primitive values."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "graph_checkpoint_id": self.graph_checkpoint_id,
            "conversation_checkpoint_id": self.conversation_checkpoint_id,
            "filesystem_checkpoint_id": self.filesystem_checkpoint_id,
            "triggering_tool_call": self.triggering_tool_call,
            "approval_state": self.approval_state.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionCheckpoint":
        """Deserialize a checkpoint envelope from primitive values."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if not isinstance(created_at, datetime):
            created_at = datetime.now(timezone.utc)

        triggering_tool_call = data.get("triggering_tool_call")
        return cls(
            id=str(data["id"]),
            session_id=str(data["session_id"]),
            graph_checkpoint_id=_optional_str(data.get("graph_checkpoint_id")),
            conversation_checkpoint_id=_optional_str(data.get("conversation_checkpoint_id")),
            filesystem_checkpoint_id=_optional_str(data.get("filesystem_checkpoint_id")),
            triggering_tool_call=(
                dict(triggering_tool_call) if isinstance(triggering_tool_call, Mapping) else None
            ),
            approval_state=ApprovalState(data.get("approval_state", ApprovalState.NOT_REQUIRED)),
            created_at=created_at,
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_trace_metadata(self) -> Dict[str, Any]:
        """Return compact metadata suitable for spans and graph events."""
        tool_call = self.triggering_tool_call or {}
        return {
            "execution_checkpoint_id": self.id,
            "session_id": self.session_id,
            "graph_checkpoint_id": self.graph_checkpoint_id,
            "conversation_checkpoint_id": self.conversation_checkpoint_id,
            "filesystem_checkpoint_id": self.filesystem_checkpoint_id,
            "approval_state": self.approval_state.value,
            "triggering_tool_call_id": tool_call.get("id"),
            "triggering_tool_name": tool_call.get("name"),
            "has_triggering_tool_arguments": bool(tool_call.get("arguments")),
            "metadata": dict(self.metadata),
        }


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


__all__ = ["ApprovalState", "ExecutionCheckpoint"]
