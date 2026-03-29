# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared runtime state types for compiled workflow execution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict


class WorkflowState(TypedDict, total=False):
    """Generic state for compiled workflow execution."""

    _workflow_id: str
    _workflow_name: str
    _current_node: str
    _node_results: Dict[str, Any]
    _error: Optional[str]
    _iteration: int
    _parallel_results: Dict[str, Any]
    _hitl_pending: bool
    _hitl_response: Optional[Dict[str, Any]]


@dataclass
class GraphNodeResult:
    """Result from executing a workflow node in the compiled graph runtime."""

    node_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls_used: int = 0


def create_initial_workflow_state(
    *,
    current_node: str = "",
    workflow_id: Optional[str] = None,
    workflow_name: str = "",
    initial_state: Optional[Dict[str, Any]] = None,
) -> WorkflowState:
    """Create the canonical initial state for compiled workflow execution."""
    state: WorkflowState = {
        "_workflow_id": workflow_id or uuid.uuid4().hex,
        "_workflow_name": workflow_name,
        "_current_node": current_node,
        "_node_results": {},
        "_error": None,
        "_iteration": 0,
        "_parallel_results": {},
        "_hitl_pending": False,
        "_hitl_response": None,
    }
    if initial_state:
        state.update(initial_state)
    return state


__all__ = [
    "create_initial_workflow_state",
    "GraphNodeResult",
    "WorkflowState",
]
