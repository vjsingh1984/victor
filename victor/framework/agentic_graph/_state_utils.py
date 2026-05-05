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

"""Shared state helpers for agentic graph nodes."""

from __future__ import annotations

from typing import Any, Dict, TypeAlias, Union

from victor.framework.agentic_graph.state import AgenticLoopStateModel
from victor.framework.graph import CopyOnWriteState

GraphStateInput: TypeAlias = Union[AgenticLoopStateModel, CopyOnWriteState, Dict[str, Any]]


def unwrap_state(state: GraphStateInput | Any) -> AgenticLoopStateModel:
    """Normalize node state to AgenticLoopStateModel."""
    if isinstance(state, CopyOnWriteState):
        unwrapped = state.get_state()
        if isinstance(unwrapped, AgenticLoopStateModel):
            return unwrapped
        if isinstance(unwrapped, dict):
            return AgenticLoopStateModel(**unwrapped)
        return unwrapped
    if isinstance(state, AgenticLoopStateModel):
        return state
    if isinstance(state, dict):
        return AgenticLoopStateModel(**state)
    return state
