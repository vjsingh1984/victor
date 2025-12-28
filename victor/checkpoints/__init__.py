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

"""Time-travel debugging via conversation state checkpoints.

This module provides LangGraph-style checkpoint functionality for conversation
state persistence, enabling replay, forking, and debugging of agent sessions.

Key Features:
- Save/restore conversation state at any point
- Fork sessions from checkpoints to explore alternatives
- Diff checkpoints to understand state changes
- Auto-checkpoint every N tool calls
- SQLite backend for embedded persistence

Usage:
    from victor.checkpoints import CheckpointManager, SQLiteCheckpointBackend

    backend = SQLiteCheckpointBackend(storage_path)
    manager = CheckpointManager(backend)

    # Save checkpoint
    checkpoint_id = await manager.save_checkpoint(session_id, state, "before refactor")

    # List checkpoints
    checkpoints = await manager.list_checkpoints(session_id)

    # Restore state
    state = await manager.load_checkpoint(checkpoint_id)

    # Fork session
    new_state = await manager.fork_from_checkpoint(checkpoint_id, new_session_id)
"""

from victor.checkpoints.protocol import (
    CheckpointMetadata,
    CheckpointDiff,
    CheckpointManagerProtocol,
    CheckpointError,
    CheckpointNotFoundError,
)
from victor.checkpoints.manager import CheckpointManager
from victor.checkpoints.backends.sqlite_backend import SQLiteCheckpointBackend

__all__ = [
    "CheckpointMetadata",
    "CheckpointDiff",
    "CheckpointManagerProtocol",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointManager",
    "SQLiteCheckpointBackend",
]
