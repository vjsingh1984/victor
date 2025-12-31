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

This module has moved to victor.storage.checkpoints.
Import from victor.storage.checkpoints instead for new code.

This module provides backward-compatible re-exports.
"""

# Re-export from new location for backward compatibility
from victor.storage.checkpoints.protocol import (
    CheckpointMetadata,
    CheckpointDiff,
    CheckpointData,
    CheckpointManagerProtocol,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointStorageError,
    DiffType,
    FieldDiff,
)
from victor.storage.checkpoints.manager import CheckpointManager
from victor.storage.checkpoints.backends.sqlite_backend import SQLiteCheckpointBackend
from victor.storage.checkpoints.tree import (
    BranchStatus,
    MergeStrategy,
    ConflictResolution,
    BranchMetadata,
    CheckpointNode,
    MergeResult,
    ReplayStep,
    CheckpointTree,
    BranchManager,
    BranchStorageProtocol,
)

__all__ = [
    # Protocol types
    "CheckpointMetadata",
    "CheckpointDiff",
    "CheckpointData",
    "CheckpointManagerProtocol",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointStorageError",
    "DiffType",
    "FieldDiff",
    # Core manager
    "CheckpointManager",
    "SQLiteCheckpointBackend",
    # Tree/Branch types
    "BranchStatus",
    "MergeStrategy",
    "ConflictResolution",
    "BranchMetadata",
    "CheckpointNode",
    "MergeResult",
    "ReplayStep",
    "CheckpointTree",
    "BranchManager",
    "BranchStorageProtocol",
]
