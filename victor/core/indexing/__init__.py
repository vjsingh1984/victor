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

"""Core indexing infrastructure for codebase management.

This module provides:
- Index-level locking to prevent concurrent indexing
- File watching for automatic cache invalidation
- Graph management with automatic updates
"""

from __future__ import annotations

__all__ = [
    "IndexLockRegistry",
    "FileWatcherService",
    "FileWatcherRegistry",
    "GraphManager",
    "initialize_file_watchers",
    "stop_file_watchers",
    "cleanup_session",
]

# Import classes and functions for export
from victor.core.indexing.index_lock import IndexLockRegistry
from victor.core.indexing.file_watcher import FileWatcherService, FileWatcherRegistry
from victor.core.indexing.graph_manager import GraphManager
from victor.core.indexing.watcher_initializer import (
    initialize_file_watchers,
    stop_file_watchers,
    cleanup_session,
)
