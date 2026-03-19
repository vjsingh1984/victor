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

"""Pydantic models for codebase indexing: symbols, file metadata, and file watcher."""

import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import watchdog for file watching
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


class IndexedSymbol(BaseModel):
    """Code symbol stored in the codebase index.

    Renamed from Symbol to be semantically distinct:
    - IndexedSymbol (here): Pydantic model for index storage
    - NativeSymbol (victor.native.protocols): Rust-extracted symbols (frozen)
    - RefactorSymbol (victor.coding.refactor.protocol): Refactoring symbol

    Note: Body content is NOT stored here - read from file via line_number/end_line.
    This keeps the index lightweight while allowing full body retrieval on demand.
    """

    name: str
    type: str  # function, class, variable, import
    file_path: str
    line_number: int
    end_line: Optional[int] = None  # end line - use with line_number to read body from file
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent_symbol: Optional[str] = None  # parent symbol name (for methods in classes)
    references: List[str] = Field(default_factory=list)  # Files that reference this symbol
    base_classes: List[str] = Field(default_factory=list)  # inheritance targets
    composition: List[tuple[str, str]] = Field(default_factory=list)  # (owner, member) for has-a


# Backward compatibility alias
Symbol = IndexedSymbol


class FileMetadata(BaseModel):
    """Metadata about a source file."""

    path: str
    language: str
    symbols: List[Symbol] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)  # Files this file depends on
    call_edges: List[tuple[str, str]] = Field(default_factory=list)  # (caller, callee) pairs
    inherit_edges: List[tuple[str, str]] = Field(default_factory=list)  # (child, base)
    implements_edges: List[tuple[str, str]] = Field(default_factory=list)  # (child, interface)
    compose_edges: List[tuple[str, str]] = Field(default_factory=list)  # (owner, member)
    references: List[str] = Field(default_factory=list)  # Identifier references (tree-sitter/AST)
    last_modified: float  # File mtime when indexed
    indexed_at: float = 0.0  # When this file was indexed
    size: int
    lines: int
    content_hash: Optional[str] = None  # SHA256 hash for change detection


class CodebaseFileHandler(FileSystemEventHandler):
    """File system event handler for tracking codebase changes.

    Tracks file modifications, creations, and deletions to mark
    the index as stale when relevant files change.
    """

    def __init__(
        self,
        on_change: Callable[[str], None],
        file_patterns: List[str] = None,
        ignore_patterns: List[str] = None,
    ):
        """Initialize file handler.

        Args:
            on_change: Callback when a file changes (receives file path)
            file_patterns: File patterns to watch (e.g., ["*.py"])
            ignore_patterns: Patterns to ignore
        """
        super().__init__()
        self.on_change = on_change
        self.file_patterns = file_patterns or ["*.py"]
        self.ignore_patterns = ignore_patterns or [
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            "venv",
            ".venv",
        ]
        self._debounce_lock = threading.Lock()
        self._pending_changes: Set[str] = set()
        self._debounce_timer: Optional[threading.Timer] = None
        self._debounce_delay = 0.5  # 500ms debounce

    def _should_process(self, path: str) -> bool:
        """Check if path should be processed."""
        path_obj = Path(path)

        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in str(path_obj):
                return False

        # Check file patterns
        for pattern in self.file_patterns:
            if path_obj.match(pattern):
                return True

        return False

    def _debounced_notify(self) -> None:
        """Notify of changes after debounce period."""
        with self._debounce_lock:
            changes = list(self._pending_changes)
            self._pending_changes.clear()
            self._debounce_timer = None

        for path in changes:
            try:
                self.on_change(path)
            except Exception as e:
                logger.warning(f"Error in file change callback: {e}")

    def _schedule_notification(self, path: str) -> None:
        """Schedule a debounced notification."""
        with self._debounce_lock:
            self._pending_changes.add(path)

            # Cancel existing timer
            if self._debounce_timer:
                self._debounce_timer.cancel()

            # Schedule new timer
            self._debounce_timer = threading.Timer(self._debounce_delay, self._debounced_notify)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def on_modified(self, event) -> None:
        """Handle file modification."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)

    def on_created(self, event) -> None:
        """Handle file creation."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)

    def on_deleted(self, event) -> None:
        """Handle file deletion."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)
