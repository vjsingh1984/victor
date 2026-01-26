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

"""File watcher implementation for cache invalidation.

This module implements a file system watcher using the watchdog library
to detect file changes and emit events for automatic cache invalidation.

Design Patterns:
    - Observer Pattern: FileChangeHandler observes file system events
    - Thread-Safe Queue: Async queue for event streaming
    - SRP: Focused on file watching, not cache invalidation logic

Usage:
    from victor.agent.cache.file_watcher import FileWatcher

    watcher = FileWatcher()
    await watcher.start()

    # Watch a single file
    await watcher.watch_file("/src/main.py")

    # Watch a directory
    await watcher.watch_directory("/src", recursive=True)

    # Get pending changes
    changes = await watcher.get_changes()
    for change in changes:
        print(f"{change.file_path}: {change.change_type}")

    await watcher.stop()
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
)

from victor.protocols import FileChangeEvent, FileChangeType, IFileWatcher


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events.

    This handler processes file system events and puts them into
    a thread-safe queue for async retrieval.

    Attributes:
        event_queue: Thread-safe queue for file change events
        event_loop: Event loop for thread-safe operations
    """

    def __init__(
        self,
        event_queue: asyncio.Queue[FileChangeEvent],
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Initialize the file change handler.

        Args:
            event_queue: Async queue to emit events to
            event_loop: Event loop for thread-safe calls
        """
        super().__init__()
        self._event_queue = event_queue
        self._event_loop = event_loop

    def on_created(self, event: FileCreatedEvent | DirCreatedEvent) -> None:
        """Handle file/directory creation.

        Args:
            event: Watchdog file created event
        """
        if event.is_directory:
            return  # Skip directory events

        # Convert bytes to str if necessary
        file_path = (
            event.src_path if isinstance(event.src_path, str) else event.src_path.decode("utf-8")
        )

        change_event = FileChangeEvent(
            file_path=str(file_path),
            change_type=FileChangeType.CREATED,
            timestamp=datetime.now().isoformat(),
        )
        self._emit_event(change_event)

    def on_modified(self, event: FileModifiedEvent | DirModifiedEvent) -> None:
        """Handle file/directory modification.

        Args:
            event: Watchdog file modified event
        """
        if event.is_directory:
            return  # Skip directory events

        # Convert bytes to str if necessary
        file_path = (
            event.src_path if isinstance(event.src_path, str) else event.src_path.decode("utf-8")
        )

        change_event = FileChangeEvent(
            file_path=str(file_path),
            change_type=FileChangeType.MODIFIED,
            timestamp=datetime.now().isoformat(),
        )
        self._emit_event(change_event)

    def on_deleted(self, event: FileDeletedEvent | DirDeletedEvent) -> None:
        """Handle file/directory deletion.

        Args:
            event: Watchdog file deleted event
        """
        if event.is_directory:
            return  # Skip directory events

        # Convert bytes to str if necessary
        file_path = (
            event.src_path if isinstance(event.src_path, str) else event.src_path.decode("utf-8")
        )

        change_event = FileChangeEvent(
            file_path=str(file_path),
            change_type=FileChangeType.DELETED,
            timestamp=datetime.now().isoformat(),
        )
        self._emit_event(change_event)

    def on_moved(self, event: FileMovedEvent | DirMovedEvent) -> None:
        """Handle file/directory move/rename.

        Args:
            event: Watchdog file moved event
        """
        if event.is_directory:
            return  # Skip directory events

        # Convert bytes to str if necessary
        dest_path = (
            event.dest_path if isinstance(event.dest_path, str) else event.dest_path.decode("utf-8")
        )
        src_path = (
            event.src_path if isinstance(event.src_path, str) else event.src_path.decode("utf-8")
        )

        change_event = FileChangeEvent(
            file_path=str(dest_path),
            change_type=FileChangeType.MOVED,
            timestamp=datetime.now().isoformat(),
            source_path=str(src_path) if src_path else None,
        )
        self._emit_event(change_event)

    def _emit_event(self, event: FileChangeEvent) -> None:
        """Emit event to queue in a thread-safe manner.

        Args:
            event: File change event to emit
        """
        try:
            # Use stored event loop reference for thread-safe calls
            self._event_loop.call_soon_threadsafe(self._event_queue.put_nowait, event)
        except RuntimeError:
            # Event loop is closed, ignore event
            pass


class FileWatcher(IFileWatcher):
    """File system watcher for cache invalidation.

    Monitors files and directories for changes and emits events
    that can be used to invalidate cached tool results.

    Attributes:
        _observer: Watchdog observer for file system events
        _event_handler: Handler for file system events
        _event_queue: Async queue for change events
        _watched_paths: Set of paths being watched
        _running: Whether the watcher is running
    """

    def __init__(self) -> None:
        """Initialize the file watcher."""
        self._observer = Observer()
        self._event_queue: asyncio.Queue[FileChangeEvent] = asyncio.Queue()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_handler: Optional[FileChangeHandler] = None
        self._watched_paths: Dict[str, Any] = {}
        self._running = False
        self._lock = threading.Lock()

    async def start(self) -> None:
        """Start the file watcher.

        This method must be called before watching any files.
        """
        with self._lock:
            if not self._running:
                # Get the current event loop and create the handler
                self._event_loop = asyncio.get_running_loop()
                self._event_handler = FileChangeHandler(
                    self._event_queue,
                    self._event_loop,
                )
                self._observer.start()
                self._running = True

    async def stop(self) -> None:
        """Stop the file watcher and release resources.

        This method stops all watching and cleans up resources.
        """
        with self._lock:
            if self._running:
                self._observer.stop()
                self._observer.join()
                self._running = False
                self._watched_paths.clear()
                self._event_handler = None
                self._event_loop = None

    def is_running(self) -> bool:
        """Check if the watcher is currently running.

        Returns:
            True if watcher is active, False otherwise
        """
        return self._running

    async def watch_file(self, file_path: str) -> None:
        """Monitor a single file for changes.

        Args:
            file_path: Absolute path to the file to watch
        """
        path = Path(file_path).resolve()

        if not path.exists():
            # Watch the parent directory, the file will be detected when created
            parent = path.parent
            await self._watch_directory(str(parent), recursive=False)
            return

        # Watch the parent directory
        parent = path.parent
        await self._watch_directory(str(parent), recursive=False)

    async def watch_directory(self, directory: str, recursive: bool = True) -> None:
        """Monitor a directory for changes.

        Args:
            directory: Absolute path to the directory
            recursive: If True, watch subdirectories as well
        """
        await self._watch_directory(directory, recursive)

    async def _watch_directory(self, directory: str, recursive: bool) -> None:
        """Internal method to watch a directory.

        Args:
            directory: Absolute path to the directory
            recursive: If True, watch subdirectories as well
        """
        if self._event_handler is None:
            raise RuntimeError("FileWatcher not started. Call start() first.")

        dir_path = Path(directory).resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        dir_str = str(dir_path)

        # Check if already watching
        with self._lock:
            if dir_str in self._watched_paths:
                return

            # Schedule the watch
            self._observer.schedule(
                self._event_handler,
                dir_str,
                recursive=recursive,
            )
            self._watched_paths[dir_str] = recursive

    async def unwatch_file(self, file_path: str) -> None:
        """Stop monitoring a file.

        Note: This is a no-op in the current implementation since
        we watch parent directories. Use unwatch_directory instead.

        Args:
            file_path: Absolute path to the file to stop watching
        """
        # Since we watch parent directories, unwatching a single file
        # would require unwatching the parent directory, which might
        # affect other files being watched. For simplicity, we treat
        # this as a no-op.
        pass

    async def unwatch_directory(self, directory: str) -> None:
        """Stop monitoring a directory.

        Args:
            directory: Absolute path to the directory to stop watching
        """
        dir_path = Path(directory).resolve()
        dir_str = str(dir_path)

        with self._lock:
            if dir_str in self._watched_paths:
                # Remove the watch using observer's unschedule method
                for emitter in self._observer.emitters.copy():
                    try:
                        if hasattr(emitter, "watch"):
                            watch_path = str(emitter.watch.path)
                            if watch_path == dir_str:
                                self._observer.unschedule(emitter.watch)
                                break
                    except Exception:
                        # Fallback: clear all emitters if unschedule fails
                        self._observer.emitters.clear()
                        break

                self._watched_paths.pop(dir_str, None)

    async def get_changes(self) -> List[FileChangeEvent]:
        """Get pending file change events.

        Returns:
            List of file change events since last call

        Note:
            This method consumes and returns all pending events.
            Subsequent calls will only return new events.
        """
        changes: List[FileChangeEvent] = []

        # Drain the queue
        while True:
            try:
                # Use get_nowait to avoid blocking
                change = self._event_queue.get_nowait()
                changes.append(change)
            except asyncio.QueueEmpty:
                break

        return changes


__all__ = ["FileWatcher", "FileChangeHandler"]
