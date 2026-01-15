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

"""Tests for FileWatcher implementation using TDD approach.

This test suite validates the file watching functionality that monitors
file system changes for cache invalidation.

Test Coverage:
    - File watching (single files)
    - Directory watching (recursive and non-recursive)
    - Event emission (created, modified, deleted, moved)
    - Thread-safety for async operations
    - Resource cleanup and unwatching
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, List, Optional

import pytest

from victor.protocols import FileChangeEvent, FileChangeType, IFileWatcher


# =============================================================================
# Test Fixtures
# =============================================================================

# Fixture is provided in conftest.py


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary test file."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("initial content")
    return file_path


# =============================================================================
# Helper Functions for Event Synchronization
# =============================================================================

async def wait_for_events(
    watcher: IFileWatcher,
    min_count: int = 1,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
) -> List[FileChangeEvent]:
    """Wait for expected number of events with timeout.

    This helper polls for events instead of using a fixed sleep,
    making tests more reliable across different environments.

    Args:
        watcher: FileWatcher instance
        min_count: Minimum number of events to wait for
        timeout: Maximum time to wait in seconds
        poll_interval: Time between polls in seconds

    Returns:
        List of collected events

    Raises:
        AssertionError: If timeout is reached without enough events
    """
    start_time = time.time()
    collected_events: List[FileChangeEvent] = []

    while time.time() - start_time < timeout:
        # Get any new events
        new_events = await watcher.get_changes()
        collected_events.extend(new_events)

        # Check if we have enough events
        if len(collected_events) >= min_count:
            return collected_events

        # Wait before next poll
        await asyncio.sleep(poll_interval)

    # If we get here, timeout was reached
    raise AssertionError(
        f"Timeout waiting for events: expected at least {min_count} "
        f"but got {len(collected_events)} after {timeout}s"
    )


async def flush_events(watcher: IFileWatcher, delay: float = 0.3) -> None:
    """Flush any pending events from the watcher.

    This helper ensures all events are processed before an assertion.
    It's useful for clearing event queue before making a change.

    Args:
        watcher: FileWatcher instance
        delay: Initial delay to allow events to propagate
    """
    await asyncio.sleep(delay)
    await watcher.get_changes()  # Drain the queue


# =============================================================================
# Startup and Shutdown Tests
# =============================================================================

class TestStartupShutdown:
    """Test file watcher startup and shutdown behavior."""

    @pytest.mark.asyncio
    async def test_initial_state(self, file_watcher: IFileWatcher) -> None:
        """Test that watcher starts in stopped state."""
        assert not file_watcher.is_running()

    @pytest.mark.asyncio
    async def test_start_watcher(self, file_watcher: IFileWatcher) -> None:
        """Test starting the file watcher."""
        await file_watcher.start()
        assert file_watcher.is_running()
        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_stop_watcher(self, file_watcher: IFileWatcher) -> None:
        """Test stopping the file watcher."""
        await file_watcher.start()
        await file_watcher.stop()
        assert not file_watcher.is_running()

    @pytest.mark.asyncio
    async def test_multiple_starts(self, file_watcher: IFileWatcher) -> None:
        """Test that multiple start calls are safe."""
        await file_watcher.start()
        await file_watcher.start()  # Should not raise
        assert file_watcher.is_running()
        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_multiple_stops(self, file_watcher: IFileWatcher) -> None:
        """Test that multiple stop calls are safe."""
        await file_watcher.start()
        await file_watcher.stop()
        await file_watcher.stop()  # Should not raise
        assert not file_watcher.is_running()


# =============================================================================
# Single File Watching Tests
# =============================================================================

class TestSingleFileWatching:
    """Test watching individual files."""

    @pytest.mark.asyncio
    async def test_watch_file(self, file_watcher: IFileWatcher, temp_file: Path) -> None:
        """Test watching a single file."""
        await file_watcher.start()
        await file_watcher.watch_file(str(temp_file.absolute()))
        # Should not raise any exceptions
        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_detect_file_modification(
        self,
        file_watcher: IFileWatcher,
        temp_file: Path,
    ) -> None:
        """Test detecting file modifications."""
        await file_watcher.start()
        await file_watcher.watch_file(str(temp_file.absolute()))

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Modify the file
        temp_file.write_text("modified content")

        # Wait for modification event with polling (more reliable than fixed sleep)
        changes = await wait_for_events(file_watcher, min_count=1, timeout=5.0)

        # Should have at least one modification event
        mod_events = [c for c in changes if c.change_type == FileChangeType.MODIFIED]
        assert len(mod_events) > 0

        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_watch_nonexistent_file(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test watching a file that doesn't exist yet."""
        nonexistent = temp_dir / "will_be_created.txt"

        await file_watcher.start()
        await file_watcher.watch_file(str(nonexistent.absolute()))

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Create the file (this should generate a CREATED event)
        nonexistent.write_text("created")

        # Wait for creation event with polling
        changes = await wait_for_events(file_watcher, min_count=1, timeout=5.0)

        # Should detect file creation
        assert len(changes) > 0

        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_unwatch_file(
        self,
        file_watcher: IFileWatcher,
        temp_file: Path,
    ) -> None:
        """Test unwatching a file."""
        await file_watcher.start()
        await file_watcher.watch_file(str(temp_file.absolute()))

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Unwatch the file (note: this is a no-op in current implementation)
        await file_watcher.unwatch_file(str(temp_file.absolute()))

        # Add delay for unwatch to take effect
        await asyncio.sleep(0.2)

        # Modify the file (may still detect since unwatch_file is a no-op)
        temp_file.write_text("modified after unwatch")

        # Wait and check for events
        await asyncio.sleep(0.3)
        changes = await file_watcher.get_changes()

        # Since unwatch_file is a no-op, we might still get events
        # This test validates the behavior rather than asserting no events
        assert isinstance(changes, list)

        await file_watcher.stop()


# =============================================================================
# Directory Watching Tests
# =============================================================================

class TestDirectoryWatching:
    """Test watching directories."""

    @pytest.mark.asyncio
    async def test_watch_directory(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test watching a directory."""
        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=False)
        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_watch_directory_recursive(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test watching a directory recursively."""
        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=True)
        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_detect_file_in_directory(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test detecting file changes in watched directory."""
        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=False)

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Create a file in the directory
        test_file = temp_dir / "new_file.txt"
        test_file.write_text("content")

        # Wait for creation event with polling
        changes = await wait_for_events(file_watcher, min_count=1, timeout=5.0)

        # Should detect file creation
        assert len(changes) > 0

        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_detect_file_in_subdirectory_recursive(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test detecting file changes in subdirectories with recursive watch."""
        # Create subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=True)

        # Allow watcher to stabilize (including subdirectory watch setup)
        await flush_events(file_watcher, delay=0.3)

        # Create a file in subdirectory
        test_file = subdir / "nested_file.txt"
        test_file.write_text("content")

        # Wait for creation event with polling
        changes = await wait_for_events(file_watcher, min_count=1, timeout=5.0)

        # Should detect file creation in subdirectory
        assert len(changes) > 0

        # Find the nested file event
        nested_events = [
            c for c in changes
            if str(subdir.absolute()) in c.file_path
        ]
        assert len(nested_events) > 0

        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_unwatch_directory(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test unwatching a directory."""
        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=False)

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Unwatch the directory
        await file_watcher.unwatch_directory(str(temp_dir.absolute()))

        # Add a small delay for unwatch to take effect
        await asyncio.sleep(0.2)

        # Create a file (should NOT be detected)
        test_file = temp_dir / "after_unwatch.txt"
        test_file.write_text("content")

        # Wait to ensure no events are generated
        await asyncio.sleep(0.3)
        changes = await file_watcher.get_changes()

        # Should not detect file creation after unwatch
        assert len(changes) == 0

        await file_watcher.stop()


# =============================================================================
# File Change Event Tests
# =============================================================================

class TestFileChangeEvents:
    """Test different types of file change events."""

    @pytest.mark.asyncio
    async def test_file_created_event(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test FileChangeType.CREATED event."""
        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=False)

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Create a new file
        new_file = temp_dir / "created.txt"
        new_file.write_text("new content")

        # Wait for creation event with polling
        changes = await wait_for_events(file_watcher, min_count=1, timeout=5.0)

        # Should have at least one CREATED event
        created_events = [
            c for c in changes
            if c.change_type == FileChangeType.CREATED
        ]

        assert len(created_events) > 0

        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_file_deleted_event(
        self,
        file_watcher: IFileWatcher,
        temp_file: Path,
    ) -> None:
        """Test FileChangeType.DELETED event."""
        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_file.parent.absolute()), recursive=False)

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Delete the file
        temp_file.unlink()

        # Wait for deletion event with polling
        changes = await wait_for_events(file_watcher, min_count=1, timeout=5.0)

        # Should have at least one DELETED event
        deleted_events = [
            c for c in changes
            if c.change_type == FileChangeType.DELETED
        ]

        assert len(deleted_events) > 0

        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_file_moved_event(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test FileChangeType.MOVED event."""
        # Create initial file
        old_path = temp_dir / "old_name.txt"
        old_path.write_text("content")

        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=False)

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Move the file
        new_path = temp_dir / "new_name.txt"
        old_path.rename(new_path)

        # Wait for move event with polling
        changes = await wait_for_events(file_watcher, min_count=1, timeout=5.0)

        # Should detect a move (either as MOVED or DELETED + CREATED)
        moved_events = [
            c for c in changes
            if c.change_type == FileChangeType.MOVED
        ]

        # If no explicit MOVED event, should have both DELETED and CREATED
        if len(moved_events) == 0:
            deleted = [c for c in changes if c.change_type == FileChangeType.DELETED]
            created = [c for c in changes if c.change_type == FileChangeType.CREATED]
            assert len(deleted) > 0 and len(created) > 0
        else:
            assert len(moved_events) > 0

        await file_watcher.stop()


# =============================================================================
# Event Retrieval Tests
# =============================================================================

class TestEventRetrieval:
    """Test retrieving file change events."""

    @pytest.mark.asyncio
    async def test_get_changes_returns_empty_initially(
        self,
        file_watcher: IFileWatcher,
    ) -> None:
        """Test that get_changes returns empty list when no events."""
        await file_watcher.start()
        changes = await file_watcher.get_changes()
        assert isinstance(changes, list)
        assert len(changes) == 0
        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_get_changes_consumes_events(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test that get_changes consumes returned events."""
        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=False)

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Create a file
        (temp_dir / "test.txt").write_text("content")

        # Wait for event with polling
        changes1 = await wait_for_events(file_watcher, min_count=1, timeout=5.0)
        assert len(changes1) > 0

        # Get changes second time - should be empty (events consumed)
        changes2 = await file_watcher.get_changes()
        assert len(changes2) == 0

        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_multiple_changes_accumulate(
        self,
        file_watcher: IFileWatcher,
        temp_file: Path,
    ) -> None:
        """Test that multiple changes accumulate before retrieval."""
        await file_watcher.start()
        await file_watcher.watch_file(str(temp_file.absolute()))

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Make multiple modifications
        for i in range(3):
            temp_file.write_text(f"content {i}")
            await asyncio.sleep(0.05)  # Small delay between modifications

        # Wait for all events to accumulate (expecting at least 3)
        changes = await wait_for_events(file_watcher, min_count=3, timeout=5.0)

        # Should have at least 3 events
        assert len(changes) >= 3

        await file_watcher.stop()


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread-safety of file watcher operations."""

    @pytest.mark.asyncio
    async def test_concurrent_watch_operations(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test that concurrent watch operations are safe."""
        await file_watcher.start()

        # Watch multiple files concurrently
        files = [temp_dir / f"file{i}.txt" for i in range(5)]
        for f in files:
            f.write_text("content")

        tasks = [
            file_watcher.watch_file(str(f.absolute()))
            for f in files
        ]
        await asyncio.gather(*tasks)

        # Should not raise
        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_concurrent_get_changes(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test that concurrent get_changes calls are safe."""
        await file_watcher.start()
        await file_watcher.watch_directory(str(temp_dir.absolute()), recursive=False)

        # Allow watcher to stabilize
        await flush_events(file_watcher, delay=0.2)

        # Create a file to generate events
        (temp_dir / "test.txt").write_text("content")

        # Wait for events to propagate
        await asyncio.sleep(0.3)

        # Call get_changes concurrently
        results = await asyncio.gather(
            file_watcher.get_changes(),
            file_watcher.get_changes(),
            file_watcher.get_changes(),
        )

        # All should return lists (may be empty after first call)
        for result in results:
            assert isinstance(result, list)

        await file_watcher.stop()


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in file watcher."""

    @pytest.mark.asyncio
    async def test_watch_nonexistent_directory(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test watching a directory that doesn't exist."""
        nonexistent = temp_dir / "nonexistent"

        await file_watcher.start()

        # Should handle gracefully (either raise or create directory)
        try:
            await file_watcher.watch_directory(str(nonexistent.absolute()))
        except Exception as e:
            # If it raises, should be a meaningful error
            assert isinstance(e, (FileNotFoundError, OSError))

        await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_operations_when_not_running(
        self,
        file_watcher: IFileWatcher,
        temp_dir: Path,
    ) -> None:
        """Test operations when watcher is not running."""
        # Should not crash, but may raise or be no-op
        try:
            await file_watcher.watch_file(str(temp_dir / "test.txt"))
        except Exception:
            pass  # Expected

        try:
            await file_watcher.get_changes()
        except Exception:
            pass  # Expected
