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

"""Tests for the change tracker undo/redo system."""

import os
import tempfile
import pytest

from victor.agent.change_tracker import (
    FileChangeHistory,
    ChangeType,
    FileChange,
    ChangeGroup,
    get_change_tracker,
    reset_change_tracker,
)


@pytest.fixture
def tracker():
    """Create a fresh change tracker for each test."""
    from pathlib import Path

    reset_change_tracker()
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FileChangeHistory(storage_dir=Path(tmpdir), max_history=50)
        yield tracker


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_types_exist(self):
        """Verify all change types exist."""
        assert ChangeType.CREATE.value == "create"
        assert ChangeType.MODIFY.value == "modify"
        assert ChangeType.DELETE.value == "delete"
        assert ChangeType.RENAME.value == "rename"


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_create_file_change(self):
        """Test creating a FileChange instance."""
        import time

        change = FileChange(
            id="test-change-1",
            change_type=ChangeType.CREATE,
            file_path="/tmp/test.py",
            timestamp=time.time(),
            tool_name="write_file",
            tool_args={"path": "/tmp/test.py"},
            original_content=None,
            new_content="print('hello')",
        )

        assert change.file_path == "/tmp/test.py"
        assert change.change_type == ChangeType.CREATE
        assert change.original_content is None
        assert change.new_content == "print('hello')"
        assert change.tool_name == "write_file"

    def test_modify_file_change(self):
        """Test creating a modification change."""
        import time

        change = FileChange(
            id="test-change-2",
            change_type=ChangeType.MODIFY,
            file_path="/tmp/test.py",
            timestamp=time.time(),
            tool_name="edit_files",
            tool_args={},
            original_content="old content",
            new_content="new content",
        )

        assert change.change_type == ChangeType.MODIFY
        assert change.original_content == "old content"
        assert change.new_content == "new content"

    def test_rename_file_change(self):
        """Test creating a rename change."""
        import time

        change = FileChange(
            id="test-change-3",
            change_type=ChangeType.RENAME,
            file_path="/tmp/new.py",
            timestamp=time.time(),
            tool_name="edit_files",
            tool_args={},
            original_path="/tmp/old.py",
        )

        assert change.change_type == ChangeType.RENAME
        assert change.original_path == "/tmp/old.py"


class TestChangeGroup:
    """Tests for ChangeGroup dataclass."""

    def test_create_change_group(self):
        """Test creating a ChangeGroup."""
        import time

        changes = [
            FileChange(
                id="test-change-1",
                change_type=ChangeType.CREATE,
                file_path="/tmp/test.py",
                timestamp=time.time(),
                tool_name="write_file",
                tool_args={},
                new_content="content",
            )
        ]
        group = ChangeGroup(
            id="test-123",
            changes=changes,
            tool_name="write_file",
            description="Create test file",
        )

        assert group.id == "test-123"
        assert group.tool_name == "write_file"
        assert group.description == "Create test file"
        assert len(group.changes) == 1
        assert group.timestamp is not None


class TestFileChangeHistory:
    """Tests for the FileChangeHistory class."""

    def test_begin_change_group(self, tracker):
        """Test beginning a change group."""
        group_id = tracker.begin_change_group("test_tool", "Test description")

        assert group_id is not None
        assert tracker._current_group is not None
        assert tracker._current_group.tool_name == "test_tool"
        assert tracker._current_group.description == "Test description"

    def test_record_change(self, tracker):
        """Test recording a change."""
        tracker.begin_change_group("write_file", "Create file")
        tracker.record_change(
            file_path="/tmp/test.py",
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="print('hello')",
            tool_name="write_file",
        )

        assert len(tracker._current_group.changes) == 1
        change = tracker._current_group.changes[0]
        assert change.file_path == "/tmp/test.py"
        assert change.change_type == ChangeType.CREATE

    def test_commit_change_group(self, tracker):
        """Test committing a change group."""
        tracker.begin_change_group("write_file", "Create file")
        tracker.record_change(
            file_path="/tmp/test.py",
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content",
            tool_name="write_file",
        )
        group = tracker.commit_change_group()

        assert group is not None
        assert tracker._current_group is None
        assert tracker.can_undo()

    def test_can_undo_empty(self, tracker):
        """Test can_undo when no changes exist."""
        assert not tracker.can_undo()

    def test_can_redo_empty(self, tracker):
        """Test can_redo when no undone changes exist."""
        assert not tracker.can_redo()


class TestUndoRedo:
    """Tests for undo/redo functionality with actual file operations."""

    def test_undo_file_create(self, tracker, temp_dir):
        """Test undoing a file creation."""
        file_path = os.path.join(temp_dir, "test.py")
        content = "print('hello')"

        # Create the file
        with open(file_path, "w") as f:
            f.write(content)

        # Record the creation
        tracker.begin_change_group("write_file", "Create test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content=content,
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Verify file exists
        assert os.path.exists(file_path)

        # Undo the creation
        success, message, files = tracker.undo()

        assert success
        assert not os.path.exists(file_path)
        assert file_path in files

    def test_redo_file_create(self, tracker, temp_dir):
        """Test redoing a file creation after undo."""
        file_path = os.path.join(temp_dir, "test.py")
        content = "print('hello')"

        # Create the file
        with open(file_path, "w") as f:
            f.write(content)

        # Record the creation
        tracker.begin_change_group("write_file", "Create test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content=content,
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Undo
        tracker.undo()
        assert not os.path.exists(file_path)

        # Redo
        success, message, files = tracker.redo()

        assert success
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            assert f.read() == content

    def test_undo_file_modify(self, tracker, temp_dir):
        """Test undoing a file modification."""
        file_path = os.path.join(temp_dir, "test.py")
        original = "original content"
        modified = "modified content"

        # Create original file
        with open(file_path, "w") as f:
            f.write(original)

        # Modify and record
        with open(file_path, "w") as f:
            f.write(modified)

        tracker.begin_change_group("write_file", "Modify test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.MODIFY,
            original_content=original,
            new_content=modified,
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Verify modified content
        with open(file_path, "r") as f:
            assert f.read() == modified

        # Undo the modification
        success, message, files = tracker.undo()

        assert success
        with open(file_path, "r") as f:
            assert f.read() == original

    def test_undo_file_delete(self, tracker, temp_dir):
        """Test undoing a file deletion."""
        file_path = os.path.join(temp_dir, "test.py")
        content = "file content"

        # Create and then delete the file
        with open(file_path, "w") as f:
            f.write(content)

        # Record and delete
        tracker.begin_change_group("edit_files", "Delete test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.DELETE,
            original_content=content,
            new_content=None,
            tool_name="edit_files",
        )
        tracker.commit_change_group()

        os.remove(file_path)
        assert not os.path.exists(file_path)

        # Undo the deletion (restore the file)
        success, message, files = tracker.undo()

        assert success
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            assert f.read() == content

    def test_multiple_undo(self, tracker, temp_dir):
        """Test multiple undo operations."""
        file1 = os.path.join(temp_dir, "file1.py")
        file2 = os.path.join(temp_dir, "file2.py")

        # Create first file
        with open(file1, "w") as f:
            f.write("content1")

        tracker.begin_change_group("write_file", "Create file1")
        tracker.record_change(
            file_path=file1,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content1",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Create second file
        with open(file2, "w") as f:
            f.write("content2")

        tracker.begin_change_group("write_file", "Create file2")
        tracker.record_change(
            file_path=file2,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content2",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        assert os.path.exists(file1)
        assert os.path.exists(file2)

        # Undo second
        success1, _, _ = tracker.undo()
        assert success1
        assert os.path.exists(file1)
        assert not os.path.exists(file2)

        # Undo first
        success2, _, _ = tracker.undo()
        assert success2
        assert not os.path.exists(file1)
        assert not os.path.exists(file2)

    def test_undo_clears_redo_on_new_change(self, tracker, temp_dir):
        """Test that new changes clear the redo stack."""
        file1 = os.path.join(temp_dir, "file1.py")
        file2 = os.path.join(temp_dir, "file2.py")

        # Create and commit first file
        with open(file1, "w") as f:
            f.write("content1")

        tracker.begin_change_group("write_file", "Create file1")
        tracker.record_change(
            file_path=file1,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content1",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Undo
        tracker.undo()
        assert tracker.can_redo()

        # New change should clear redo stack
        with open(file2, "w") as f:
            f.write("content2")

        tracker.begin_change_group("write_file", "Create file2")
        tracker.record_change(
            file_path=file2,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content2",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        assert not tracker.can_redo()


class TestGetChangeTracker:
    """Tests for singleton pattern."""

    def test_get_change_tracker_returns_same_instance(self):
        """Test that get_change_tracker returns the same instance."""
        reset_change_tracker()
        tracker1 = get_change_tracker()
        tracker2 = get_change_tracker()
        assert tracker1 is tracker2

    def test_reset_change_tracker(self):
        """Test resetting the tracker creates a new instance."""
        reset_change_tracker()
        tracker1 = get_change_tracker()
        reset_change_tracker()
        tracker2 = get_change_tracker()
        assert tracker1 is not tracker2


class TestHistory:
    """Tests for change history."""

    def test_get_history_empty(self, tracker):
        """Test getting history when empty."""
        history = tracker.get_history()
        assert history == []

    def test_get_history_with_changes(self, tracker, temp_dir):
        """Test getting history with changes."""
        file_path = os.path.join(temp_dir, "test.py")

        with open(file_path, "w") as f:
            f.write("content")

        tracker.begin_change_group("write_file", "Create test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        history = tracker.get_history()

        assert len(history) == 1
        assert history[0]["tool_name"] == "write_file"
        assert history[0]["description"] == "Create test.py"
        assert history[0]["undone"] is False  # Not undone = applied

    def test_get_history_limit(self, tracker, temp_dir):
        """Test history limit."""
        # Create multiple changes
        for i in range(5):
            file_path = os.path.join(temp_dir, f"test{i}.py")
            with open(file_path, "w") as f:
                f.write(f"content{i}")

            tracker.begin_change_group("write_file", f"Create test{i}.py")
            tracker.record_change(
                file_path=file_path,
                change_type=ChangeType.CREATE,
                original_content=None,
                new_content=f"content{i}",
                tool_name="write_file",
            )
            tracker.commit_change_group()

        history = tracker.get_history(limit=3)
        assert len(history) == 3
