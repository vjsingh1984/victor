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

"""Tests for file_editor_tool module."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.file_editor_tool import edit_files


class TestEditFiles:
    """Tests for edit_files function."""

    @pytest.mark.asyncio
    async def test_edit_files_empty_operations(self):
        """Test editing with empty operations."""
        result = await edit_files(operations=[])
        assert result["success"] is False
        assert "No operations" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_files_invalid_operation(self):
        """Test editing with invalid operation type."""
        result = await edit_files(operations=[{"path": "/tmp/test.txt"}])
        assert result["success"] is False
        assert "type" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_edit_files_create_success(self):
        """Test successful file creation."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/new_file.txt"

        try:
            result = await edit_files(
                operations=[{"type": "create", "path": temp_path, "content": "Hello World"}]
            )
            assert result["success"] is True
            assert result["operations_queued"] == 1

            # Verify file content
            with open(temp_path) as f:
                content = f.read()
            assert content == "Hello World"
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_edit_files_modify_nonexistent_file(self):
        """Test modifying nonexistent file."""
        result = await edit_files(
            operations=[
                {"type": "modify", "path": "/nonexistent/file.txt", "content": "new content"}
            ]
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_edit_files_preview_mode(self):
        """Test preview mode."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/test.txt"
        Path(temp_path).write_text("original content")

        try:
            result = await edit_files(
                operations=[{"type": "modify", "path": temp_path, "content": "updated content"}],
                preview=True,
                auto_commit=False,
            )
            # Preview should succeed but not apply changes
            assert result["success"] is True

            # Verify file content unchanged
            with open(temp_path) as f:
                content = f.read()
            assert content == "original content"
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_edit_files_json_string_operations(self):
        """Test passing operations as JSON string."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/json_test.txt"

        try:
            import json

            ops_json = json.dumps([{"type": "create", "path": temp_path, "content": "From JSON"}])
            result = await edit_files(operations=ops_json)
            assert result["success"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_edit_files_invalid_json_string(self):
        """Test invalid JSON string operations."""
        result = await edit_files(operations="not valid json")
        assert result["success"] is False
        assert "JSON" in result["error"]
