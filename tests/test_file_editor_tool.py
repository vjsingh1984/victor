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

"""Tests for file editor tool integration."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.file_editor_tool import edit


@pytest.mark.asyncio
async def test_file_editor_tool():
    """Test file editor tool operations using edit_files function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"

        # Test 1: Create file
        result = await edit(
            ops=[
                {
                    "type": "create",
                    "path": str(test_file),
                    "content": "def hello():\n    print('Hello, World!')\n",
                }
            ],
            desc="Create test file",
        )
        assert result["success"], f"Create failed: {result.get('error')}"

        # Verify file was created
        assert test_file.exists(), "File was not created"
        content = test_file.read_text()
        assert "Hello, World!" in content, "File content incorrect"

        # Test 2: Modify file
        new_content = "def hello():\n    print('Hello, Victor!')\n"
        result = await edit(
            ops=[
                {
                    "type": "modify",
                    "path": str(test_file),
                    "new_content": new_content,
                }
            ],
            desc="Modify test file",
        )
        assert result["success"], f"Modify failed: {result.get('error')}"

        # Verify modification
        content = test_file.read_text()
        assert "Victor" in content, "File modification failed"

        # Test 3: Preview mode (dry run) - must pass commit=False to not apply
        result = await edit(
            ops=[
                {
                    "type": "modify",
                    "path": str(test_file),
                    "new_content": "# This won't be applied\n",
                }
            ],
            preview=True,
            commit=False,  # Required for true dry run
            desc="Preview test",
        )
        assert result["success"], f"Preview failed: {result.get('error')}"

        # Verify file wasn't changed
        content = test_file.read_text()
        assert "Victor" in content, "Preview mode modified file!"

        # Test 4: Delete file
        result = await edit(
            ops=[
                {
                    "type": "delete",
                    "path": str(test_file),
                }
            ],
            desc="Delete test file",
        )
        assert result["success"], f"Delete failed: {result.get('error')}"

        # Verify file was deleted
        assert not test_file.exists(), "File was not deleted"


@pytest.mark.asyncio
async def test_file_editor_multiple_operations():
    """Test multiple file operations in one call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "file1.py"
        file2 = Path(tmpdir) / "file2.py"

        # Create multiple files
        result = await edit(
            ops=[
                {
                    "type": "create",
                    "path": str(file1),
                    "content": "# File 1\n",
                },
                {
                    "type": "create",
                    "path": str(file2),
                    "content": "# File 2\n",
                },
            ],
            desc="Create multiple files",
        )
        assert result["success"], f"Multiple create failed: {result.get('error')}"

        # Verify both files exist
        assert file1.exists(), "File 1 not created"
        assert file2.exists(), "File 2 not created"


@pytest.mark.asyncio
async def test_file_editor_invalid_operation():
    """Test handling of invalid operations."""
    result = await edit(
        ops=[
            {
                "type": "invalid_op",
                "path": "/nonexistent/file.py",
            }
        ],
        desc="Invalid operation test",
    )
    # Should handle gracefully
    assert isinstance(result, dict)
