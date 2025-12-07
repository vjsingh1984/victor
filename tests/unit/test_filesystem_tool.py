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

"""Tests for filesystem tool."""

import os
import tempfile
import pytest

from victor.tools.filesystem import read_file, write_file, list_directory


@pytest.mark.asyncio
async def test_read_file_success():
    """Test successful file reading."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, World!")
        temp_path = f.name

    try:
        result = await read_file(path=temp_path)

        assert result == "Hello, World!"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_read_file_not_found():
    """Test reading non-existent file."""
    with pytest.raises(FileNotFoundError):
        await read_file(path="/nonexistent/path/file.txt")


@pytest.mark.asyncio
async def test_read_file_not_a_file():
    """Test reading a directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(IsADirectoryError):
            await read_file(path=tmpdir)


@pytest.mark.asyncio
async def test_read_file_exception_handling():
    """Test exception handling in read_file (permission error)."""
    from unittest.mock import patch

    # Create a real file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("test")
        temp_path = f.name

    try:
        # Mock aiofiles.open to raise an exception
        with patch("aiofiles.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                await read_file(path=temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_write_file_success():
    """Test successful file writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_file.txt")
        content = "Test content\nMultiple lines"

        result = await write_file(path=file_path, content=content)

        assert "Successfully created" in result
        assert f"{len(content)} characters" in result
        assert "/undo" in result  # Verify undo hint is shown

        # Verify file was actually written
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            written_content = f.read()
            assert written_content == content


@pytest.mark.asyncio
async def test_write_file_creates_directories():
    """Test that write_file creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested path that doesn't exist
        file_path = os.path.join(tmpdir, "subdir1", "subdir2", "test.txt")
        content = "Test content"

        result = await write_file(path=file_path, content=content)

        assert "Successfully created" in result

        # Verify directories and file were created
        assert os.path.exists(file_path)
        assert os.path.isfile(file_path)


@pytest.mark.asyncio
async def test_write_file_overwrites_existing():
    """Test that write_file overwrites existing files."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Original content")
        temp_path = f.name

    try:
        new_content = "New content"
        result = await write_file(path=temp_path, content=new_content)

        assert "Successfully modified" in result  # File existed, so it's a modification

        # Verify content was overwritten
        with open(temp_path, "r") as f:
            assert f.read() == new_content
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_write_file_exception_handling():
    """Test exception handling in write_file (permission error)."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")

        # Mock aiofiles.open to raise an exception
        with patch("aiofiles.open", side_effect=PermissionError("Write denied")):
            with pytest.raises(PermissionError):
                await write_file(path=file_path, content="test content")


@pytest.mark.asyncio
async def test_list_directory_success():
    """Test listing directory contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(tmpdir, "file2.txt")
        subdir = os.path.join(tmpdir, "subdir")

        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")
        os.mkdir(subdir)

        result = await list_directory(path=tmpdir)

        assert len(result) == 3
        names = [item["name"] for item in result]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

        # Check types
        types = {item["name"]: item["type"] for item in result}
        assert types["file1.txt"] == "file"
        assert types["file2.txt"] == "file"
        assert types["subdir"] == "directory"


@pytest.mark.asyncio
async def test_list_directory_recursive():
    """Test listing directory contents recursively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        subdir = os.path.join(tmpdir, "subdir")
        os.mkdir(subdir)
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(subdir, "file2.txt")

        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")

        result = await list_directory(path=tmpdir, recursive=True)

        assert len(result) == 3  # subdir, file1.txt, subdir/file2.txt
        paths = [item["path"] for item in result]
        assert "subdir" in paths
        assert "file1.txt" in paths
        assert os.path.join("subdir", "file2.txt") in paths or "subdir/file2.txt" in paths


@pytest.mark.asyncio
async def test_list_directory_not_found():
    """Test listing non-existent directory."""
    with pytest.raises(FileNotFoundError):
        await list_directory(path="/nonexistent/directory")


@pytest.mark.asyncio
async def test_list_directory_not_a_directory():
    """Test listing a file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        temp_path = f.name

    try:
        with pytest.raises(NotADirectoryError):
            await list_directory(path=temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_list_directory_empty():
    """Test listing empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await list_directory(path=tmpdir)

        assert result == []


@pytest.mark.asyncio
async def test_write_file_to_directory_raises_error():
    """Test that write_file raises IsADirectoryError for directory path (line 91)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(IsADirectoryError) as excinfo:
            await write_file(path=tmpdir, content="test content")

        assert "Cannot write to directory" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_binary_extension_rejected():
    """Test that read_file rejects files with binary extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with a binary extension
        binary_file = os.path.join(tmpdir, "test.pyc")
        with open(binary_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03")

        with pytest.raises(ValueError) as excinfo:
            await read_file(path=binary_file)

        assert "Cannot read binary file" in str(excinfo.value)
        assert ".pyc" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_binary_content_rejected():
    """Test that read_file rejects files with binary content (non-UTF-8)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a text file but with binary content
        binary_file = os.path.join(tmpdir, "test.txt")
        with open(binary_file, "wb") as f:
            # Write invalid UTF-8 bytes
            f.write(b"\x80\x81\x82\x83")

        with pytest.raises(ValueError) as excinfo:
            await read_file(path=binary_file)

        assert "Cannot read file" in str(excinfo.value)
        assert "binary" in str(excinfo.value).lower()
        assert "non-UTF-8 content" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_coverage_file_rejected():
    """Test that read_file rejects .coverage files (SQLite databases)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        coverage_file = os.path.join(tmpdir, ".coverage")
        with open(coverage_file, "wb") as f:
            f.write(b"SQLite format 3\x00")

        with pytest.raises(ValueError) as excinfo:
            await read_file(path=coverage_file)

        assert "Cannot read binary file" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_db_extension_rejected():
    """Test that read_file rejects .db and .sqlite files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in [".db", ".sqlite", ".sqlite3"]:
            db_file = os.path.join(tmpdir, f"test{ext}")
            with open(db_file, "wb") as f:
                f.write(b"\x00\x01\x02\x03")

            with pytest.raises(ValueError) as excinfo:
                await read_file(path=db_file)

            assert "Cannot read binary file" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_directory_error_with_suggestion():
    """Test that read_file gives helpful suggestion when trying to read a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(IsADirectoryError) as excinfo:
            await read_file(path=tmpdir)

        error_msg = str(excinfo.value)
        assert "Cannot read directory as file" in error_msg
        assert "list_directory" in error_msg
        assert "Suggestion" in error_msg
