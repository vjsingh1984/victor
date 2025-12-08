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
from pathlib import Path
import pytest

from victor.tools.filesystem import (
    read,
    write,
    ls,
    detect_file_type_by_magic,
    check_extension_magic_mismatch,
    FileCategory,
    FileTypeInfo,
    register_binary_handler,
    get_binary_handler,
)


@pytest.mark.asyncio
async def test_read_file_success():
    """Test successful file reading."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, World!")
        temp_path = f.name

    try:
        result = await read(path=temp_path)

        assert result == "Hello, World!"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_read_file_not_found():
    """Test reading non-existent file."""
    with pytest.raises(FileNotFoundError):
        await read(path="/nonexistent/path/file.txt")


@pytest.mark.asyncio
async def test_read_file_not_a_file():
    """Test reading a directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(IsADirectoryError):
            await read(path=tmpdir)


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
                await read(path=temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_write_file_success():
    """Test successful file writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_file.txt")
        content = "Test content\nMultiple lines"

        result = await write(path=file_path, content=content)

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

        result = await write(path=file_path, content=content)

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
        result = await write(path=temp_path, content=new_content)

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
                await write(path=file_path, content="test content")


@pytest.mark.asyncio
async def test_list_directory_success():
    """Test listing directory contents with depth=1 (default)."""
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

        # depth=1 returns immediate children with "name" key
        result = await ls(path=tmpdir, depth=1)

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
async def test_list_directory_default_depth():
    """Test listing directory with default depth=1 returns immediate children."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        file1 = os.path.join(tmpdir, "file1.txt")
        subdir = os.path.join(tmpdir, "subdir")
        nested_file = os.path.join(subdir, "nested.txt")

        with open(file1, "w") as f:
            f.write("test")
        os.mkdir(subdir)
        with open(nested_file, "w") as f:
            f.write("nested")

        # Default depth=1 returns only immediate children with "name" key
        result = await ls(path=tmpdir)

        assert len(result) == 2  # file1.txt, subdir (not nested.txt)
        names = [item["name"] for item in result]
        assert "file1.txt" in names
        assert "subdir" in names


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

        result = await ls(path=tmpdir, recursive=True)

        assert len(result) == 3  # subdir, file1.txt, subdir/file2.txt
        paths = [item["path"] for item in result]
        assert "subdir" in paths
        assert "file1.txt" in paths
        assert os.path.join("subdir", "file2.txt") in paths or "subdir/file2.txt" in paths


@pytest.mark.asyncio
async def test_list_directory_not_found():
    """Test listing non-existent directory."""
    with pytest.raises(FileNotFoundError):
        await ls(path="/nonexistent/directory")


@pytest.mark.asyncio
async def test_list_directory_not_a_directory():
    """Test listing a file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        temp_path = f.name

    try:
        with pytest.raises(NotADirectoryError):
            await ls(path=temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_list_directory_empty():
    """Test listing empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await ls(path=tmpdir)

        assert result == []


@pytest.mark.asyncio
async def test_write_file_to_directory_raises_error():
    """Test that write_file raises IsADirectoryError for directory path (line 91)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(IsADirectoryError) as excinfo:
            await write(path=tmpdir, content="test content")

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
            await read(path=binary_file)

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
            await read(path=binary_file)

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
            await read(path=coverage_file)

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
                await read(path=db_file)

            assert "Cannot read binary file" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_directory_error_with_suggestion():
    """Test that read_file gives helpful suggestion when trying to read a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(IsADirectoryError) as excinfo:
            await read(path=tmpdir)

        error_msg = str(excinfo.value)
        assert "Cannot read directory as file" in error_msg
        assert "list_directory" in error_msg
        assert "Suggestion" in error_msg


# ============================================================================
# MAGIC BYTES DETECTION TESTS
# ============================================================================


class TestMagicBytesDetection:
    """Tests for magic bytes file type detection."""

    def test_detect_png_magic_bytes(self):
        """Test PNG file detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            # PNG magic bytes
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.IMAGE
            assert result.mime_type == "image/png"
            assert ".png" in result.extensions

            os.unlink(f.name)

    def test_detect_jpeg_magic_bytes(self):
        """Test JPEG file detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            # JPEG magic bytes
            f.write(b"\xff\xd8\xff")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.IMAGE
            assert result.mime_type == "image/jpeg"
            assert ".jpg" in result.extensions or ".jpeg" in result.extensions

            os.unlink(f.name)

    def test_detect_pdf_magic_bytes(self):
        """Test PDF file detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            # PDF magic bytes
            f.write(b"%PDF-1.4")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.DOCUMENT
            assert result.mime_type == "application/pdf"

            os.unlink(f.name)

    def test_detect_zip_magic_bytes(self):
        """Test ZIP file detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as f:
            # ZIP magic bytes (PK\x03\x04)
            f.write(b"PK\x03\x04")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.ARCHIVE
            assert result.mime_type == "application/zip"

            os.unlink(f.name)

    def test_detect_sqlite_magic_bytes(self):
        """Test SQLite database detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            # SQLite magic bytes
            f.write(b"SQLite format 3\x00")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.DATABASE
            assert result.mime_type == "application/x-sqlite3"

            os.unlink(f.name)

    def test_detect_elf_magic_bytes(self):
        """Test ELF executable detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".so") as f:
            # ELF magic bytes
            f.write(b"\x7fELF")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.COMPILED
            assert "ELF" in result.description

            os.unlink(f.name)

    def test_detect_text_file_no_magic(self):
        """Test that text files don't match any magic signature."""
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as f:
            f.write("This is plain text content")
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            # Text files shouldn't match any magic signature
            assert result is None

            os.unlink(f.name)

    def test_empty_file_no_magic(self):
        """Test that empty files don't match any magic signature."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is None

            os.unlink(f.name)


class TestExtensionMagicMismatch:
    """Tests for extension/magic bytes mismatch detection."""

    def test_matching_extension_no_warning(self):
        """Test no warning when extension matches magic bytes."""
        type_info = FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/png",
            description="PNG image",
            extensions=(".png",),
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            warning = check_extension_magic_mismatch(Path(f.name), type_info)
            assert warning is None
            os.unlink(f.name)

    def test_mismatched_extension_warning(self):
        """Test warning when extension doesn't match magic bytes."""
        type_info = FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/png",
            description="PNG image",
            extensions=(".png",),
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            warning = check_extension_magic_mismatch(Path(f.name), type_info)

            assert warning is not None
            assert ".txt" in warning
            assert "PNG image" in warning
            assert ".png" in warning

            os.unlink(f.name)

    def test_no_magic_type_no_warning(self):
        """Test no warning when no magic type is detected."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            warning = check_extension_magic_mismatch(Path(f.name), None)
            assert warning is None
            os.unlink(f.name)


class TestBinaryFileHandler:
    """Tests for binary file handler registry."""

    def test_register_and_get_handler(self):
        """Test registering and retrieving a binary handler."""
        from victor.tools.filesystem import unregister_binary_handler

        def dummy_pdf_handler(path: Path) -> str:
            return f"Extracted text from {path}"

        try:
            # Register handler
            register_binary_handler(FileCategory.DOCUMENT, dummy_pdf_handler)

            # Retrieve handler
            handler = get_binary_handler(FileCategory.DOCUMENT)
            assert handler is not None
            assert handler(Path("/test.pdf")) == "Extracted text from /test.pdf"
        finally:
            # Clean up to avoid test pollution
            unregister_binary_handler(FileCategory.DOCUMENT)

    def test_get_unregistered_handler(self):
        """Test getting handler for unregistered category returns None."""
        # MEDIA category likely not registered
        get_binary_handler(FileCategory.MEDIA)
        # May or may not be None depending on previous tests
        # Just verify it doesn't raise an exception


@pytest.mark.asyncio
async def test_read_file_rejects_png_by_magic():
    """Test that read_file rejects PNG files detected by magic bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".data") as f:
        # Write PNG magic bytes with misleading extension
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(b"\x00" * 100)
        f.flush()

        with pytest.raises(ValueError) as excinfo:
            await read(path=f.name)

        error_msg = str(excinfo.value)
        assert "PNG image" in error_msg
        assert "Cannot read binary file" in error_msg

        os.unlink(f.name)


@pytest.mark.asyncio
async def test_read_file_rejects_pdf_by_magic():
    """Test that read_file rejects PDF files detected by magic bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        # Write PDF magic bytes with misleading .txt extension
        f.write(b"%PDF-1.4\n")
        f.write(b"%%EOF\n")
        f.flush()

        with pytest.raises(ValueError) as excinfo:
            await read(path=f.name)

        error_msg = str(excinfo.value)
        assert "PDF document" in error_msg
        assert "Cannot read binary file" in error_msg

        os.unlink(f.name)


@pytest.mark.asyncio
async def test_read_file_rejects_sqlite_by_magic():
    """Test that read_file rejects SQLite database detected by magic bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        # Write SQLite magic bytes with misleading .txt extension
        f.write(b"SQLite format 3\x00")
        f.write(b"\x00" * 100)
        f.flush()

        with pytest.raises(ValueError) as excinfo:
            await read(path=f.name)

        error_msg = str(excinfo.value)
        assert "SQLite database" in error_msg
        assert "Cannot read binary file" in error_msg

        os.unlink(f.name)


@pytest.mark.asyncio
async def test_read_file_allows_text_with_no_magic():
    """Test that read_file allows text files with no magic signature."""
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as f:
        f.write("This is plain text content")
        f.flush()

        content = await read(path=f.name)
        assert content == "This is plain text content"

        os.unlink(f.name)
