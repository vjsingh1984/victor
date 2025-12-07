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

"""Tests for token-efficient filesystem tools."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.filesystem import read_file, list_directory


class TestReadFileSearch:
    """Tests for read_file search capability."""

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
            f.flush()

            result = await read_file(f.name, search="def foo")

            assert "1 matches" in result
            assert "def foo" in result
            # Should not include full file
            assert "def bar" not in result or "context" in result.lower()

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_search_with_context(self):
        """Test search with context lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("line 1\nline 2\nMATCH HERE\nline 4\nline 5\n")
            f.flush()

            result = await read_file(f.name, search="MATCH", context_lines=2)

            assert "MATCH HERE" in result
            # Should include context
            assert "line 2" in result or "line 4" in result

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_search_regex(self):
        """Test regex search."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    pass\ndef bar(x):\n    pass\n")
            f.flush()

            result = await read_file(f.name, search=r"def \w+\(", regex=True)

            assert "2 matches" in result

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_search_no_matches(self):
        """Test search with no matches."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("line one\nline two\nline three\n")
            f.flush()

            result = await read_file(f.name, search="nonexistent")

            assert "No matches" in result

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_full_read_without_search(self):
        """Test that full read still works without search."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            content = "line 1\nline 2\nline 3\n"
            f.write(content)
            f.flush()

            result = await read_file(f.name)

            assert result == content

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_offset_limit_still_works(self):
        """Test that offset/limit functionality still works."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            f.flush()

            result = await read_file(f.name, offset=1, limit=2)

            assert "line 2" in result
            assert "line 3" in result
            assert "Lines 2-3" in result

            Path(f.name).unlink()


class TestListDirectoryFilters:
    """Tests for list_directory filtering."""

    @pytest.mark.asyncio
    async def test_pattern_filter(self):
        """Test glob pattern filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "foo.py").touch()
            (Path(tmpdir) / "bar.py").touch()
            (Path(tmpdir) / "baz.txt").touch()
            (Path(tmpdir) / "qux.js").touch()

            result = await list_directory(tmpdir, pattern="*.py")

            # Should return dict with metadata when filtered
            assert isinstance(result, dict)
            assert result["count"] == 2
            assert all(".py" in item["name"] for item in result["items"])

    @pytest.mark.asyncio
    async def test_extensions_filter(self):
        """Test extensions filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "foo.py").touch()
            (Path(tmpdir) / "bar.ts").touch()
            (Path(tmpdir) / "baz.txt").touch()

            result = await list_directory(tmpdir, extensions="py,ts")

            assert isinstance(result, dict)
            assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_dirs_only_filter(self):
        """Test dirs_only filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").touch()
            (Path(tmpdir) / "subdir").mkdir()

            result = await list_directory(tmpdir, dirs_only=True)

            assert isinstance(result, dict)
            assert result["count"] == 1
            assert result["items"][0]["type"] == "directory"

    @pytest.mark.asyncio
    async def test_files_only_filter(self):
        """Test files_only filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").touch()
            (Path(tmpdir) / "subdir").mkdir()

            result = await list_directory(tmpdir, files_only=True)

            assert isinstance(result, dict)
            assert result["count"] == 1
            assert result["items"][0]["type"] == "file"

    @pytest.mark.asyncio
    async def test_max_items_limit(self):
        """Test max_items limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(20):
                (Path(tmpdir) / f"file{i}.py").touch()

            result = await list_directory(tmpdir, pattern="*.py", max_items=5)

            assert isinstance(result, dict)
            assert result["count"] == 5
            assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_unfiltered_returns_list(self):
        """Test that unfiltered call returns simple list for backwards compat."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").touch()
            (Path(tmpdir) / "subdir").mkdir()

            result = await list_directory(tmpdir)

            # Should return list (not dict) for backwards compatibility
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_recursive_with_pattern(self):
        """Test recursive with pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "foo.py").touch()
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "bar.py").touch()
            (subdir / "baz.txt").touch()

            result = await list_directory(tmpdir, recursive=True, pattern="*.py")

            assert isinstance(result, dict)
            assert result["count"] == 2


class TestTokenEfficiencyComparison:
    """Tests demonstrating token efficiency improvements."""

    @pytest.mark.asyncio
    async def test_search_vs_full_read_efficiency(self):
        """Demonstrate token savings with search mode."""
        # Create a larger file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write 100 lines
            for i in range(100):
                if i == 50:
                    f.write("TARGET_FUNCTION_TO_FIND = True\n")
                else:
                    f.write(f"# Comment line {i} with some content\n")
            f.flush()

            # Full read
            full_result = await read_file(f.name)

            # Search mode
            search_result = await read_file(f.name, search="TARGET_FUNCTION")

            # Search should be much smaller
            assert len(search_result) < len(full_result) / 5
            assert "TARGET_FUNCTION" in search_result

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_filtered_list_efficiency(self):
        """Demonstrate token savings with filtered listing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mixed files
            for i in range(50):
                (Path(tmpdir) / f"file{i}.py").touch()
            for i in range(50):
                (Path(tmpdir) / f"file{i}.txt").touch()

            # Full listing
            full_result = await list_directory(tmpdir)

            # Filtered listing
            filtered_result = await list_directory(tmpdir, extensions="py")

            # Filtered should be smaller
            assert len(filtered_result["items"]) < len(full_result)
            assert filtered_result["count"] == 50
