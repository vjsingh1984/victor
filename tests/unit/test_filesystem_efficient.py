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

from victor.tools.filesystem import read, ls


class TestReadFileSearch:
    """Tests for read_file search capability."""

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
            f.flush()

            result = await read(f.name, search="def foo")

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

            result = await read(f.name, search="MATCH", ctx=2)

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

            result = await read(f.name, search=r"def \w+\(", regex=True)

            assert "2 matches" in result

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_search_no_matches(self):
        """Test search with no matches."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("line one\nline two\nline three\n")
            f.flush()

            result = await read(f.name, search="nonexistent")

            assert "No matches" in result

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_full_read_without_search(self):
        """Test that full read still works without search."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            content = "line 1\nline 2\nline 3\n"
            f.write(content)
            f.flush()

            result = await read(f.name)

            assert result == content

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_offset_limit_still_works(self):
        """Test that offset/limit functionality still works."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            f.flush()

            result = await read(f.name, offset=1, limit=2)

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

            # With pattern, result is a dict with count, filter, items, truncated
            result = await ls(tmpdir, pattern="*.py", depth=1)

            assert isinstance(result, dict)
            assert result["count"] == 2
            assert result["filter"] == "*.py"
            assert all(".py" in item["name"] for item in result["items"])

    @pytest.mark.asyncio
    async def test_extensions_via_pattern(self):
        """Test filtering by extension using pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "foo.py").touch()
            (Path(tmpdir) / "bar.ts").touch()
            (Path(tmpdir) / "baz.txt").touch()

            # With pattern, result is a dict
            result = await ls(tmpdir, pattern="*.py")

            assert isinstance(result, dict)
            assert result["count"] == 1
            assert result["items"][0]["name"] == "foo.py"

    @pytest.mark.asyncio
    async def test_type_detection_dirs(self):
        """Test that directories are correctly detected with type='directory'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").touch()
            (Path(tmpdir) / "subdir").mkdir()

            result = await ls(tmpdir)

            # Filter directories manually
            dirs = [item for item in result if item["type"] == "directory"]
            assert len(dirs) == 1
            assert dirs[0]["name"] == "subdir"

    @pytest.mark.asyncio
    async def test_type_detection_files(self):
        """Test that files are correctly detected with type='file'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").touch()
            (Path(tmpdir) / "subdir").mkdir()

            result = await ls(tmpdir)

            # Filter files manually
            files = [item for item in result if item["type"] == "file"]
            assert len(files) == 1
            assert files[0]["name"] == "file.py"

    @pytest.mark.asyncio
    async def test_limit_parameter(self):
        """Test limit parameter truncates results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(20):
                (Path(tmpdir) / f"file{i}.py").touch()

            result = await ls(tmpdir, pattern="*.py", limit=5)

            # Result is a dict when using pattern
            assert isinstance(result, dict)
            assert result["count"] == 5
            assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_unfiltered_returns_list(self):
        """Test that unfiltered call returns simple list for backwards compat."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").touch()
            (Path(tmpdir) / "subdir").mkdir()

            result = await ls(tmpdir)

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

            result = await ls(tmpdir, recursive=True, pattern="*.py")

            # With pattern, result is a dict
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
            full_result = await read(f.name)

            # Search mode
            search_result = await read(f.name, search="TARGET_FUNCTION")

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

            # Full listing (returns list)
            full_result = await ls(tmpdir)

            # Filtered listing using pattern (returns dict)
            filtered_result = await ls(tmpdir, pattern="*.py")

            # Filtered should have fewer items
            assert filtered_result["count"] < len(full_result)
            assert filtered_result["count"] == 50


class TestTraversalOrdering:
    """Tests for directory traversal ordering (breadth-first by default)."""

    @pytest.mark.asyncio
    async def test_breadth_first_ordering_default(self):
        """Test that default traversal visits all siblings before children (BFS)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure:
            # root/
            #   a/
            #     a1.txt
            #   b/
            #     b1.txt
            #   c.txt
            a_dir = Path(tmpdir) / "a"
            b_dir = Path(tmpdir) / "b"
            a_dir.mkdir()
            b_dir.mkdir()
            (a_dir / "a1.txt").touch()
            (b_dir / "b1.txt").touch()
            (Path(tmpdir) / "c.txt").touch()

            result = await ls(tmpdir, depth=2)

            # BFS: all depth-1 items should come before any depth-2 items
            paths = [item["path"] for item in result]

            # Separate depth-1 and depth-2 items
            depth1_items = [p for p in paths if "/" not in p and "\\" not in p]
            depth2_items = [p for p in paths if "/" in p or "\\" in p]

            # Verify depth-1 items all come before depth-2 items in the result
            depth1_indices = [paths.index(p) for p in depth1_items]
            depth2_indices = [paths.index(p) for p in depth2_items]

            if depth2_indices:  # Only check if there are depth-2 items
                assert max(depth1_indices) < min(depth2_indices), (
                    f"BFS should have all depth-1 items before depth-2. " f"Got paths: {paths}"
                )

    @pytest.mark.asyncio
    async def test_traversal_sees_all_top_level_first(self):
        """Test that traversal sees top-level directories before nested files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure:
            # root/
            #   a/
            #     a1.txt
            #   b/
            #     b1.txt
            a_dir = Path(tmpdir) / "a"
            b_dir = Path(tmpdir) / "b"
            a_dir.mkdir()
            b_dir.mkdir()
            (a_dir / "a1.txt").touch()
            (b_dir / "b1.txt").touch()

            result = await ls(tmpdir, depth=2)
            paths = [item["path"] for item in result]

            # Both 'a' and 'b' directories should appear before any nested files
            a_idx = paths.index("a")
            b_idx = paths.index("b")
            nested_indices = [i for i, p in enumerate(paths) if "/" in p or "\\" in p]

            if nested_indices:
                assert a_idx < min(nested_indices), "a should appear before nested files"
                assert b_idx < min(nested_indices), "b should appear before nested files"

    @pytest.mark.asyncio
    async def test_truncation_sees_top_level_dirs(self):
        """Test that with limit, we see top-level directories first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 5 directories each with 5 files
            for i in range(5):
                d = Path(tmpdir) / f"dir{i}"
                d.mkdir()
                for j in range(5):
                    (d / f"file{j}.txt").touch()

            # With limit=8 and BFS, should see all 5 directories first
            result = await ls(tmpdir, depth=2, limit=8)

            # Result is a dict with 'items' when using non-default depth/limit
            assert isinstance(result, dict)
            items = result["items"]

            # Should see all 5 top-level directories
            top_level_dirs = [
                item["path"]
                for item in items
                if item["type"] == "directory"
                and "/" not in item["path"]
                and "\\" not in item["path"]
            ]

            # BFS should capture all 5 dirs in the first 8 entries
            assert (
                len(top_level_dirs) == 5
            ), f"BFS should see all 5 top dirs with limit=8. Got: {top_level_dirs}"


class TestDepthParameter:
    """Tests for the depth parameter."""

    @pytest.mark.asyncio
    async def test_depth_1_only_immediate_children(self):
        """Test depth=1 returns only immediate children."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").touch()
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").touch()

            result = await ls(tmpdir, depth=1)

            assert len(result) == 2
            names = [item["name"] for item in result]
            assert "file.txt" in names
            assert "subdir" in names
            # nested.txt should NOT be included

    @pytest.mark.asyncio
    async def test_depth_2_includes_nested(self):
        """Test depth=2 includes one level of nesting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").touch()
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").touch()
            nested_dir = subdir / "nested_dir"
            nested_dir.mkdir()
            (nested_dir / "deep.txt").touch()

            result = await ls(tmpdir, depth=2)

            paths = [item["path"] for item in result]
            assert "file.txt" in paths
            assert "subdir" in paths
            assert "subdir/nested.txt" in paths or "subdir\\nested.txt" in paths
            assert "subdir/nested_dir" in paths or "subdir\\nested_dir" in paths
            # deep.txt should NOT be included (depth 3)

    @pytest.mark.asyncio
    async def test_depth_3_includes_two_levels(self):
        """Test depth=3 includes two levels of nesting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            nested_dir = subdir / "nested_dir"
            nested_dir.mkdir()
            (nested_dir / "deep.txt").touch()

            result = await ls(tmpdir, depth=3)

            paths = [item["path"] for item in result]
            assert any("deep.txt" in p for p in paths)

    @pytest.mark.asyncio
    async def test_depth_0_returns_empty(self):
        """Test depth=0 returns nothing (no items at depth 0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").touch()

            # depth=0 should be treated as "don't explore"
            # Current implementation starts at depth 1, so depth=0 means no entries
            result = await ls(tmpdir, depth=0)

            # With depth=0, we get nothing since the iterator never yields
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_recursive_ignores_depth(self):
        """Test that recursive=True ignores depth parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 4 levels deep
            d = Path(tmpdir)
            for i in range(4):
                d = d / f"level{i}"
                d.mkdir()
            (d / "deep.txt").touch()

            # depth=1 with recursive=True should still find deep.txt
            result = await ls(tmpdir, recursive=True, depth=1)

            paths = [item["path"] for item in result]
            assert any("deep.txt" in p for p in paths)


class TestLimitParameter:
    """Tests for the limit parameter."""

    @pytest.mark.asyncio
    async def test_limit_truncates(self):
        """Test limit parameter truncates results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(100):
                (Path(tmpdir) / f"file{i:03d}.txt").touch()

            result = await ls(tmpdir, limit=10)

            # Non-default limit returns dict with items
            assert isinstance(result, dict)
            assert result["count"] == 10
            assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_limit_with_recursive(self):
        """Test limit works with recursive traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many nested files
            for i in range(5):
                d = Path(tmpdir) / f"dir{i}"
                d.mkdir()
                for j in range(10):
                    (d / f"file{j}.txt").touch()

            result = await ls(tmpdir, recursive=True, limit=15)

            # Non-default limit returns dict with items
            assert isinstance(result, dict)
            assert result["count"] == 15

    @pytest.mark.asyncio
    async def test_limit_default_1000(self):
        """Test default limit is 1000."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fewer than 1000 files
            for i in range(50):
                (Path(tmpdir) / f"file{i}.txt").touch()

            result = await ls(tmpdir)

            # Default behavior returns list
            assert isinstance(result, list)
            assert len(result) == 50

    @pytest.mark.asyncio
    async def test_limit_string_coercion(self):
        """Test limit handles string input from models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                (Path(tmpdir) / f"file{i}.txt").touch()

            # Simulate model passing string - non-default returns dict
            result = await ls(tmpdir, limit="5")

            assert isinstance(result, dict)
            assert result["count"] == 5


class TestListDirectoryEdgeCases:
    """Edge case tests for list_directory."""

    @pytest.mark.asyncio
    async def test_empty_directory(self):
        """Test listing empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await ls(tmpdir)
            assert result == []

    @pytest.mark.asyncio
    async def test_permission_error_handled(self):
        """Test permission errors are silently handled."""
        import os
        import stat

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create accessible and inaccessible directories
            accessible = Path(tmpdir) / "accessible"
            restricted = Path(tmpdir) / "restricted"
            accessible.mkdir()
            restricted.mkdir()
            (accessible / "file.txt").touch()
            (restricted / "secret.txt").touch()

            # Make restricted unreadable (skip on Windows)
            if os.name != "nt":
                os.chmod(restricted, 0o000)
                try:
                    result = await ls(tmpdir, depth=2)
                    # Should still get accessible results
                    paths = [item.get("path", item.get("name")) for item in result]
                    assert "accessible" in paths
                finally:
                    os.chmod(restricted, stat.S_IRWXU)

    @pytest.mark.asyncio
    async def test_depth_entry_includes_depth_info(self):
        """Test that results include depth information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").touch()
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").touch()

            result = await ls(tmpdir, depth=2)

            # Check depth info is included
            for item in result:
                assert "depth" in item
                assert isinstance(item["depth"], int)

            # Verify depth values
            depth_by_name = {}
            for item in result:
                name = item.get("path", item.get("name"))
                depth_by_name[name] = item["depth"]

            assert depth_by_name.get("file.txt") == 1 or depth_by_name.get("file.txt", 0) == 1
            assert depth_by_name.get("subdir") == 1 or depth_by_name.get("subdir", 0) == 1

    @pytest.mark.asyncio
    async def test_sorted_output(self):
        """Test that output is sorted alphabetically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create in random order
            (Path(tmpdir) / "zebra.txt").touch()
            (Path(tmpdir) / "apple.txt").touch()
            (Path(tmpdir) / "mango.txt").touch()

            result = await ls(tmpdir)
            names = [item["name"] for item in result]

            assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_mixed_files_and_dirs(self):
        """Test correct type detection for mixed content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").touch()
            (Path(tmpdir) / "subdir").mkdir()
            (Path(tmpdir) / "another.py").touch()

            result = await ls(tmpdir)

            types = {item["name"]: item["type"] for item in result}
            assert types["file.txt"] == "file"
            assert types["subdir"] == "directory"
            assert types["another.py"] == "file"

    @pytest.mark.asyncio
    async def test_path_key_for_recursive(self):
        """Test that recursive results use 'path' key with relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "file.txt").touch()

            result = await ls(tmpdir, recursive=True)

            # Should use 'path' key with relative path
            for item in result:
                assert "path" in item
                # Path should not be absolute
                assert not Path(item["path"]).is_absolute()

    @pytest.mark.asyncio
    async def test_name_key_for_shallow(self):
        """Test that depth=1 results use 'name' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").touch()

            result = await ls(tmpdir, depth=1)

            # Should use 'name' key
            for item in result:
                assert "name" in item

    @pytest.mark.asyncio
    async def test_depth_string_coercion(self):
        """Test depth handles string input from models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").touch()
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").touch()

            # Simulate model passing string
            result = await ls(tmpdir, depth="2")

            # Should include nested file
            paths = [item["path"] for item in result]
            assert any("nested.txt" in p for p in paths)
