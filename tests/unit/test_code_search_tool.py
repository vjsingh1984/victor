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

"""Tests for code_search_tool module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from victor.tools.code_search_tool import (
    _latest_mtime,
    _gather_files,
    _keyword_score,
    code_search,
)


class TestLatestMtime:
    """Tests for _latest_mtime function."""

    def test_latest_mtime(self):
        """Test getting latest modification time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/file1.py").write_text("test")
            mtime = _latest_mtime(Path(tmpdir))
            assert mtime > 0

    def test_latest_mtime_empty_dir(self):
        """Test latest mtime on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mtime = _latest_mtime(Path(tmpdir))
            assert mtime == 0.0


class TestGatherFiles:
    """Tests for _gather_files function."""

    def test_gather_files_default_extensions(self):
        """Test gathering files with default extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/file.py").write_text("test")
            Path(f"{tmpdir}/file.txt").write_text("test")
            Path(f"{tmpdir}/file.xyz").write_text("test")

            files = _gather_files(tmpdir, exts=None, max_files=100)
            # Should include .py and .txt but not .xyz
            assert any(f.endswith(".py") for f in files)
            assert any(f.endswith(".txt") for f in files)
            assert not any(f.endswith(".xyz") for f in files)

    def test_gather_files_custom_extensions(self):
        """Test gathering files with custom extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/file.py").write_text("test")
            Path(f"{tmpdir}/file.js").write_text("test")

            files = _gather_files(tmpdir, exts=[".js"], max_files=100)
            assert len(files) == 1
            assert files[0].endswith(".js")

    def test_gather_files_max_limit(self):
        """Test gathering files respects max limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                Path(f"{tmpdir}/file{i}.py").write_text("test")

            files = _gather_files(tmpdir, exts=None, max_files=5)
            assert len(files) == 5

    def test_gather_files_excludes_git(self):
        """Test that .git directories are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(f"{tmpdir}/.git")
            git_dir.mkdir()
            Path(f"{git_dir}/config.py").write_text("test")
            Path(f"{tmpdir}/file.py").write_text("test")

            files = _gather_files(tmpdir, exts=None, max_files=100)
            assert len(files) == 1
            assert ".git" not in files[0]


class TestKeywordScore:
    """Tests for _keyword_score function."""

    def test_single_keyword(self):
        """Test scoring with single keyword."""
        score = _keyword_score("hello world hello", "hello")
        assert score == 2  # "hello" appears twice

    def test_multiple_keywords(self):
        """Test scoring with multiple keywords."""
        score = _keyword_score("hello world foo bar", "hello world")
        assert score >= 2  # Each keyword counted

    def test_case_insensitive(self):
        """Test scoring is case insensitive."""
        score = _keyword_score("HELLO World", "hello")
        assert score == 1

    def test_no_match(self):
        """Test scoring with no matches."""
        score = _keyword_score("foo bar", "xyz")
        assert score == 0


class TestCodeSearch:
    """Tests for code_search function."""

    @pytest.mark.asyncio
    async def test_code_search_basic(self):
        """Test basic code search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/test.py").write_text("def hello():\n    pass")
            Path(f"{tmpdir}/other.py").write_text("def world():\n    pass")

            result = await code_search("hello", root=tmpdir, k=5)
            assert result["success"] is True
            assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_code_search_no_results(self):
        """Test code search with no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/test.py").write_text("def foo():\n    pass")

            result = await code_search("nonexistent_query_xyz", root=tmpdir, k=5)
            assert result["success"] is True
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_code_search_with_extensions(self):
        """Test code search with specific extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/test.py").write_text("hello python")
            Path(f"{tmpdir}/test.js").write_text("hello javascript")

            result = await code_search("hello", root=tmpdir, exts=[".py"], k=5)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_code_search_file_read_error(self):
        """Test code search handles file read errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/test.py").write_text("hello")

            original_open = open

            def mock_open_error(path, *args, **kwargs):
                if str(path).endswith(".py"):
                    raise IOError("Read error")
                return original_open(path, *args, **kwargs)

            with patch("builtins.open", mock_open_error):
                result = await code_search("hello", root=tmpdir, k=5)
                # Should still succeed but with no results
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_code_search_empty_dir(self):
        """Test code search with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await code_search("query", root=tmpdir, k=5)
            assert result["success"] is True
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_code_search_max_files_limit(self):
        """Test code search respects max_files limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                Path(f"{tmpdir}/file{i}.py").write_text(f"hello {i}")

            result = await code_search("hello", root=tmpdir, max_files=5, k=10)
            assert result["success"] is True
            # Should be limited by max_files
            assert result["count"] <= 5

    @pytest.mark.asyncio
    async def test_code_search_results_sorted(self):
        """Test that results are sorted by score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/low.py").write_text("hello")
            Path(f"{tmpdir}/high.py").write_text("hello hello hello hello")

            result = await code_search("hello", root=tmpdir, k=5)
            assert result["success"] is True
            if result["count"] >= 2:
                assert result["results"][0]["score"] >= result["results"][1]["score"]

    @pytest.mark.asyncio
    async def test_code_search_exception_handling(self):
        """Test code_search handles general exceptions (covers lines 116-117)."""
        with patch("victor.tools.code_search_tool._gather_files") as mock_gather:
            mock_gather.side_effect = OSError("Permission denied")
            result = await code_search("query", root="/some/path", k=5)

            assert result["success"] is False
            assert "error" in result
            assert "Permission denied" in result["error"]


class TestSemanticCodeSearch:
    """Tests for semantic_code_search function."""

    @pytest.mark.asyncio
    async def test_semantic_search_no_root(self):
        """Test semantic search with non-existent root."""
        from victor.tools.code_search_tool import semantic_code_search

        result = await semantic_code_search(
            "query", root="/nonexistent/path/xyz", context={"settings": MagicMock()}
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_semantic_search_no_settings(self):
        """Test semantic search without settings."""
        from victor.tools.code_search_tool import semantic_code_search

        result = await semantic_code_search("query", root=".", context={})
        assert result["success"] is False
        assert "settings" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_semantic_search_no_context(self):
        """Test semantic search without context."""
        from victor.tools.code_search_tool import semantic_code_search

        result = await semantic_code_search("query", root=".", context=None)
        assert result["success"] is False
        assert "settings" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_semantic_search_import_error(self):
        """Test semantic search handles missing dependencies."""
        from victor.tools.code_search_tool import semantic_code_search

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("victor.tools.code_search_tool._get_or_build_index") as mock_index:
                mock_index.side_effect = ImportError("lancedb not installed")
                result = await semantic_code_search(
                    "query", root=tmpdir, context={"settings": MagicMock()}
                )
                assert result["success"] is False
                assert "dependencies" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_semantic_search_general_error(self):
        """Test semantic search handles general errors."""
        from victor.tools.code_search_tool import semantic_code_search

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("victor.tools.code_search_tool._get_or_build_index") as mock_index:
                mock_index.side_effect = Exception("Some error")
                result = await semantic_code_search(
                    "query", root=tmpdir, context={"settings": MagicMock()}
                )
                assert result["success"] is False

    @pytest.mark.asyncio
    async def test_semantic_search_success(self):
        """Test successful semantic search."""
        from victor.tools.code_search_tool import semantic_code_search, _INDEX_CACHE

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/test.py").write_text("print('hello')")
            root_key = str(Path(tmpdir).resolve())

            mock_index = MagicMock()
            mock_index.semantic_search = AsyncMock(return_value=[{"path": "test.py", "score": 0.9}])

            with patch("victor.tools.code_search_tool._get_or_build_index") as mock_get:
                mock_get.return_value = (mock_index, True)
                # Set up cache entry
                _INDEX_CACHE[root_key] = {"indexed_at": 123456}
                try:
                    result = await semantic_code_search(
                        "hello", root=tmpdir, context={"settings": MagicMock()}
                    )
                    assert result["success"] is True
                    assert result["count"] == 1
                finally:
                    # Clean up cache
                    if root_key in _INDEX_CACHE:
                        del _INDEX_CACHE[root_key]


class TestGetOrBuildIndex:
    """Tests for _get_or_build_index function."""

    @pytest.mark.asyncio
    async def test_get_cached_index(self):
        """Test returning cached index."""
        from victor.tools.code_search_tool import _get_or_build_index, _INDEX_CACHE
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            # Use str(root_path) to match what the function uses internally
            root_key = str(root_path)

            # Create a file so directory has mtime
            Path(f"{tmpdir}/test.py").write_text("test")
            current_mtime = Path(f"{tmpdir}/test.py").stat().st_mtime

            # Create a cached index with mtime after the file
            mock_index = MagicMock()
            _INDEX_CACHE[root_key] = {
                "index": mock_index,
                "latest_mtime": current_mtime + 1000,  # Future time so no rebuild
                "indexed_at": time.time(),
            }

            mock_settings = MagicMock(spec=[])
            try:
                index, rebuilt = await _get_or_build_index(root_path, mock_settings)
                assert index is mock_index
                assert rebuilt is False
            finally:
                if root_key in _INDEX_CACHE:
                    del _INDEX_CACHE[root_key]

    @pytest.mark.asyncio
    async def test_build_new_index(self):
        """Test building new index."""
        from victor.tools.code_search_tool import _get_or_build_index, _INDEX_CACHE

        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            root_key = str(root_path)
            Path(f"{tmpdir}/test.py").write_text("test")

            mock_index_instance = MagicMock()
            mock_index_instance.index_codebase = AsyncMock()

            mock_settings = MagicMock(spec=[])
            mock_settings.codebase_vector_store = "lancedb"
            mock_settings.codebase_embedding_provider = "sentence-transformers"
            mock_settings.codebase_embedding_model = "all-MiniLM-L12-v2"
            mock_settings.unified_embedding_model = "all-MiniLM-L12-v2"
            mock_settings.codebase_persist_directory = None

            with patch("victor.codebase.indexer.CodebaseIndex") as MockCodebaseIndex:
                MockCodebaseIndex.return_value = mock_index_instance

                try:
                    index, rebuilt = await _get_or_build_index(root_path, mock_settings)
                    assert rebuilt is True
                    mock_index_instance.index_codebase.assert_called_once()
                finally:
                    if root_key in _INDEX_CACHE:
                        del _INDEX_CACHE[root_key]

    @pytest.mark.asyncio
    async def test_force_reindex(self):
        """Test force reindex."""
        from victor.tools.code_search_tool import _get_or_build_index, _INDEX_CACHE
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            root_key = str(root_path)

            # Create a file so directory has mtime
            Path(f"{tmpdir}/test.py").write_text("test")
            current_mtime = Path(f"{tmpdir}/test.py").stat().st_mtime

            # Create a cached index
            mock_old_index = MagicMock()
            _INDEX_CACHE[root_key] = {
                "index": mock_old_index,
                "latest_mtime": current_mtime + 1000,
                "indexed_at": time.time(),
            }

            mock_new_index = MagicMock()
            mock_new_index.index_codebase = AsyncMock()

            mock_settings = MagicMock(spec=[])
            mock_settings.codebase_vector_store = "lancedb"
            mock_settings.codebase_embedding_provider = "sentence-transformers"
            mock_settings.codebase_embedding_model = "all-MiniLM-L12-v2"
            mock_settings.unified_embedding_model = "all-MiniLM-L12-v2"
            mock_settings.codebase_persist_directory = None

            with patch("victor.codebase.indexer.CodebaseIndex") as MockCodebaseIndex:
                MockCodebaseIndex.return_value = mock_new_index

                try:
                    index, rebuilt = await _get_or_build_index(
                        root_path, mock_settings, force_reindex=True
                    )
                    assert rebuilt is True
                    assert index is mock_new_index
                finally:
                    if root_key in _INDEX_CACHE:
                        del _INDEX_CACHE[root_key]
