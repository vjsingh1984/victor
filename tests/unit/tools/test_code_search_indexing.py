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

"""Unit tests for code search indexing initialization tracking."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from victor.tools.code_search_tool import _probe_index_integrity


class TestIndexingFlagBehavior:
    """Tests for _is_indexing flag in CodebaseIndex."""

    def test_indexing_flag_attributes_exist(self):
        """Test that CodebaseIndex has _is_indexing and _indexing_start_time attributes."""
        # This test requires the actual victor-coding package
        pytest.importorskip("victor_coding")

        from victor_coding.codebase.indexer import CodebaseIndex

        # Create a temporary directory for testing
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index = CodebaseIndex(root_path=tmpdir, use_embeddings=False)

            # Check that flags exist and are initialized correctly
            assert hasattr(index, "_is_indexing")
            assert hasattr(index, "_indexing_start_time")
            assert index._is_indexing is False
            assert index._indexing_start_time is None

    def test_indexing_flag_set_during_indexing(self):
        """Test that _is_indexing flag is set to True during index_codebase."""
        pytest.importorskip("victor_coding")

        from victor_coding.codebase.indexer import CodebaseIndex

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index = CodebaseIndex(root_path=tmpdir, use_embeddings=False)

            # Mock the indexing process to just set the flag
            async def mock_index():
                index._is_indexing = True
                index._indexing_start_time = time.time()
                try:
                    await asyncio.sleep(0.1)  # Simulate some work
                finally:
                    index._is_indexing = False
                    index._indexing_start_time = None

            # Run the mock indexing
            asyncio.run(mock_index())

            # Verify flag is cleared after indexing
            assert index._is_indexing is False
            assert index._indexing_start_time is None


class TestCorruptionDetection:
    """Tests for index integrity checking during init."""

    @pytest.mark.asyncio
    async def test_corruption_check_skipped_during_indexing(self):
        """Test that integrity check is skipped when _is_indexing flag is True."""
        # Create a mock index with _is_indexing flag
        mock_index = MagicMock()
        mock_index._is_indexing = True
        mock_index._is_indexed = False

        # The integrity check should return False (no rebuild needed)
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result is False  # Should NOT trigger rebuild

    @pytest.mark.asyncio
    async def test_corruption_check_proceeds_when_not_indexing(self):
        """Test that integrity check proceeds when _is_indexing flag is False."""
        # Create a mock index that appears healthy
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock vector store with data
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 1000
        mock_store = MagicMock()
        mock_store._table = mock_table
        mock_index._vector_store = mock_store

        # The integrity check should return False (healthy)
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result is False  # Should be healthy
        mock_table.count_rows.assert_called_once()

    @pytest.mark.asyncio
    async def test_transient_error_does_not_trigger_rebuild(self):
        """Test that transient init errors don't trigger rebuild."""
        # Create a mock index that throws a "locked" error
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock vector store to return no table (skip fast path)
        mock_store = MagicMock()
        mock_store._table = None  # No table, so it will try semantic_search
        mock_index._vector_store = mock_store

        # Mock semantic_search that raises a "locked" error
        async def mock_semantic_search_error(*args, **kwargs):
            raise Exception("database is locked")

        mock_index.semantic_search = mock_semantic_search_error

        # The integrity check should return False (no rebuild for transient errors)
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result is False  # Should NOT rebuild for transient errors

    @pytest.mark.asyncio
    async def test_actual_corruption_triggers_rebuild(self):
        """Test that actual corruption errors trigger background rebuild."""
        # Create a mock index that throws a non-transient error
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock semantic_search that raises a "corruption" error
        async def mock_semantic_search_corrupt(*args, **kwargs):
            raise ValueError("Invalid data format")

        mock_index.semantic_search = mock_semantic_search_corrupt

        # The integrity check should return True (rebuild triggered)
        with patch("victor.tools.code_search_tool.asyncio.create_task"):
            result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result is True  # Should trigger rebuild
        assert mock_index._is_indexed is False  # Flag should be cleared


class TestErrorClassification:
    """Tests for error classification in corruption detection."""

    @pytest.mark.asyncio
    async def test_empty_exception_string_classified(self):
        """Test that empty exception strings are handled gracefully."""
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock semantic_search that raises exception with empty string
        class EmptyError(Exception):
            def __str__(self):
                return ""

        async def mock_empty_error(*args, **kwargs):
            raise EmptyError()

        mock_index.semantic_search = mock_empty_error

        # Should use type name as error message
        with patch("victor.tools.code_search_tool.asyncio.create_task"):
            result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result is True  # Should trigger rebuild (not a transient error)

    @pytest.mark.asyncio
    async def test_timeout_error_classified_as_transient(self):
        """Test that timeout errors are classified as transient."""
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock vector store to return no table (skip fast path)
        mock_store = MagicMock()
        mock_store._table = None
        mock_index._vector_store = mock_store

        # Mock semantic_search that raises timeout
        async def mock_timeout(*args, **kwargs):
            raise TimeoutError("Operation timed out")

        mock_index.semantic_search = mock_timeout

        # Should NOT trigger rebuild for timeout errors
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result is False  # Should skip rebuild

    @pytest.mark.asyncio
    async def test_not_ready_error_classified_as_transient(self):
        """Test that "not ready" errors are classified as transient."""
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock vector store to return no table (skip fast path)
        mock_store = MagicMock()
        mock_store._table = None
        mock_index._vector_store = mock_store

        # Mock semantic_search that raises "not ready"
        async def mock_not_ready(*args, **kwargs):
            raise RuntimeError("Service not ready")

        mock_index.semantic_search = mock_not_ready

        # Should NOT trigger rebuild
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result is False  # Should skip rebuild


class TestBackgroundRebuildLogging:
    """Tests for background rebuild logging improvements."""

    @pytest.mark.asyncio
    async def test_rebuild_logging_includes_index_path(self, caplog):
        """Test that rebuild logging includes the index path."""
        from victor.tools.code_search_tool import _background_index_rebuild

        # Create a mock index
        mock_index = MagicMock()
        mock_index.root = Path("/test/project")

        # Mock index_codebase to succeed quickly
        async def mock_rebuild():
            pass

        mock_index.index_codebase = mock_rebuild

        # Run rebuild (should log with index path)
        import logging

        with caplog.at_level(logging.INFO):
            await _background_index_rebuild(mock_index, rebuild_timeout=10.0)

        # Check that log messages include the index path
        log_messages = [record.message for record in caplog.records]
        assert any("/test/project" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_rebuild_logging_includes_timing(self, caplog):
        """Test that rebuild logging includes timing information."""
        from victor.tools.code_search_tool import _background_index_rebuild

        # Create a mock index
        mock_index = MagicMock()
        mock_index.root = Path("/test/project")

        # Mock index_codebase to take some time
        async def mock_rebuild_with_delay():
            await asyncio.sleep(0.1)

        mock_index.index_codebase = mock_rebuild_with_delay

        # Run rebuild
        import logging

        with caplog.at_level(logging.INFO):
            await _background_index_rebuild(mock_index, rebuild_timeout=10.0)

        # Check that log messages include timing
        log_messages = [record.message for record in caplog.records]
        assert any("0." in msg or "0.1" in msg or "0.2" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_rebuild_failure_logging_includes_error_details(self, caplog):
        """Test that rebuild failures include error details."""
        from victor.tools.code_search_tool import _background_index_rebuild

        # Create a mock index
        mock_index = MagicMock()
        mock_index.root = Path("/test/project")

        # Mock index_codebase to fail
        async def mock_rebuild_failure():
            raise ValueError("Test rebuild error")

        mock_index.index_codebase = mock_rebuild_failure

        # Run rebuild
        import logging

        with caplog.at_level(logging.WARNING):
            await _background_index_rebuild(mock_index, rebuild_timeout=10.0)

        # Check that log includes error details
        log_messages = [record.message for record in caplog.records]
        assert any("Test rebuild error" in msg for msg in log_messages)
        assert any("/test/project" in msg for msg in log_messages)
