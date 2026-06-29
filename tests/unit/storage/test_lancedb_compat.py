"""Tests for LanceDB backward/forward compatibility layer."""

from unittest.mock import MagicMock

from victor.storage.vector_stores._lancedb_compat import get_table_names


class TestLanceDBCompat:
    """Tests for get_table_names compatibility helper."""

    def test_table_names_new_api(self):
        """Should use table_names() on new LanceDB versions."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["table1", "table2"]
        del mock_db.list_tables  # Simulate new API only
        result = get_table_names(mock_db)
        assert result == ["table1", "table2"]

    def test_table_names_old_api_with_tables_attr(self):
        """Should fall back to list_tables().tables on old LanceDB versions."""
        mock_db = MagicMock()
        del mock_db.table_names  # Simulate old API only
        mock_response = MagicMock()
        mock_response.tables = ["table1"]
        mock_db.list_tables.return_value = mock_response
        result = get_table_names(mock_db)
        assert result == ["table1"]

    def test_table_names_old_api_list_return(self):
        """Should handle old API returning a plain list."""
        mock_db = MagicMock()
        del mock_db.table_names
        mock_db.list_tables.return_value = ["t1", "t2"]
        result = get_table_names(mock_db)
        assert result == ["t1", "t2"]

    def test_empty_on_no_api(self):
        """Should return empty list if neither API is available."""
        mock_db = MagicMock(spec=[])
        result = get_table_names(mock_db)
        assert result == []

    def test_table_names_returns_none(self):
        """Should return empty list when table_names() returns None."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = None
        result = get_table_names(mock_db)
        assert result == []

    def test_prefers_new_api_over_old(self):
        """When both APIs exist, prefer list_tables() (current) over table_names() (deprecated).

        list_tables() is the canonical method on current LanceDB versions and is
        tried first so call-sites never trigger the ``table_names() is
        deprecated`` warning emitted by recent releases. table_names() is the
        legacy fallback used only when list_tables() is absent.
        """
        mock_db = MagicMock()
        # list_tables() is the current canonical API.
        mock_db.list_tables.return_value = ["new_api"]
        # table_names() is the deprecated legacy API and must NOT be called.
        mock_db.table_names.return_value = ["old_api"]
        result = get_table_names(mock_db)
        assert result == ["new_api"]
        mock_db.table_names.assert_not_called()
