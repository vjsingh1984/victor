"""Tests for entry point caching in entry_point_loader."""

from unittest.mock import patch, MagicMock

from victor.framework.entry_point_loader import _cached_entry_points


class TestEntryPointCaching:
    """Verify that entry_points() results are cached per group."""

    def setup_method(self):
        _cached_entry_points.cache_clear()

    def teardown_method(self):
        _cached_entry_points.cache_clear()

    @patch("victor.framework.entry_point_loader.entry_points")
    def test_cached_entry_points_called_once_per_group(self, mock_ep):
        """entry_points() should be called only once per group."""
        mock_ep.return_value = []

        _cached_entry_points("victor.safety_rules")
        _cached_entry_points("victor.safety_rules")
        _cached_entry_points("victor.safety_rules")

        assert mock_ep.call_count == 1

    @patch("victor.framework.entry_point_loader.entry_points")
    def test_different_groups_cached_separately(self, mock_ep):
        """Different groups should each trigger one call."""
        mock_ep.return_value = []

        _cached_entry_points("victor.safety_rules")
        _cached_entry_points("victor.commands")

        assert mock_ep.call_count == 2

    @patch("victor.framework.entry_point_loader.entry_points")
    def test_cache_clear_allows_rescan(self, mock_ep):
        """cache_clear() should allow re-scanning entry points."""
        mock_ep.return_value = []

        _cached_entry_points("victor.safety_rules")
        assert mock_ep.call_count == 1

        _cached_entry_points.cache_clear()
        _cached_entry_points("victor.safety_rules")
        assert mock_ep.call_count == 2

    @patch("victor.framework.entry_point_loader.entry_points")
    def test_returns_tuple(self, mock_ep):
        """Result should be a tuple (hashable) not a list."""
        ep1 = MagicMock()
        ep1.name = "test"
        mock_ep.return_value = [ep1]

        result = _cached_entry_points("victor.test")
        assert isinstance(result, tuple)
        assert len(result) == 1
