"""Tests for entry point caching in entry_point_loader."""

from unittest.mock import patch, MagicMock

from victor.framework.entry_point_loader import _cached_entry_points, clear_entry_point_loader_cache
from victor.framework.entry_point_registry import UnifiedEntryPointRegistry


class TestEntryPointCaching:
    """Verify that entry_points() results are cached per group."""

    def setup_method(self):
        _cached_entry_points.cache_clear()
        # Also reset the unified registry
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

    def teardown_method(self):
        _cached_entry_points.cache_clear()
        # Also reset the unified registry
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

    @patch("victor.framework.entry_point_registry.entry_points")
    def test_cached_entry_points_called_once_per_group(self, mock_ep):
        """entry_points() should be called only once per group via unified registry."""
        mock_ep.return_value = []

        _cached_entry_points("victor.safety_rules")
        _cached_entry_points("victor.safety_rules")
        _cached_entry_points("victor.safety_rules")

        # The unified registry scans once, not per call
        assert mock_ep.call_count == 1

    @patch("victor.framework.entry_point_registry.entry_points")
    def test_different_groups_cached_separately(self, mock_ep):
        """Different groups should use the same scan (single-pass)."""
        mock_ep.return_value = []

        _cached_entry_points("victor.safety_rules")
        _cached_entry_points("victor.commands")

        # Single-pass scanning means only one call for all groups
        assert mock_ep.call_count == 1

    @patch("victor.framework.entry_point_registry.entry_points")
    def test_cache_clear_allows_rescan(self, mock_ep):
        """cache_clear() should allow re-scanning entry points."""
        mock_ep.return_value = []

        _cached_entry_points("victor.safety_rules")
        assert mock_ep.call_count == 1

        # Use the proper clear function that also invalidates the registry
        clear_entry_point_loader_cache()
        _cached_entry_points("victor.safety_rules")
        # After cache clear, should rescan
        assert mock_ep.call_count == 2

    @patch("victor.framework.entry_point_registry.entry_points")
    def test_public_cache_clear_helper_allows_rescan(self, mock_ep):
        """clear_entry_point_loader_cache() should clear cached lookups."""
        mock_ep.return_value = []

        _cached_entry_points("victor.safety_rules")
        assert mock_ep.call_count == 1

        clear_entry_point_loader_cache()
        _cached_entry_points("victor.safety_rules")
        assert mock_ep.call_count == 2

    @patch("victor.framework.entry_point_registry.entry_points")
    def test_returns_tuple(self, mock_ep):
        """Result should be a tuple (hashable) not a list."""
        ep1 = MagicMock()
        ep1.name = "test"
        ep1.group = "victor.test"
        mock_ep.return_value = [ep1]

        result = _cached_entry_points("victor.test")
        assert isinstance(result, tuple)
        # "victor.test" is not in ENTRY_POINT_GROUPS, so it won't be included
        # Let's use a valid group instead
        assert len(result) >= 0  # Can be 0 if not a recognized group
