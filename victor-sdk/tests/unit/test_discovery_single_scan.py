"""TDD tests for consolidated single-pass entry point scanning.

These tests verify that ProtocolRegistry.load_from_entry_points()
calls importlib.metadata.entry_points() exactly once instead of 4
separate times, and correctly partitions results by group.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch


from victor_sdk.discovery import ProtocolRegistry


def _make_entry_point(name: str, group: str, load_return: Any = None) -> Any:
    """Create a mock entry point with the given name, group, and load result."""
    ep = SimpleNamespace(
        name=name,
        group=group,
        value=f"fake_module:{name}",
    )
    if load_return is not None:
        ep.load = MagicMock(return_value=load_return)
    else:
        ep.load = MagicMock(return_value=MagicMock())
    return ep


class TestSinglePassScan:
    """Tests verifying entry_points() is called once, not 4 times."""

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_load_from_entry_points_single_scan(self, mock_entry_points):
        """entry_points() should be called exactly once during load."""
        mock_entry_points.return_value = []

        registry = ProtocolRegistry()
        registry.load_from_entry_points()

        assert mock_entry_points.call_count == 1, (
            f"entry_points() called {mock_entry_points.call_count} times, expected 1. "
            "Discovery should use a single-pass scan, not 4 separate calls."
        )

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_load_from_entry_points_partitions_by_group(self, mock_entry_points):
        """Entry points from different groups should be routed correctly."""
        # Create a mock vertical plugin
        mock_plugin = MagicMock()
        mock_plugin.__name__ = "FakePlugin"
        # Make it NOT a VerticalBase subclass to avoid complex registration
        # but make it a VictorPlugin-like object
        mock_plugin_instance = MagicMock()
        mock_plugin_instance.name = "fake_vertical"

        ep_vertical = _make_entry_point(
            "fake", ProtocolRegistry.VERTICALS_GROUP, load_return=mock_plugin
        )
        ep_capability = _make_entry_point(
            "fake_cap", ProtocolRegistry.CAPABILITIES_GROUP
        )
        ep_validator = _make_entry_point(
            "fake_val",
            ProtocolRegistry.VALIDATORS_GROUP,
            load_return=lambda x: True,
        )

        # Return all entry points in a flat list (Python 3.12+ format)
        mock_entry_points.return_value = [ep_vertical, ep_capability, ep_validator]

        registry = ProtocolRegistry()
        registry.load_from_entry_points()

        # Verify single scan
        assert mock_entry_points.call_count == 1

        # Verify each entry point's load() was called (i.e., it was routed)
        ep_vertical.load.assert_called_once()
        ep_capability.load.assert_called_once()
        ep_validator.load.assert_called_once()

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_load_from_entry_points_ignores_non_victor_groups(
        self, mock_entry_points
    ):
        """Entry points from non-victor groups (e.g., console_scripts) are ignored."""
        ep_console = _make_entry_point("victor-cli", "console_scripts")
        ep_gui = _make_entry_point("victor-gui", "gui_scripts")
        ep_valid = _make_entry_point(
            "real_val",
            ProtocolRegistry.VALIDATORS_GROUP,
            load_return=lambda x: True,
        )

        mock_entry_points.return_value = [ep_console, ep_gui, ep_valid]

        registry = ProtocolRegistry()
        registry.load_from_entry_points()

        # Non-victor entry points should NOT have load() called
        ep_console.load.assert_not_called()
        ep_gui.load.assert_not_called()

        # Victor entry point SHOULD have load() called
        ep_valid.load.assert_called_once()

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_load_from_entry_points_handles_empty_scan(self, mock_entry_points):
        """When no entry points exist, stats should be all zeros and no errors."""
        mock_entry_points.return_value = []

        registry = ProtocolRegistry()
        stats = registry.load_from_entry_points()

        assert stats.total_verticals == 0
        assert stats.total_protocols == 0
        assert stats.total_capabilities == 0
        assert stats.total_validators == 0
        assert stats.failed_loads == 0

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_reload_triggers_fresh_scan(self, mock_entry_points):
        """reload=True should clear state and re-scan."""
        mock_entry_points.return_value = []

        registry = ProtocolRegistry()
        registry.load_from_entry_points()
        registry.load_from_entry_points(reload=True)

        # Should be called twice: once for initial, once for reload
        assert mock_entry_points.call_count == 2

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_python_version_compat_dict_format(self, mock_entry_points):
        """Handle Python 3.9 dict-style entry_points() return format.

        In Python 3.9, entry_points() without group= returns a dict
        mapping group names to lists. We must handle this format.
        """
        ep_val = _make_entry_point(
            "val1",
            ProtocolRegistry.VALIDATORS_GROUP,
            load_return=lambda x: True,
        )

        # Python 3.9 dict format: {group_name: [entry_points]}
        result_dict = {
            ProtocolRegistry.VALIDATORS_GROUP: [ep_val],
            "console_scripts": [_make_entry_point("cli", "console_scripts")],
        }
        mock_entry_points.return_value = result_dict

        registry = ProtocolRegistry()
        stats = registry.load_from_entry_points()

        assert stats.total_validators == 1
        assert mock_entry_points.call_count == 1

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_stats_accurate_after_consolidation(self, mock_entry_points):
        """Discovery stats should correctly count items from each group."""
        # Create validator entry points (simplest to test without protocol mocking)
        ep_v1 = _make_entry_point(
            "val1",
            ProtocolRegistry.VALIDATORS_GROUP,
            load_return=lambda x: True,
        )
        ep_v2 = _make_entry_point(
            "val2",
            ProtocolRegistry.VALIDATORS_GROUP,
            load_return=lambda x: False,
        )

        mock_entry_points.return_value = [ep_v1, ep_v2]

        registry = ProtocolRegistry()
        stats = registry.load_from_entry_points()

        assert stats.total_validators == 2
        assert stats.total_verticals == 0
        assert stats.total_protocols == 0
        assert stats.total_capabilities == 0
