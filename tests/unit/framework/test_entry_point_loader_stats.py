"""Tests for entry-point loader telemetry counters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from victor.core.tool_types import EmptyToolDependencyProvider
from victor.framework.entry_point_loader import (
    clear_entry_point_loader_cache,
    get_entry_point_loader_stats,
    load_tool_dependency_provider_from_entry_points,
    reset_entry_point_loader_stats,
)


def setup_function() -> None:
    """Reset loader stats/cache before each test."""
    reset_entry_point_loader_stats(clear_cache=True)


def teardown_function() -> None:
    """Reset loader stats/cache after each test."""
    reset_entry_point_loader_stats(clear_cache=True)


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_stats_track_tool_dependency_entry_point_resolution(mock_cached_eps):
    """Entry-point resolution should update tool dependency stats."""
    provider = object()
    ep = MagicMock()
    ep.name = "coding"
    ep.load.return_value = lambda: provider
    mock_cached_eps.return_value = (ep,)

    resolved = load_tool_dependency_provider_from_entry_points("coding")
    stats = get_entry_point_loader_stats()

    assert resolved is provider
    assert stats["tool_dependency_calls"] == 1
    assert stats["tool_dependency_entry_point_resolutions"] == 1
    assert stats["tool_dependency_fallback_resolutions"] == 0
    assert stats["tool_dependency_none_returns"] == 0


@patch("victor.core.tool_dependency_loader.create_vertical_tool_dependency_provider")
@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_stats_track_tool_dependency_fallback_resolution(
    mock_cached_eps, mock_create_vertical_provider
):
    """Fallback resolution should update tool dependency stats."""
    provider = object()
    mock_cached_eps.return_value = ()
    mock_create_vertical_provider.return_value = provider

    resolved = load_tool_dependency_provider_from_entry_points("coding")
    stats = get_entry_point_loader_stats()

    assert resolved is provider
    assert stats["tool_dependency_calls"] == 1
    assert stats["tool_dependency_entry_point_resolutions"] == 0
    assert stats["tool_dependency_fallback_resolutions"] == 1
    assert stats["tool_dependency_none_returns"] == 0


@patch("victor.core.tool_dependency_loader.create_vertical_tool_dependency_provider")
@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_stats_track_tool_dependency_none_returns(mock_cached_eps, mock_create_vertical_provider):
    """Empty fallback providers should increment none-return counters."""
    mock_cached_eps.return_value = ()
    mock_create_vertical_provider.return_value = EmptyToolDependencyProvider("coding")

    resolved = load_tool_dependency_provider_from_entry_points("coding")
    stats = get_entry_point_loader_stats()

    assert resolved is None
    assert stats["tool_dependency_calls"] == 1
    assert stats["tool_dependency_none_returns"] == 1


def test_stats_track_cache_clear_operations() -> None:
    """Explicit cache clearing should increment cache clear telemetry."""
    clear_entry_point_loader_cache()
    stats = get_entry_point_loader_stats()
    assert stats["cache_clears"] == 1


def test_reset_stats_with_clear_cache_keeps_zero_baseline() -> None:
    """reset_entry_point_loader_stats(clear_cache=True) should leave counters at zero."""
    clear_entry_point_loader_cache()
    reset_entry_point_loader_stats(clear_cache=True)
    stats = get_entry_point_loader_stats()
    assert stats["cache_clears"] == 0
