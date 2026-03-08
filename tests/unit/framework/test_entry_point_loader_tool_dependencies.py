"""Tests for tool dependency provider loading in entry_point_loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from victor.core.tool_types import EmptyToolDependencyProvider
from victor.framework.entry_point_loader import load_tool_dependency_provider_from_entry_points


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_load_tool_dependency_provider_prefers_entry_points(mock_cached_eps):
    """Entry point provider should be used when available."""
    provider = object()

    ep = MagicMock()
    ep.name = "coding"
    ep.load.return_value = lambda: provider
    mock_cached_eps.return_value = (ep,)

    resolved = load_tool_dependency_provider_from_entry_points("coding")

    assert resolved is provider


@patch("victor.core.tool_dependency_loader.create_vertical_tool_dependency_provider")
@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_load_tool_dependency_provider_uses_core_fallback_when_entry_point_missing(
    mock_cached_eps, mock_create_vertical_provider
):
    """Fallback should use core vertical provider resolution when no entry point matches."""
    provider = object()
    mock_cached_eps.return_value = ()
    mock_create_vertical_provider.return_value = provider

    resolved = load_tool_dependency_provider_from_entry_points("coding")

    assert resolved is provider
    mock_create_vertical_provider.assert_called_once_with("coding")


@patch("victor.core.tool_dependency_loader.create_vertical_tool_dependency_provider")
@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_load_tool_dependency_provider_returns_none_for_empty_fallback(
    mock_cached_eps, mock_create_vertical_provider
):
    """EmptyToolDependencyProvider fallback should map to None for compatibility."""
    mock_cached_eps.return_value = ()
    mock_create_vertical_provider.return_value = EmptyToolDependencyProvider("coding")

    resolved = load_tool_dependency_provider_from_entry_points("coding")

    assert resolved is None


@patch("victor.core.tool_dependency_loader.create_vertical_tool_dependency_provider")
@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_load_tool_dependency_provider_returns_none_for_unknown_vertical(
    mock_cached_eps, mock_create_vertical_provider
):
    """Unknown vertical fallback should return None."""
    mock_cached_eps.return_value = ()
    mock_create_vertical_provider.side_effect = ValueError("unknown vertical")

    resolved = load_tool_dependency_provider_from_entry_points("mlops")

    assert resolved is None
