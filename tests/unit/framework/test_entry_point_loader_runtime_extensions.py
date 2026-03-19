"""Tests for provider-style runtime extension entry-point resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from victor.framework.entry_point_loader import (
    load_rl_config_provider_from_entry_points,
    load_runtime_extension_from_entry_points,
    reset_entry_point_loader_stats,
)


def setup_function() -> None:
    """Reset loader stats/cache before each test."""
    reset_entry_point_loader_stats(clear_cache=True)


def teardown_function() -> None:
    """Reset loader stats/cache after each test."""
    reset_entry_point_loader_stats(clear_cache=True)


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_load_runtime_extension_normalizes_aliases_and_instantiates_classes(
    mock_cached_eps,
) -> None:
    """Generic runtime-extension loading should normalize aliases and instantiate classes."""

    class PromptContributor:
        pass

    ep = MagicMock()
    ep.name = "data-analysis"
    ep.load.return_value = PromptContributor
    mock_cached_eps.return_value = (ep,)

    resolved = load_runtime_extension_from_entry_points(
        "data_analysis",
        "victor.prompt_contributors",
    )

    assert isinstance(resolved, PromptContributor)


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_load_runtime_extension_returns_prebuilt_instances(mock_cached_eps) -> None:
    """Generic runtime-extension loading should preserve non-callable entry-point values."""
    contributor = object()

    ep = MagicMock()
    ep.name = "coding"
    ep.load.return_value = contributor
    mock_cached_eps.return_value = (ep,)

    resolved = load_runtime_extension_from_entry_points(
        "coding",
        "victor.prompt_contributors",
    )

    assert resolved is contributor


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_load_rl_config_provider_returns_provider_instance(mock_cached_eps) -> None:
    """RL config provider entry-point loading should return provider instances."""

    class RLConfigProvider:
        def get_rl_config(self) -> dict[str, bool]:
            return {"enabled": True}

    ep = MagicMock()
    ep.name = "coding"
    ep.load.return_value = RLConfigProvider
    mock_cached_eps.return_value = (ep,)

    resolved = load_rl_config_provider_from_entry_points("coding")

    assert isinstance(resolved, RLConfigProvider)
    assert resolved.get_rl_config() == {"enabled": True}
