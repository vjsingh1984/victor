"""Tests for provider-style runtime extension entry-point resolution."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

from victor.framework.entry_point_loader import (
    load_rl_config_provider_from_entry_points,
    load_runtime_extension_from_entry_points,
    list_installed_verticals,
    reset_entry_point_loader_stats,
)


def setup_function() -> None:
    """Reset loader stats/cache before each test."""
    reset_entry_point_loader_stats(clear_cache=True)


def teardown_function() -> None:
    """Reset loader stats/cache after each test."""
    reset_entry_point_loader_stats(clear_cache=True)


def _mock_registry_get(group_entries):
    """Create a mock UnifiedEntryPointRegistry that returns values from group_entries.

    Args:
        group_entries: dict mapping (group, normalized_name) -> loaded value
    """
    mock_registry = MagicMock()
    mock_registry.scan_all.return_value = None

    def _get(group, name):
        return group_entries.get((group, name))

    mock_registry.get.side_effect = _get
    return mock_registry


@patch("victor.framework.entry_point_loader.UnifiedEntryPointRegistry")
def test_load_runtime_extension_normalizes_aliases_and_instantiates_classes(
    mock_registry_cls,
) -> None:
    """Generic runtime-extension loading should normalize aliases and instantiate classes."""

    class PromptContributor:
        pass

    # normalize_vertical_name("data_analysis") returns "dataanalysis"
    mock_registry = _mock_registry_get(
        {("victor.prompt_contributors", "dataanalysis"): PromptContributor}
    )
    mock_registry_cls.get_instance.return_value = mock_registry

    resolved = load_runtime_extension_from_entry_points(
        "data_analysis",
        "victor.prompt_contributors",
    )

    assert isinstance(resolved, PromptContributor)


@patch("victor.framework.entry_point_loader.UnifiedEntryPointRegistry")
def test_load_runtime_extension_returns_prebuilt_instances(mock_registry_cls) -> None:
    """Generic runtime-extension loading should preserve non-callable entry-point values."""
    contributor = object()

    mock_registry = _mock_registry_get({("victor.prompt_contributors", "coding"): contributor})
    mock_registry_cls.get_instance.return_value = mock_registry

    resolved = load_runtime_extension_from_entry_points(
        "coding",
        "victor.prompt_contributors",
    )

    assert resolved is contributor


@patch("victor.framework.entry_point_loader.UnifiedEntryPointRegistry")
def test_load_rl_config_provider_returns_provider_instance(mock_registry_cls) -> None:
    """RL config provider entry-point loading should return provider instances."""

    class RLConfigProvider:
        def get_rl_config(self) -> dict[str, bool]:
            return {"enabled": True}

    mock_registry = _mock_registry_get({("victor.rl_configs", "coding"): RLConfigProvider})
    mock_registry_cls.get_instance.return_value = mock_registry

    resolved = load_rl_config_provider_from_entry_points("coding")

    assert isinstance(resolved, RLConfigProvider)
    assert resolved.get_rl_config() == {"enabled": True}


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_list_installed_verticals_warns_when_legacy_group_is_present(
    mock_cached_entry_points,
    caplog,
) -> None:
    """Installed vertical inventory should log warning for deprecated raw entry-points."""

    class _EntryPoint:
        def __init__(self, name: str) -> None:
            self.name = name

    def _mock_group(group: str):
        if group == "victor.plugins":
            return (_EntryPoint("coding"),)
        if group == "victor.verticals":
            return (_EntryPoint("legacy-security"),)
        return ()

    mock_cached_entry_points.side_effect = _mock_group
    reset_entry_point_loader_stats(clear_cache=True)

    verticals = list_installed_verticals()

    assert verticals == ["coding", "legacy-security"]
    assert any("victor.verticals" in r.message for r in caplog.records)


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_list_installed_verticals_warns_only_once_for_legacy_group(
    mock_cached_entry_points,
    caplog,
) -> None:
    """Repeated inventory calls should not spam log warnings."""

    class _EntryPoint:
        def __init__(self, name: str) -> None:
            self.name = name

    def _mock_group(group: str):
        if group == "victor.plugins":
            return ()
        if group == "victor.verticals":
            return (_EntryPoint("legacy-security"),)
        return ()

    mock_cached_entry_points.side_effect = _mock_group
    reset_entry_point_loader_stats(clear_cache=True)

    list_installed_verticals()
    list_installed_verticals()

    legacy_warnings = [r for r in caplog.records if "victor.verticals" in r.message]
    assert len(legacy_warnings) == 1
