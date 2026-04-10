"""Tests for vertical-name normalization in entry_point_loader helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from victor.framework.entry_point_loader import (
    load_rl_config_from_entry_points,
    load_safety_rules_from_entry_points,
    register_commands_from_entry_points,
    register_escape_hatches_from_entry_points,
)


@patch("victor.framework.entry_point_loader.UnifiedEntryPointRegistry")
def test_load_rl_config_normalizes_vertical_aliases(mock_registry_cls):
    """RL config lookup should match alias spellings (data-analysis/data_analysis)."""
    config = {"enabled": True}

    # load_rl_config_from_entry_points -> load_rl_config_provider_from_entry_points
    # -> load_runtime_extension_from_entry_points which uses UnifiedEntryPointRegistry.
    # The registry.get() returns the loaded entry point target (a callable factory).
    # _resolve_loaded_entry_point_target will call it since it's callable.
    mock_registry = MagicMock()
    mock_registry.scan_all.return_value = None

    def _get(group, name):
        # normalize_vertical_name("data_analysis") returns "dataanalysis"
        if group == "victor.rl_configs" and name == "dataanalysis":
            return lambda: config
        return None

    mock_registry.get.side_effect = _get
    mock_registry_cls.get_instance.return_value = mock_registry

    resolved = load_rl_config_from_entry_points("data_analysis")

    assert resolved == config


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_load_safety_rules_filters_with_normalized_vertical_names(mock_cached_eps):
    """Safety-rule loading should normalize caller filters and entry-point names."""
    enforcer = object()
    called = {"count": 0}

    ep = MagicMock()
    ep.name = "data-analysis"
    ep.load.return_value = lambda _enforcer: called.__setitem__("count", called["count"] + 1)
    mock_cached_eps.return_value = (ep,)

    loaded = load_safety_rules_from_entry_points(enforcer, vertical_names=["data_analysis"])

    assert loaded == 1
    assert called["count"] == 1


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_register_escape_hatches_filters_with_normalized_vertical_names(
    mock_cached_eps,
):
    """Escape-hatch registration should normalize caller filters and entry-point names."""
    registry = object()
    called = {"count": 0}

    ep = MagicMock()
    ep.name = "data-analysis"
    ep.load.return_value = lambda _registry: called.__setitem__("count", called["count"] + 1)
    mock_cached_eps.return_value = (ep,)

    registered = register_escape_hatches_from_entry_points(
        registry, vertical_names=["data_analysis"]
    )

    assert registered == 1
    assert called["count"] == 1


@patch("victor.framework.entry_point_loader._cached_entry_points")
def test_register_commands_filters_with_normalized_vertical_names(mock_cached_eps):
    """Command registration should normalize caller filters and entry-point names."""
    app = object()
    called = {"count": 0}

    ep = MagicMock()
    ep.name = "data-analysis"
    ep.load.return_value = lambda _app: called.__setitem__("count", called["count"] + 1)
    mock_cached_eps.return_value = (ep,)

    registered = register_commands_from_entry_points(app, vertical_names=["data_analysis"])

    assert registered == 1
    assert called["count"] == 1
