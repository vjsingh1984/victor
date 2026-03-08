"""Tests for VerticalLoader contract enforcement and cache invalidation hooks."""

from typing import List

from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor.core.verticals.vertical_loader import VerticalLoader


def _make_vertical(name: str, api_version: int):
    """Create a minimal concrete vertical for loader tests."""

    class _TestVertical(VerticalBase):
        description = f"Test vertical {name}"
        VERTICAL_API_VERSION = api_version

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "test prompt"

    _TestVertical.name = name
    return _TestVertical


def test_loader_rejects_vertical_with_unsupported_api_version(monkeypatch):
    """VerticalLoader must reject entry-point verticals below min API version."""
    loader = VerticalLoader()
    loader._discovered_verticals = {}
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    vertical_name = "api_too_old_vertical"
    VerticalRegistry.unregister(vertical_name)
    old_vertical = _make_vertical(vertical_name, api_version=0)

    monkeypatch.setattr(loader, "_load_entry_point", lambda *_: old_vertical)
    loader._load_vertical_entries({"old_plugin": "fake.module:OldVertical"})

    assert vertical_name not in loader._discovered_verticals
    assert VerticalRegistry.get(vertical_name) is None


def test_loader_accepts_vertical_with_supported_api_version(monkeypatch):
    """VerticalLoader should accept and register compatible verticals."""
    loader = VerticalLoader()
    loader._discovered_verticals = {}
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    vertical_name = "api_current_vertical"
    VerticalRegistry.unregister(vertical_name)
    current_vertical = _make_vertical(vertical_name, api_version=1)

    monkeypatch.setattr(loader, "_load_entry_point", lambda *_: current_vertical)
    loader._load_vertical_entries({"current_plugin": "fake.module:CurrentVertical"})

    assert "current_plugin" in loader._discovered_verticals
    assert loader._discovered_verticals["current_plugin"] is current_vertical
    assert VerticalRegistry.get(vertical_name) is current_vertical

    VerticalRegistry.unregister(vertical_name)


def test_refresh_plugins_clears_entry_point_loader_caches(monkeypatch):
    """refresh_plugins() should clear framework + tool dependency entry-point caches."""
    loader = VerticalLoader()
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    framework_called = {"value": False}
    tool_dep_called = {"value": False}
    provider_cache_called = {"value": False}

    def _clear_framework_cache():
        framework_called["value"] = True

    def _clear_tool_dep_cache():
        tool_dep_called["value"] = True

    def _clear_provider_cache():
        provider_cache_called["value"] = True
        return 0

    monkeypatch.setattr(
        "victor.framework.entry_point_loader.clear_entry_point_loader_cache",
        _clear_framework_cache,
    )
    monkeypatch.setattr(
        "victor.core.tool_dependency_loader.clear_tool_dependency_entry_point_cache",
        _clear_tool_dep_cache,
    )
    monkeypatch.setattr(
        "victor.core.tool_dependency_loader.clear_vertical_tool_dependency_provider_cache",
        _clear_provider_cache,
    )

    loader.refresh_plugins()

    assert framework_called["value"] is True
    assert tool_dep_called["value"] is True
    assert provider_cache_called["value"] is True


def test_loader_skips_name_conflict_with_existing_vertical(monkeypatch):
    """VerticalLoader should skip entry-point verticals conflicting by name."""
    loader = VerticalLoader()
    loader._discovered_verticals = {}
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    existing_name = "conflict_vertical"
    existing_vertical = _make_vertical(existing_name, api_version=1)
    conflicting_vertical = _make_vertical(existing_name, api_version=1)
    VerticalRegistry.register(existing_vertical)

    monkeypatch.setattr(loader, "_load_entry_point", lambda *_: conflicting_vertical)
    loader._load_vertical_entries({"conflict_plugin": "fake.module:ConflictingVertical"})

    assert "conflict_plugin" not in loader._discovered_verticals
    assert VerticalRegistry.get(existing_name) is existing_vertical

    VerticalRegistry.unregister(existing_name)


def test_discovery_stats_include_dependency_and_entry_point_snapshots(monkeypatch):
    """get_discovery_stats() should include dependency + framework entry-point telemetry."""
    loader = VerticalLoader()
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    dependency_expected = {"total_requests": 3, "entry_point_resolutions": 2}
    framework_expected = {"tool_dependency_calls": 5, "cache_hits": 4}
    monkeypatch.setattr(
        "victor.core.tool_dependency_loader.get_tool_dependency_resolution_stats",
        lambda: dependency_expected,
    )
    monkeypatch.setattr(
        "victor.framework.entry_point_loader.get_entry_point_loader_stats",
        lambda: framework_expected,
    )

    stats = loader.get_discovery_stats()

    assert "tool_dependency_resolution" in stats
    assert stats["tool_dependency_resolution"] == dependency_expected
    assert "framework_entry_point_loader" in stats
    assert stats["framework_entry_point_loader"] == framework_expected
