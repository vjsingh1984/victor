"""Tests for VerticalLoader contract enforcement and cache invalidation hooks."""

import logging

from typing import List

from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor.core.verticals.vertical_loader import VerticalLoader
from victor_sdk import VerticalBase as SdkVerticalBase


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


def _make_sdk_vertical(name: str, api_version: int):
    """Create a minimal SDK-only vertical for loader tests."""

    vertical_name = name

    class _SdkTestVertical(SdkVerticalBase):
        name = vertical_name
        description = f"SDK test vertical {vertical_name}"
        VERTICAL_API_VERSION = api_version

        @classmethod
        def get_name(cls) -> str:
            return cls.name

        @classmethod
        def get_description(cls) -> str:
            return cls.description

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "sdk test prompt"

    return _SdkTestVertical


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


def test_loader_wraps_sdk_only_verticals_at_activation_time(monkeypatch):
    """SDK-only entry-point verticals should be discovered raw and activated as runtime shims."""

    loader = VerticalLoader()
    loader._discovered_verticals = {}
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    vertical_name = "sdk_only_vertical"
    VerticalRegistry.unregister(vertical_name)
    sdk_vertical = _make_sdk_vertical(vertical_name, api_version=1)

    monkeypatch.setattr(loader, "_load_entry_point", lambda *_: sdk_vertical)
    loader._load_vertical_entries({"sdk_plugin": "fake.module:SdkVertical"})
    loaded = loader.load(vertical_name)

    assert loader._discovered_verticals["sdk_plugin"] is sdk_vertical
    assert VerticalRegistry.get(vertical_name) is sdk_vertical
    assert loaded is loader.active_vertical
    assert loaded is not sdk_vertical
    assert loaded.__victor_sdk_source__ is sdk_vertical
    assert loaded.get_definition().name == vertical_name

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


def test_discover_verticals_force_refresh_bypasses_loader_cache(monkeypatch):
    """force_refresh should bypass the loader cache and rescan entry points."""

    loader = VerticalLoader()
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    first_vertical = _make_vertical("refresh_one", api_version=1)
    refreshed_vertical = _make_vertical("refresh_two", api_version=1)
    call_flags: list[bool] = []

    class _Cache:
        def get_entry_points(self, group: str, force_refresh: bool = False):
            assert group == "victor.verticals"
            call_flags.append(force_refresh)
            if force_refresh:
                return {"refresh_two": "fake.module:RefreshTwo"}
            return {"refresh_one": "fake.module:RefreshOne"}

        def invalidate(self, group: str):
            return 1

    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_cache",
        lambda: _Cache(),
    )
    monkeypatch.setattr(
        loader,
        "_load_entry_point",
        lambda name, value: refreshed_vertical if name == "refresh_two" else first_vertical,
    )

    first_result = loader.discover_verticals()
    cached_result = loader.discover_verticals()
    refreshed_result = loader.discover_verticals(force_refresh=True)

    assert call_flags == [False, True]
    assert list(first_result.keys()) == ["refresh_one"]
    assert cached_result is first_result
    assert list(refreshed_result.keys()) == ["refresh_two"]


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
    entry_point_cache_expected = {
        "groups_cached": 1,
        "groups": {"victor.verticals": {"entries": 2}},
    }
    monkeypatch.setattr(
        "victor.core.tool_dependency_loader.get_tool_dependency_resolution_stats",
        lambda: dependency_expected,
    )
    monkeypatch.setattr(
        "victor.framework.entry_point_loader.get_entry_point_loader_stats",
        lambda: framework_expected,
    )
    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_cache",
        lambda: type(
            "_Cache",
            (),
            {"get_cache_stats": staticmethod(lambda: entry_point_cache_expected)},
        )(),
    )

    stats = loader.get_discovery_stats()

    assert "tool_dependency_resolution" in stats
    assert stats["tool_dependency_resolution"] == dependency_expected
    assert "framework_entry_point_loader" in stats
    assert stats["framework_entry_point_loader"] == framework_expected
    assert "entry_point_cache" in stats
    assert stats["entry_point_cache"] == entry_point_cache_expected


def test_discover_verticals_logs_structured_telemetry(monkeypatch, caplog):
    """discover_verticals() should emit structured logging with cache context."""
    loader = VerticalLoader()
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    telemetry_stats = {
        "vertical": {"calls": 1, "cache_hits": 0, "scans": 1, "last_discovery_ms": 12.5},
        "entry_point_cache": {
            "groups_cached": 1,
            "groups": {"victor.verticals": {"entries": 2}},
        },
    }
    monkeypatch.setattr(loader, "get_discovery_stats", lambda: telemetry_stats)
    monkeypatch.setattr(
        loader,
        "_discover_verticals_internal",
        lambda force_refresh=False: (
            {
                "coding": _make_vertical("logging_coding", api_version=1),
                "research": _make_vertical("logging_research", api_version=1),
            },
            False,
            12.5,
        ),
    )

    with caplog.at_level(logging.INFO, logger="victor.core.verticals.vertical_loader"):
        loader.discover_verticals(force_refresh=True)

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "VERTICAL_DISCOVERY"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.discovery_kind == "vertical"
    assert record.discovered_count == 2
    assert record.force_refresh is True
    assert record.cache_hit is False
    assert record.entry_point_cache_group == "victor.verticals"
    assert record.entry_point_cache_group_stats == {"entries": 2}
    assert record.entry_point_groups_cached == 1


def test_refresh_plugins_logs_structured_telemetry(monkeypatch, caplog):
    """refresh_plugins() should emit structured logging with refresh/cache context."""
    loader = VerticalLoader()
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    class _Cache:
        def invalidate(self, group: str) -> int:
            return 1

    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_cache",
        lambda: _Cache(),
    )
    monkeypatch.setattr(
        "victor.core.verticals.extension_loader.VerticalExtensionLoader.clear_extension_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "victor.framework.vertical_service.clear_vertical_integration_pipeline_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        "victor.framework.entry_point_loader.clear_entry_point_loader_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        "victor.core.tool_dependency_loader.clear_tool_dependency_entry_point_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        "victor.core.tool_dependency_loader.clear_vertical_tool_dependency_provider_cache",
        lambda: 0,
    )
    monkeypatch.setattr(
        loader,
        "get_discovery_stats",
        lambda: {
            "refresh": {"count": 1, "last_refresh_ms": 4.2},
            "entry_point_cache": {"groups_cached": 0, "groups": {}},
        },
    )

    with caplog.at_level(logging.INFO, logger="victor.core.verticals.vertical_loader"):
        loader.refresh_plugins()

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "VERTICAL_PLUGIN_REFRESH"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.refresh_count == 1
    assert record.duration_ms == 4.2
    assert record.entry_point_groups_cached == 0
    assert record.refresh_stats == {"count": 1, "last_refresh_ms": 4.2}
