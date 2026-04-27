"""Tests for VerticalLoader contract enforcement and cache invalidation hooks."""

import logging

from typing import List
from unittest.mock import Mock

import pytest

from victor.core import tool_dependency_loader
from victor.core.verticals.adapters import ensure_runtime_vertical
from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor.core.verticals.manifest_contract import (
    VerticalRuntimeProvenance,
    set_vertical_runtime_provenance,
)
from victor.core.verticals.vertical_loader import VerticalLoader
from victor.framework import entry_point_loader
from victor_sdk import VerticalBase as SdkVerticalBase
from victor_sdk.verticals.manifest import ExtensionManifest


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

    assert vertical_name in loader._discovered_verticals
    assert loader._discovered_verticals[vertical_name] is current_vertical
    assert VerticalRegistry.get(vertical_name) is current_vertical

    VerticalRegistry.unregister(vertical_name)


def test_loader_passes_sdk_verticals_through_at_activation_time(monkeypatch):
    """SDK entry-point verticals should be adapted at activation time."""

    loader = VerticalLoader()
    loader._discovered_verticals = {}
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    vertical_name = "sdk_only_vertical"
    sdk_vertical = _make_sdk_vertical(vertical_name, api_version=1)
    VerticalRegistry.unregister(vertical_name)

    activated = []
    monkeypatch.setattr(loader, "_negotiate_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "_validate_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "_activate", lambda vertical: activated.append(vertical))

    VerticalRegistry.register(sdk_vertical)
    resolved = loader.load(vertical_name)

    expected = ensure_runtime_vertical(sdk_vertical)

    assert resolved is expected
    assert activated == [expected]
    assert issubclass(resolved, VerticalBase)
    assert VerticalRegistry.get(vertical_name) is sdk_vertical

    VerticalRegistry.unregister(vertical_name)


def test_vertical_registry_register_attaches_manifest_for_legacy_sdk_vertical():
    """Legacy registration should synthesize and attach a normalized manifest."""

    class _LegacySdkVertical(SdkVerticalBase):
        name = "legacy_manifest_vertical"
        description = "Legacy manifest test"
        version = "2.3.4"

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
            return "legacy prompt"

    VerticalRegistry.unregister(_LegacySdkVertical.name)

    VerticalRegistry.register(_LegacySdkVertical)

    manifest = getattr(_LegacySdkVertical, "_victor_manifest", None)
    assert isinstance(manifest, ExtensionManifest)
    assert manifest.name == "legacy_manifest_vertical"
    assert manifest.version == "2.3.4"

    VerticalRegistry.unregister(_LegacySdkVertical.name)


def test_loader_validation_does_not_invoke_runtime_prompt_or_tool_methods(monkeypatch):
    """Entry-point validation should not execute runtime-heavy vertical methods."""

    loader = VerticalLoader()
    loader._discovered_verticals = {}
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    class _RuntimeSensitiveVertical(SdkVerticalBase):
        name = "runtime_sensitive_vertical"
        description = "Should validate without executing runtime hooks"

        @classmethod
        def get_name(cls) -> str:
            return cls.name

        @classmethod
        def get_description(cls) -> str:
            return cls.description

        @classmethod
        def get_tools(cls) -> List[str]:
            raise AssertionError("get_tools must not run during discovery")

        @classmethod
        def get_system_prompt(cls) -> str:
            raise AssertionError("get_system_prompt must not run during discovery")

        @classmethod
        def get_manifest(cls) -> ExtensionManifest:
            return ExtensionManifest(name=cls.name, version="1.0.0", api_version=1)

    VerticalRegistry.unregister(_RuntimeSensitiveVertical.name)
    monkeypatch.setattr(loader, "_load_entry_point", lambda *_: _RuntimeSensitiveVertical)

    loader._load_vertical_entries({"runtime_sensitive": "fake.module:RuntimeSensitiveVertical"})

    assert loader._discovered_verticals[_RuntimeSensitiveVertical.name] is _RuntimeSensitiveVertical
    assert VerticalRegistry.get(_RuntimeSensitiveVertical.name) is _RuntimeSensitiveVertical

    VerticalRegistry.unregister(_RuntimeSensitiveVertical.name)


def test_loader_resolves_requested_entry_point_without_importing_all_verticals(
    monkeypatch,
):
    """Single-vertical resolution should only import the requested entry point."""

    loader = VerticalLoader()
    requested = _make_sdk_vertical("requested_vertical", api_version=1)

    class _EntryPoint:
        def __init__(self, value: str) -> None:
            self.value = value

    class _Registry:
        def invalidate(self) -> None:
            return None

        def get_group(self, group: str):
            assert group == "victor.plugins"
            return type(
                "_Group",
                (),
                {
                    "entry_points": {
                        "requested_vertical": (_EntryPoint("pkg.requested:Vertical"), False),
                        "unused_vertical": (_EntryPoint("pkg.unused:Vertical"), False),
                    }
                },
            )()

    def _load_entry_point(name: str, value: str):
        if name != "requested_vertical":
            raise AssertionError(f"unexpected import: {name} -> {value}")
        return requested

    VerticalRegistry.unregister("requested_vertical")
    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_registry",
        lambda: _Registry(),
    )
    monkeypatch.setattr(loader, "_load_entry_point", _load_entry_point)

    resolved = loader.resolve("requested_vertical")

    assert resolved is ensure_runtime_vertical(requested)
    assert issubclass(resolved, VerticalBase)
    assert VerticalRegistry.get("requested_vertical") is requested

    VerticalRegistry.unregister("requested_vertical")


def test_ensure_runtime_vertical_reuses_cached_adapter_for_sdk_vertical():
    """SDK vertical adaptation should be idempotent and cache adapter classes."""

    requested = _make_sdk_vertical("cached_sdk_vertical", api_version=1)

    first = ensure_runtime_vertical(requested)
    second = ensure_runtime_vertical(requested)

    assert first is second
    assert issubclass(first, VerticalBase)
    assert first is not requested


def test_manifest_negotiation_enforces_min_framework_version(monkeypatch):
    """Manifest negotiation should honor min_framework_version on normalized manifests."""

    loader = VerticalLoader()
    vertical = _make_sdk_vertical("future_framework_vertical", api_version=2)
    vertical._victor_manifest = ExtensionManifest(
        api_version=2,
        name="future_framework_vertical",
        version="1.0.0",
        min_framework_version=">=9.9.9",
    )

    monkeypatch.setattr(
        "victor.core.verticals.compatibility_gate.get_framework_version",
        lambda: "1.2.3",
    )
    monkeypatch.setattr(
        "victor.core.verticals.compatibility_gate.CapabilityNegotiator.negotiate",
        lambda self, manifest: Mock(compatible=True, warnings=[], errors=[]),
    )
    monkeypatch.setattr(
        "victor.core.verticals.compatibility_gate.get_compatibility_matrix",
        lambda: Mock(
            is_loaded=lambda: True,
            load_default_rules=lambda: None,
            check_compatibility=lambda **kwargs: Mock(
                is_incompatible=False,
                message="",
                required_features=[],
                status=Mock(value="compatible"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not meet requirement >=9.9.9"):
        loader._negotiate_manifest(vertical)


def test_manifest_negotiation_uses_runtime_framework_version_for_matrix(monkeypatch):
    """Version-matrix compatibility should use the normalized framework version helper."""

    loader = VerticalLoader()
    vertical = _make_sdk_vertical("matrix_version_vertical", api_version=2)
    vertical._victor_manifest = ExtensionManifest(
        api_version=2,
        name="matrix_version_vertical",
        version="1.0.0",
        min_framework_version=">=0.0.1",
    )
    observed: dict[str, str] = {}

    def _check_compatibility(*, vertical_name: str, vertical_version: str, framework_version: str):
        observed["vertical_name"] = vertical_name
        observed["vertical_version"] = vertical_version
        observed["framework_version"] = framework_version
        return Mock(
            is_incompatible=False,
            message="",
            required_features=[],
            status=Mock(value="compatible"),
        )

    monkeypatch.setattr(
        "victor.core.verticals.compatibility_gate.get_framework_version",
        lambda: "2.4.6",
    )
    monkeypatch.setattr(
        "victor.core.verticals.compatibility_gate.CapabilityNegotiator.negotiate",
        lambda self, manifest: Mock(compatible=True, warnings=[], errors=[]),
    )
    monkeypatch.setattr(
        "victor.core.verticals.compatibility_gate.get_compatibility_matrix",
        lambda: Mock(
            is_loaded=lambda: True,
            load_default_rules=lambda: None,
            check_compatibility=_check_compatibility,
        ),
    )

    loader._negotiate_manifest(vertical)

    assert observed == {
        "vertical_name": "matrix_version_vertical",
        "vertical_version": "1.0.0",
        "framework_version": "2.4.6",
    }


def test_discover_vertical_names_uses_entry_point_metadata_only(monkeypatch):
    """Fast name discovery should not import entry-point modules."""

    loader = VerticalLoader()

    class _EntryPoint:
        def __init__(self, value: str) -> None:
            self.value = value

    class _Registry:
        def invalidate(self) -> None:
            return None

        def get_group(self, group: str):
            assert group == "victor.plugins"
            return type(
                "_Group",
                (),
                {
                    "entry_points": {
                        "coding": (_EntryPoint("victor_coding:Assistant"), False),
                        "research": (_EntryPoint("victor_research:Assistant"), False),
                    }
                },
            )()

    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_registry",
        lambda: _Registry(),
    )
    monkeypatch.setattr(
        loader,
        "_load_entry_point",
        lambda *_: (_ for _ in ()).throw(AssertionError("name discovery must not import")),
    )

    assert loader.discover_vertical_names(force_refresh=True) == ["coding", "research"]


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
        entry_point_loader, "clear_entry_point_loader_cache", _clear_framework_cache
    )
    monkeypatch.setattr(
        tool_dependency_loader,
        "clear_tool_dependency_entry_point_cache",
        _clear_tool_dep_cache,
    )
    monkeypatch.setattr(
        tool_dependency_loader,
        "clear_vertical_tool_dependency_provider_cache",
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

    class _EntryPoint:
        def __init__(self, value: str) -> None:
            self.value = value

    class _Registry:
        def __init__(self) -> None:
            self.invalidations = 0

        def get_group(self, group: str):
            assert group == "victor.plugins"
            force_refresh = self.invalidations > 0
            call_flags.append(force_refresh)
            entries = (
                {"refresh_two": (_EntryPoint("fake.module:RefreshTwo"), False)}
                if force_refresh
                else {"refresh_one": (_EntryPoint("fake.module:RefreshOne"), False)}
            )
            return type("_Group", (), {"entry_points": entries})()

        def invalidate(self) -> None:
            self.invalidations += 1

    registry = _Registry()

    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        loader,
        "_load_entry_point",
        lambda name, value: (refreshed_vertical if name == "refresh_two" else first_vertical),
    )

    first_result = loader.discover_verticals()
    cached_result = loader.discover_verticals()
    refreshed_result = loader.discover_verticals(force_refresh=True)

    assert call_flags == [False, True]
    assert list(first_result.keys()) == ["refresh_one"]
    assert cached_result is first_result
    assert list(refreshed_result.keys()) == ["refresh_two"]


def test_discover_tools_uses_shared_entry_point_values(monkeypatch):
    """Tool discovery should use the shared entry-point discovery helper."""
    loader = VerticalLoader()
    loader._emit_observability_event = lambda *args, **kwargs: None
    loader._emit_observability_event_async = lambda *args, **kwargs: None

    call_flags: list[tuple[str, bool]] = []
    tool_cls = type("SharedDiscoveredTool", (), {})

    def _get_values(group: str, *, force: bool = False):
        call_flags.append((group, force))
        return {"tool_a": "fake.module:ToolA"}

    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_values",
        _get_values,
    )
    monkeypatch.setattr(loader, "_load_entry_point", lambda *_: tool_cls)

    first_result = loader.discover_tools()
    cached_result = loader.discover_tools()
    refreshed_result = loader.discover_tools(force_refresh=True)

    assert call_flags == [("victor.tools", False), ("victor.tools", True)]
    assert first_result["tool_a"] is tool_cls
    assert cached_result is first_result
    assert refreshed_result["tool_a"] is tool_cls


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


def test_vertical_registry_prefers_explicit_external_provenance_over_contrib() -> None:
    """Registry conflict resolution should use explicit provenance metadata."""
    vertical_name = "provenance_conflict_vertical"
    VerticalRegistry.unregister(vertical_name)

    contrib_vertical = _make_vertical(vertical_name, api_version=1)
    contrib_vertical.__module__ = "victor.runtime.shadowed"
    set_vertical_runtime_provenance(contrib_vertical, VerticalRuntimeProvenance.CONTRIB)

    external_vertical = _make_vertical(vertical_name, api_version=1)
    external_vertical.__module__ = "victor.verticals.contrib.shadowed_external"
    set_vertical_runtime_provenance(external_vertical, VerticalRuntimeProvenance.EXTERNAL)

    VerticalRegistry.register(contrib_vertical)
    VerticalRegistry.register(external_vertical)

    assert VerticalRegistry.get(vertical_name) is external_vertical

    VerticalRegistry.unregister(vertical_name)


def test_loader_resolve_prefers_explicit_external_provenance(monkeypatch) -> None:
    """Resolve should honor explicit provenance instead of module-path heuristics."""
    loader = VerticalLoader()
    vertical_name = "resolve_provenance_vertical"
    VerticalRegistry.unregister(vertical_name)

    registered_vertical = _make_vertical(vertical_name, api_version=1)
    registered_vertical.__module__ = "victor.runtime.shadowed"
    set_vertical_runtime_provenance(registered_vertical, VerticalRuntimeProvenance.CONTRIB)
    VerticalRegistry.register(registered_vertical)

    entry_point_vertical = _make_vertical(vertical_name, api_version=1)
    entry_point_vertical.__module__ = "victor.verticals.contrib.entrypoint_shadow"
    set_vertical_runtime_provenance(entry_point_vertical, VerticalRuntimeProvenance.EXTERNAL)

    monkeypatch.setattr(loader, "_get_vertical_entry_points", lambda force_refresh=False: {})
    monkeypatch.setattr(loader, "_import_from_entrypoint", lambda *_args, **_kwargs: entry_point_vertical)

    resolved = loader.resolve(vertical_name)

    assert resolved is ensure_runtime_vertical(entry_point_vertical)

    VerticalRegistry.unregister(vertical_name)


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
        "vertical": {
            "calls": 1,
            "cache_hits": 0,
            "scans": 1,
            "last_discovery_ms": 12.5,
        },
        "entry_point_cache": {
            "groups_cached": 1,
            "groups": {"victor.plugins": {"entries": 2}},
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
    assert record.entry_point_cache_group == "victor.plugins"
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

    class _Registry:
        def __init__(self) -> None:
            self.invalidated = 0

        def invalidate(self) -> None:
            self.invalidated += 1

    registry = _Registry()

    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_cache",
        lambda: _Cache(),
    )
    monkeypatch.setattr(
        "victor.core.verticals.vertical_loader.get_entry_point_registry",
        lambda: registry,
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

    assert registry.invalidated == 1

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
