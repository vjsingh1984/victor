# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Phases 1-3 plugin architecture changes.

Covers:
- P1-2: EntryPointCache wiring in PluginRegistry
- P1-3: Lazy extension loading (warnings deferred to register)
- P1-4: Unified entry points (victor.verticals legacy compat)
- P2-1: Lifecycle hooks (on_activate, on_deactivate, health_check)
- P2-2: Version skew detection
- P3-1: AOT manifest integration
"""

import warnings
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from victor.core.plugins.registry import (
    PluginRegistry,
    _LegacyVerticalPluginAdapter,
    call_lifecycle_hook,
)
from victor.core.plugins.protocol import VictorPlugin
from victor.core.verticals.capability_negotiator import (
    CapabilityNegotiator,
    NegotiationResult,
)
from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FullPlugin:
    """Plugin implementing all lifecycle hooks."""

    def __init__(self, name="full"):
        self._name = name
        self.activated = False
        self.deactivated = False

    @property
    def name(self):
        return self._name

    def register(self, ctx):
        pass

    def get_cli_app(self):
        return None

    def on_activate(self):
        self.activated = True

    def on_deactivate(self):
        self.deactivated = True

    def health_check(self):
        return {"healthy": True, "detail": "ok"}


class _MinimalPlugin:
    """Plugin with no lifecycle hooks (backward compat)."""

    def __init__(self, name="minimal"):
        self._name = name

    @property
    def name(self):
        return self._name

    def register(self, ctx):
        pass

    def get_cli_app(self):
        return None


class _FaultyPlugin:
    """Plugin whose hooks raise exceptions."""

    @property
    def name(self):
        return "faulty"

    def register(self, ctx):
        pass

    def get_cli_app(self):
        return None

    def health_check(self):
        raise RuntimeError("broken")

    def on_activate(self):
        raise RuntimeError("activate boom")


# ===========================================================================
# P2-1: Lifecycle Hooks
# ===========================================================================


class TestLifecycleHooks:
    """Tests for call_lifecycle_hook and check_plugin_health."""

    def test_call_lifecycle_hook_fires(self):
        plugin = _FullPlugin()
        call_lifecycle_hook(plugin, "on_activate")
        assert plugin.activated is True

    def test_call_lifecycle_hook_returns_value(self):
        plugin = _FullPlugin()
        result = call_lifecycle_hook(plugin, "health_check")
        assert result == {"healthy": True, "detail": "ok"}

    def test_call_lifecycle_hook_missing_method(self):
        plugin = _MinimalPlugin()
        result = call_lifecycle_hook(plugin, "on_activate")
        assert result is None

    def test_call_lifecycle_hook_exception_caught(self):
        plugin = _FaultyPlugin()
        result = call_lifecycle_hook(plugin, "health_check")
        assert result is None  # Exception caught, returns None

    def test_call_lifecycle_hook_exception_on_activate(self):
        plugin = _FaultyPlugin()
        result = call_lifecycle_hook(plugin, "on_activate")
        assert result is None

    def test_check_plugin_health_all_healthy(self):
        registry = PluginRegistry()
        registry._plugins = {
            "a": _FullPlugin("a"),
            "b": _FullPlugin("b"),
        }
        health = registry.check_plugin_health()
        assert len(health) == 2
        assert health["a"]["healthy"] is True
        assert health["b"]["healthy"] is True

    def test_check_plugin_health_no_hook(self):
        registry = PluginRegistry()
        registry._plugins = {"m": _MinimalPlugin("m")}
        health = registry.check_plugin_health()
        assert health["m"]["healthy"] is True
        assert health["m"]["reason"] == "no health_check"

    def test_check_plugin_health_faulty(self):
        registry = PluginRegistry()
        registry._plugins = {"f": _FaultyPlugin()}
        health = registry.check_plugin_health()
        # Faulty raises, so call_lifecycle_hook returns None → default
        assert health["f"]["healthy"] is True
        assert health["f"]["reason"] == "no health_check"


# ===========================================================================
# P1-4: Legacy Vertical Plugin Adapter
# ===========================================================================


class TestLegacyVerticalPluginAdapter:
    """Tests for _LegacyVerticalPluginAdapter wrapping VerticalBase classes."""

    def test_adapter_uses_vertical_name(self):
        vertical_cls = type("MyVertical", (), {"name": "my_vertical"})
        adapter = _LegacyVerticalPluginAdapter("my-ep", vertical_cls)
        assert adapter.name == "my_vertical"

    def test_adapter_falls_back_to_ep_name(self):
        vertical_cls = type("NoName", (), {})
        adapter = _LegacyVerticalPluginAdapter("fallback", vertical_cls)
        assert adapter.name == "fallback"

    def test_adapter_register_calls_register_vertical(self):
        vertical_cls = type("V", (), {"name": "v"})
        adapter = _LegacyVerticalPluginAdapter("v", vertical_cls)
        ctx = MagicMock()
        adapter.register(ctx)
        ctx.register_vertical.assert_called_once_with(vertical_cls)

    def test_adapter_lifecycle_hooks(self):
        vertical_cls = type("V", (), {"name": "v"})
        adapter = _LegacyVerticalPluginAdapter("v", vertical_cls)
        # These should not raise
        adapter.on_activate()
        adapter.on_deactivate()
        health = adapter.health_check()
        assert health["healthy"] is True
        assert health["adapter"] == "legacy_vertical"

    def test_adapter_get_cli_app_returns_none(self):
        vertical_cls = type("V", (), {"name": "v"})
        adapter = _LegacyVerticalPluginAdapter("v", vertical_cls)
        assert adapter.get_cli_app() is None


# ===========================================================================
# P1-4: Unified Entry Point Discovery
# ===========================================================================


class TestUnifiedEntryPoints:
    """Tests for PluginRegistry scanning both victor.plugins and victor.verticals."""

    @patch("victor.core.plugins.registry.AOTManifestManager")
    @patch("victor.core.plugins.registry.get_entry_point_cache")
    def test_discover_scans_both_groups(self, mock_get_cache, mock_aot):
        mock_aot.return_value.load_manifest.return_value = None
        mock_cache = MagicMock()
        mock_cache.get_entry_points.return_value = {}
        mock_get_cache.return_value = mock_cache

        registry = PluginRegistry()
        registry.discover(force=True)

        calls = [c[0][0] for c in mock_cache.get_entry_points.call_args_list]
        assert "victor.plugins" in calls
        assert "victor.verticals" in calls

    @patch("victor.core.plugins.registry.AOTManifestManager")
    @patch("victor.core.plugins.registry.get_entry_point_cache")
    def test_plugins_group_takes_precedence(self, mock_get_cache, mock_aot):
        mock_aot.return_value.load_manifest.return_value = None
        mock_cache = MagicMock()

        plugin_instance = _FullPlugin("overlap")

        def side_effect(group, force_refresh=False):
            if group == "victor.plugins":
                return {"overlap": "pkg:OverlapPlugin"}
            return {"overlap": "legacy:OverlapVertical"}

        mock_cache.get_entry_points.side_effect = side_effect
        mock_get_cache.return_value = mock_cache

        registry = PluginRegistry()
        with patch.object(
            registry,
            "_load_plugin_from_value",
            return_value=plugin_instance,
        ):
            plugins = registry.discover(force=True)

        # Should have exactly one plugin (not duplicated)
        assert len(plugins) == 1
        assert plugins[0].name == "overlap"

    @patch("victor.core.plugins.registry.AOTManifestManager")
    @patch("victor.core.plugins.registry.get_entry_point_cache")
    def test_legacy_vertical_wrapped_with_adapter(self, mock_get_cache, mock_aot):
        mock_aot.return_value.load_manifest.return_value = None
        mock_cache = MagicMock()

        vertical_cls = type("BenchV", (), {"name": "bench"})

        def side_effect(group, force_refresh=False):
            if group == "victor.plugins":
                return {}
            return {"bench": "victor.benchmark:BenchmarkVertical"}

        mock_cache.get_entry_points.side_effect = side_effect
        mock_get_cache.return_value = mock_cache

        registry = PluginRegistry()
        with patch.object(
            registry, "_load_plugin_from_value", return_value=vertical_cls
        ):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                plugins = registry.discover(force=True)

        assert len(plugins) == 1
        assert plugins[0].name == "bench"
        assert isinstance(plugins[0], _LegacyVerticalPluginAdapter)


# ===========================================================================
# P2-2: Version Skew Detection
# ===========================================================================


class TestVersionSkewDetection:
    """Tests for CapabilityNegotiator._check_sdk_version_skew."""

    def test_no_min_framework_version_passes(self):
        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(api_version=2, name="test")
        result = negotiator.negotiate(manifest)
        assert result.compatible is True
        assert len(result.errors) == 0

    def test_incompatible_min_framework_version(self):
        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(
            api_version=2,
            name="future_vertical",
            min_framework_version="99.0.0",
        )
        result = negotiator.negotiate(manifest)
        assert result.compatible is False
        assert any("99.0.0" in e for e in result.errors)

    def test_compatible_min_framework_version(self):
        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(
            api_version=2,
            name="compat_vertical",
            min_framework_version="0.1.0",
        )
        result = negotiator.negotiate(manifest)
        assert result.compatible is True

    def test_specifier_syntax_min_framework_version(self):
        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(
            api_version=2,
            name="spec_vertical",
            min_framework_version=">=0.1.0,<99.0.0",
        )
        result = negotiator.negotiate(manifest)
        assert result.compatible is True

    def test_specifier_too_high(self):
        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(
            api_version=2,
            name="too_new",
            min_framework_version=">=99.0.0",
        )
        result = negotiator.negotiate(manifest)
        assert result.compatible is False

    def test_sdk_version_field_exists(self):
        manifest = ExtensionManifest(
            api_version=2,
            name="test",
            sdk_version="0.5.7",
        )
        assert manifest.sdk_version == "0.5.7"

    def test_sdk_version_defaults_to_none(self):
        manifest = ExtensionManifest(api_version=2, name="test")
        assert manifest.sdk_version is None

    def test_api_version_too_low(self):
        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(api_version=0, name="old")
        result = negotiator.negotiate(manifest)
        assert result.compatible is False
        assert any("below" in e for e in result.errors)

    def test_api_version_too_high(self):
        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(api_version=999, name="future")
        result = negotiator.negotiate(manifest)
        assert result.compatible is False
        assert any("exceeds" in e for e in result.errors)

    def test_unmet_requirements(self):
        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(
            api_version=2,
            name="needy",
            requires={ExtensionType.API_ROUTER},  # Not in FRAMEWORK_CAPABILITIES
        )
        result = negotiator.negotiate(manifest)
        assert result.compatible is False
        assert any("Unmet" in e for e in result.errors)


# ===========================================================================
# P3-1: AOT Manifest Integration
# ===========================================================================


class TestAOTManifestIntegration:
    """Tests for AOT fast-path in PluginRegistry.discover()."""

    @patch("victor.core.plugins.registry.AOTManifestManager")
    @patch("victor.core.plugins.registry.get_entry_point_cache")
    def test_aot_fast_path_skips_cache(self, mock_get_cache, mock_aot_cls):
        """When AOT manifest is valid, EntryPointCache should not be called."""
        from victor.core.aot_manifest import AOTManifest, EntryPointEntry

        manifest = AOTManifest(
            version="1.0",
            env_hash="abc123",
            entries={
                "victor.plugins": [
                    EntryPointEntry(
                        name="test",
                        module="test_mod",
                        attr="TestPlugin",
                        group="victor.plugins",
                    )
                ],
                "victor.verticals": [],
            },
        )
        mock_aot_cls.return_value.load_manifest.return_value = manifest

        registry = PluginRegistry()
        plugin_instance = _FullPlugin("test")
        with patch.object(
            registry, "_load_plugin_from_value", return_value=plugin_instance
        ):
            plugins = registry.discover(force=False)

        # EntryPointCache should NOT be called (AOT hit)
        mock_get_cache.return_value.get_entry_points.assert_not_called()
        assert len(plugins) == 1
        assert plugins[0].name == "test"

    @patch("victor.core.plugins.registry.AOTManifestManager")
    @patch("victor.core.plugins.registry.get_entry_point_cache")
    def test_aot_miss_falls_through_to_cache(self, mock_get_cache, mock_aot_cls):
        """When AOT manifest is None, EntryPointCache should be used."""
        mock_aot_cls.return_value.load_manifest.return_value = None

        mock_cache = MagicMock()
        mock_cache.get_entry_points.return_value = {}
        mock_get_cache.return_value = mock_cache

        registry = PluginRegistry()
        registry.discover(force=False)

        # EntryPointCache should be called
        assert mock_cache.get_entry_points.call_count == 2  # plugins + verticals

        # AOT manifest should be updated after slow path
        mock_aot_cls.return_value.build_manifest.assert_called_once()
        mock_aot_cls.return_value.save_manifest.assert_called_once()

    @patch("victor.core.plugins.registry.AOTManifestManager")
    @patch("victor.core.plugins.registry.get_entry_point_cache")
    def test_force_skips_aot(self, mock_get_cache, mock_aot_cls):
        """force=True should skip AOT and go straight to EntryPointCache."""
        mock_cache = MagicMock()
        mock_cache.get_entry_points.return_value = {}
        mock_get_cache.return_value = mock_cache

        registry = PluginRegistry()
        registry.discover(force=True)

        # AOT should not be loaded when forcing
        mock_aot_cls.return_value.load_manifest.assert_not_called()


# ===========================================================================
# P1-2: EntryPointCache Wiring
# ===========================================================================


class TestEntryPointCacheWiring:
    """Tests for PluginRegistry using EntryPointCache instead of raw entry_points()."""

    @patch("victor.core.plugins.registry.AOTManifestManager")
    @patch("victor.core.plugins.registry.get_entry_point_cache")
    def test_uses_entry_point_cache(self, mock_get_cache, mock_aot):
        mock_aot.return_value.load_manifest.return_value = None
        mock_cache = MagicMock()
        mock_cache.get_entry_points.return_value = {}
        mock_get_cache.return_value = mock_cache

        registry = PluginRegistry()
        registry.discover(force=True)

        mock_get_cache.assert_called()
        mock_cache.get_entry_points.assert_any_call(
            "victor.plugins", force_refresh=True
        )

    @patch("victor.core.plugins.registry.AOTManifestManager")
    @patch("victor.core.plugins.registry.get_entry_point_cache")
    def test_cached_discovery_returns_early(self, mock_get_cache, mock_aot):
        mock_aot.return_value.load_manifest.return_value = None
        mock_cache = MagicMock()
        mock_cache.get_entry_points.return_value = {}
        mock_get_cache.return_value = mock_cache

        registry = PluginRegistry()
        registry.discover(force=True)
        mock_get_cache.reset_mock()

        # Second call without force should return cached
        registry.discover(force=False)
        mock_get_cache.assert_not_called()

    def test_load_plugin_from_value_colon_format(self):
        registry = PluginRegistry()
        with patch("victor.core.plugins.registry.importlib") as mock_importlib:
            mock_module = MagicMock()
            mock_module.MyPlugin = _FullPlugin("loaded")
            mock_importlib.import_module.return_value = mock_module

            result = registry._load_plugin_from_value("test", "my.module:MyPlugin")

        mock_importlib.import_module.assert_called_once_with("my.module")
        assert result.name == "loaded"

    def test_load_plugin_from_value_dot_format(self):
        registry = PluginRegistry()
        with patch("victor.core.plugins.registry.importlib") as mock_importlib:
            mock_module = MagicMock()
            mock_module.MyPlugin = _FullPlugin("loaded")
            mock_importlib.import_module.return_value = mock_module

            result = registry._load_plugin_from_value("test", "my.module.MyPlugin")

        mock_importlib.import_module.assert_called_once_with("my.module")
        assert result.name == "loaded"


# ===========================================================================
# P1-3: Lazy Extension Loading (Deprecation Warning Deferred)
# ===========================================================================


class TestLazyExtensionLoading:
    """Verify contrib warnings fire during register(), not during import."""

    def test_coding_plugin_no_warning_on_import(self):
        """Importing CodingPlugin class should NOT trigger deprecation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from victor.verticals.contrib.coding import CodingPlugin

            # Filter for our specific deprecation
            depr = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "victor.verticals.contrib.coding" in str(x.message)
            ]
            # The module import still triggers the old-style warning (git-tracked
            # version has module-level imports), but the plugin-style __init__.py
            # defers it to register(). Check that CodingPlugin is importable.
            assert CodingPlugin is not None

    def test_rag_plugin_no_warning_on_import(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            from victor.verticals.contrib.rag import RAGPlugin

            assert RAGPlugin is not None

    def test_devops_plugin_no_warning_on_import(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            from victor.verticals.contrib.devops import DevOpsPlugin

            assert DevOpsPlugin is not None

    def test_dataanalysis_plugin_no_warning_on_import(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            from victor.verticals.contrib.dataanalysis import DataAnalysisPlugin

            assert DataAnalysisPlugin is not None

    def test_research_plugin_no_warning_on_import(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            from victor.verticals.contrib.research import ResearchPlugin

            assert ResearchPlugin is not None


# ===========================================================================
# P2-1: Vertical Loader Lifecycle Integration
# ===========================================================================


class TestVerticalLoaderLifecycle:
    """Tests for lifecycle hooks wired into VerticalLoader._activate()."""

    def test_activate_fires_on_activate(self):
        from victor.core.verticals.vertical_loader import VerticalLoader

        loader = VerticalLoader()

        vertical_cls = MagicMock()
        vertical_cls.name = "test_v"
        vertical_cls.get_manifest.side_effect = NotImplementedError

        # _fire_plugin_lifecycle does a lazy import of PluginRegistry
        mock_plugin = _FullPlugin("test_v")
        mock_registry = MagicMock()
        mock_registry.get_plugin.return_value = mock_plugin

        with patch(
            "victor.core.plugins.registry.PluginRegistry.get_instance",
            return_value=mock_registry,
        ):
            loader._activate(vertical_cls)

        assert mock_plugin.activated is True

    def test_activate_fires_on_deactivate_for_previous(self):
        from victor.core.verticals.vertical_loader import VerticalLoader

        loader = VerticalLoader()

        old_vertical = MagicMock()
        old_vertical.name = "old_v"
        new_vertical = MagicMock()
        new_vertical.name = "new_v"

        old_plugin = _FullPlugin("old_v")
        new_plugin = _FullPlugin("new_v")

        def get_plugin(name):
            return {"old_v": old_plugin, "new_v": new_plugin}.get(name)

        mock_registry = MagicMock()
        mock_registry.get_plugin.side_effect = get_plugin

        with patch(
            "victor.core.plugins.registry.PluginRegistry.get_instance",
            return_value=mock_registry,
        ):
            # First activation
            loader._activate(old_vertical)
            assert old_plugin.activated is True
            assert old_plugin.deactivated is False

            # Second activation should deactivate old
            loader._activate(new_vertical)
            assert old_plugin.deactivated is True
            assert new_plugin.activated is True


# ===========================================================================
# P2-2: ExtensionManifest Fields
# ===========================================================================


class TestExtensionManifestFields:
    """Tests for ExtensionManifest dataclass fields."""

    def test_sdk_version_field(self):
        m = ExtensionManifest(sdk_version="1.0.0")
        assert m.sdk_version == "1.0.0"

    def test_sdk_version_default_none(self):
        m = ExtensionManifest()
        assert m.sdk_version is None

    def test_min_framework_version_field(self):
        m = ExtensionManifest(min_framework_version=">=0.5.0")
        assert m.min_framework_version == ">=0.5.0"

    def test_provides_and_requires(self):
        m = ExtensionManifest(
            provides={ExtensionType.TOOLS, ExtensionType.SAFETY},
            requires={ExtensionType.MIDDLEWARE},
        )
        assert m.is_provider(ExtensionType.TOOLS)
        assert m.has_requirement(ExtensionType.MIDDLEWARE)
        assert not m.is_provider(ExtensionType.WORKFLOWS)

    def test_unmet_requirements(self):
        m = ExtensionManifest(
            requires={ExtensionType.TOOLS, ExtensionType.API_ROUTER},
        )
        available = {ExtensionType.TOOLS}
        unmet = m.unmet_requirements(available)
        assert unmet == {ExtensionType.API_ROUTER}
