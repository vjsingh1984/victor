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

"""Integration tests for plugin-driven vertical loading.

These tests reflect the post-extraction architecture:
- external packages publish ``VictorPlugin`` objects via ``victor.plugins``
- plugin registration populates ``VerticalRegistry`` with vertical classes
- SDK-only vertical definitions are adapted into runtime ``VerticalBase``
  classes when the framework resolves or activates them

The legacy ``victor.verticals`` path remains covered as a compatibility shim.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from victor.core.plugins.registry import PluginRegistry
from victor.core.verticals.adapters import ensure_runtime_vertical
from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor.core.verticals.vertical_loader import VerticalLoader
from victor_sdk import PluginContext, VictorPlugin
from victor_sdk import VerticalBase as SdkVerticalBase
from victor_sdk.verticals import register_vertical


@register_vertical(
    name="sdk_external",
    version="1.0.0",
    min_framework_version=">=0.6.0",
    plugin_namespace="sdk.external",
)
class SdkOnlyExternalVertical(SdkVerticalBase):
    """SDK-only external vertical used to verify runtime adaptation."""

    name = "sdk_external"
    description = "SDK-only external vertical"
    version = "1.0.0"

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are an SDK-only test assistant."

    @classmethod
    def customize_config(cls, config):
        return config.with_metadata(runtime_adapter="sdk")


class _SdkExternalPlugin(VictorPlugin):
    """Plugin wrapper that registers the SDK-only vertical."""

    @property
    def name(self) -> str:
        return "sdk_external"

    def register(self, context: PluginContext) -> None:
        context.register_vertical(SdkOnlyExternalVertical)

    def get_cli_app(self) -> Optional[Any]:
        return None

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    async def on_activate_async(self) -> None:
        pass

    async def on_deactivate_async(self) -> None:
        pass

    def health_check(self) -> Dict[str, Any]:
        return {"healthy": True}


@register_vertical(
    name="legacy_sdk_external",
    version="1.0.0",
    min_framework_version=">=0.6.0",
    plugin_namespace="legacy.sdk.external",
)
class LegacySdkExternalVertical(SdkVerticalBase):
    """SDK vertical exposed via the legacy victor.verticals compatibility path."""

    name = "legacy_sdk_external"
    description = "Legacy SDK external vertical"
    version = "1.0.0"

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
        return "Legacy SDK-only prompt."


@pytest.fixture(autouse=True)
def reset_vertical_state():
    """Restore global registries after each test."""

    original_registry = dict(VerticalRegistry._registry)
    original_provenance = dict(VerticalRegistry._provenance)
    original_discovered = VerticalRegistry._external_discovered
    original_legacy_warning = VerticalRegistry._legacy_entry_point_warning_emitted
    original_plugin_instance = PluginRegistry._instance

    yield

    VerticalRegistry._registry = original_registry
    VerticalRegistry._provenance = original_provenance
    VerticalRegistry._external_discovered = original_discovered
    VerticalRegistry._legacy_entry_point_warning_emitted = original_legacy_warning
    PluginRegistry._instance = original_plugin_instance


@pytest.mark.integration
def test_plugin_registry_registers_sdk_vertical_into_vertical_registry(monkeypatch):
    """Canonical plugin discovery should register verticals through PluginContext."""

    registry = PluginRegistry()
    monkeypatch.setattr(
        "victor.core.plugins.registry.get_entry_point_cache",
        lambda: MagicMock(
            get_entry_points=MagicMock(
                return_value={"sdk_external": "victor_sdk_external.plugin:plugin"}
            )
        ),
    )
    monkeypatch.setattr(
        registry,
        "_load_plugin_from_value",
        lambda *_args, **_kwargs: _SdkExternalPlugin(),
    )

    registry.register_all(MagicMock())

    assert VerticalRegistry.get("sdk_external") is SdkOnlyExternalVertical


@pytest.mark.integration
def test_vertical_loader_resolve_adapts_registered_sdk_vertical():
    """SDK-only verticals should remain raw in the registry and adapt at runtime."""

    VerticalRegistry.register(SdkOnlyExternalVertical)

    resolved = VerticalLoader().resolve("sdk_external")

    assert VerticalRegistry.get("sdk_external") is SdkOnlyExternalVertical
    assert resolved is ensure_runtime_vertical(SdkOnlyExternalVertical)
    assert issubclass(resolved, VerticalBase)
    assert resolved is not SdkOnlyExternalVertical

    config = resolved.get_config(use_cache=False)
    assert config.metadata["runtime_adapter"] == "sdk"
    assert config.tools.tools == {"read", "write"}


@pytest.mark.integration
def test_vertical_loader_load_activates_sdk_vertical_through_adapter(monkeypatch):
    """Activation should operate on the runtime adapter, not the raw SDK class."""

    loader = VerticalLoader()
    VerticalRegistry.register(SdkOnlyExternalVertical)

    activated = []
    monkeypatch.setattr(loader, "_negotiate_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "_validate_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "_activate", lambda vertical: activated.append(vertical))

    resolved = loader.load("sdk_external")

    expected = ensure_runtime_vertical(SdkOnlyExternalVertical)
    assert resolved is expected
    assert activated == [expected]


@pytest.mark.integration
def test_legacy_vertical_registry_discovery_still_supports_victor_verticals(
    monkeypatch,
):
    """Legacy raw-vertical discovery should remain available for compatibility."""

    class _LegacyEntryPoint:
        name = "legacy_sdk_external"
        value = "legacy_pkg:LegacySdkExternalVertical"

        def load(self):
            return LegacySdkExternalVertical

    class _LegacyGroup:
        entry_points = {"legacy_sdk_external": (_LegacyEntryPoint(), False)}

    class _LegacyRegistry:
        def get_group(self, group_name: str):
            assert group_name == "victor.verticals"
            return _LegacyGroup()

    VerticalRegistry.reset_discovery()
    monkeypatch.setattr(
        "victor.framework.entry_point_registry.get_entry_point_registry",
        lambda: _LegacyRegistry(),
    )

    with pytest.warns(DeprecationWarning, match="victor.verticals"):
        discovered = VerticalRegistry.discover_external_verticals()

    assert discovered["legacy_sdk_external"] is LegacySdkExternalVertical
    assert VerticalRegistry.get("legacy_sdk_external") is LegacySdkExternalVertical


@pytest.mark.integration
def test_legacy_vertical_registry_warning_is_emitted_only_once(monkeypatch):
    """Compatibility discovery should warn once per reset cycle."""

    class _LegacyEntryPoint:
        name = "legacy_sdk_external"
        value = "legacy_pkg:LegacySdkExternalVertical"

        def load(self):
            return LegacySdkExternalVertical

    class _LegacyGroup:
        entry_points = {"legacy_sdk_external": (_LegacyEntryPoint(), False)}

    class _LegacyRegistry:
        def get_group(self, group_name: str):
            assert group_name == "victor.verticals"
            return _LegacyGroup()

    VerticalRegistry.reset_discovery()
    monkeypatch.setattr(
        "victor.framework.entry_point_registry.get_entry_point_registry",
        lambda: _LegacyRegistry(),
    )

    with pytest.warns(DeprecationWarning, match="victor.verticals"):
        VerticalRegistry.discover_external_verticals()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        VerticalRegistry.discover_external_verticals()
    assert not recorded

    VerticalRegistry.reset_discovery()
    monkeypatch.setattr(
        "victor.framework.entry_point_registry.get_entry_point_registry",
        lambda: _LegacyRegistry(),
    )
    with pytest.warns(DeprecationWarning, match="victor.verticals"):
        VerticalRegistry.discover_external_verticals()


@pytest.mark.integration
def test_canonical_and_legacy_entry_point_groups_are_explicit():
    """Current architecture keeps plugins canonical and legacy discovery optional."""

    assert PluginRegistry.ENTRY_POINT_GROUP == "victor.plugins"
    assert VerticalRegistry.ENTRY_POINT_GROUP == "victor.verticals"
