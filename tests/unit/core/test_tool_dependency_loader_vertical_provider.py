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

"""Tests for vertical tool dependency provider resolution fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from victor.core.tool_dependency_base import BaseToolDependencyProvider
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
from victor.core.tool_types import EmptyToolDependencyProvider
import victor.core.tool_dependency_loader as loader_mod


@pytest.fixture(autouse=True)
def _clear_tool_dependency_ep_cache() -> None:
    """Keep entry-point cache isolated between tests."""
    loader_mod.clear_tool_dependency_entry_point_cache()
    yield
    loader_mod.clear_tool_dependency_entry_point_cache()


def test_entry_point_provider_has_priority_over_fallbacks(monkeypatch) -> None:
    """Entry point providers should short-circuit module/resource fallbacks."""
    sentinel = EmptyToolDependencyProvider("coding")

    class _FakeEntryPoint:
        name = "coding"

        @staticmethod
        def load():
            return lambda: sentinel

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])

    def _unexpected_import(_module_name: str):
        raise AssertionError("fallback import should not execute when entry point resolves")

    monkeypatch.setattr(loader_mod, "import_module_with_fallback", _unexpected_import)

    provider = create_vertical_tool_dependency_provider("coding")

    assert provider is sentinel


def test_entry_point_alias_name_matches_normalized_vertical(monkeypatch) -> None:
    """Entry point names with alias spellings should resolve correctly."""
    sentinel = EmptyToolDependencyProvider("dataanalysis")

    class _FakeEntryPoint:
        name = "data-analysis"

        @staticmethod
        def load():
            return lambda: sentinel

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [_FakeEntryPoint()])

    def _unexpected_import(_module_name: str):
        raise AssertionError("fallback import should not execute when entry point resolves")

    monkeypatch.setattr(loader_mod, "import_module_with_fallback", _unexpected_import)

    provider = create_vertical_tool_dependency_provider("data_analysis")

    assert provider is sentinel


def test_module_factory_fallback_is_used_when_entry_points_missing(monkeypatch) -> None:
    """Resolver should load module-level get_provider before YAML/resource fallbacks."""
    sentinel = EmptyToolDependencyProvider("coding")
    fake_module = SimpleNamespace(get_provider=lambda: sentinel)

    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])

    calls: list[str] = []

    def _fake_import(module_name: str):
        calls.append(module_name)
        return fake_module, "victor_coding.tool_dependencies"

    monkeypatch.setattr(loader_mod, "import_module_with_fallback", _fake_import)

    provider = create_vertical_tool_dependency_provider("coding")

    assert provider is sentinel
    assert calls == ["victor.coding.tool_dependencies"]


@pytest.mark.parametrize(
    "vertical",
    ["coding", "devops", "research", "rag", "dataanalysis"],
)
def test_package_resource_yaml_fallback_loads_contrib_config(monkeypatch, vertical: str) -> None:
    """Known verticals should load bundled YAML when providers are unavailable."""
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    provider = create_vertical_tool_dependency_provider(vertical)

    assert isinstance(provider, BaseToolDependencyProvider)
    assert not isinstance(provider, EmptyToolDependencyProvider)
    assert provider.get_required_tools() or provider.get_dependencies()


def test_unknown_vertical_raises_value_error_when_unresolved(monkeypatch) -> None:
    """Unknown verticals should still fail fast when no provider can be resolved."""
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    with pytest.raises(ValueError, match="Unknown vertical"):
        create_vertical_tool_dependency_provider("mlops")


def test_known_vertical_returns_empty_provider_when_resource_lookup_fails(monkeypatch) -> None:
    """Known verticals should degrade to EmptyToolDependencyProvider on lookup failure."""
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    def _missing_resource(_package_name: str):
        raise ModuleNotFoundError("simulated missing package")

    monkeypatch.setattr(loader_mod, "files", _missing_resource)

    provider = create_vertical_tool_dependency_provider("coding")

    assert isinstance(provider, EmptyToolDependencyProvider)
    assert provider.vertical == "coding"


@pytest.mark.parametrize("alias_name", ["data-analysis", "data_analysis"])
def test_dataanalysis_aliases_resolve_to_supported_vertical(monkeypatch, alias_name: str) -> None:
    """Historical data-analysis spellings should resolve via compatibility fallback."""
    monkeypatch.setattr(loader_mod, "entry_points", lambda group: [])
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    provider = create_vertical_tool_dependency_provider(alias_name)

    assert isinstance(provider, BaseToolDependencyProvider)
    assert not isinstance(provider, EmptyToolDependencyProvider)


def test_tool_dependency_entry_point_queries_are_cached(monkeypatch) -> None:
    """Entry point scans should be cached across repeated resolution calls."""
    call_count = 0

    def _fake_entry_points(group: str):
        nonlocal call_count
        assert group == "victor.tool_dependencies"
        call_count += 1
        return []

    monkeypatch.setattr(loader_mod, "entry_points", _fake_entry_points)
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    with pytest.raises(ValueError, match="Unknown vertical"):
        create_vertical_tool_dependency_provider("mlops")
    with pytest.raises(ValueError, match="Unknown vertical"):
        create_vertical_tool_dependency_provider("mlops")

    assert call_count == 1


def test_clear_tool_dependency_entry_point_cache_forces_rescan(monkeypatch) -> None:
    """Cache clear helper should force a fresh entry-point query."""
    call_count = 0

    def _fake_entry_points(group: str):
        nonlocal call_count
        assert group == "victor.tool_dependencies"
        call_count += 1
        return []

    monkeypatch.setattr(loader_mod, "entry_points", _fake_entry_points)
    monkeypatch.setattr(loader_mod, "import_module_with_fallback", lambda _: (None, None))

    with pytest.raises(ValueError, match="Unknown vertical"):
        create_vertical_tool_dependency_provider("mlops")
    loader_mod.clear_tool_dependency_entry_point_cache()
    with pytest.raises(ValueError, match="Unknown vertical"):
        create_vertical_tool_dependency_provider("mlops")

    assert call_count == 2
