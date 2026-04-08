"""Regression tests for explicit runtime-extension entry-point resolution."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from victor.core.verticals.extension_loader import VerticalExtensionLoader
from victor.core.verticals.metadata import VerticalMetadataProvider
from victor.framework import entry_point_loader


class ResearchAssistant(VerticalExtensionLoader, VerticalMetadataProvider):
    """Minimal vertical class used to test entry-point-based extension resolution."""

    name = "research"
    description = "Research assistant"
    version = "1.0.0"


def _register_package(monkeypatch: pytest.MonkeyPatch, module_name: str) -> ModuleType:
    """Register a package module with a package-style ``__path__``."""
    module = ModuleType(module_name)
    module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _register_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    **attributes: object,
) -> ModuleType:
    """Register a plain module with the provided attributes."""
    module = ModuleType(module_name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


@pytest.fixture(autouse=True)
def _clear_shared_caches() -> None:
    """Reset shared extension caches before and after each test."""
    ResearchAssistant.clear_extension_cache(clear_all=True)
    yield
    ResearchAssistant.clear_extension_cache(clear_all=True)


@pytest.mark.parametrize(
    ("method_name", "group", "module_name", "class_name"),
    [
        (
            "get_prompt_contributor",
            "victor.prompt_contributors",
            "victor_research.runtime.prompts",
            "ResearchPromptContributor",
        ),
        (
            "get_mode_config_provider",
            "victor.mode_configs",
            "victor_research.runtime.mode_config",
            "ResearchModeConfigProvider",
        ),
        (
            "get_capability_provider",
            "victor.capability_providers",
            "victor_research.runtime.capabilities",
            "ResearchCapabilityProvider",
        ),
        (
            "get_workflow_provider",
            "victor.workflow_providers",
            "victor_research.runtime.workflows",
            "ResearchWorkflowProvider",
        ),
        (
            "get_team_spec_provider",
            "victor.team_spec_providers",
            "victor_research.runtime.teams",
            "ResearchTeamSpecProvider",
        ),
        (
            "get_service_provider",
            "victor.service_providers",
            "victor_research.runtime.service_provider",
            "ResearchServiceProvider",
        ),
    ],
)
def test_provider_style_runtime_extensions_prefer_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    group: str,
    module_name: str,
    class_name: str,
) -> None:
    """Provider-style runtime helpers should prefer explicit entry points over modules."""

    EntryPointExtension = type(f"EntryPoint{class_name}", (), {})
    RuntimeExtension = type(class_name, (), {})

    _register_package(monkeypatch, "victor_research")
    _register_package(monkeypatch, "victor_research.runtime")
    _register_module(monkeypatch, module_name, **{class_name: RuntimeExtension})
    monkeypatch.setattr(
        ResearchAssistant,
        "_extension_module_available",
        classmethod(lambda cls, module_path: module_path.startswith("victor_research")),
    )

    calls = {"count": 0}

    def _load_from_entry_points(vertical: str, entry_point_group: str) -> object | None:
        assert vertical == "research"
        if entry_point_group != group:
            return None
        calls["count"] += 1
        return EntryPointExtension()

    monkeypatch.setattr(
        entry_point_loader,
        "load_runtime_extension_from_entry_points",
        _load_from_entry_points,
    )

    resolved = getattr(ResearchAssistant, method_name)()
    again = getattr(ResearchAssistant, method_name)()

    assert isinstance(resolved, EntryPointExtension)
    assert again is resolved
    assert calls["count"] == 1


def test_tool_dependency_provider_uses_entry_point_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool dependency helpers should route through the framework entry-point loader."""
    provider = object()
    calls = {"count": 0}

    def _load(vertical: str) -> object | None:
        assert vertical == "research"
        calls["count"] += 1
        return provider

    monkeypatch.setattr(
        entry_point_loader,
        "load_tool_dependency_provider_from_entry_points",
        _load,
    )

    resolved = ResearchAssistant.get_tool_dependency_provider()
    again = ResearchAssistant.get_tool_dependency_provider()

    assert resolved is provider
    assert again is provider
    assert calls["count"] == 1


def test_rl_config_provider_uses_entry_point_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RL config helpers should route through the framework entry-point loader."""

    class EntryPointRLConfig:
        pass

    calls = {"count": 0}

    def _load(vertical: str) -> object | None:
        assert vertical == "research"
        calls["count"] += 1
        return EntryPointRLConfig()

    monkeypatch.setattr(
        entry_point_loader, "load_rl_config_provider_from_entry_points", _load
    )

    resolved = ResearchAssistant.get_rl_config_provider()
    again = ResearchAssistant.get_rl_config_provider()

    assert isinstance(resolved, EntryPointRLConfig)
    assert again is resolved
    assert calls["count"] == 1
