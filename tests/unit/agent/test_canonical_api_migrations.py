# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.factory.coordination_builders import CoordinationBuildersMixin
from victor.agent.factory.tool_builders import ToolBuildersMixin
from victor.agent.service_provider import OrchestratorServiceProvider


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _ToolBuilderHarness(ToolBuildersMixin):
    def __init__(self, settings):
        self.settings = settings


class _CoordinationBuilderHarness(CoordinationBuildersMixin):
    def __init__(self, settings):
        self.settings = settings


def test_service_provider_prefers_canonical_nested_semantic_selection_setting():
    settings = SimpleNamespace(
        tool_selection=SimpleNamespace(use_semantic_tool_selection=False),
        use_semantic_tool_selection=True,
        embedding_model="legacy-embedding-model",
    )
    provider = OrchestratorServiceProvider(settings)

    with patch("victor.tools.semantic_selector.SemanticToolSelector") as selector_cls:
        selector = provider._create_semantic_tool_selector()

    selector_cls.assert_not_called()
    assert selector.select_tools("query", ["read", "grep", "edit"], max_tools=2) == ["read", "grep"]


def test_tool_builders_prefers_canonical_nested_semantic_selection_setting():
    settings = SimpleNamespace(
        tool_selection=SimpleNamespace(use_semantic_tool_selection=False),
        tools=SimpleNamespace(
            embedding_model="canonical-embedding-model",
            embedding_provider="sentence-transformers",
        ),
        provider=SimpleNamespace(ollama_base_url="http://localhost:11434"),
        use_semantic_tool_selection=True,
    )
    harness = _ToolBuilderHarness(settings)

    with patch("victor.tools.semantic_selector.SemanticToolSelector") as selector_cls:
        selector = harness.create_semantic_selector()

    selector_cls.assert_not_called()
    assert selector is None


def test_coordination_builders_prefers_canonical_nested_semantic_selection_setting():
    settings = SimpleNamespace(
        tool_selection=SimpleNamespace(use_semantic_tool_selection=False),
        use_semantic_tool_selection=True,
    )
    harness = _CoordinationBuilderHarness(settings)

    use_semantic, preload_task = harness.setup_semantic_selection()

    assert use_semantic is False
    assert preload_task is None


def test_tool_builders_initialize_plugin_system_prefers_canonical_registrar_method():
    settings = SimpleNamespace(plugin_enabled=True)
    harness = _ToolBuilderHarness(settings)
    initialize_plugins = MagicMock(return_value=2)
    registrar = SimpleNamespace(
        initialize_plugins=initialize_plugins,
        plugin_manager="plugin-manager",
    )

    def _legacy_load_plugin_tools():
        raise AssertionError("ToolBuilders should not call ToolRegistrar private plugin helpers")

    registrar._load_plugin_tools = _legacy_load_plugin_tools

    plugin_manager = harness.initialize_plugin_system(registrar)

    initialize_plugins.assert_called_once_with()
    assert plugin_manager == "plugin-manager"


def test_tool_builders_prefers_canonical_nested_tool_cache_settings():
    settings = SimpleNamespace(
        tools=SimpleNamespace(
            tool_cache_enabled=False,
            tool_cache_ttl=123,
            tool_cache_allowlist=["code_search"],
        ),
        tool_cache_dir_override="/tmp/canonical-cache",
        tool_cache_enabled=True,
        tool_cache_ttl=999,
        tool_cache_allowlist=["legacy"],
    )
    harness = _ToolBuilderHarness(settings)

    cache = harness.create_tool_cache()

    assert cache is None


def test_internal_modules_use_sdk_tool_dependency_types_directly():
    repo_root = _repo_root()
    module_paths = [
        repo_root / "victor/core/tool_dependency_loader.py",
        repo_root / "victor/core/tool_types.py",
        repo_root / "victor/core/entry_points/tool_dependency_provider.py",
    ]

    for module_path in module_paths:
        source = module_path.read_text()
        assert "from victor.core.tool_dependency_base import" not in source
        assert "victor_sdk.verticals.tool_dependencies" in source


def test_response_processor_uses_canonical_tool_registry_module():
    source = (_repo_root() / "victor/agent/response_processor.py").read_text()

    assert "from victor.tools import ToolRegistry" not in source
    assert "from victor.tools.registry import ToolRegistry" in source
