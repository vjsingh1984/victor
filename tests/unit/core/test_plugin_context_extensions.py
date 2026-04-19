# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for extended PluginContext methods (S2 of plugin-vertical consolidation)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_register_rl_config_buffers_payload():
    from victor.core.plugins.context import HostPluginContext

    ctx = HostPluginContext(container=None)
    ctx.register_rl_config("coding", {"alpha": 0.1})
    assert ctx.pending_rl_configs == {"coding": {"alpha": 0.1}}


def test_register_bootstrap_service_buffers_factory():
    from victor.core.plugins.context import HostPluginContext

    def factory(container, settings, context):  # pragma: no cover - inert hook
        return None

    ctx = HostPluginContext(container=None)
    ctx.register_bootstrap_service(factory, phase="vertical_services")
    pending = ctx.pending_bootstrap_services
    assert len(pending) == 1
    assert pending[0] == ("vertical_services", factory)


def test_register_mcp_server_buffers_spec():
    from victor.core.plugins.context import HostPluginContext

    spec = SimpleNamespace(name="example", url="http://localhost:9000")
    ctx = HostPluginContext(container=None)
    ctx.register_mcp_server(spec)
    assert ctx.pending_mcp_servers == [spec]


def test_register_tool_dependency_seeds_cache():
    from victor.core.plugins.context import HostPluginContext
    from victor.core.tool_dependency_loader import (
        _vertical_provider_cache,
        _vertical_provider_cache_lock,
        clear_vertical_tool_dependency_provider_cache,
    )

    clear_vertical_tool_dependency_provider_cache()

    class StubProvider:
        def get_dependencies(self):
            return {}

        def get_recommended_sequence(self, _task):
            return []

    provider = StubProvider()
    ctx = HostPluginContext(container=None)
    ctx.register_tool_dependency("stubvertical", provider)

    with _vertical_provider_cache_lock:
        canon = _vertical_provider_cache.get(("stubvertical", True))
        noncanon = _vertical_provider_cache.get(("stubvertical", False))

    assert canon is provider
    assert noncanon is provider
    clear_vertical_tool_dependency_provider_cache()


def test_register_safety_rule_no_container_is_noop():
    """When SafetyEnforcer isn't registered, the method logs + returns without error."""
    from victor.core.plugins.context import HostPluginContext

    class StubRule:
        name = "stub_rule"

    ctx = HostPluginContext(container=None)
    # Should not raise even though no SafetyEnforcer is registered.
    ctx.register_safety_rule(StubRule())


def test_register_escape_hatch_condition(monkeypatch):
    from victor.core.plugins.context import HostPluginContext

    recorded = {}

    class FakeRegistry:
        def register_condition(self, name, fn, **opts):
            recorded["condition"] = (name, fn, opts)

        def register_transform(self, name, fn, **opts):
            recorded["transform"] = (name, fn, opts)

    monkeypatch.setattr(
        "victor.framework.escape_hatch_registry.get_escape_hatch_registry",
        lambda: FakeRegistry(),
    )

    ctx = HostPluginContext(container=None)
    ctx.register_escape_hatch(
        SimpleNamespace(kind="condition", name="allow_everything", fn=lambda ctx: True)
    )

    assert "condition" in recorded
    assert recorded["condition"][0] == "allow_everything"


def test_pluginctx_protocol_has_new_methods():
    """The SDK Protocol includes the new register_* methods."""
    from victor_sdk.core.plugins import PluginContext

    for name in (
        "register_safety_rule",
        "register_tool_dependency",
        "register_escape_hatch",
        "register_rl_config",
        "register_bootstrap_service",
        "register_mcp_server",
    ):
        assert hasattr(PluginContext, name), f"SDK PluginContext missing {name}"
