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

"""Integration test verifying CLI behavior when coding vertical is not installed."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

import pytest
from typer.testing import CliRunner

from victor.core.container import ServiceContainer, set_container


@pytest.mark.integration
def test_chat_missing_coding_vertical_emits_capability_warning(monkeypatch):
    """victor chat --vertical coding should emit structured warning when victor-coding is absent."""
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.ui.commands.chat import chat_app

    # Ensure victor_coding modules look absent for this test run.
    for key in list(sys.modules.keys()):
        if key.startswith("victor_coding"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    async def fake_from_settings(cls, *args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError("stop")  # Fail fast after bootstrap/vertical resolution

    monkeypatch.setattr(
        AgentOrchestrator,
        "from_settings",
        classmethod(fake_from_settings),
    )

    usage_events: List[Tuple[str, Dict[str, Any]]] = []

    class StubUsageLogger:
        def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
            usage_events.append((event_type, data))

        def is_enabled(self) -> bool:
            return True

    monkeypatch.setattr(
        "victor.core.bootstrap._create_usage_logger",
        lambda *args, **kwargs: StubUsageLogger(),
    )

    emitted_events: List[Tuple[str, Dict[str, Any], str]] = []

    class StubObservabilityBus:
        def emit_sync(
            self, topic: str, data: Dict[str, Any], source: str | None = None
        ) -> None:
            emitted_events.append((topic, data, source or ""))

    monkeypatch.setattr(
        "victor.core.events.get_observability_bus",
        lambda: StubObservabilityBus(),
    )

    from victor.core import bootstrap as bootstrap_module
    from victor.core.plugins.registry import PluginRegistry
    from victor.core.verticals.base import VerticalRegistry
    from victor.core.verticals.vertical_loader import (
        VerticalLoader,
        get_vertical_loader,
    )

    original_discover = VerticalLoader.discover_verticals
    original_plugin_discover = PluginRegistry.discover
    original_list_names = VerticalRegistry.list_names

    def discover_without_coding(
        self, force_refresh: bool = False, emit_event: bool = True
    ):
        discovered = original_discover(
            self, force_refresh=force_refresh, emit_event=emit_event
        )
        if "coding" in discovered:
            discovered = dict(discovered)
            discovered.pop("coding", None)
            if hasattr(self, "_discovered_verticals") and isinstance(
                self._discovered_verticals, dict  # type: ignore[attr-defined]
            ):
                filtered_cache = dict(self._discovered_verticals)
                filtered_cache.pop("coding", None)
                self._discovered_verticals = filtered_cache  # type: ignore[attr-defined]
        return discovered

    def discover_plugins_without_coding(self, force: bool = False):
        discovered = original_plugin_discover(self, force=force)
        filtered = [
            plugin for plugin in discovered if getattr(plugin, "name", None) != "coding"
        ]
        self._plugins = {plugin.name: plugin for plugin in filtered}
        return filtered

    def list_names_without_coding(cls):
        return [name for name in original_list_names() if name != "coding"]

    monkeypatch.setattr(VerticalLoader, "discover_verticals", discover_without_coding)
    monkeypatch.setattr(PluginRegistry, "discover", discover_plugins_without_coding)
    monkeypatch.setattr(
        VerticalRegistry, "list_names", classmethod(list_names_without_coding)
    )
    loader = get_vertical_loader()
    loader._discovered_verticals = None
    plugin_registry = PluginRegistry.get_instance()
    plugin_registry._plugins = {}
    plugin_registry._discovered = False

    VerticalRegistry.unregister("coding")
    bootstrap_module._REPORTED_MISSING_VERTICALS.clear()
    reported_verticals: List[str | None] = []
    original_report = bootstrap_module._report_capability_health

    def tracking_report(
        vertical_name: str | None, container: ServiceContainer | None = None
    ) -> None:
        reported_verticals.append(vertical_name)
        original_report(vertical_name, container)

    monkeypatch.setattr(
        bootstrap_module,
        "_report_capability_health",
        tracking_report,
    )

    # Reset global container so bootstrap_container installs services fresh for this test.
    set_container(ServiceContainer())

    runner = CliRunner()

    # Prevent CLI logging configuration from muting the warning we want to inspect.
    monkeypatch.setattr("victor.ui.commands.utils.setup_logging", lambda *_, **__: None)

    result = runner.invoke(chat_app, ["--vertical", "coding", "hello world"])

    # CLI should exit with failure because we stubbed the orchestrator, but bootstrap must succeed.
    assert result.exit_code != 0

    assert "coding" in reported_verticals
    assert "service extension failed to load" not in result.stdout

    assert any(
        event == "missing_vertical" for event, _ in usage_events
    ), "Usage metric not recorded"

    assert emitted_events and emitted_events[0][0] == "capabilities.vertical.missing"

    # Restore container to a clean state for other tests.
    plugin_registry._plugins = {}
    plugin_registry._discovered = False
    set_container(ServiceContainer())
