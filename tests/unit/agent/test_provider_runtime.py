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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.runtime.provider_runtime import (
    LazyRuntimeProxy,
    create_provider_runtime_components,
)


def test_lazy_runtime_proxy_initializes_once():
    calls = {"count": 0}
    target = SimpleNamespace(value=42)

    def _factory():
        calls["count"] += 1
        return target

    proxy = LazyRuntimeProxy(factory=_factory, name="test_component")
    assert proxy.initialized is False

    assert proxy.value == 42
    assert proxy.initialized is True
    assert calls["count"] == 1

    assert proxy.value == 42
    assert calls["count"] == 1


def test_lazy_runtime_proxy_setattr_delegates_to_instance():
    target = SimpleNamespace(value=1)

    proxy = LazyRuntimeProxy(factory=lambda: target, name="test_component")
    assert proxy.initialized is False

    proxy.value = 7

    assert proxy.initialized is True
    assert target.value == 7
    assert proxy.value == 7


def test_create_provider_runtime_components_provider_coordinator_lazy():
    factory = MagicMock()
    manager = MagicMock()
    manager._provider_switcher = object()
    manager._health_monitor = object()
    settings = SimpleNamespace(max_rate_limit_retries=7, provider_health_checks=False)

    runtime = create_provider_runtime_components(
        factory=factory,
        settings=settings,
        provider_manager=manager,
    )

    assert runtime.provider_coordinator.initialized is False

    with patch("victor.agent.provider_coordinator.ProviderCoordinator") as coordinator_cls:
        coordinator_instance = MagicMock()
        coordinator_cls.return_value = coordinator_instance

        resolved = runtime.provider_coordinator.get_instance()
        assert resolved is coordinator_instance
        assert runtime.provider_coordinator.initialized is True
        coordinator_cls.assert_called_once()

        kwargs = coordinator_cls.call_args.kwargs
        assert kwargs["provider_manager"] is manager
        config = kwargs["config"]
        assert config.max_rate_limit_retries == 7
        assert config.enable_health_monitoring is False


def test_create_provider_runtime_components_switch_coordinator_lazy():
    factory = MagicMock()
    manager = MagicMock()
    manager._provider_switcher = object()
    manager._health_monitor = object()
    settings = SimpleNamespace(max_rate_limit_retries=3, provider_health_checks=True)

    switch_coordinator = object()
    factory.create_provider_switch_coordinator.return_value = switch_coordinator

    runtime = create_provider_runtime_components(
        factory=factory,
        settings=settings,
        provider_manager=manager,
    )

    assert runtime.provider_switch_coordinator.initialized is False
    resolved = runtime.provider_switch_coordinator.get_instance()
    assert resolved is switch_coordinator
    assert runtime.provider_switch_coordinator.initialized is True

    factory.create_provider_switch_coordinator.assert_called_once_with(
        provider_switcher=manager._provider_switcher,
        health_monitor=manager._health_monitor,
    )
