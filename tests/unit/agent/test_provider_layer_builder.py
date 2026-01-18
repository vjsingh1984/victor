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

"""Tests for ProviderLayerBuilder."""

from unittest.mock import MagicMock, patch

from victor.agent.builders.provider_layer_builder import ProviderLayerBuilder


def test_provider_layer_builder_wires_components():
    """ProviderLayerBuilder assigns provider components."""
    settings = MagicMock()
    settings.max_rate_limit_retries = 5
    settings.provider_health_checks = True

    factory = MagicMock()
    factory.create_tool_calling_matrix.return_value = ({"m": True}, {"cap": True})
    provider_manager = MagicMock()
    provider_manager._provider_switcher = "switcher"
    provider_manager._health_monitor = "monitor"
    factory.create_provider_manager_with_adapter.return_value = (
        provider_manager,
        "provider-instance",
        "model-id",
        "provider-name",
        "tool-adapter",
        "tool-caps",
    )
    factory.create_provider_switch_coordinator.return_value = "switch-coordinator"

    orchestrator = MagicMock()

    with (
        patch(
            "victor.agent.provider_coordinator.ProviderCoordinatorConfig",
            return_value="config",
        ) as config_cls,
        patch(
            "victor.agent.provider_coordinator.ProviderCoordinator",
            return_value="provider-coordinator",
        ) as coordinator_cls,
    ):
        builder = ProviderLayerBuilder(settings=settings, factory=factory)
        components = builder.build(
            orchestrator,
            provider="provider",
            model="model",
            provider_name="provider-name",
        )

    config_cls.assert_called_once_with(
        max_rate_limit_retries=5,
        enable_health_monitoring=True,
    )
    coordinator_cls.assert_called_once_with(
        provider_manager=provider_manager,
        config="config",
    )
    assert orchestrator.provider == "provider-instance"
    assert orchestrator.model == "model-id"
    assert orchestrator.provider_name == "provider-name"
    assert orchestrator.tool_adapter == "tool-adapter"
    assert orchestrator._tool_calling_caps_internal == "tool-caps"
    assert orchestrator._provider_coordinator == "provider-coordinator"
    assert orchestrator._provider_switch_coordinator == "switch-coordinator"
    assert components["provider_manager"] == provider_manager
