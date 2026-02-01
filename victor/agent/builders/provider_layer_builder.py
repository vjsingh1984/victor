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

"""Provider layer builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory
    from victor.providers.base import BaseProvider


class ProviderLayerBuilder(FactoryAwareBuilder):
    """Build provider-related components for the orchestrator."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(  # type: ignore[override]
        self,
        orchestrator: "AgentOrchestrator",
        provider: "BaseProvider",
        model: str,
        provider_name: Optional[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build provider layer components and attach them to orchestrator."""
        factory = self._ensure_factory(
            provider=provider,
            model=model,
            provider_name=provider_name,
        )
        components: dict[str, Any] = {}

        # Tool calling matrix for managing provider capabilities (via factory)
        orchestrator.tool_calling_models, orchestrator.tool_capabilities = (
            factory.create_tool_calling_matrix()
        )
        components["tool_calling_models"] = orchestrator.tool_calling_models
        components["tool_capabilities"] = orchestrator.tool_capabilities

        # Initialize ProviderManager with tool adapter (via factory)
        (
            orchestrator._provider_manager,
            orchestrator.provider,
            orchestrator.model,
            orchestrator.provider_name,
            orchestrator.tool_adapter,
            orchestrator._tool_calling_caps_internal,
        ) = factory.create_provider_manager_with_adapter(provider, model, provider_name or "")
        components["provider_manager"] = orchestrator._provider_manager
        components["provider"] = orchestrator.provider
        components["model"] = orchestrator.model
        components["provider_name"] = orchestrator.provider_name
        components["tool_adapter"] = orchestrator.tool_adapter
        components["tool_calling_caps"] = orchestrator._tool_calling_caps_internal

        # ProviderCoordinator: Wraps ProviderManager with rate limiting and health monitoring (TD-002)
        from victor.agent.provider_coordinator import (
            ProviderCoordinator,
            ProviderCoordinatorConfig,
        )

        orchestrator._provider_coordinator = ProviderCoordinator(
            provider_manager=orchestrator._provider_manager,
            config=ProviderCoordinatorConfig(
                max_rate_limit_retries=getattr(self.settings, "max_rate_limit_retries", 3),
                enable_health_monitoring=getattr(self.settings, "provider_health_checks", True),
            ),
        )
        components["provider_coordinator"] = orchestrator._provider_coordinator

        # ProviderSwitchCoordinator: Coordinate provider/model switching workflow (via factory)
        orchestrator._provider_switch_coordinator = factory.create_provider_switch_coordinator(
            provider_switcher=orchestrator._provider_manager._provider_switcher,
            health_monitor=orchestrator._provider_manager._health_monitor,
        )
        components["provider_switch_coordinator"] = orchestrator._provider_switch_coordinator

        # ProviderLifecycleManager: Handles post-switch hooks (Phase 1.2)
        from victor.agent.protocols import ProviderLifecycleProtocol

        if hasattr(orchestrator, "_container") and orchestrator._container:
            manager = orchestrator._container.get_optional(ProviderLifecycleProtocol)
            if manager:
                orchestrator._provider_lifecycle_manager = manager
                components["provider_lifecycle_manager"] = manager

        self._register_components(components)
        return components
