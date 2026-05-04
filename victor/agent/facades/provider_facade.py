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

"""Provider domain facade for orchestrator decomposition.

Groups LLM provider instance, model selection, switching, health monitoring,
and rate coordination components behind a single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade. It does not own
provider behavior.

Deprecated coordinator properties are retained as compatibility accessors.
They are no longer owned by ``provider_runtime``; when needed, the facade
materializes compatibility shims lazily and binds them to the canonical
``ProviderService``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProviderFacade:
    """Groups LLM provider, model, and coordination components.

    Satisfies ``ProviderFacadeProtocol`` structurally.  The orchestrator creates
    this facade after all provider-domain components are initialized, passing
    references to the already-created instances.

    Components managed:
        - provider: Active LLM provider instance
        - model: Active model identifier
        - provider_name: Provider label (e.g., lmstudio, vllm)
        - temperature: Sampling temperature
        - max_tokens: Maximum tokens to generate
        - thinking: Extended thinking mode flag
        - runtime_state_host: Canonical runtime owner for mutable provider config
        - provider_manager: ProviderManager for lifecycle management
        - provider_runtime: Provider runtime boundary components
        - provider_coordinator: Deprecated compatibility accessor derived from
          provider_runtime when supplied explicitly or synthesized lazily
        - provider_switch_coordinator: Deprecated compatibility accessor
          derived from provider_runtime when supplied explicitly or synthesized lazily
    """

    def __init__(
        self,
        *,
        provider: Any,
        model: str,
        provider_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking: bool = False,
        runtime_state_host: Optional[Any] = None,
        provider_manager: Any,
        provider_runtime: Optional[Any] = None,
        provider_service: Optional[Any] = None,
        provider_coordinator: Optional[Any] = None,
        provider_switch_coordinator: Optional[Any] = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._provider_name = provider_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._thinking = thinking
        self._runtime_state_host = runtime_state_host
        self._provider_manager = provider_manager
        self._provider_runtime = provider_runtime
        self._provider_service = provider_service
        self._provider_coordinator = provider_coordinator
        self._provider_switch_coordinator = provider_switch_coordinator

        logger.debug(
            "ProviderFacade initialized (provider=%s, model=%s, thinking=%s)",
            provider_name,
            model,
            thinking,
        )

    def _get_runtime_component(self, component_name: str) -> Optional[Any]:
        """Resolve a compatibility component from provider runtime if available."""
        runtime = self._provider_runtime
        if runtime is None:
            return None
        return getattr(runtime, component_name, None)

    def _synthesize_provider_coordinator(self) -> Optional[Any]:
        """Create the deprecated ProviderCoordinator shim lazily when needed."""
        if self._provider_coordinator is not None:
            return self._provider_coordinator
        if self._provider_runtime is None and self._provider_service is None:
            return None
        if self._provider_manager is None:
            return None

        from victor.agent.provider.coordinator import create_provider_coordinator

        coordinator = create_provider_coordinator(provider_manager=self._provider_manager)
        if self._provider_service is not None and hasattr(
            coordinator, "bind_provider_service"
        ):
            coordinator.bind_provider_service(self._provider_service)
        self._provider_coordinator = coordinator
        logger.debug("Materialized deprecated provider_coordinator compatibility shim")
        return coordinator

    def _synthesize_provider_switch_coordinator(self) -> Optional[Any]:
        """Create the deprecated ProviderSwitchCoordinator shim lazily when needed."""
        if self._provider_switch_coordinator is not None:
            return self._provider_switch_coordinator
        if self._provider_runtime is None and self._provider_service is None:
            return None
        if self._provider_manager is None:
            return None

        provider_switcher = getattr(self._provider_manager, "_provider_switcher", None)
        if provider_switcher is None:
            return None

        from victor.agent.provider.switch_coordinator import (
            create_provider_switch_coordinator,
        )

        coordinator = create_provider_switch_coordinator(
            provider_switcher=provider_switcher,
            health_monitor=getattr(self._provider_manager, "_health_monitor", None),
        )
        self._provider_switch_coordinator = coordinator
        logger.debug("Materialized deprecated provider_switch_coordinator compatibility shim")
        return coordinator

    # ------------------------------------------------------------------
    # Properties (satisfy ProviderFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def provider(self) -> Any:
        """Active LLM provider instance."""
        if self._runtime_state_host is not None:
            return getattr(self._runtime_state_host, "provider", self._provider)
        return self._provider

    @provider.setter
    def provider(self, value: Any) -> None:
        """Update the active provider instance."""
        if self._runtime_state_host is not None:
            self._runtime_state_host.provider = value
        self._provider = value

    @property
    def model(self) -> str:
        """Active model identifier."""
        if self._runtime_state_host is not None:
            return getattr(self._runtime_state_host, "model", self._model)
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Update the active model identifier."""
        if self._runtime_state_host is not None:
            self._runtime_state_host.model = value
        self._model = value

    @property
    def provider_name(self) -> Optional[str]:
        """Provider label (e.g., lmstudio, vllm)."""
        if self._runtime_state_host is not None:
            return getattr(self._runtime_state_host, "provider_name", self._provider_name)
        return self._provider_name

    @provider_name.setter
    def provider_name(self, value: Optional[str]) -> None:
        """Update the provider label."""
        if self._runtime_state_host is not None:
            self._runtime_state_host.provider_name = value
        self._provider_name = value

    @property
    def temperature(self) -> float:
        """Sampling temperature."""
        if self._runtime_state_host is not None:
            return getattr(self._runtime_state_host, "temperature", self._temperature)
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Update the sampling temperature."""
        if self._runtime_state_host is not None:
            self._runtime_state_host.temperature = value
        self._temperature = value

    @property
    def max_tokens(self) -> int:
        """Maximum tokens to generate."""
        if self._runtime_state_host is not None:
            return getattr(self._runtime_state_host, "max_tokens", self._max_tokens)
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        """Update the maximum tokens."""
        if self._runtime_state_host is not None:
            self._runtime_state_host.max_tokens = value
        self._max_tokens = value

    @property
    def thinking(self) -> bool:
        """Extended thinking mode flag."""
        if self._runtime_state_host is not None:
            return getattr(self._runtime_state_host, "thinking", self._thinking)
        return self._thinking

    @thinking.setter
    def thinking(self, value: bool) -> None:
        """Update the thinking mode flag."""
        if self._runtime_state_host is not None:
            self._runtime_state_host.thinking = value
        self._thinking = value

    @property
    def provider_manager(self) -> Any:
        """ProviderManager for lifecycle management."""
        return self._provider_manager

    @property
    def provider_runtime(self) -> Optional[Any]:
        """Provider runtime boundary components."""
        return self._provider_runtime

    @property
    def provider_coordinator(self) -> Optional[Any]:
        """Provider coordination service."""
        if self._provider_coordinator is not None:
            return self._provider_coordinator
        runtime_component = self._get_runtime_component("provider_coordinator")
        if runtime_component is not None:
            return runtime_component
        return self._synthesize_provider_coordinator()

    @property
    def provider_switch_coordinator(self) -> Optional[Any]:
        """Provider switching coordinator."""
        if self._provider_switch_coordinator is not None:
            return self._provider_switch_coordinator
        runtime_component = self._get_runtime_component("provider_switch_coordinator")
        if runtime_component is not None:
            return runtime_component
        return self._synthesize_provider_switch_coordinator()
