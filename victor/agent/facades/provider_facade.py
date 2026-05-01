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
The orchestrator delegates property access through this facade.

Deprecated coordinator properties are retained as compatibility accessors,
but canonical runtime code should source those handles from
``provider_runtime`` rather than wiring them independently.
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
        - provider_manager: ProviderManager for lifecycle management
        - provider_runtime: Provider runtime boundary components
        - provider_coordinator: Deprecated compatibility accessor derived from
          provider_runtime when not supplied explicitly
        - provider_switch_coordinator: Deprecated compatibility accessor
          derived from provider_runtime when not supplied explicitly
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
        provider_manager: Any,
        provider_runtime: Optional[Any] = None,
        provider_coordinator: Optional[Any] = None,
        provider_switch_coordinator: Optional[Any] = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._provider_name = provider_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._thinking = thinking
        self._provider_manager = provider_manager
        self._provider_runtime = provider_runtime
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

    # ------------------------------------------------------------------
    # Properties (satisfy ProviderFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def provider(self) -> Any:
        """Active LLM provider instance."""
        return self._provider

    @provider.setter
    def provider(self, value: Any) -> None:
        """Update the active provider instance."""
        self._provider = value

    @property
    def model(self) -> str:
        """Active model identifier."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Update the active model identifier."""
        self._model = value

    @property
    def provider_name(self) -> Optional[str]:
        """Provider label (e.g., lmstudio, vllm)."""
        return self._provider_name

    @provider_name.setter
    def provider_name(self, value: Optional[str]) -> None:
        """Update the provider label."""
        self._provider_name = value

    @property
    def temperature(self) -> float:
        """Sampling temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Update the sampling temperature."""
        self._temperature = value

    @property
    def max_tokens(self) -> int:
        """Maximum tokens to generate."""
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        """Update the maximum tokens."""
        self._max_tokens = value

    @property
    def thinking(self) -> bool:
        """Extended thinking mode flag."""
        return self._thinking

    @thinking.setter
    def thinking(self, value: bool) -> None:
        """Update the thinking mode flag."""
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
        return self._get_runtime_component("provider_coordinator")

    @property
    def provider_switch_coordinator(self) -> Optional[Any]:
        """Provider switching coordinator."""
        if self._provider_switch_coordinator is not None:
            return self._provider_switch_coordinator
        return self._get_runtime_component("provider_switch_coordinator")
