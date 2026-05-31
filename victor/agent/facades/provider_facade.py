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

Groups LLM provider instance, model selection, and configuration components
behind a single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade. It does not own
provider behavior.

Migration Notes (2026-05-01):
- provider_coordinator and provider_switch_coordinator removed in v1.0.0
- Use ProviderService for all provider operations
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProviderFacade:
    """Groups LLM provider, model, and configuration components.

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
        - provider_service: Provider service for all provider operations

    Migration Notes (2026-05-01):
        - provider_coordinator and provider_switch_coordinator removed
        - Use ProviderService for all provider operations
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

        logger.debug(
            "ProviderFacade initialized (provider=%s, model=%s, thinking=%s)",
            provider_name,
            model,
            thinking,
        )

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
