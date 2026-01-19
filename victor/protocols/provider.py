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

"""Provider protocol for model access.

This module defines the ProviderProtocol for accessing LLM provider
information and configuration. This protocol isolates provider-related
functionality following the Interface Segregation Principle (ISP).

Design Principles:
    - ISP: Protocol contains only provider-related properties
    - DIP: Depend on this abstraction, not concrete implementations
    - OCP: Extend via protocol composition, not modification
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProviderProtocol(Protocol):
    """Provider and model access interface.

    This protocol defines properties for accessing information about
    the current LLM provider and model configuration.

    Implementations:
        - AgentOrchestrator (via IAgentOrchestrator)
        - Mock implementations for testing
    """

    @property
    def provider(self) -> Any:
        """Get the current LLM provider instance.

        Returns:
            BaseProvider instance or similar provider object

        Examples:
            >>> provider = orchestrator.provider
            >>> print(provider.name)
        """
        ...

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider.

        Returns:
            Provider name (e.g., 'anthropic', 'openai', 'ollama')

        Examples:
            >>> name = orchestrator.provider_name
            >>> print(f"Using {name}")
        """
        ...

    @property
    def model(self) -> str:
        """Get the current model identifier.

        Returns:
            Model name (e.g., 'claude-sonnet-4-20250514', 'gpt-4o')

        Examples:
            >>> model = orchestrator.model
            >>> print(f"Using model: {model}")
        """
        ...

    @property
    def temperature(self) -> float:
        """Get the temperature setting for sampling.

        Returns:
            Temperature value (0.0 to 1.0+)

        Examples:
            >>> temp = orchestrator.temperature
            >>> print(f"Temperature: {temp}")
        """
        ...


__all__ = [
    "ProviderProtocol",
]
