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

"""Capability Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols for capability, chain, and persona providers.
Following ISP, these protocols are focused on declaring vertical capabilities.

Usage:
    from victor.core.verticals.protocols.capability_provider import (
        CapabilityProviderProtocol,
        ChainProviderProtocol,
        PersonaProviderProtocol,
    )

    class CodingCapabilityProvider(CapabilityProviderProtocol):
        def get_capabilities(self) -> Dict[str, Any]:
            return {
                "code_review": True,
                "refactoring": True,
                "max_file_size": 100000,
            }
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


# =============================================================================
# Capability Provider Protocol
# =============================================================================


@runtime_checkable
class CapabilityProviderProtocol(Protocol):
    """Protocol for capability configuration providers.

    Enables verticals to declare which capabilities they support,
    allowing runtime discovery and feature toggling.

    Example:
        class CodingCapabilityProvider(CapabilityProviderProtocol):
            def get_capabilities(self) -> Dict[str, Any]:
                return {
                    "code_review": True,
                    "refactoring": True,
                    "test_generation": True,
                    "max_file_size": 100000,
                }
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Get capability definitions for this vertical.

        Returns:
            Dict mapping capability names to their configurations
            (typically bool for feature flags, or values for limits)
        """
        ...


# =============================================================================
# Chain Provider Protocol
# =============================================================================


@runtime_checkable
class ChainProviderProtocol(Protocol):
    """Protocol for chain configuration providers.

    Enables verticals to define chains of operations that can be executed
    in sequence, supporting DIP by providing a protocol interface rather
    than concrete implementations.

    Example:
        class CodingChainProvider(ChainProviderProtocol):
            def get_chains(self) -> Dict[str, Any]:
                return {
                    "refactor": RefactorChain(steps=[...]),
                    "test_and_fix": TestFixChain(steps=[...]),
                }
    """

    def get_chains(self) -> Dict[str, Any]:
        """Get chain definitions for this vertical.

        Returns:
            Dict mapping chain names to chain configurations or instances
        """
        ...


# =============================================================================
# Persona Provider Protocol
# =============================================================================


@runtime_checkable
class PersonaProviderProtocol(Protocol):
    """Protocol for persona configuration providers.

    Enables verticals to define different personas (behavioral profiles)
    that affect how the agent interacts and responds.

    Example:
        class CodingPersonaProvider(PersonaProviderProtocol):
            def get_personas(self) -> Dict[str, Any]:
                return {
                    "senior_dev": {"name": "Senior Developer", "style": "thorough"},
                    "junior_dev": {"name": "Junior Developer", "style": "verbose"},
                }
    """

    def get_personas(self) -> Dict[str, Any]:
        """Get persona definitions for this vertical.

        Returns:
            Dict mapping persona names to persona configurations
        """
        ...


__all__ = [
    "CapabilityProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
]
