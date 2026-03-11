"""Capability-related protocol definitions.

These protocols define how verticals provide capability configurations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional, Callable

from victor_sdk.core.types import CapabilityRequirementLike


@runtime_checkable
class CapabilityProvider(Protocol):
    """Protocol for providing capability configurations.

    Capabilities represent high-level features that verticals can
    provide or depend on.
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Return capability configurations.

        Returns:
            Dictionary of capability configurations
        """
        ...

    def has_capability(self, capability_name: str) -> bool:
        """Check if a capability is available.

        Args:
            capability_name: Name of the capability

        Returns:
            True if capability is available
        """
        ...

    def get_capability_requirements(self) -> List[CapabilityRequirementLike]:
        """Return list of required capabilities.

        Returns:
            List of capability names or typed requirements that must be available
        """
        ...


@runtime_checkable
class ChainProvider(Protocol):
    """Protocol for providing capability chains.

    Chains allow multiple capabilities to be composed together.
    """

    def get_chain_definitions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return chain definitions.

        Returns:
            Dictionary mapping chain names to lists of capability configs
        """
        ...

    def get_chain(self, chain_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get a specific chain by name.

        Args:
            chain_name: Name of the chain

        Returns:
            List of capability configs or None
        """
        ...


@runtime_checkable
class PersonaProvider(Protocol):
    """Protocol for providing persona configurations.

    Personas define agent personalities and behaviors.
    """

    def get_persona_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Return persona definitions.

        Returns:
            Dictionary mapping persona names to persona configs
        """
        ...

    def get_default_persona(self) -> str:
        """Return the default persona name.

        Returns:
            Default persona identifier
        """
        ...

    def get_persona(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific persona by name.

        Args:
            persona_name: Name of the persona

        Returns:
            Persona configuration or None
        """
        ...
