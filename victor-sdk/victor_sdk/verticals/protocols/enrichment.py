"""Enrichment-related protocol definitions.

These protocols define how verticals provide enrichment strategies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Callable


@runtime_checkable
class EnrichmentProvider(Protocol):
    """Protocol for providing context enrichment strategies.

    Enrichment strategies add relevant context to agent inputs
    to improve performance.
    """

    def get_enrichment_strategies(self) -> Dict[str, Any]:
        """Return enrichment strategy configurations.

        Returns:
            Dictionary of strategy configurations
        """
        ...

    def enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich execution context with additional information.

        Args:
            context: Original execution context

        Returns:
            Enriched context with additional data
        """
        ...

    def get_enrichment_functions(self) -> List[Callable[[Dict[str, Any]], Dict[str, Any]]]:
        """Return list of enrichment functions.

        Returns:
            List of functions that enrich context
        """
        ...


@runtime_checkable
class EnrichmentStrategy(Protocol):
    """Protocol for individual enrichment strategies.

    Enrichment strategies are reusable context enrichment patterns.
    """

    def get_strategy_name(self) -> str:
        """Return the name of this enrichment strategy."""
        ...

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this enrichment strategy to context.

        Args:
            context: Original context

        Returns:
            Enriched context
        """
        ...
