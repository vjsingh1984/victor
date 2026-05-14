"""Memory protocol definitions for external verticals.

These protocols let external verticals interact with Victor's shared memory
services without importing from victor-ai internals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class MemoryCoordinatorProtocol(Protocol):
    """Protocol for the shared memory coordinator service."""

    async def search_all(
        self,
        query: str,
        limit: int = 20,
        memory_types: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_relevance: float = 0.0,
    ) -> List[Any]:
        """Search across all registered memory providers."""
        ...

    async def search_type(
        self,
        memory_type: Any,
        query: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> List[Any]:
        """Search a specific memory type."""
        ...

    async def store(
        self,
        memory_type: Any,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a value in a memory backend."""
        ...

    async def get(
        self,
        memory_type: Any,
        key: str,
    ) -> Optional[Any]:
        """Retrieve a value from a memory backend."""
        ...

    def register_provider(self, provider: Any) -> None:
        """Register a memory provider."""
        ...

    def unregister_provider(self, memory_type: Any) -> bool:
        """Unregister a memory provider."""
        ...

    def get_registered_types(self) -> List[Any]:
        """Return registered memory provider types."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return memory-coordinator statistics."""
        ...


__all__ = ["MemoryCoordinatorProtocol"]
