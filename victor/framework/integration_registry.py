"""Integration plan registry for cross-session plan persistence.

Replaces WeakKeyDictionary-based caching with a dedicated registry
service that persists plans by orchestrator identity, surviving GC
cycles within a process session.
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class IntegrationPlanRegistry:
    """Singleton registry for integration plan metadata.

    Stores applied integration plans indexed by orchestrator ID,
    providing better persistence than WeakKeyDictionary which loses
    plans when orchestrators are garbage collected.

    Thread-safe via internal lock.
    """

    _instance: Optional["IntegrationPlanRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._plans: Dict[int, Dict[str, Any]] = {}
        self._access_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "IntegrationPlanRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for tests)."""
        with cls._lock:
            cls._instance = None

    def get_plan(self, orchestrator: Any) -> Optional[Dict[str, Any]]:
        """Get previously applied plan for an orchestrator.

        Args:
            orchestrator: Orchestrator instance (uses id() for lookup)

        Returns:
            Deep copy of the plan, or None if not found
        """
        orch_id = id(orchestrator)
        with self._access_lock:
            plan = self._plans.get(orch_id)
            if plan is not None:
                return copy.deepcopy(plan)
        return None

    def set_plan(self, orchestrator: Any, plan: Dict[str, Any]) -> None:
        """Store applied plan for an orchestrator.

        Args:
            orchestrator: Orchestrator instance
            plan: Integration plan metadata
        """
        orch_id = id(orchestrator)
        with self._access_lock:
            self._plans[orch_id] = copy.deepcopy(plan)

    def remove_plan(self, orchestrator: Any) -> None:
        """Remove plan for an orchestrator."""
        orch_id = id(orchestrator)
        with self._access_lock:
            self._plans.pop(orch_id, None)

    def clear(self) -> None:
        """Clear all stored plans."""
        with self._access_lock:
            self._plans.clear()

    @property
    def size(self) -> int:
        """Number of stored plans."""
        with self._access_lock:
            return len(self._plans)
