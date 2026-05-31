"""Progressive tool loading — lazy init, cost preference, parameter escalation.

FEP-0003 implementation: reduces startup time via deferred tool initialization,
prefers cheaper tools when scores are close, and supports progressive parameter
escalation for expensive operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from victor.tools.enums import CostTier

logger = logging.getLogger(__name__)


class LazyToolProxy:
    """Proxy that defers tool instantiation until first use.

    Wraps a factory callable. The tool is not created until execute()
    is called. Once created, the instance is cached for reuse.

    This reduces startup time by avoiding initialization of tools
    that may never be used in a session (e.g., Docker tool when
    no containers are involved).
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        cost_tier: CostTier = CostTier.FREE,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._factory = factory
        self._instance: Optional[Any] = None
        self._cost_tier = cost_tier
        self._description = description
        self._parameters = parameters or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def cost_tier(self) -> CostTier:
        return self._cost_tier

    @property
    def description(self) -> str:
        if self._instance:
            return getattr(self._instance, "description", self._description)
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        if self._instance:
            return getattr(self._instance, "parameters", self._parameters)
        return self._parameters

    @property
    def is_loaded(self) -> bool:
        return self._instance is not None

    def _ensure_loaded(self) -> Any:
        """Load the tool on first use."""
        if self._instance is None:
            logger.debug("Lazy-loading tool: %s", self._name)
            self._instance = self._factory()
        return self._instance

    async def execute(self, exec_ctx: Dict[str, Any], **kwargs: Any) -> Any:
        """Execute the tool, loading it first if needed."""
        tool = self._ensure_loaded()
        return await tool.execute(exec_ctx, **kwargs)

    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema without loading the tool."""
        return {
            "name": self._name,
            "description": self._description,
            "parameters": self._parameters,
        }

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the loaded tool."""
        if name.startswith("_"):
            raise AttributeError(name)
        tool = self._ensure_loaded()
        return getattr(tool, name)


def apply_cost_preference(
    candidates: List[Tuple[str, float, CostTier]],
    preference_boost: float = 0.05,
) -> List[Tuple[str, float, CostTier]]:
    """Re-rank tool candidates by applying a score boost to cheaper tools.

    When two tools have similar scores, the cheaper one wins. The boost
    is applied per tier: FREE gets full boost, LOW gets 2/3, MEDIUM gets 1/3.

    Args:
        candidates: List of (name, score, cost_tier) tuples
        preference_boost: Max score boost for FREE tier tools

    Returns:
        Re-ranked candidates (highest adjusted score first)
    """
    if not candidates:
        return []

    tier_boost = {
        CostTier.FREE: preference_boost,
        CostTier.LOW: preference_boost * 0.67,
        CostTier.MEDIUM: preference_boost * 0.33,
        CostTier.HIGH: 0.0,
    }

    adjusted = [(name, score + tier_boost.get(tier, 0.0), tier) for name, score, tier in candidates]

    return sorted(adjusted, key=lambda x: -x[1])


@dataclass
class ProgressiveParams:
    """Progressive parameter escalation for expensive operations.

    Starts with conservative values (e.g., max_results=5) and
    escalates on feedback (e.g., "not enough results").

    Usage:
        pp = ProgressiveParams(initial=5, max_value=100, escalation_factor=2.0)
        results = search(max_results=pp.current)
        if not enough:
            pp.escalate()
            results = search(max_results=pp.current)
    """

    initial: int
    max_value: int
    escalation_factor: float = 2.0
    _current: Optional[int] = None

    @property
    def current(self) -> int:
        if self._current is None:
            return self.initial
        return self._current

    def escalate(self) -> int:
        """Escalate to next level. Returns new value."""
        new_value = int(self.current * self.escalation_factor)
        self._current = min(new_value, self.max_value)
        return self._current

    def reset(self) -> None:
        """Reset to initial value."""
        self._current = None
