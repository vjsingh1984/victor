"""LLM decision service protocol.

Defines the interface for the centralized LLM-assisted decision service
that provides structured classification when heuristic confidence is low.

The service follows a heuristic-first approach: fast keyword/regex checks
run first (0ms), and the LLM is only consulted as a fallback when the
heuristic confidence falls below a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from victor.agent.decisions.schemas import DecisionType


@dataclass
class DecisionResult:
    """Result of a decision call (heuristic or LLM-augmented).

    Attributes:
        decision_type: Which decision was requested.
        result: The parsed decision object (Pydantic model instance).
        source: How the decision was made.
        confidence: Confidence score 0.0-1.0.
        latency_ms: Wall-clock time for the decision call.
        tokens_used: Tokens consumed (0 for heuristic/cache).
    """

    decision_type: DecisionType
    result: Any
    source: (
        str  # "heuristic" | "llm" | "cache" | "timeout_fallback" | "budget_exhausted"
    )
    confidence: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class DecisionMetrics:
    """Aggregate metrics for decision service usage."""

    total_calls: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    timeouts: int = 0
    budget_exhaustions: int = 0
    parse_failures: int = 0
    avg_latency_ms: float = 0.0
    _latency_sum: float = field(default=0.0, repr=False)


@runtime_checkable
class LLMDecisionServiceProtocol(Protocol):
    """Protocol for LLM-based decision making.

    Provides a unified interface for consulting the LLM when heuristic
    classifiers have low confidence. Implements budget control, caching,
    and timeout protection to keep decision calls fast and bounded.

    This protocol follows the Interface Segregation Principle (ISP)
    by focusing only on decision-making operations.

    Example:
        class MyDecisionService(LLMDecisionServiceProtocol):
            async def decide(self, decision_type, context):
                result = await self._call_llm(decision_type, context)
                return DecisionResult(...)
    """

    async def decide(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Make a decision, consulting the LLM if heuristic confidence is low.

        Fast path: if heuristic_confidence >= threshold, returns the heuristic
        result immediately without an LLM call.

        Args:
            decision_type: Type of decision to make.
            context: Context dict with template placeholders for the prompt.
            heuristic_result: Pre-computed heuristic result (returned as-is if confident).
            heuristic_confidence: Confidence of the heuristic result (0.0-1.0).

        Returns:
            DecisionResult with the decision and metadata.
        """
        ...

    async def decide_async(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Async alias for decide() — use from async context.

        Avoids the thread-spawning overhead of decide_sync().
        Delegates directly to decide().

        Args:
            decision_type: Type of decision to make.
            context: Context dict with template placeholders.
            heuristic_result: Pre-computed heuristic result.
            heuristic_confidence: Confidence of the heuristic result.

        Returns:
            DecisionResult with the decision and metadata.
        """
        ...

    def decide_sync(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Synchronous version of decide().

        If an event loop is already running, returns the heuristic fallback
        rather than blocking. Safe to call from sync code paths.

        Args:
            decision_type: Type of decision to make.
            context: Context dict with template placeholders.
            heuristic_result: Pre-computed heuristic result.
            heuristic_confidence: Confidence of the heuristic result.

        Returns:
            DecisionResult with the decision and metadata.
        """
        ...

    @property
    def budget_remaining(self) -> int:
        """Number of LLM decision calls remaining in the current turn budget."""
        ...

    def reset_budget(self) -> None:
        """Reset the per-turn LLM call budget. Called at the start of each turn."""
        ...

    def is_healthy(self) -> bool:
        """Check if the decision service is operational."""
        ...

    def get_metrics(self) -> DecisionMetrics:
        """Get aggregate metrics for monitoring."""
        ...
