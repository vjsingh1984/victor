"""Caching for tool execution decisions (hot path optimization).

This module provides caching for tool execution decisions to avoid repeated
validation, normalization, and signature computation in the hot path.

Performance target: 10-20% improvement in tool execution throughput.
"""

from dataclasses import dataclass
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolValidationResult:
    """Cached validation result.

    Attributes:
        is_valid: Whether the tool is valid and enabled
        tool_exists: Whether the tool exists in registry
        is_enabled: Whether the tool is enabled
    """

    is_valid: bool
    tool_exists: bool
    is_enabled: bool


@dataclass
class ToolNormalizationResult:
    """Cached normalization result.

    Attributes:
        normalized_args: Normalized arguments dictionary
        strategy: Normalization strategy used
        signature: Computed call signature
    """

    normalized_args: dict[str, Any]
    strategy: str
    signature: str


class ToolExecutionDecisionCache:
    """Cache for tool execution decisions.

    Provides performance optimization by caching:
    - Tool validation results (tool names don't change during session)
    - Argument normalization (same args normalize same way)
    - Signature computation (expensive hashing)

    Thread-safety: Not thread-safe by design. ToolPipeline creates instances
    per conversation, so no cross-thread contention.

    Attributes:
        _max_size: Maximum cache size before eviction
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries per cache before eviction
        """
        self._validation_cache: dict[str, ToolValidationResult] = {}
        self._normalization_cache: dict[
            tuple[str, str | frozenset[tuple[str, str]] | None], ToolNormalizationResult
        ] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def is_valid_tool(self, tool_name: str, tool_registry: Any) -> ToolValidationResult:
        """Get cached validation result for a tool.

        Args:
            tool_name: Name of the tool to validate
            tool_registry: Tool registry instance

        Returns:
            ToolValidationResult with validation status
        """
        if tool_name not in self._validation_cache:
            # Cache miss - validate and cache
            self._misses += 1

            tool_exists = tool_registry.get(tool_name) is not None
            is_enabled = tool_registry.is_tool_enabled(tool_name) if tool_exists else False

            self._validation_cache[tool_name] = ToolValidationResult(
                is_valid=is_enabled,
                tool_exists=tool_exists,
                is_enabled=is_enabled,
            )

            # Evict if over limit (FIFO)
            if len(self._validation_cache) > self._max_size:
                oldest = next(iter(self._validation_cache))
                del self._validation_cache[oldest]
                logger.debug(
                    f"Validation cache evicted oldest entry (size: {len(self._validation_cache)})"
                )
        else:
            self._hits += 1

        return self._validation_cache[tool_name]

    def get_normalized_args(
        self,
        tool_name: str,
        raw_args: Any,
        normalizer: Any,
    ) -> ToolNormalizationResult:
        """Get cached normalization result for tool arguments.

        Args:
            tool_name: Name of the tool
            raw_args: Raw arguments (may be string, dict, or None)
            normalizer: ArgumentNormalizer instance

        Returns:
            ToolNormalizationResult with normalized args, strategy, and signature
        """
        # Create hashable key from args (handle strings, dicts, None)
        key_items: str | frozenset[tuple[str, str]] | None
        try:
            if isinstance(raw_args, str):
                # For string args, use the string itself as key
                key_items = raw_args
            elif isinstance(raw_args, dict):
                # For dict args, create hashable representation
                key_items = frozenset((k, str(v)) for k, v in raw_args.items())
            else:
                # For None or other types, use empty representation
                key_items = None
        except Exception:
            # Fallback for unhashable values - convert to string
            key_items = str(raw_args)

        key: tuple[str, str | frozenset[tuple[str, str]] | None] = (tool_name, key_items)

        if key not in self._normalization_cache:
            # Cache miss - normalize and cache
            self._misses += 1

            # Normalize arguments (normalizer handles strings, dicts, None)
            normalized, strategy = normalizer.normalize_arguments(raw_args, tool_name)

            # Compute signature
            signature = self._compute_signature(tool_name, normalized)

            # Convert strategy to string for serialization
            strategy_str = str(strategy.value) if hasattr(strategy, "value") else str(strategy)

            self._normalization_cache[key] = ToolNormalizationResult(
                normalized_args=normalized, strategy=strategy_str, signature=signature
            )

            # Evict if over limit (FIFO)
            if len(self._normalization_cache) > self._max_size:
                oldest = next(iter(self._normalization_cache))
                del self._normalization_cache[oldest]
                logger.debug(
                    f"Normalization cache evicted oldest entry (size: {len(self._normalization_cache)})"
                )
        else:
            self._hits += 1

        return self._normalization_cache[key]

    def _compute_signature(self, tool_name: str, args: dict[str, Any]) -> str:
        """Compute signature for tool call.

        Uses JSON serialization with sorted keys for deterministic hashing.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Signature string
        """
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            args_str = str(args)
        return f"{tool_name}:{args_str}"

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats including hits, misses, hit rate, and sizes
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "validation_cache_size": len(self._validation_cache),
            "normalization_cache_size": len(self._normalization_cache),
        }

    def clear(self) -> None:
        """Clear all caches and reset statistics."""
        self._validation_cache.clear()
        self._normalization_cache.clear()
        self._hits = 0
        self._misses = 0
        logger.debug("Tool execution decision cache cleared")
