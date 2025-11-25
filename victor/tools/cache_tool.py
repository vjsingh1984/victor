"""Cache management tool for controlling Victor's caching system.

Features:
- Cache statistics
- Cache clearing
- Cache warmup
- Cache configuration
"""

from typing import Any, Dict, List
import logging

from victor.tools.base import BaseTool, ToolParameter, ToolResult
from victor.cache.manager import CacheManager
from victor.cache.config import CacheConfig

logger = logging.getLogger(__name__)


class CacheTool(BaseTool):
    """Tool for cache management and monitoring."""

    def __init__(self, cache_manager: CacheManager):
        """Initialize cache tool.

        Args:
            cache_manager: Cache manager instance to control
        """
        super().__init__()
        self.cache = cache_manager

    @property
    def name(self) -> str:
        """Get tool name."""
        return "cache"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Cache management and monitoring.

Control Victor's tiered caching system (memory + disk).

Operations:
- stats: Get cache statistics and hit rates
- clear: Clear cache (all or by namespace)
- info: Show cache configuration
- warmup: Preload cache with data

Example workflows:
1. View cache stats:
   cache(operation="stats")

2. Clear all cache:
   cache(operation="clear")

3. Clear specific namespace:
   cache(operation="clear", namespace="responses")

4. Show configuration:
   cache(operation="info")
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
        [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: stats, clear, info",
                required=True,
            ),
            ToolParameter(
                name="namespace",
                type="string",
                description="Cache namespace (for clear operation)",
                required=False,
            ),
        ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute cache operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with cache information
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "stats":
                return await self._get_stats(kwargs)
            elif operation == "clear":
                return await self._clear_cache(kwargs)
            elif operation == "info":
                return await self._get_info(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Cache operation failed")
            return ToolResult(
                success=False, output="", error=f"Cache error: {str(e)}"
            )

    async def _get_stats(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Get cache statistics."""
        stats = self.cache.get_stats()

        report = []
        report.append("Cache Statistics")
        report.append("=" * 70)
        report.append("")

        report.append("Performance:")
        report.append(
            f"  Memory Hit Rate: {stats.get('memory_hit_rate', 0):.2%}"
        )
        report.append(f"  Disk Hit Rate: {stats.get('disk_hit_rate', 0):.2%}")
        report.append("")

        report.append("Counts:")
        report.append(f"  Memory Hits: {stats.get('memory_hits', 0)}")
        report.append(f"  Memory Misses: {stats.get('memory_misses', 0)}")
        report.append(f"  Disk Hits: {stats.get('disk_hits', 0)}")
        report.append(f"  Disk Misses: {stats.get('disk_misses', 0)}")
        report.append(f"  Total Sets: {stats.get('sets', 0)}")
        report.append("")

        if "memory_size" in stats:
            report.append("Memory Cache:")
            report.append(
                f"  Current Size: {stats['memory_size']}/{stats.get('memory_max_size', 0)}"
            )
            report.append("")

        if "disk_size" in stats:
            report.append("Disk Cache:")
            report.append(f"  Entries: {stats['disk_size']}")
            if "disk_volume" in stats:
                volume_mb = stats["disk_volume"] / (1024 * 1024)
                report.append(f"  Volume: {volume_mb:.2f} MB")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _clear_cache(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Clear cache."""
        namespace = kwargs.get("namespace")

        count = self.cache.clear(namespace)

        if namespace:
            message = f"Cleared {count} entries from namespace '{namespace}'"
        else:
            message = f"Cleared all cache ({count} entries)"

        return ToolResult(
            success=True,
            output=message,
            error="",
        )

    async def _get_info(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Get cache configuration info."""
        config = self.cache.config

        report = []
        report.append("Cache Configuration")
        report.append("=" * 70)
        report.append("")

        report.append("Architecture: Tiered (L1 Memory + L2 Disk)")
        report.append("")

        report.append("L1 Memory Cache:")
        report.append(f"  Enabled: {config.enable_memory}")
        if config.enable_memory:
            report.append(f"  Max Size: {config.memory_max_size} entries")
            report.append(f"  TTL: {config.memory_ttl} seconds ({config.memory_ttl//60} min)")
        report.append("")

        report.append("L2 Disk Cache:")
        report.append(f"  Enabled: {config.enable_disk}")
        if config.enable_disk:
            size_mb = config.disk_max_size / (1024 * 1024)
            report.append(f"  Max Size: {size_mb:.0f} MB")
            report.append(
                f"  TTL: {config.disk_ttl} seconds ({config.disk_ttl//86400} days)"
            )
            report.append(f"  Path: {config.disk_path}")
        report.append("")

        report.append("Features:")
        report.append("  ✓ Automatic tiering (memory → disk → source)")
        report.append("  ✓ TTL-based expiration")
        report.append("  ✓ Thread-safe operations")
        report.append("  ✓ Persistent across restarts (disk cache)")
        report.append("  ✓ Zero external dependencies")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )
