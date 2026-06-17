"""Tool deduplication system for Victor AI framework.

This module provides unified deduplication across native, LangChain, and MCP tools
with priority-based conflict resolution and naming convention enforcement.

Example:
    from victor.tools.deduplication import ToolDeduplicator, DeduplicationConfig

    config = DeduplicationConfig()
    deduplicator = ToolDeduplicator(config)
    deduplicated_tools = deduplicator.deduplicate(tools)
"""

from victor.tools.deduplication.tool_deduplicator import (
    DeduplicationConfig,
    DeduplicationResult,
    ToolDeduplicator,
    ToolSource,
)

__all__ = [
    "ToolDeduplicator",
    "DeduplicationConfig",
    "DeduplicationResult",
    "ToolSource",
]
