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

"""Executes compiled tool execution graphs.

This module provides fast execution of pre-compiled tool execution graphs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from victor.agent.tool_graph import ToolExecutionGraph

logger = logging.getLogger(__name__)


class ToolExecutionEngine:
    """Executes pre-compiled graphs (fast path).

    The engine provides optimized execution of pre-compiled tool execution
    graphs with caching support.
    """

    def __init__(self, tool_pipeline: Any) -> None:
        """Initialize execution engine.

        Args:
            tool_pipeline: ToolPipeline instance
        """
        self._tool_pipeline = tool_pipeline
        self._graph_cache: Dict[str, ToolExecutionGraph] = {}

    async def execute(self, graph: ToolExecutionGraph, context: Dict[str, Any]) -> Any:
        """Execute pre-compiled graph.

        Args:
            graph: Compiled execution graph
            context: Execution context

        Returns:
            PipelineExecutionResult
        """
        from victor.agent.tool_pipeline import PipelineExecutionResult

        results = []
        successful = 0
        failed = 0

        for node in graph.nodes:
            # Execute node
            tool_call = {"name": node.tool_name, "arguments": {}}

            # Execute using tool pipeline
            call_result = await self._tool_pipeline._execute_single_call(tool_call, context)

            results.append(call_result)

            if call_result.success:
                successful += 1
            else:
                failed += 1

        return PipelineExecutionResult(
            results=results,
            total_calls=len(results),
            successful_calls=successful,
            failed_calls=failed,
        )

    def cache_graph(self, key: str, graph: ToolExecutionGraph) -> None:
        """Cache compiled graph.

        Args:
            key: Cache key
            graph: Compiled graph to cache
        """
        self._graph_cache[key] = graph
        logger.debug(f"Cached tool execution graph: {key}")

    def get_cached_graph(self, key: str) -> ToolExecutionGraph | None:
        """Get cached graph.

        Args:
            key: Cache key

        Returns:
            Cached graph if found, None otherwise
        """
        return self._graph_cache.get(key)

    def clear_cache(self) -> None:
        """Clear graph cache."""
        self._graph_cache.clear()
        logger.debug("Cleared tool execution graph cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cached_graphs": len(self._graph_cache),
            "cache_keys": list(self._graph_cache.keys()),
        }
