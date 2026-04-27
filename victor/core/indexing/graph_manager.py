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

"""Graph management with automatic updates.

This module provides automatic graph invalidation and updates based on
file system changes detected by the file watcher service.

Pattern: Singleton + Observer + Cache + Repository
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from victor.core.indexing.file_watcher import (
    FileChangeEvent,
    FileChangeType,
    FileWatcherRegistry,
)
from victor.core.indexing.index_lock import IndexLockRegistry

logger = logging.getLogger(__name__)

__all__ = ["GraphManager"]


class GraphManager:
    """Manages code graph with automatic updates.

    Provides automatic cache invalidation and incremental updates
    based on file system changes. Subscribes to file watcher events
    and maintains per-mode graph caching.

    Pattern: Singleton + Observer + Cache

    Example:
        >>> manager = GraphManager.get_instance()
        >>> graph, built = await manager.get_or_build_graph(
        ...     root=Path("/my/project"),
        ...     mode="pagerank",
        ...     exec_ctx=None
        ... )
        >>> # Graph is automatically updated when files change
    """

    _instance: Optional["GraphManager"] = None

    def __init__(self):
        """Initialize GraphManager."""
        self._graph_cache: Dict[str, Dict] = {}
        self._cache_lock = asyncio.Lock()
        self._watcher_subscribed: set[str] = set()

    @classmethod
    def get_instance(cls) -> "GraphManager":
        """Get singleton instance.

        Returns:
            GraphManager singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_or_build_graph(
        self,
        root: Path,
        mode: str,
        exec_ctx: Optional[Dict] = None,
    ) -> Tuple[Dict[str, any], bool]:
        """Get cached graph or build/update it.

        Args:
            root: Root path of codebase
            mode: Graph mode (pagerank, centrality, trace, etc.)
            exec_ctx: Execution context

        Returns:
            Tuple of (graph, was_built)
        """
        root_str = str(root.resolve())
        cache_key = f"{root_str}:{mode}"

        # Check cache without lock (fast path)
        cache_entry = self._graph_cache.get(cache_key)
        if cache_entry and not cache_entry.get("stale", False):
            logger.info(f"[GraphManager] Cache hit for {cache_key}")
            return cache_entry["graph"], False

        # Acquire lock for this path and mode
        lock_registry = IndexLockRegistry.get_instance()
        # Use mode-specific lock path to allow concurrent different modes
        path_lock = await lock_registry.acquire_lock(root / mode)

        async with path_lock:
            # Double-check cache inside lock
            cache_entry = self._graph_cache.get(cache_key)
            if cache_entry and not cache_entry.get("stale", False):
                return cache_entry["graph"], False

            # Build graph
            logger.info(f"[GraphManager] Building graph for {cache_key}")

            # Subscribe to file changes (only once per root)
            await self._ensure_file_watcher(root, exec_ctx)

            # Import graph tool to build graph
            from victor.tools.graph_tool import graph

            try:
                result = await graph(
                    path=str(root),
                    mode=mode,
                    reindex=True,  # Force fresh graph
                    _exec_ctx=exec_ctx,
                )

                # Cache result
                self._graph_cache[cache_key] = {
                    "graph": result,
                    "built_at": datetime.now().timestamp(),
                    "stale": False,
                }

                logger.info(f"[GraphManager] Graph built for {cache_key}")
                return result, True

            except Exception as e:
                logger.error(f"[GraphManager] Failed to build graph for {cache_key}: {e}")
                raise

    async def _ensure_file_watcher(
        self,
        root: Path,
        exec_ctx: Optional[Dict] = None,
    ) -> None:
        """Ensure file watcher is subscribed for this root.

        Args:
            root: Root path to watch
            exec_ctx: Execution context
        """
        root_str = str(root.resolve())

        # Check if already subscribed
        if root_str in self._watcher_subscribed:
            return

        # Get file watcher
        watcher_registry = FileWatcherRegistry.get_instance()
        file_watcher = await watcher_registry.get_watcher(root)

        # Subscribe to file changes
        file_watcher.subscribe(
            lambda e: asyncio.create_task(self._on_file_change(e, root, exec_ctx))
        )

        # Mark as subscribed
        self._watcher_subscribed.add(root_str)

        logger.info(f"[GraphManager] Subscribed to file watcher for {root_str}")

    async def _on_file_change(
        self,
        event: FileChangeEvent,
        root: Path,
        exec_ctx: Optional[Dict] = None,
    ) -> None:
        """Handle file change event.

        Invalidates all cached graphs for this root.

        Args:
            event: File change event
            root: Root path of codebase
            exec_ctx: Execution context
        """
        root_str = str(root.resolve())

        # Mark all graphs for this root as stale
        stale_count = 0
        for key, cache_entry in self._graph_cache.items():
            if key.startswith(root_str):
                cache_entry["stale"] = True
                stale_count += 1
                logger.debug(f"[GraphManager] Graph marked stale: {key}")

        if stale_count > 0:
            logger.info(
                f"[GraphManager] Marked {stale_count} graph(s) stale for {root_str} "
                f"due to {event.change_type.value}: {event.path}"
            )

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about graph cache.

        Returns:
            Dict with current statistics
        """
        total_graphs = len(self._graph_cache)
        stale_graphs = sum(1 for e in self._graph_cache.values() if e.get("stale", False))
        watched_roots = len(self._watcher_subscribed)

        return {
            "total_graphs": total_graphs,
            "fresh_graphs": total_graphs - stale_graphs,
            "stale_graphs": stale_graphs,
            "watched_roots": watched_roots,
            "graph_details": {
                key: {
                    "built_at": entry.get("built_at"),
                    "stale": entry.get("stale", False),
                }
                for key, entry in self._graph_cache.items()
            },
        }

    async def invalidate_root(self, root: Path) -> int:
        """Invalidate all cached graphs for a specific root.

        Args:
            root: Root path to invalidate

        Returns:
            Number of graphs invalidated
        """
        root_str = str(root.resolve())
        invalidated = 0

        async with self._cache_lock:
            for key, cache_entry in self._graph_cache.items():
                if key.startswith(root_str):
                    cache_entry["stale"] = True
                    invalidated += 1

        if invalidated > 0:
            logger.info(f"[GraphManager] Invalidated {invalidated} graph(s) for {root_str}")

        return invalidated

    async def clear_cache(self, root: Optional[Path] = None) -> int:
        """Clear cached graphs.

        Args:
            root: Optional root path to clear. If None, clears all graphs.

        Returns:
            Number of graphs cleared
        """
        cleared = 0

        async with self._cache_lock:
            if root is None:
                cleared = len(self._graph_cache)
                self._graph_cache.clear()
                logger.info(f"[GraphManager] Cleared all {cleared} graph(s)")
            else:
                root_str = str(root.resolve())
                keys_to_remove = [key for key in self._graph_cache if key.startswith(root_str)]

                for key in keys_to_remove:
                    del self._graph_cache[key]
                    cleared += 1

                logger.info(f"[GraphManager] Cleared {cleared} graph(s) for {root_str}")

        return cleared
