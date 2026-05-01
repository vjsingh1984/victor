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
from typing import Any, Dict, Optional, Tuple

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
        self._background_refresh: Dict[str, Dict[str, Any]] = {}
        self._refresh_tasks: Dict[str, asyncio.Task[None]] = {}

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

            # Subscribe to file changes and keep the persisted graph fresh.
            await self.ensure_background_refresh(root, exec_ctx=exec_ctx)

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

    async def ensure_background_refresh(
        self,
        root: Path,
        *,
        enable_ccg: bool = True,
        exec_ctx: Optional[Dict] = None,
        poll_interval_seconds: float = 1.0,
        debounce_seconds: float = 0.3,
        build_now: bool = False,
    ) -> Optional[Any]:
        """Ensure file watching plus incremental persisted-graph refresh for a root.

        Args:
            root: Root path to watch and incrementally index
            enable_ccg: Whether refreshes should rebuild Code Context Graph data
            exec_ctx: Optional execution context
            poll_interval_seconds: Watcher poll interval
            debounce_seconds: Watcher debounce window
            build_now: If True, run one incremental refresh immediately

        Returns:
            Initial refresh stats when ``build_now`` is True, otherwise None.
        """
        root = root.resolve()
        root_str = str(root)
        self._background_refresh[root_str] = {
            "enable_ccg": enable_ccg,
            "exec_ctx": exec_ctx,
            "pending": False,
        }

        initial_stats = None
        if build_now:
            initial_stats = await self._refresh_graph_index(root)

        await self._ensure_file_watcher(
            root,
            exec_ctx,
            poll_interval_seconds=poll_interval_seconds,
            debounce_seconds=debounce_seconds,
        )
        return initial_stats

    async def _ensure_file_watcher(
        self,
        root: Path,
        exec_ctx: Optional[Dict] = None,
        *,
        poll_interval_seconds: float = 1.0,
        debounce_seconds: float = 0.3,
    ) -> None:
        """Ensure file watcher is subscribed for this root.

        Args:
            root: Root path to watch
            exec_ctx: Execution context
            poll_interval_seconds: Poll interval for file detection
            debounce_seconds: Debounce interval before publishing events
        """
        root_str = str(root.resolve())

        # Check if already subscribed
        if root_str in self._watcher_subscribed:
            return

        # Get file watcher
        watcher_registry = FileWatcherRegistry.get_instance()
        file_watcher = await watcher_registry.get_watcher(
            root,
            poll_interval_seconds=poll_interval_seconds,
            debounce_seconds=debounce_seconds,
        )

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

        if root_str in self._background_refresh:
            self._schedule_background_refresh(root)

        if stale_count > 0:
            logger.info(
                f"[GraphManager] Marked {stale_count} graph(s) stale for {root_str} "
                f"due to {event.change_type.value}: {event.path}"
            )

    def _schedule_background_refresh(self, root: Path) -> None:
        """Schedule incremental refresh of the persisted graph for a root."""
        root_str = str(root.resolve())
        refresh_config = self._background_refresh.get(root_str)
        if refresh_config is None:
            return

        active_task = self._refresh_tasks.get(root_str)
        if active_task is not None and not active_task.done():
            refresh_config["pending"] = True
            return

        refresh_config["pending"] = False
        self._refresh_tasks[root_str] = asyncio.create_task(self._run_refresh_loop(root.resolve()))

    async def _run_refresh_loop(self, root: Path) -> None:
        """Run one or more coalesced incremental refresh passes for a root."""
        root_str = str(root.resolve())
        current_task = asyncio.current_task()
        try:
            while True:
                await self._refresh_graph_index(root)
                refresh_config = self._background_refresh.get(root_str)
                if not refresh_config or not refresh_config.get("pending", False):
                    break
                refresh_config["pending"] = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[GraphManager] Background refresh failed for %s: %s", root_str, exc)
        finally:
            task = self._refresh_tasks.get(root_str)
            if task is current_task:
                self._refresh_tasks.pop(root_str, None)

    async def _refresh_graph_index(self, root: Path) -> Any:
        """Incrementally refresh the persisted graph and synthetic edges for a root."""
        from victor.core.graph_rag import GraphIndexConfig, GraphIndexingPipeline
        from victor.core.indexing.index_lock import IndexLockRegistry
        from victor.core.indexing.graph_enrichment import ensure_project_graph_enriched
        from victor.storage.graph import create_graph_store
        from victor.tools.common import latest_mtime

        root = root.resolve()
        root_str = str(root)
        refresh_config = self._background_refresh.get(root_str, {})
        enable_ccg = bool(refresh_config.get("enable_ccg", True))

        lock_registry = IndexLockRegistry.get_instance()
        path_lock = await lock_registry.acquire_lock(root)

        async with path_lock:
            graph_store = create_graph_store("sqlite", root)
            config = GraphIndexConfig(
                root_path=root,
                enable_ccg=enable_ccg,
                enable_embeddings=False,
                enable_subgraph_cache=False,
            )
            pipeline = GraphIndexingPipeline(graph_store, config)
            stats = await pipeline.index_repository()

            repo_mtime = latest_mtime(root)
            ensure_project_graph_enriched(root, latest_mtime=repo_mtime)

        if stats.files_processed or stats.files_deleted:
            logger.info(
                "[GraphManager] Incremental graph refresh complete for %s "
                "(changed=%d deleted=%d unchanged=%d)",
                root_str,
                stats.files_processed,
                stats.files_deleted,
                stats.files_unchanged,
            )
        else:
            logger.debug("[GraphManager] Graph already current for %s", root_str)

        return stats

    async def wait_for_refresh(self, root: Path, timeout: float = 5.0) -> bool:
        """Wait for an active background refresh to complete for a root."""
        root_str = str(root.resolve())
        task = self._refresh_tasks.get(root_str)
        if task is None:
            return True

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def stop_background_refresh(self, root: Optional[Path] = None) -> int:
        """Cancel active background refresh tasks and clear refresh configuration."""
        if root is None:
            roots = list(self._background_refresh.keys())
        else:
            roots = [str(root.resolve())]

        stopped = 0
        for root_str in roots:
            task = self._refresh_tasks.pop(root_str, None)
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                stopped += 1
            self._background_refresh.pop(root_str, None)

        return stopped

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
            "background_refresh_roots": len(self._background_refresh),
            "active_refresh_tasks": sum(
                1 for task in self._refresh_tasks.values() if not task.done()
            ),
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
