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

"""Hierarchical tool cache with dependency-aware invalidation.

This module implements a sophisticated caching system for tool results
that supports namespace isolation, dependency tracking, and cascading
invalidation.

Design Patterns:
    - Strategy Pattern: Pluggable cache backends via ICacheBackend
    - Observer Pattern: Dependency tracking for cascading invalidation
    - Decorator Pattern: Namespace-aware cache operations
    - SRP: Separates cache storage, dependency tracking, and invalidation

Usage:
    from victor.agent.cache.tool_cache_manager import ToolCacheManager
    from victor.storage.cache.tiered_cache import TieredCache

    # Create manager with default backend
    manager = ToolCacheManager(backend=TieredCache())

    # Cache tool result with namespace
    await manager.set_tool_result(
        tool_name="code_search",
        args_hash="abc123",
        result={"files": ["main.py"]},
        namespace=CacheNamespace.SESSION,
        ttl_seconds=3600
    )

    # Invalidate tool and its dependents
    count = await manager.invalidate_tool("code_search", cascade=True)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from victor.protocols import ICacheBackend, FileChangeEvent
from victor.agent.cache.dependency_graph import ToolDependencyGraph


class CacheNamespace(Enum):
    """Cache namespace levels for isolation.

    Namespaces provide hierarchical isolation:
    - GLOBAL: Shared across all sessions (e.g., static code analysis)
    - SESSION: Shared within a session (e.g., file read results)
    - REQUEST: Isolated to a single request (e.g., intermediate computations)
    - TOOL: Isolated per tool (e.g., tool-specific state)

    Hierarchy (parent to children):
    - GLOBAL → SESSION → REQUEST → TOOL
    """

    GLOBAL = "global"
    SESSION = "session"
    REQUEST = "request"
    TOOL = "tool"

    def get_child_namespaces(self) -> List[CacheNamespace]:
        """Get child namespaces in the hierarchy.

        Returns:
            List of child namespaces (more specific levels)

        Example:
            CacheNamespace.GLOBAL.get_child_namespaces()
            # [CacheNamespace.SESSION, CacheNamespace.REQUEST, CacheNamespace.TOOL]

            CacheNamespace.TOOL.get_child_namespaces()
            # []
        """
        hierarchy = [
            CacheNamespace.GLOBAL,
            CacheNamespace.SESSION,
            CacheNamespace.REQUEST,
            CacheNamespace.TOOL,
        ]
        try:
            current_index = hierarchy.index(self)
            return hierarchy[current_index + 1 :]
        except ValueError:
            return []
        except IndexError:
            return []


@dataclass
class InvalidationResult:
    """Result of a cache invalidation operation.

    Attributes:
        namespace: The namespace that was invalidated
        keys_cleared: Number of cache keys cleared
        recursive: Whether child namespaces were also cleared
        metadata: Additional metadata about the invalidation
    """

    namespace: str
    keys_cleared: int
    recursive: bool = False
    metadata: Dict[str, Any] | None = None


@dataclass
class CacheEntry:
    """A cached tool result entry.

    Attributes:
        tool_name: Name of the tool that produced this result
        args_hash: Hash of the tool arguments
        result: The cached result
        namespace: Cache namespace for this entry
        dependent_tools: Tools that depend on this entry
        file_dependencies: Files this result depends on
        created_at: Timestamp when entry was created
        expires_at: Optional expiration timestamp
        metadata: Additional metadata
    """

    tool_name: str
    args_hash: str
    result: Any
    namespace: CacheNamespace
    dependent_tools: Set[str]
    file_dependencies: Set[str]
    created_at: str
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] | None = None


class ToolCacheManager:
    """Hierarchical tool cache with dependency-aware invalidation.

    This manager provides:
    1. Namespace-aware caching (GLOBAL, SESSION, REQUEST, TOOL)
    2. Tool-to-tool dependency tracking
    3. Tool-to-file dependency tracking
    4. Cascading invalidation
    5. Pluggable cache backends via ICacheBackend protocol

    The manager implements ICacheBackend protocol for dependency injection.
    """

    def __init__(
        self,
        backend: ICacheBackend,
        default_ttl: int = 3600,
        enable_dependency_tracking: bool = True,
        enable_file_watching: bool = False,
    ) -> None:
        """Initialize the tool cache manager.

        Args:
            backend: Cache backend for storage (implements ICacheBackend)
            default_ttl: Default TTL for cache entries in seconds
            enable_dependency_tracking: Enable dependency graph tracking
            enable_file_watching: Enable automatic file watching for invalidation
        """
        self._backend = backend
        self._default_ttl = default_ttl
        self._enable_dependency_tracking = enable_dependency_tracking
        self._dependency_graph = ToolDependencyGraph()
        self._namespace_prefixes: Dict[CacheNamespace, str] = {
            CacheNamespace.GLOBAL: "global",
            CacheNamespace.SESSION: "session",
            CacheNamespace.REQUEST: "request",
            CacheNamespace.TOOL: "tool",
        }
        # Track cache keys by tool name for invalidation
        # Structure: {(tool_name, namespace): set(cache_keys)}
        self._tool_keys: Dict[tuple[str, str], Set[str]] = {}

        # File watching support
        self._enable_file_watching = enable_file_watching
        self._file_watcher: Optional[Any] = None  # FileWatcher instance
        self._file_watch_task: Optional[asyncio.Task[Any]] = None
        self._logger = logging.getLogger(__name__)

    def _make_cache_key(
        self,
        tool_name: str,
        args_hash: str,
        namespace: CacheNamespace,
    ) -> str:
        """Create a cache key with namespace prefix.

        Args:
            tool_name: Name of the tool
            args_hash: Hash of tool arguments
            namespace: Cache namespace

        Returns:
            Namespaced cache key
        """
        prefix = self._namespace_prefixes[namespace]
        return f"{prefix}:{tool_name}:{args_hash}"

    def _hash_args(self, args: Dict[str, Any]) -> str:
        """Create a stable hash for tool arguments.

        Args:
            args: Tool arguments dictionary

        Returns:
            SHA256 hash of arguments
        """
        try:
            data = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            data = str(args)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    async def get_tool_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        namespace: CacheNamespace = CacheNamespace.SESSION,
    ) -> Optional[Any]:
        """Get cached tool result.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            namespace: Cache namespace to query

        Returns:
            Cached result if found and not expired, None otherwise

        Example:
            result = await manager.get_tool_result(
                tool_name="code_search",
                args={"query": "authentication", "root": "/src"},
                namespace=CacheNamespace.SESSION
            )
        """
        args_hash = self._hash_args(args)
        cache_key = self._make_cache_key(tool_name, args_hash, namespace)
        namespace_str = namespace.value

        return await self._backend.get(cache_key, namespace_str)

    async def set_tool_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        namespace: CacheNamespace = CacheNamespace.SESSION,
        ttl_seconds: Optional[int] = None,
        dependent_tools: Optional[Set[str]] = None,
        file_dependencies: Optional[Set[str]] = None,
    ) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Result to cache
            namespace: Cache namespace for this entry
            ttl_seconds: TTL in seconds (None = use default)
            dependent_tools: Tools that depend on this result
            file_dependencies: Files this result depends on

        Example:
            await manager.set_tool_result(
                tool_name="code_search",
                args={"query": "authentication", "root": "/src"},
                result={"files": ["auth.py"]},
                namespace=CacheNamespace.SESSION,
                ttl_seconds=3600,
                file_dependencies={"/src/auth.py"}
            )
        """
        args_hash = self._hash_args(args)
        cache_key = self._make_cache_key(tool_name, args_hash, namespace)
        namespace_str = namespace.value
        ttl = ttl_seconds or self._default_ttl

        # Store in backend
        await self._backend.set(cache_key, result, namespace_str, ttl)

        # Track cache key for this tool/namespace combination
        key = (tool_name, namespace_str)
        if key not in self._tool_keys:
            self._tool_keys[key] = set()
        self._tool_keys[key].add(cache_key)

        # Track dependencies if enabled
        if self._enable_dependency_tracking:
            if dependent_tools:
                for dep_tool in dependent_tools:
                    self._dependency_graph.add_tool_dependency(dep_tool, tool_name)

            if file_dependencies:
                for file_path in file_dependencies:
                    self._dependency_graph.add_file_dependency(tool_name, file_path)

    async def invalidate_tool(
        self,
        tool_name: str,
        cascade: bool = True,
        namespace: Optional[CacheNamespace] = None,
    ) -> int:
        """Invalidate cached results for a tool.

        Args:
            tool_name: Name of the tool to invalidate
            cascade: If True, also invalidate dependent tools
            namespace: Specific namespace to clear (None = all namespaces)

        Returns:
            Number of cache entries invalidated

        Example:
            count = await manager.invalidate_tool(
                "code_search",
                cascade=True
            )
            print(f"Invalidated {count} entries")
        """
        invalidated = 0
        tools_to_invalidate = {tool_name}

        # Add dependent tools if cascading
        if cascade and self._enable_dependency_tracking:
            transitive = self._dependency_graph.get_transitive_dependents(tool_name)
            tools_to_invalidate.update(transitive)

        # Invalidate each tool
        for tool in tools_to_invalidate:
            if namespace:
                # Clear specific namespace
                namespace_str = namespace.value
                key = (tool, namespace_str)

                # Delete all tracked keys for this tool/namespace
                if key in self._tool_keys:
                    for cache_key in list(self._tool_keys[key]):
                        deleted = await self._backend.delete(cache_key, namespace_str)
                        if deleted:
                            invalidated += 1
                    # Clear tracking for this tool/namespace
                    del self._tool_keys[key]
            else:
                # Clear all namespaces
                for ns in CacheNamespace:
                    namespace_str = ns.value
                    key = (tool, namespace_str)

                    # Delete all tracked keys for this tool/namespace
                    if key in self._tool_keys:
                        for cache_key in list(self._tool_keys[key]):
                            deleted = await self._backend.delete(cache_key, namespace_str)
                            if deleted:
                                invalidated += 1
                        # Clear tracking for this tool/namespace
                        del self._tool_keys[key]

        return invalidated

    async def invalidate_file_dependencies(
        self,
        file_path: str,
    ) -> int:
        """Invalidate all tools that depend on a file.

        When a file is modified, this method invalidates all cached
        results from tools that depend on that file.

        Args:
            file_path: Path to the modified file

        Returns:
            Number of cache entries invalidated

        Example:
            count = await manager.invalidate_file_dependencies("/src/main.py")
            print(f"Invalidated {count} tool results due to file change")
        """
        if not self._enable_dependency_tracking:
            return 0

        invalidated = 0
        dependent_tools = self._dependency_graph.get_file_dependents(file_path)

        for tool_name in dependent_tools:
            count = await self.invalidate_tool(tool_name, cascade=False)
            invalidated += count

        return invalidated

    async def invalidate_namespace(
        self,
        namespace: CacheNamespace,
        recursive: bool = False,
    ) -> InvalidationResult:
        """Invalidate all entries in a namespace with optional recursive clearing.

        Args:
            namespace: Namespace to clear
            recursive: If True, also clear all child namespaces

        Returns:
            InvalidationResult with keys_cleared count and metadata

        Example:
            # Clear single namespace
            result = await manager.invalidate_namespace(CacheNamespace.SESSION)
            print(f"Cleared {result.keys_cleared} session entries")

            # Clear namespace and all children
            result = await manager.invalidate_namespace(
                CacheNamespace.GLOBAL, recursive=True
            )
            print(f"Cleared {result.keys_cleared} entries across {result.metadata['child_count'] + 1} namespaces")
        """
        count = 0
        namespaces_to_clear = [namespace]

        # Add child namespaces if recursive
        if recursive:
            children = namespace.get_child_namespaces()
            namespaces_to_clear.extend(children)

        # Clear each namespace
        cleared_namespace_strings = []
        for ns in namespaces_to_clear:
            namespace_str = ns.value
            cleared = await self._backend.clear_namespace(namespace_str)
            count += cleared
            cleared_namespace_strings.append(namespace_str)

        return InvalidationResult(
            namespace=namespace.value,
            keys_cleared=count,
            recursive=recursive,
            metadata={
                "cleared_namespaces": cleared_namespace_strings,
                "child_count": len(namespaces_to_clear) - 1,
            },
        )

    async def clear_all(self) -> int:
        """Clear all cached entries across all namespaces.

        Returns:
            Number of entries cleared
        """
        total = 0
        for namespace in CacheNamespace:
            result = await self.invalidate_namespace(namespace)
            total += result.keys_cleared
        return total

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics

        Example:
            stats = await manager.get_stats()
            # {
            #     "backend_stats": {...},
            #     "dependency_stats": {...}
            # }
        """
        backend_stats = await self._backend.get_stats()
        dependency_stats = self._dependency_graph.get_stats()

        return {
            "backend_stats": backend_stats,
            "dependency_stats": dependency_stats,
        }

    # ICacheBackend protocol implementation

    async def get(
        self,
        key: str,
        namespace: str,
    ) -> Optional[Any]:
        """Get value from cache (ICacheBackend protocol)."""
        return await self._backend.get(key, namespace)

    async def set(
        self,
        key: str,
        value: Any,
        namespace: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set value in cache (ICacheBackend protocol)."""
        await self._backend.set(key, value, namespace, ttl_seconds)

    async def delete(
        self,
        key: str,
        namespace: str,
    ) -> bool:
        """Delete value from cache (ICacheBackend protocol)."""
        return await self._backend.delete(key, namespace)

    async def clear_namespace(
        self,
        namespace: str,
    ) -> int:
        """Clear namespace (ICacheBackend protocol).

        Note: This method accepts a string namespace for ICacheBackend protocol.
        For namespace enum support, use invalidate_namespace().
        """
        return await self._backend.clear_namespace(namespace)

    # =========================================================================
    # File Watching Integration
    # =========================================================================

    async def start_file_watching(self, file_paths: List[str]) -> None:
        """Start file watching for automatic cache invalidation.

        Monitors the specified files and directories for changes and
        automatically invalidates dependent cache entries when changes occur.

        Args:
            file_paths: List of file or directory paths to watch

        Raises:
            RuntimeError: If file watching is not enabled or already running
            ImportError: If watchdog library is not installed

        Example:
            # Watch specific files
            await manager.start_file_watching(["/src/main.py", "/src/auth.py"])

            # Watch directories
            await manager.start_file_watching(["/src", "/tests"])

            # File changes will automatically invalidate dependent caches
        """
        if not self._enable_file_watching:
            raise RuntimeError(
                "File watching is not enabled. " "Set enable_file_watching=True in constructor."
            )

        if self._file_watcher is not None:
            raise RuntimeError("File watching is already running")

        try:
            from victor.agent.cache.file_watcher import FileWatcher
        except ImportError as e:
            raise ImportError(
                "File watching requires the 'watchdog' library. "
                "Install it with: pip install watchdog"
            ) from e

        # Create and start file watcher
        self._file_watcher = FileWatcher()
        await self._file_watcher.start()

        # Watch each path
        for path in file_paths:
            from pathlib import Path

            p = Path(path)
            if p.is_file():
                await self._file_watcher.watch_file(path)
            elif p.is_dir():
                await self._file_watcher.watch_directory(path, recursive=True)
            else:
                self._logger.warning(f"Path does not exist, skipping: {path}")

        # Start background task to process file changes
        self._file_watch_task = asyncio.create_task(self._process_file_changes())

        self._logger.info(f"File watching started for {len(file_paths)} path(s)")

    async def stop_file_watching(self) -> None:
        """Stop file watching and clean up resources.

        Stops monitoring file system changes and cancels the background
        processing task.

        Example:
            await manager.stop_file_watching()
        """
        if self._file_watch_task is not None:
            self._file_watch_task.cancel()
            try:
                await self._file_watch_task
            except asyncio.CancelledError:
                pass
            self._file_watch_task = None

        if self._file_watcher is not None:
            await self._file_watcher.stop()
            self._file_watcher = None

        self._logger.info("File watching stopped")

    async def _process_file_changes(self) -> None:
        """Background task to process file change events.

        Automatically invalidates cache entries when watched files change.
        """
        if self._file_watcher is None:
            return

        try:
            while True:
                # Get pending changes (non-blocking)
                changes = await self._file_watcher.get_changes()

                if not changes:
                    # No changes, wait a bit before checking again
                    await asyncio.sleep(0.5)
                    continue

                # Process each change
                for change in changes:
                    await self._on_file_changed(change)

        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            raise
        except Exception as e:
            self._logger.error(f"Error processing file changes: {e}")

    async def _on_file_changed(self, event: FileChangeEvent) -> None:
        """Handle file change event for cache invalidation.

        Args:
            event: File change event
        """
        file_path = event.file_path

        # Invalidate dependent caches
        try:
            invalidated = await self.invalidate_file_dependencies(file_path)

            if invalidated > 0:
                self._logger.info(
                    f"Invalidated {invalidated} cache entries " f"due to file change: {file_path}"
                )
        except Exception as e:
            self._logger.error(f"Failed to invalidate cache for {file_path}: {e}")


__all__ = [
    "ToolCacheManager",
    "CacheNamespace",
    "CacheEntry",
    "InvalidationResult",
]
