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

"""Cache protocols for dependency inversion and tool cache invalidation.

This module defines protocols for cache storage and tool cache invalidation,
following the Dependency Inversion Principle (DIP) and Interface Segregation
Principle (ISP).

Design Principles:
    - DIP: High-level modules depend on these protocols, not concrete implementations
    - ISP: Focused, minimal protocols - implement only what you need
    - OCP: New cache backends and invalidation strategies can be added without modification
    - Strategy Pattern: Different backends for different use cases

Protocols:
    ICacheBackend: Basic cache storage with namespace isolation
    ICacheInvalidator: Cascading cache invalidation with dependency tracking
    IIdempotentTool: Tool protocol for idempotency-aware caching
    ICacheDependencyTracker: Dependency tracking for intelligent invalidation

Supported Backends:
    - Memory: In-memory caching (default)
    - Redis: Distributed caching
    - SQLite: Persistent caching
    - Tiered: Multi-level caching (L1 memory + L2 Redis)
    - Custom: User-provided implementations

Usage:
    # Implement a cache backend
    class RedisCacheBackend(ICacheBackend):
        async def get(self, key: str, namespace: str) -> Optional[Any]:
            return await self._redis.get(f"{namespace}:{key}")

        async def set(self, key: str, value: Any, namespace: str, ttl_seconds: Optional[int]) -> None:
            await self._redis.setex(f"{namespace}:{key}", ttl_seconds or 3600, pickle.dumps(value))

    # Implement tool idempotency
    class ReadFileTool(BaseTool, IIdempotentTool):
        def is_idempotent(self) -> bool:
            return True  # Reading is idempotent

        def get_idempotency_key(self, **kwargs) -> str:
            return f"read_file:{kwargs['path']}"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable
from collections.abc import Iterator


# =============================================================================
# File Change Detection Types
# =============================================================================


class FileChangeType(str, Enum):
    """Types of file system changes.

    Attributes:
        CREATED: A new file was created
        MODIFIED: An existing file was modified
        DELETED: A file was deleted
        MOVED: A file was moved or renamed
    """

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChangeEvent:
    """Event representing a file system change.

    Attributes:
        file_path: Absolute path to the file that changed
        change_type: Type of change that occurred
        timestamp: ISO timestamp of when the change was detected (can be datetime or str)
        source_path: Original path if file was moved (None otherwise)
        metadata: Additional metadata about the change
    """

    file_path: str
    change_type: FileChangeType
    timestamp: str | datetime  # Accept both, normalize to str in __post_init__
    source_path: Optional[str] = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the file change event."""
        if isinstance(self.timestamp, datetime):
            # Convert datetime to ISO string if needed
            object.__setattr__(self, "timestamp", self.timestamp.isoformat())

        if self.change_type == FileChangeType.MOVED and not self.source_path:
            raise ValueError("source_path is required for MOVED events")


# =============================================================================
# File Watcher Protocol
# =============================================================================


@runtime_checkable
class IFileWatcher(Protocol):
    """Protocol for monitoring file system changes.

    Implementations track file modifications and emit events when files change.
    This enables automatic cache invalidation when dependent files are modified.

    Use Cases:
        - Watch source files for code analysis tools
        - Monitor configuration files for caching
        - Track documentation files for RAG systems
        - Detect asset file changes for build tools

    Implementation Guidelines:
        - Use watchdog library for cross-platform file watching
        - Support recursive directory watching
        - Be thread-safe for async/await usage
        - Queue events for later retrieval (non-blocking)
        - Handle file system errors gracefully

    Example Implementation:
        class WatchdogFileWatcher(IFileWatcher):
            def __init__(self):
                self._observer = Observer()
                self._event_handler = FileChangeHandler()
                self._watched_paths: Dict[str, Any] = {}
                self._change_queue: asyncio.Queue[FileChangeEvent] = asyncio.Queue()

            async def watch_file(self, file_path: str) -> None:
                path = Path(file_path).resolve()
                parent = str(path.parent)
                if parent not in self._watched_paths:
                    self._observer.schedule(
                        self._event_handler,
                        parent,
                        recursive=False
                    )
                    self._watched_paths[parent] = None
                self._observer.start()

            async def get_changes(self) -> List[FileChangeEvent]:
                changes = []
                while not self._change_queue.empty():
                    changes.append(await self._change_queue.get())
                return changes
    """

    async def watch_file(self, file_path: str) -> None:
        """Monitor a single file for changes.

        Args:
            file_path: Absolute path to the file to watch

        Example:
            await watcher.watch_file("/src/main.py")
        """
        ...

    async def watch_directory(self, directory: str, recursive: bool = True) -> None:
        """Monitor a directory for changes.

        Args:
            directory: Absolute path to the directory
            recursive: If True, watch subdirectories as well

        Example:
            # Watch directory recursively
            await watcher.watch_directory("/src", recursive=True)

            # Watch only immediate files
            await watcher.watch_directory("/src", recursive=False)
        """
        ...

    async def unwatch_file(self, file_path: str) -> None:
        """Stop monitoring a file.

        Args:
            file_path: Absolute path to the file to stop watching

        Example:
            await watcher.unwatch_file("/src/main.py")
        """
        ...

    async def unwatch_directory(self, directory: str) -> None:
        """Stop monitoring a directory.

        Args:
            directory: Absolute path to the directory to stop watching

        Example:
            await watcher.unwatch_directory("/src")
        """
        ...

    async def get_changes(self) -> list[FileChangeEvent]:
        """Get pending file change events.

        Returns:
            List of file change events since last call

        Example:
            changes = await watcher.get_changes()
            for change in changes:
                print(f"{change.file_path}: {change.change_type}")
        """
        ...

    async def start(self) -> None:
        """Start the file watcher.

        This method should be called before watching any files.
        Some implementations may start automatically on first watch.

        Example:
            await watcher.start()
            await watcher.watch_file("/src/main.py")
        """
        ...

    async def stop(self) -> None:
        """Stop the file watcher and release resources.

        Example:
            await watcher.stop()
        """
        ...

    def is_running(self) -> bool:
        """Check if the watcher is currently running.

        Returns:
            True if watcher is active, False otherwise

        Example:
            if watcher.is_running():
                print("Watcher is active")
        """
        ...


# =============================================================================
# Dependency Extraction Protocol
# =============================================================================


@runtime_checkable
class IDependencyExtractor(Protocol):
    """Protocol for extracting file dependencies from tool arguments.

    Implementations analyze tool arguments and automatically identify
    file paths that should be tracked for cache invalidation.

    Use Cases:
        - Extract file paths from read_file tool arguments
        - Extract directory paths from search tool arguments
        - Extract glob patterns from file listing tools
        - Extract module paths from import analysis tools

    Implementation Guidelines:
        - Support common argument patterns (path, file, directory, etc.)
        - Handle both strings and lists of strings
        - Validate paths exist (optional, for error detection)
        - Normalize paths (resolve relative paths, symlinks)
        - Be fast (called on every tool invocation)

    Example Implementation:
        class DependencyExtractor(IDependencyExtractor):
            def extract_file_dependencies(
                self,
                tool_name: str,
                arguments: Dict[str, Any]
            ) -> Set[str]:
                dependencies = set()

                # Check common argument patterns
                for key in ['path', 'file', 'files', 'directory', 'dir']:
                    if key in arguments:
                        value = arguments[key]
                        if isinstance(value, str):
                            dependencies.add(value)
                        elif isinstance(value, list):
                            dependencies.update(value)

                return dependencies
    """

    def extract_file_dependencies(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> set[str]:
        """Extract file paths from tool arguments.

        Analyzes tool arguments and identifies file paths that should
        be tracked for cache invalidation.

        Args:
            tool_name: Name of the tool being invoked
            arguments: Tool arguments dictionary

        Returns:
            Set of absolute file paths extracted from arguments

        Example:
            deps = extractor.extract_file_dependencies(
                "read_file",
                {"path": "/src/main.py", "encoding": "utf-8"}
            )
            # => {"/src/main.py"}

            deps = extractor.extract_file_dependencies(
                "code_search",
                {
                    "query": "authentication",
                    "files": ["/src/auth.py", "/src/login.py"],
                    "exclude": ["/src/test_*"]
                }
            )
            # => {"/src/auth.py", "/src/login.py"}
        """
        ...


# =============================================================================
# Cache Namespace Hierarchy
# =============================================================================


class CacheNamespace(str, Enum):
    """Cache namespace levels for hierarchical isolation.

    Namespaces provide hierarchical isolation to prevent cache pollution
    and enable granular invalidation. The hierarchy is:

    GLOBAL (shared across all sessions)
        └── SESSION (shared within a session)
            └── REQUEST (isolated to a single request)
                └── TOOL (isolated per tool invocation)

    Inheritance Rules:
        - Invalidating a parent namespace invalidates all children
        - Clearing SESSION also clears REQUEST and TOOL caches
        - GLOBAL is never automatically invalidated (manual only)

    Use Cases:
        GLOBAL: Static code analysis, project metadata, expensive computations
        SESSION: File read results, codebase search results, AST data
        REQUEST: Intermediate computations, temporary data
        TOOL: Tool-specific state, configuration data

    Example:
        # Cache static analysis result (shared across sessions)
        await cache.set(result, namespace=CacheNamespace.GLOBAL)

        # Cache file read (session-scoped)
        await cache.set(file_content, namespace=CacheNamespace.SESSION)

        # Invalidate session and all its children
        await cache.invalidate_namespace(CacheNamespace.SESSION)
    """

    GLOBAL = "global"
    SESSION = "session"
    REQUEST = "request"
    TOOL = "tool"

    def get_child_namespaces(self) -> list["CacheNamespace"]:
        """Get all child namespaces in the hierarchy.

        Returns:
            List of child namespaces (empty for leaf nodes)

        Example:
            CacheNamespace.SESSION.get_child_namespaces()
            # => [CacheNamespace.REQUEST, CacheNamespace.TOOL]
        """
        hierarchy = {
            CacheNamespace.GLOBAL: [CacheNamespace.SESSION],
            CacheNamespace.SESSION: [CacheNamespace.REQUEST, CacheNamespace.TOOL],
            CacheNamespace.REQUEST: [CacheNamespace.TOOL],
            CacheNamespace.TOOL: [],
        }
        return hierarchy.get(self, [])

    def get_parent_namespace(self) -> Optional["CacheNamespace"]:
        """Get parent namespace in the hierarchy.

        Returns:
            Parent namespace or None for GLOBAL root

        Example:
            CacheNamespace.SESSION.get_parent_namespace()
            # => CacheNamespace.GLOBAL
        """
        parents = {
            CacheNamespace.GLOBAL: None,
            CacheNamespace.SESSION: CacheNamespace.GLOBAL,
            CacheNamespace.REQUEST: CacheNamespace.SESSION,
            CacheNamespace.TOOL: CacheNamespace.REQUEST,
        }
        return parents.get(self)


# =============================================================================
# Cache Entry Metadata Types
# =============================================================================


@dataclass
class CacheEntryMetadata:
    """Metadata for a cache entry.

    Attributes:
        tool_name: Name of the tool that produced this entry
        args_hash: Hash of the tool arguments (for idempotency)
        dependent_tools: Tools that depend on this entry
        file_dependencies: Files this result depends on
        created_at: ISO timestamp of creation
        expires_at: Optional ISO timestamp of expiration
        access_count: Number of times this entry was accessed
        last_accessed: ISO timestamp of last access
        size_bytes: Estimated size in memory (if available)
        namespace: Cache namespace for this entry
        tags: Optional tags for custom invalidation strategies
    """

    tool_name: str
    args_hash: str
    dependent_tools: set[str]
    file_dependencies: set[str]
    created_at: str
    expires_at: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[str] = None
    size_bytes: Optional[int] = None
    namespace: CacheNamespace = CacheNamespace.SESSION
    tags: set[str] | None = None


@dataclass
class InvalidationResult:
    """Result from a cache invalidation operation.

    Attributes:
        success: Whether the invalidation succeeded
        entries_invalidated: Number of cache entries invalidated
        tools_invalidated: Number of tools affected
        namespaces_invalidated: Namespaces that were cleared
        error_message: Error message if invalidation failed
        cascade_depth: Depth of cascading invalidation (0 = no cascade)
        metadata: Additional metadata about the invalidation
    """

    success: bool
    entries_invalidated: int
    tools_invalidated: int
    namespaces_invalidated: list[str]
    error_message: str | None = None
    cascade_depth: int = 0
    metadata: dict[str, Any] | None = None


@dataclass
class CacheStatistics:
    """Statistics about cache performance and usage.

    Attributes:
        backend_type: Type of cache backend
        total_entries: Total number of cached entries
        hit_rate: Cache hit rate (0.0 to 1.0)
        miss_rate: Cache miss rate (0.0 to 1.0)
        memory_used: Memory usage in bytes (if available)
        eviction_count: Number of entries evicted due to TTL/size
        namespace_stats: Statistics per namespace
        tool_stats: Statistics per tool
        dependency_stats: Dependency graph statistics
    """

    backend_type: str
    total_entries: int
    hit_rate: float
    miss_rate: float
    memory_used: Optional[int]
    eviction_count: int
    namespace_stats: dict[str, dict[str, Any]]
    tool_stats: dict[str, dict[str, Any]]
    dependency_stats: dict[str, int]


# =============================================================================
# Core Cache Backend Protocol
# =============================================================================


@runtime_checkable
class ICacheBackend(Protocol):
    """Protocol for cache storage backends.

    Implementations provide cache storage with namespace isolation,
    TTL support, and statistics tracking. The protocol enables
    pluggable backends for different deployment scenarios:

    - Memory: Fast, local caching (default)
    - Redis: Distributed caching for multi-process deployments
    - SQLite: Persistent caching across restarts
    - Tiered: Multi-level caching with automatic promotion/demotion
    - Custom: User-provided implementations

    Namespace isolation prevents cache key collisions between
    different components (tools, sessions, global, etc.).

    Implementation Guidelines:
        - All methods must be async for non-blocking operation
        - TTL must be enforced (auto-expire entries after ttl_seconds)
        - Namespaces must be isolated (clear_namespace doesn't affect other namespaces)
        - Thread/process-safe for concurrent access
        - Handle serialization/deserialization internally

    Example Implementation:
        class RedisCacheBackend(ICacheBackend):
            def __init__(self, redis_url: str):
                self._redis = aioredis.from_url(redis_url)

            async def get(self, key: str, namespace: str) -> Optional[Any]:
                data = await self._redis.get(f"{namespace}:{key}")
                return pickle.loads(data) if data else None

            async def set(self, key: str, value: Any, namespace: str, ttl_seconds: Optional[int]) -> None:
                serialized = pickle.dumps(value)
                await self._redis.setex(f"{namespace}:{key}", ttl_seconds or 3600, serialized)
    """

    async def get(self, key: str, namespace: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key
            namespace: Namespace for isolation (e.g., 'tool', 'session', 'global')

        Returns:
            Cached value or None if not found or expired

        Example:
            value = await cache.get("result_123", "tool")
        """
        ...

    async def set(
        self,
        key: str,
        value: Any,
        namespace: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be pickle-able)
            namespace: Namespace for isolation
            ttl_seconds: Time-to-live in seconds (None = use backend default)

        Example:
            await cache.set("result_123", tool_result, "tool", ttl_seconds=300)
        """
        ...

    async def delete(self, key: str, namespace: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key to delete
            namespace: Namespace of the key

        Returns:
            True if key was deleted, False if not found

        Example:
            deleted = await cache.delete("result_123", "tool")
        """
        ...

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of keys deleted

        Example:
            count = await cache.clear_namespace("tool")  # Clear all tool caches
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with backend-specific statistics

        Common keys:
            - backend_type: Type of backend ('memory', 'redis', 'sqlite')
            - keys: Number of cached items
            - memory_used: Memory usage in bytes (if tracked)
            - hit_rate: Cache hit rate (if tracked)
            - eviction_count: Number of entries evicted

        Example:
            stats = await cache.get_stats()
            # {'backend_type': 'redis', 'keys': 1234, 'memory_used': 4567890}
        """
        ...


# =============================================================================
# Cache Invalidation Protocol
# =============================================================================


@runtime_checkable
class ICacheInvalidator(Protocol):
    """Protocol for advanced cache invalidation strategies.

    This protocol extends basic cache operations with intelligent
    invalidation based on dependencies, namespaces, and tool relationships.

    Use Cases:
        - Invalidate a tool and all its dependents (cascading)
        - Invalidate caches when files are modified
        - Namespace-aware hierarchical invalidation
        - Tag-based invalidation for custom strategies

    Implementation Guidelines:
        - Track tool-to-tool dependencies for cascading
        - Track tool-to-file dependencies for file-based invalidation
        - Respect namespace hierarchy (invalidate children when parent is cleared)
        - Support efficient bulk operations (batch invalidation)

    Example Implementation:
        class ToolCacheInvalidator(ICacheInvalidator):
            async def invalidate_tool(self, tool_name: str, cascade: bool) -> InvalidationResult:
                # Invalidate tool's cache entries
                count = await self._clear_tool_entries(tool_name)

                # Cascade to dependents if requested
                if cascade:
                    dependents = self._dependency_graph.get_transitive_dependents(tool_name)
                    for dep in dependents:
                        count += await self._clear_tool_entries(dep)

                return InvalidationResult(
                    success=True,
                    entries_invalidated=count,
                    tools_invalidated=len(dependents) + 1,
                    namespaces_invalidated=[],
                    cascade_depth=len(dependents)
                )
    """

    async def invalidate_tool(
        self,
        tool_name: str,
        cascade: bool = True,
        namespace: Optional[CacheNamespace] = None,
    ) -> InvalidationResult:
        """Invalidate cached results for a tool.

        Args:
            tool_name: Name of the tool to invalidate
            cascade: If True, also invalidate dependent tools (transitive)
            namespace: Specific namespace to clear (None = all namespaces)

        Returns:
            InvalidationResult with details about what was invalidated

        Example:
            result = await invalidator.invalidate_tool("code_search", cascade=True)
            print(f"Invalidated {result.entries_invalidated} entries")
            print(f"Affected {result.tools_invalidated} tools")
        """
        ...

    async def invalidate_file_dependencies(
        self,
        file_path: str,
        cascade: bool = True,
    ) -> InvalidationResult:
        """Invalidate all tools that depend on a file.

        When a file is modified, this method invalidates all cached
        results from tools that depend on that file.

        Args:
            file_path: Absolute path to the modified file
            cascade: If True, also invalidate dependents of affected tools

        Returns:
            InvalidationResult with details about what was invalidated

        Example:
            result = await invalidator.invalidate_file_dependencies("/src/main.py")
            print(f"Invalidated {result.tools_invalidated} tools due to file change")
        """
        ...

    async def invalidate_namespace(
        self,
        namespace: CacheNamespace,
        recursive: bool = True,
    ) -> InvalidationResult:
        """Invalidate all entries in a namespace.

        Args:
            namespace: Namespace to clear
            recursive: If True, also clear child namespaces

        Returns:
            InvalidationResult with details about what was invalidated

        Example:
            # Clear session and all its children (request, tool)
            result = await invalidator.invalidate_namespace(
                CacheNamespace.SESSION,
                recursive=True
            )
        """
        ...

    async def invalidate_by_tags(
        self,
        tags: set[str],
        operator: str = "OR",  # "OR" or "AND"
    ) -> InvalidationResult:
        """Invalidate cache entries matching tags.

        Args:
            tags: Set of tags to match
            operator: "OR" = match any tag, "AND" = match all tags

        Returns:
            InvalidationResult with details about what was invalidated

        Example:
            # Invalidate all entries tagged with "temp" or "experimental"
            result = await invalidator.invalidate_by_tags({"temp", "experimental"}, operator="OR")
        """
        ...


# =============================================================================
# Tool Idempotency Protocol
# =============================================================================


@runtime_checkable
class IIdempotentTool(Protocol):
    """Protocol for tools that support idempotency-aware caching.

    Idempotent tools can be safely cached and retried without
    side effects. Implementations provide idempotency keys
    to identify duplicate calls.

    Benefits of Implementing This Protocol:
        - Automatic result caching for idempotent tools
        - Duplicate call detection (avoid redundant work)
        - Safe retry on failures without side effects
        - Improved performance for expensive read operations

    Idempotent Tools (safe to cache):
        - read_file: Reading the same file returns the same result
        - grep: Searching for the same pattern returns the same result
        - code_search: Semantic search with the same query returns same results
        - ast_parse: Parsing the same file returns the same AST

    Non-Idempotent Tools (should not use result caching):
        - write_file: Each call modifies state
        - shell_execute: Each execution may have different results
        - git_push: Each push changes remote state
        - tool_execute: Dynamic tool calls have varying results

    Implementation Guidelines:
        - is_idempotent() should be fast (cached property, not computation)
        - get_idempotency_key() must be deterministic (same args = same key)
        - Include all relevant arguments in the key (exclude irrelevant ones)
        - Handle argument normalization (e.g., resolve relative paths)

    Example Implementation:
        class ReadFileTool(BaseTool, IIdempotentTool):
            def is_idempotent(self) -> bool:
                return True  # File reading is idempotent

            def get_idempotency_key(self, **kwargs) -> str:
                # Include file path and relevant options
                path = Path(kwargs["path"]).resolve()  # Normalize path
                encoding = kwargs.get("encoding", "utf-8")
                return f"read_file:{path}:{encoding}"
    """

    def is_idempotent(self) -> bool:
        """Check if tool is idempotent.

        Returns:
            True if tool is idempotent, False otherwise

        Example:
            def is_idempotent(self) -> bool:
                # File reading is idempotent
                return True
        """
        ...

    def get_idempotency_key(self, **kwargs: Any) -> str:
        """Generate key for idempotency tracking.

        The key uniquely identifies a tool call such that
        calls with the same key should return the same result.

        Guidelines:
            - Include all arguments that affect the result
            - Normalize arguments (e.g., resolve paths, sort lists)
            - Use a stable format (same args = same key)
            - Keep keys reasonably short (avoid including large data)

        Args:
            **kwargs: Tool arguments

        Returns:
            String key for idempotency tracking

        Examples:
            # For read_file, key is based on file path and encoding
            def get_idempotency_key(self, **kwargs) -> str:
                path = Path(kwargs["path"]).resolve()
                encoding = kwargs.get("encoding", "utf-8")
                return f"read_file:{path}:{encoding}"

            # For grep, key is based on pattern and path
            def get_idempotency_key(self, **kwargs) -> str:
                pattern = kwargs["pattern"]
                path = Path(kwargs["path"]).resolve()
                case_sensitive = kwargs.get("case_sensitive", True)
                return f"grep:{path}:{pattern}:{case_sensitive}"
        """
        ...


# =============================================================================
# Dependency Tracking Protocol
# =============================================================================


@runtime_checkable
class ICacheDependencyTracker(Protocol):
    """Protocol for tracking dependencies for intelligent cache invalidation.

    This protocol enables automatic cache invalidation based on dependency
    tracking. When a dependency changes, all dependent cache entries are
    automatically invalidated.

    Dependency Types:
        1. Tool-to-Tool: Tool A depends on Tool B's results
           Example: code_search depends on read_file
        2. Tool-to-File: Tool A depends on a file's contents
           Example: ast_parse depends on .py file
        3. Tool-to-Resource: Tool A depends on external resource
           Example: database_query depends on table schema

    Implementation Guidelines:
        - Use a directed acyclic graph (DAG) to track dependencies
        - Detect and handle cycles (error or graceful handling)
        - Support transitive dependency queries
        - Efficient lookup for file-based invalidation
        - Thread-safe for concurrent updates

    Example Implementation:
        class DependencyTracker(ICacheDependencyTracker):
            def __init__(self):
                self._tool_deps: Dict[str, Set[str]] = {}  # tool -> tools it depends on
                self._file_deps: Dict[str, Set[str]] = {}  # tool -> files it depends on

            async def add_tool_dependency(self, tool: str, depends_on: str) -> None:
                if tool not in self._tool_deps:
                    self._tool_deps[tool] = set()
                self._tool_deps[tool].add(depends_on)

            async def get_dependent_tools(self, tool: str) -> Set[str]:
                # Get all tools that depend on this tool (direct and transitive)
                dependents = set()
                # Implementation would traverse the dependency graph
                return dependents
    """

    async def add_tool_dependency(self, tool: str, depends_on: str) -> None:
        """Register a tool-to-tool dependency.

        When `depends_on` is invalidated, `tool` should also be invalidated.

        Args:
            tool: The tool that has a dependency
            depends_on: The tool that `tool` depends on

        Example:
            await tracker.add_tool_dependency("code_search", "read_file")
            # When read_file is invalidated, code_search should also be invalidated
        """
        ...

    async def add_file_dependency(self, tool: str, file_path: str) -> None:
        """Register a tool-to-file dependency.

        When `file_path` is modified, `tool` should be invalidated.

        Args:
            tool: The tool that depends on the file
            file_path: Absolute path to the file

        Example:
            await tracker.add_file_dependency("ast_parse", "/src/main.py")
            # When /src/main.py is modified, ast_parse should be invalidated
        """
        ...

    async def get_dependent_tools(self, tool: str) -> set[str]:
        """Get all tools that depend on the given tool.

        Returns both direct and transitive dependents.

        Args:
            tool: The tool to query

        Returns:
            Set of tool names that depend on this tool

        Example:
            await tracker.add_tool_dependency("code_search", "read_file")
            await tracker.add_tool_dependency("ast_analysis", "read_file")

            dependents = await tracker.get_dependent_tools("read_file")
            # => {"code_search", "ast_analysis"}
        """
        ...

    async def get_file_dependents(self, file_path: str) -> set[str]:
        """Get all tools that depend on the given file.

        Args:
            file_path: Absolute path to the file

        Returns:
            Set of tool names that depend on this file

        Example:
            await tracker.add_file_dependency("code_search", "/src/main.py")
            await tracker.add_file_dependency("linter", "/src/main.py")

            dependents = await tracker.get_file_dependents("/src/main.py")
            # => {"code_search", "linter"}
        """
        ...

    async def remove_tool(self, tool: str) -> None:
        """Remove a tool from the dependency graph.

        This removes all dependencies and dependents related to the tool.

        Args:
            tool: The tool to remove

        Example:
            await tracker.remove_tool("deprecated_tool")
        """
        ...

    async def get_stats(self) -> dict[str, int]:
        """Get statistics about the dependency graph.

        Returns:
            Dictionary with graph statistics

        Example:
            stats = await tracker.get_stats()
            # {
            #     "total_tools": 10,
            #     "total_files": 25,
            #     "tool_dependencies": 15,
            #     "file_dependencies": 30
            # }
        """
        ...


# =============================================================================
# Combined Protocols
# =============================================================================


@runtime_checkable
class IAdvancedCacheBackend(ICacheBackend, ICacheInvalidator, Protocol):
    """Combined protocol for full-featured cache backends.

    This protocol combines basic cache operations with advanced
    invalidation strategies. Use this when you need both capabilities
    in a single implementation.

    Example:
        class RedisAdvancedCache(RedisCacheBackend, IAdvancedCacheBackend):
            # Implement both ICacheBackend and ICacheInvalidator
            async def invalidate_tool(self, tool_name: str, cascade: bool) -> InvalidationResult:
                # Implementation using Redis SCAN for bulk deletion
                ...
    """

    pass


@runtime_checkable
class ICacheManager(ICacheBackend, ICacheInvalidator, ICacheDependencyTracker, Protocol):
    """Unified cache management protocol.

    This is the full-featured protocol that production cache managers
    should implement. It combines:
        - ICacheBackend: Basic cache operations
        - ICacheInvalidator: Advanced invalidation strategies
        - ICacheDependencyTracker: Dependency tracking

    Implementations provide a complete caching solution with:
        - Pluggable storage backends
        - Namespace-aware hierarchical caching
        - Dependency-based automatic invalidation
        - Tool idempotency support
        - Comprehensive statistics and monitoring

    Example:
        class ToolCacheManager(ICacheManager):
            def __init__(self, backend: ICacheBackend):
                self._backend = backend
                self._dependency_tracker = DependencyTracker()
                # Implement all three protocols
    """

    pass


# =============================================================================
# Dict-like Cache Namespace Protocol
# =============================================================================


@runtime_checkable
class ICacheNamespace(Protocol):
    """Protocol for dict-like cache namespace.

    This protocol breaks the circular dependency between:
    - victor/tools/context.py (needs CacheNamespace for type hints)
    - victor/tools/cache_manager.py (implements CacheNamespace)

    By using this protocol, context.py can reference the cache interface
    without importing the concrete CacheNamespace implementation.

    The protocol provides a dict-like interface with additional features
    like TTL support, statistics, and thread safety.

    Example:
        from victor.protocols.cache import ICacheNamespace

        def use_cache(cache: ICacheNamespace) -> None:
            cache["key"] = "value"
            value = cache.get("key")
            if "key" in cache:
                del cache["key"]

    Implementation:
        class CacheNamespace(ICacheNamespace):
            def __init__(self, name: str):
                self.name = name
                self._data: Dict[str, Any] = {}

            def get(self, key: str, default: Any = None) -> Any:
                return self._data.get(key, default)

            def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
                self._data[key] = value

            # ... implement other methods
    """

    name: str
    """Namespace identifier."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a cached value.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        ...

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (None for no expiry)
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete a cached value.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        ...

    def clear(self) -> int:
        """Clear all entries in the namespace.

        Returns:
            Number of entries cleared
        """
        ...

    def keys(self) -> Iterator[str]:
        """Iterate over cache keys.

        Yields:
            Cache keys (excluding expired entries)
        """
        ...

    def __getitem__(self, key: str) -> Any:
        """Get a cached value (dict-like interface)."""
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a cached value (dict-like interface)."""
        ...

    def __delitem__(self, key: str) -> None:
        """Delete a cached value (dict-like interface)."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...

    def __len__(self) -> int:
        """Get number of entries in cache."""
        ...


__all__ = [
    # File change detection
    "FileChangeType",
    "FileChangeEvent",
    "IFileWatcher",
    "IDependencyExtractor",
    # Namespace types
    "CacheNamespace",
    # Result types
    "CacheEntryMetadata",
    "InvalidationResult",
    "CacheStatistics",
    # Core protocols
    "ICacheBackend",
    "ICacheInvalidator",
    "IIdempotentTool",
    "ICacheDependencyTracker",
    # Combined protocols
    "IAdvancedCacheBackend",
    "ICacheManager",
    # Dict-like namespace protocol
    "ICacheNamespace",
]
