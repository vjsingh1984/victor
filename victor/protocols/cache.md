# Cache Protocols Reference Guide

## Overview

The cache protocols provide a comprehensive, dependency-inverted architecture for tool cache invalidation in Victor. This guide explains the design, usage, and implementation of these protocols.

## Architecture

### Protocol Hierarchy

```
ICacheBackend (Basic storage)
    ↓
ICacheInvalidator (Advanced invalidation)
    ↓
ICacheDependencyTracker (Dependency management)
    ↓
ICacheManager (Full-featured management)
```

### Combined Protocols

- **IAdvancedCacheBackend**: ICacheBackend + ICacheInvalidator
- **ICacheManager**: ICacheBackend + ICacheInvalidator + ICacheDependencyTracker

## Protocols

### 1. ICacheBackend

**Purpose**: Basic cache storage with namespace isolation

**Use When**: You need simple cache operations without advanced features

**Methods**:
```python
async def get(key: str, namespace: str) -> Optional[Any]:
    """Get value from cache."""

async def set(key: str, value: Any, namespace: str, ttl_seconds: Optional[int] = None) -> None:
    """Set value in cache."""

async def delete(key: str, namespace: str) -> bool:
    """Delete key from cache."""

async def clear_namespace(namespace: str) -> int:
    """Clear all keys in a namespace."""

async def get_stats() -> Dict[str, Any]:
    """Get cache statistics."""
```

**Implementation Example**:
```python
class MemoryCacheBackend(ICacheBackend):
    def __init__(self):
        self._storage: Dict[str, Any] = {}

    async def get(self, key: str, namespace: str) -> Optional[Any]:
        return self._storage.get(f"{namespace}:{key}")

    async def set(self, key: str, value: Any, namespace: str, ttl_seconds: Optional[int] = None) -> None:
        self._storage[f"{namespace}:{key}"] = value

    async def delete(self, key: str, namespace: str) -> bool:
        full_key = f"{namespace}:{key}"
        if full_key in self._storage:
            del self._storage[full_key]
            return True
        return False

    async def clear_namespace(self, namespace: str) -> int:
        count = 0
        for key in list(self._storage.keys()):
            if key.startswith(f"{namespace}:"):
                del self._storage[key]
                count += 1
        return count

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "backend_type": "memory",
            "keys": len(self._storage),
        }
```

### 2. ICacheInvalidator

**Purpose**: Advanced cache invalidation strategies

**Use When**: You need intelligent invalidation based on dependencies, files, or tags

**Methods**:
```python
async def invalidate_tool(
    tool_name: str,
    cascade: bool = True,
    namespace: Optional[CacheNamespace] = None,
) -> InvalidationResult:
    """Invalidate cached results for a tool."""

async def invalidate_file_dependencies(
    file_path: str,
    cascade: bool = True,
) -> InvalidationResult:
    """Invalidate all tools that depend on a file."""

async def invalidate_namespace(
    namespace: CacheNamespace,
    recursive: bool = True,
) -> InvalidationResult:
    """Invalidate all entries in a namespace."""

async def invalidate_by_tags(
    tags: Set[str],
    operator: str = "OR",
) -> InvalidationResult:
    """Invalidate cache entries matching tags."""
```

**Implementation Example**:
```python
class ToolCacheInvalidator(ICacheInvalidator):
    def __init__(self, backend: ICacheBackend, tracker: ICacheDependencyTracker):
        self._backend = backend
        self._tracker = tracker

    async def invalidate_tool(
        self,
        tool_name: str,
        cascade: bool = True,
        namespace: Optional[CacheNamespace] = None,
    ) -> InvalidationResult:
        """Invalidate tool and its dependents."""
        invalidated = 0
        tools_affected = {tool_name}

        # Get dependents if cascading
        if cascade:
            dependents = await self._tracker.get_dependent_tools(tool_name)
            tools_affected.update(dependents)

        # Invalidate each tool
        for tool in tools_affected:
            # Implementation would delete cache entries
            invalidated += 1

        return InvalidationResult(
            success=True,
            entries_invalidated=invalidated,
            tools_invalidated=len(tools_affected),
            namespaces_invalidated=[namespace.value] if namespace else [],
            cascade_depth=len(tools_affected) - 1,
        )
```

### 3. ICacheDependencyTracker

**Purpose**: Track dependencies for intelligent cache invalidation

**Use When**: You need to track tool-to-tool and tool-to-file dependencies

**Methods**:
```python
async def add_tool_dependency(tool: str, depends_on: str) -> None:
    """Register a tool-to-tool dependency."""

async def add_file_dependency(tool: str, file_path: str) -> None:
    """Register a tool-to-file dependency."""

async def get_dependent_tools(tool: str) -> Set[str]:
    """Get all tools that depend on the given tool."""

async def get_file_dependents(file_path: str) -> Set[str]:
    """Get all tools that depend on the given file."""

async def remove_tool(tool: str) -> None:
    """Remove a tool from the dependency graph."""

async def get_stats() -> Dict[str, int]:
    """Get statistics about the dependency graph."""
```

**Implementation Example**:
```python
class DependencyTracker(ICacheDependencyTracker):
    def __init__(self):
        self._tool_deps: Dict[str, Set[str]] = defaultdict(set)
        self._file_deps: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_tool: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_file: Dict[str, Set[str]] = defaultdict(set)

    async def add_tool_dependency(self, tool: str, depends_on: str) -> None:
        self._tool_deps[tool].add(depends_on)
        self._reverse_tool[depends_on].add(tool)

    async def add_file_dependency(self, tool: str, file_path: str) -> None:
        self._file_deps[tool].add(file_path)
        self._reverse_file[file_path].add(tool)

    async def get_dependent_tools(self, tool: str) -> Set[str]:
        return self._reverse_tool.get(tool, set()).copy()

    async def get_file_dependents(self, file_path: str) -> Set[str]:
        return self._reverse_file.get(file_path, set()).copy()

    async def remove_tool(self, tool: str) -> None:
        # Remove all dependencies
        for dep in self._tool_deps.get(tool, set()):
            self._reverse_tool[dep].discard(tool)
        self._tool_deps.pop(tool, None)

        # Remove all dependents
        for dependent in self._reverse_tool.get(tool, set()):
            self._tool_deps[dependent].discard(tool)
        self._reverse_tool.pop(tool, None)

    async def get_stats(self) -> Dict[str, int]:
        return {
            "total_tools": len(self._tool_deps),
            "total_files": len(self._file_deps),
            "tool_dependencies": sum(len(deps) for deps in self._tool_deps.values()),
            "file_dependencies": sum(len(deps) for deps in self._file_deps.values()),
        }
```

### 4. IIdempotentTool

**Purpose**: Mark tools as idempotent for safe caching

**Use When**: Implementing tools that can be safely cached and retried

**Methods**:
```python
def is_idempotent(self) -> bool:
    """Check if tool is idempotent."""

def get_idempotency_key(self, **kwargs) -> str:
    """Generate key for idempotency tracking."""
```

**Implementation Example**:
```python
class ReadFileTool(BaseTool, IIdempotentTool):
    name = "read_file"
    description = "Read the contents of a file"

    def is_idempotent(self) -> bool:
        """File reading is idempotent."""
        return True

    def get_idempotency_key(self, **kwargs) -> str:
        """Generate key based on file path and encoding."""
        path = Path(kwargs["path"]).resolve()  # Normalize path
        encoding = kwargs.get("encoding", "utf-8")
        return f"read_file:{path}:{encoding}"

    async def execute(self, **kwargs):
        # Tool implementation
        ...
```

## Data Structures

### CacheNamespace

Enum defining cache namespace hierarchy:

```python
class CacheNamespace(str, Enum):
    GLOBAL = "global"   # Shared across all sessions
    SESSION = "session"  # Shared within a session
    REQUEST = "request"  # Isolated to a single request
    TOOL = "tool"        # Isolated per tool
```

**Methods**:
- `get_child_namespaces()`: Get child namespaces
- `get_parent_namespace()`: Get parent namespace

**Usage**:
```python
# Cache at session level
await cache.set(result, namespace=CacheNamespace.SESSION)

# Invalidate session and all children
await invalidator.invalidate_namespace(CacheNamespace.SESSION, recursive=True)
```

### InvalidationResult

Result from cache invalidation operations:

```python
@dataclass
class InvalidationResult:
    success: bool                      # Whether invalidation succeeded
    entries_invalidated: int           # Number of cache entries invalidated
    tools_invalidated: int             # Number of tools affected
    namespaces_invalidated: List[str]  # Namespaces that were cleared
    error_message: str | None = None   # Error message if failed
    cascade_depth: int = 0             # Depth of cascading invalidation
    metadata: Dict[str, Any] | None = None  # Additional metadata
```

### CacheEntryMetadata

Metadata for a cache entry:

```python
@dataclass
class CacheEntryMetadata:
    tool_name: str
    args_hash: str
    dependent_tools: Set[str]
    file_dependencies: Set[str]
    created_at: str
    expires_at: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[str] = None
    size_bytes: Optional[int] = None
    namespace: CacheNamespace = CacheNamespace.SESSION
    tags: Set[str] | None = None
```

### CacheStatistics

Statistics about cache performance:

```python
@dataclass
class CacheStatistics:
    backend_type: str
    total_entries: int
    hit_rate: float
    miss_rate: float
    memory_used: Optional[int]
    eviction_count: int
    namespace_stats: Dict[str, Dict[str, Any]]
    tool_stats: Dict[str, Dict[str, Any]]
    dependency_stats: Dict[str, int]
```

## Usage Patterns

### Pattern 1: Simple Caching

```python
# Create backend
backend = MemoryCacheBackend()

# Use cache
await backend.set("key", {"data": "value"}, "session", ttl_seconds=300)
value = await backend.get("key", "session")
```

### Pattern 2: Tool Result Caching

```python
# Create manager with backend
manager = ToolCacheManager(backend=MemoryCacheBackend())

# Cache tool result
await manager.set_tool_result(
    tool_name="code_search",
    args={"query": "authentication"},
    result={"files": ["auth.py"]},
    namespace=CacheNamespace.SESSION,
    ttl_seconds=3600,
)

# Retrieve cached result
result = await manager.get_tool_result(
    tool_name="code_search",
    args={"query": "authentication"},
    namespace=CacheNamespace.SESSION,
)
```

### Pattern 3: Dependency-Aware Caching

```python
# Cache with file dependencies
await manager.set_tool_result(
    tool_name="ast_parse",
    args={"file": "/src/main.py"},
    result=ast_tree,
    namespace=CacheNamespace.SESSION,
    file_dependencies={"/src/main.py"},
)

# When file is modified, invalidate automatically
await manager.invalidate_file_dependencies("/src/main.py")
```

### Pattern 4: Cascading Invalidation

```python
# Add tool dependencies
await tracker.add_tool_dependency("code_search", "read_file")
await tracker.add_tool_dependency("ast_analysis", "read_file")

# Invalidate read_file and all dependents
result = await invalidator.invalidate_tool("read_file", cascade=True)
# This invalidates: read_file, code_search, ast_analysis
```

### Pattern 5: Idempotent Tool Caching

```python
class GrepTool(BaseTool, IIdempotentTool):
    def is_idempotent(self) -> bool:
        return True  # Searching is idempotent

    def get_idempotency_key(self, **kwargs) -> str:
        path = Path(kwargs["path"]).resolve()
        pattern = kwargs["pattern"]
        return f"grep:{path}:{pattern}"

    async def execute(self, **kwargs):
        # Check cache first
        if self.is_idempotent():
            cache_key = self.get_idempotency_key(**kwargs)
            cached = await cache.get(cache_key, "tool")
            if cached:
                return cached

        # Execute tool
        result = await self._grep(**kwargs)

        # Cache result
        if self.is_idempotent():
            await cache.set(cache_key, result, "tool", ttl_seconds=300)

        return result
```

## Best Practices

### 1. Choose the Right Namespace

- **GLOBAL**: Static, expensive computations that rarely change
- **SESSION**: User/session-specific data that changes occasionally
- **REQUEST**: Temporary data for single request
- **TOOL**: Tool-specific state and configuration

### 2. Set Appropriate TTLs

```python
# Short TTL for rapidly changing data
await cache.set(data, namespace=CacheNamespace.REQUEST, ttl_seconds=60)

# Long TTL for static data
await cache.set(data, namespace=CacheNamespace.GLOBAL, ttl_seconds=86400)
```

### 3. Track Dependencies

```python
# Always track file dependencies for file-based tools
await manager.set_tool_result(
    tool_name="ast_parse",
    args={"file": path},
    result=ast,
    file_dependencies={path},  # Critical for invalidation
)
```

### 4. Use Cascading Invalidation

```python
# Enable cascading for transitive dependencies
await invalidator.invalidate_tool("read_file", cascade=True)
```

### 5. Monitor Cache Performance

```python
# Regularly check cache statistics
stats = await manager.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory: {stats['memory_used']} bytes")

# Alert on low hit rates
if stats['hit_rate'] < 0.5:
    logger.warning(f"Low cache hit rate: {stats['hit_rate']:.2%}")
```

## Testing

### Unit Testing

```python
import pytest
from victor.protocols import ICacheBackend

class MockCacheBackend(ICacheBackend):
    """Mock implementation for testing."""

    def __init__(self):
        self._storage = {}

    async def get(self, key: str, namespace: str) -> Optional[Any]:
        return self._storage.get(f"{namespace}:{key}")

    async def set(self, key: str, value: Any, namespace: str, ttl_seconds: Optional[int] = None) -> None:
        self._storage[f"{namespace}:{key}"] = value

    async def delete(self, key: str, namespace: str) -> bool:
        full_key = f"{namespace}:{key}"
        if full_key in self._storage:
            del self._storage[full_key]
            return True
        return False

    async def clear_namespace(self, namespace: str) -> int:
        count = 0
        for key in list(self._storage.keys()):
            if key.startswith(f"{namespace}:"):
                del self._storage[key]
                count += 1
        return count

    async def get_stats(self) -> Dict[str, Any]:
        return {"backend_type": "mock", "keys": len(self._storage)}

@pytest.mark.asyncio
async def test_cache_operations():
    cache = MockCacheBackend()

    # Test set and get
    await cache.set("key", "value", "session")
    assert await cache.get("key", "session") == "value"

    # Test delete
    assert await cache.delete("key", "session") is True
    assert await cache.get("key", "session") is None

    # Test clear namespace
    await cache.set("key1", "value1", "session")
    await cache.set("key2", "value2", "session")
    count = await cache.clear_namespace("session")
    assert count == 2
```

## Migration Guide

### From Simple Caching to Protocol-Based Caching

**Before**:
```python
# Direct dictionary usage
cache = {}
cache["key"] = value
value = cache.get("key")
```

**After**:
```python
# Protocol-based caching
from victor.protocols import ICacheBackend

cache = MemoryCacheBackend()
await cache.set("key", value, "session")
value = await cache.get("key", "session")
```

### Adding Dependency Tracking

**Before**:
```python
# Manual invalidation
async def on_file_change(file_path):
    # Manually clear all caches
    cache.clear()
```

**After**:
```python
# Automatic dependency-based invalidation
async def on_file_change(file_path):
    await invalidator.invalidate_file_dependencies(file_path, cascade=True)
```

## Performance Considerations

### 1. Async Operations

All protocol methods are async for non-blocking operation:

```python
# Good: Parallel cache operations
results = await asyncio.gather(
    cache.get("key1", "session"),
    cache.get("key2", "session"),
    cache.get("key3", "session"),
)

# Bad: Sequential operations
result1 = await cache.get("key1", "session")
result2 = await cache.get("key2", "session")
result3 = await cache.get("key3", "session")
```

### 2. Namespace Isolation

Namespaces prevent cache pollution and enable efficient invalidation:

```python
# Good: Separate namespaces for different data types
await cache.set(user_data, "user")
await cache.set(session_data, "session")
await cache.set(tool_result, "tool")

# Clear only what's needed
await cache.clear_namespace("session")  # Only session data
```

### 3. TTL Management

Set appropriate TTLs to balance performance and freshness:

```python
# Short TTL for frequently changing data
await cache.set(data, namespace="request", ttl_seconds=60)

# Long TTL for static data
await cache.set(data, namespace="global", ttl_seconds=86400)
```

## Troubleshooting

### Issue: Low Cache Hit Rate

**Solution**:
- Check if TTL is too short
- Verify namespace isolation (different data in different namespaces)
- Monitor cache key generation (are keys consistent?)
- Review dependency invalidation (are dependencies too broad?)

### Issue: Memory Usage Growing

**Solution**:
- Reduce TTL values
- Implement eviction policies
- Clear namespaces regularly
- Monitor cache size with `get_stats()`

### Issue: Stale Data

**Solution**:
- Verify file dependency tracking is working
- Check cascading invalidation is enabled
- Review tool dependency graph
- Implement manual invalidation for critical data

## References

- SOLID Principles: https://en.wikipedia.org/wiki/SOLID
- Python Protocols: https://docs.python.org/3/library/typing.html#typing.Protocol
- Victor Architecture: See CLAUDE.md

## Contributing

When adding new cache backends or invalidation strategies:

1. Implement the appropriate protocol
2. Add comprehensive docstrings with examples
3. Write unit tests
4. Update this guide
5. Follow Victor's coding standards

## License

Apache 2.0 - See LICENSE file for details
