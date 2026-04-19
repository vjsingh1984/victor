# Indexed Registration Architecture Design

## Overview

This document describes the indexed registration architecture for achieving O(1) lookup performance in ToolRegistry, replacing linear list searches with hash-based composite indexes.

## Problem Statement

Current ToolRegistry uses linear list searches for queries:
- `get_by_tag(tag)` - O(n) scan through all tools
- `get_by_role(role)` - O(n) scan through all tools
- `discover_by_expertise(expertise)` - O(n) scan with filter

For 1000 tools, these queries become expensive, especially when called repeatedly during tool selection.

## Solution: Composite Indexes

### Design Pattern
- **Primary Pattern**: Registry Pattern + Index Pattern (from database design)
- **Secondary Pattern**: Composite Key Pattern (for multi-criteria queries)
- **Invalidation Pattern**: Observer Pattern (automatic index updates)

### Index Structure

```python
class ToolIndexes:
    """Composite indexes for O(1) tool lookups."""

    def __init__(self):
        # Primary indexes: single attribute lookups
        self._by_name: Dict[str, ToolID] = {}
        self._by_tag: Dict[str, Set[ToolID]] = defaultdict(set)
        self._by_category: Dict[ExecutionCategory, Set[ToolID]] = defaultdict(set)
        self._by_role: Dict[str, Set[ToolID]] = defaultdict(set)

        # Composite indexes: multi-attribute queries
        self._by_tag_and_category: Dict[Tuple[str, ExecutionCategory], Set[ToolID]] = defaultdict(set)
        self._by_role_and_tag: Dict[Tuple[str, str], Set[ToolID]] = defaultdict(set)

        # Reverse index: ToolID -> Tool object
        self._tools: Dict[ToolID, BaseTool] = {}
```

### Lookup Complexity

| Query Type | Current | With Index | Improvement |
|------------|---------|------------|-------------|
| get_by_name | O(n) | O(1) | ∞× faster |
| get_by_tag | O(n) | O(1) | ∞× faster |
| get_by_category | O(n) | O(1) | ∞× faster |
| get_by_role | O(n) | O(1) | ∞× faster |
| by_tag_and_category | O(n²) | O(1) | ∞× faster |

## Index Maintenance

### Registration Flow

```python
def register(self, tool: BaseTool) -> None:
    """Register tool with automatic index updates."""
    tool_id = self._generate_id(tool)

    # 1. Store tool (O(1))
    self._tools[tool_id] = tool

    # 2. Update primary indexes (O(1) each)
    self._by_name[tool.name] = tool_id
    self._by_tag[tool.tag].add(tool_id)
    self._by_category[tool.category].add(tool_id)

    # 3. Update composite indexes (O(1) each)
    for tag in tool.tags:
        self._by_tag_and_category[(tag, tool.category)].add(tool_id)
    for role in tool.roles:
        self._by_role_and_tag[(role, tag)].add(tool_id)

    # 4. Notify observers (cache invalidation)
    self._notify_observers(tool_id)
```

### Unregistration Flow

```python
def unregister(self, tool_name: str) -> None:
    """Unregister tool with automatic index cleanup."""
    tool_id = self._by_name[tool_name]
    tool = self._tools[tool_id]

    # 1. Remove from all indexes (O(k) where k=number of indexes)
    del self._by_name[tool_name]
    for tag in tool.tags:
        self._by_tag[tag].discard(tool_id)
        # Clean up empty tag sets
        if not self._by_tag[tag]:
            del self._by_tag[tag]

    # 2. Remove composite indexes
    for key in list(self._by_tag_and_category.keys()):
        key.discard(tool_id)
        if not key:
            del self._by_tag_and_category[key]

    # 3. Remove tool storage
    del self._tools[tool_id]
```

## Memory Trade-offs

### Memory Overhead Analysis

For N tools with:
- Average tags per tool: 3
- Average roles per tool: 2
- Composite indexes: tag×category, role×tag

**Memory Complexity**: O(N × k) where k = avg indexes per tool

**Estimated Overhead**:
- 100 tools: ~50KB additional memory
- 1000 tools: ~500KB additional memory
- 10000 tools: ~5MB additional memory

**Trade-off Justification**: Memory is cheap; latency is expensive. 5MB for 10000 tools is negligible compared to O(n²) lookup costs.

## Concurrency Model

### Thread Safety

```python
class ThreadSafeToolIndexes:
    """Thread-safe indexes with read-write locks."""

    def __init__(self):
        self._lock = threading.RLock()
        self._indexes = ToolIndexes()

    def register(self, tool: BaseTool) -> None:
        with self._lock:
            self._indexes.register(tool)

    def get_by_tag(self, tag: str) -> List[BaseTool]:
        with self._lock:
            # Reads are exclusive with writes
            return self._indexes.get_by_tag(tag)
```

### Lock-Free Reads (Optimization)

For read-heavy workloads, use Copy-on-Write:

```python
class LockFreeToolIndexes:
    """Lock-free indexes using immutable snapshots."""

    def __init__(self):
        self._state_version = 0
        self._state_lock = threading.Lock()
        self._current_state = ToolIndexes()

    def get_by_tag(self, tag: str) -> List[BaseTool]:
        # Lock-free read
        state = self._current_state
        return state.get_by_tag(tag)

    def register(self, tool: BaseTool) -> None:
        with self._state_lock:
            # Copy-on-write: create new state
            new_state = copy.deepcopy(self._current_state)
            new_state.register(tool)
            self._current_state = new_state
            self._state_version += 1
```

**Trade-off**:
- **Pros**: Readers never block, excellent throughput
- **Cons**: Writes are more expensive (copy overhead)
- **Best for**: Read-heavy workloads (90% reads, 10% writes)

## Invalidation Strategy

### Cache Invalidation Triggers

1. **On Registration**: Invalidate affected query caches
   - Register tool with tag "test" → invalidate "test" cache
   - No need to invalidate unrelated queries

2. **On Unregistration**: Full cache clear (conservative)
   - Tool removed → clear all query caches
   - **Optimization**: Selective invalidation (future work)

3. **On Schema Change**: Full cache clear
   - Tool metadata structure changes → clear all

### Cache Keys

```python
CacheKey = Union[
    str,              # "tag:test"
    tuple,            # ("tag:test", "category:read_only")
    frozenset,        # frozenset({"tag:a", "tag:b"})
]

_query_cache: Dict[CacheKey, List[ToolID]] = {}
```

## Implementation Phases

### Phase 1: Core Indexes (Week 1)
- Implement primary indexes (name, tag, category, role)
- Add index maintenance in registration/unregistration
- Add unit tests for index correctness

### Phase 2: Composite Indexes (Week 2)
- Implement composite indexes (tag+category, role+tag)
- Add multi-criteria query API
- Add performance benchmarks

### Phase 3: Thread Safety (Week 3)
- Add read-write locks for basic thread safety
- Implement lock-free reads for read-heavy workloads
- Add concurrency tests

### Phase 4: Integration (Week 4)
- Integrate with existing ToolRegistry
- Add backward compatibility layer
- Migration guide and deprecation timeline

## API Changes

### New Query API

```python
# Before (O(n) linear scan)
tools = [t for t in registry.list_all() if "test" in t.tags]

# After (O(1) indexed lookup)
tools = registry.get_by_tag("test")
```

### Batch Registration API

```python
# Register multiple tools with single cache invalidation
registry.register_batch([
    tool1, tool2, tool3, ...
])  # O(n) with single cache invalidation at end
```

## Performance Targets

| Operation | Current | Target | Improvement |
|------------|---------|--------|-------------|
| Register 100 tools | 4ms | 2ms | 2× faster |
| Query by tag (1000 tools) | ~10ms | <0.1ms | 100× faster |
| Query by tag+category (1000 tools) | ~20ms | <0.2ms | 100× faster |
| Register batch (100 tools) | N/A | <5ms | New capability |

## Monitoring

### Metrics to Track

1. **Index Size**: Track growth of each index over time
2. **Query Performance**: p50/p95/p99 latencies for each query type
3. **Cache Hit Ratio**: Percentage of queries served from cache
4. **Memory Usage**: Monitor index memory overhead
5. **Lock Contention**: Track thread blocking time

### Alerts

- Cache hit ratio < 80% (indicates poor cache utilization)
- p95 query latency > 1ms (indicates performance degradation)
- Index memory > 10MB (indicates cleanup needed)

## References

- **Registry Pattern**: Gang of Four
- **Index Organization**: Database System Concepts (Silberschatz)
- **Copy-on-Write:**
- **Lock-Free Data Structures**: Herlihy & Shavit

## Conclusion

The indexed architecture provides O(1) lookup performance with acceptable memory overhead. The design prioritizes read performance (common case) over write performance (rare case), matching the expected workload pattern of registration once, query many times.

**Next Step**: Implement Task #20 (Batch Registration API) to realize the expected performance gains.
