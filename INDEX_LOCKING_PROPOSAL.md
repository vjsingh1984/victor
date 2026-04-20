# Proposal: Index-Level Locking for code_search

## Problem

Currently, `code_search` has **NO locking** during index creation:
- Multiple concurrent calls for the same path trigger **multiple indexing runs**
- Causes resource waste (double/triple/quadruple work)
- Risk of LanceDB corruption from concurrent writes
- Each indexing run can take **1-5 minutes** for large codebases

## Example Scenario

```python
# LLM makes 5 parallel tool calls
code_search(path="../", query="A")  # Starts indexing
code_search(path="../", query="B")  # Starts indexing AGAIN (no lock!)
code_search(path="../", query="C")  # Starts indexing AGAIN
code_search(path="../", query="D")  # Starts indexing AGAIN
code_search(path="../", query="E")  # Starts indexing AGAIN

# Result: 5x resource usage, potential corruption
```

## Proposed Solution

### Option 1: Per-Path Async Lock (Recommended)

```python
# In code_search_tool.py

import asyncio
from typing import Dict

# Global registry of path-specific locks
_index_locks: Dict[str, asyncio.Lock] = {}
_lock_registry_lock = asyncio.Lock()

async def _get_index_lock(path: str) -> asyncio.Lock:
    """Get or create a lock for this specific path."""
    if path not in _index_locks:
        async with _lock_registry_lock:
            if path not in _index_locks:
                _index_locks[path] = asyncio.Lock()
    return _index_locks[path]

async def _get_or_build_index(...):
    """Acquire lock before building index."""
    # ... existing cache check code ...

    # NEW: Acquire lock for this path
    path_lock = await _get_index_lock(str(root))
    async with path_lock:
        # Double-check cache inside lock (another task might have built it)
        if str(root) in index_cache:
            return index_cache[str(root)]["index"], False

        # Build index (exclusive access for this path)
        index = await _build_index(...)
        index_cache[str(root)] = {...}
        return index, True
```

**Benefits**:
- ✅ Single indexing run per path (no double work)
- ✅ Concurrent searches allowed (different paths)
- ✅ Concurrent searches allowed (same path, after index built)
- ✅ Thread-safe with asyncio primitives
- ✅ Automatic cleanup possible (remove lock after idle)

**Trade-offs**:
- ⚠️ Small memory overhead (one Lock per active path)
- ⚠️ Sequential indexing for same path (not a problem - desired behavior)

### Option 2: Build-in-Progress Flag

```python
# Track which paths are currently being built
_building_indexes: Dict[str, asyncio.Task] = {}

async def _get_or_build_index(...):
    path_str = str(root)

    # Check if already being built
    if path_str in _building_indexes:
        # Wait for existing build to complete
        await _building_indexes[path_str]
        return index_cache[path_str]["index"], False

    # Create new build task
    build_task = asyncio.create_task(_build_index(...))
    _building_indexes[path_str] = build_task

    try:
        index = await build_task
        index_cache[path_str] = {...}
        return index, True
    finally:
        del _building_indexes[path_str]
```

**Benefits**:
- ✅ No explicit locks (uses task tracking)
- ✅ All waiters get same result
- ✅ Automatic cleanup on completion

**Trade-offs**:
- ⚠️ Slightly more complex (task lifecycle management)
- ⚠️ Need to handle task cancellation

## Recommendation

**Implement Option 1** (Per-Path Async Lock):
- Simpler to understand and maintain
- Standard asyncio pattern for resource protection
- Easy to test (single responsibility)
- Can add metrics later (lock wait time, contention)

## Additional Improvements

1. **Add metrics**:
   ```python
   - lock_wait_time_ms
   - lock_contention_count
   - parallel_indexing_averted_count
   ```

2. **Add timeout**:
   ```python
   async with asyncio.timeout(300):  # 5 minute max
       async with path_lock:
           index = await _build_index(...)
   ```

3. **Add logging**:
   ```python
   logger.info("Acquiring index lock for %s", path)
   logger.info("Index lock acquired in %.2fs", elapsed)
   logger.info("Building index for %s (exclusive access)", path)
   ```

## Impact

**Before** (current state):
- 5 concurrent code_search(path=../) → 5 indexing runs
- Resource usage: 5x
- Time: 5 minutes each (all run in parallel)
- Risk: Database corruption

**After** (with locking):
- 5 concurrent code_search(path=../) → 1 indexing run
- Resource usage: 1x
- Time: 5 minutes total (4 calls wait, 1 builds)
- Result: All 5 get same cached index
- Safety: No corruption risk

## Files to Modify

1. `victor/tools/code_search_tool.py`:
   - Add `_index_locks` dict
   - Add `_get_index_lock()` function
   - Modify `_get_or_build_index()` to use lock

2. `tests/unit/tools/test_code_search_tool.py`:
   - Add test for concurrent indexing
   - Verify only one index built
   - Verify all waiters get same result

## Estimated Effort

- Implementation: 2-3 hours
- Testing: 1-2 hours
- Documentation: 30 minutes
- **Total: 4-6 hours**
