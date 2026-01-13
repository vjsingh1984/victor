# Thread Safety Stress Tests

## Quick Reference

**File:** `test_thread_safety_stress.py`  
**Tests:** 28 test methods in 7 classes  
**Lines:** 1,249

## Test Classes Overview

| Class | Tests | Focus |
|-------|-------|-------|
| `TestEventBackendConcurrencyStress` | 3 | Event backend subscribe/unsubscribe |
| `TestEmbeddingCacheConcurrencyStress` | 3 | Embedding cache read/write/evict |
| `TestToolPipelineMetricsConcurrency` | 3 | Metrics recording integrity |
| `TestServiceContainerConcurrency` | 3 | Service registration/resolution |
| `TestDeadlockDetection` | 3 | Deadlock detection under load |
| `TestDataStructureCorruption` | 3 | Data structure integrity |
| `TestHighConcurrencyScenarios` | 3 | 100+ concurrent operations |

## Running Tests

```bash
# All tests
pytest tests/unit/core/threading/test_thread_safety_stress.py -v

# Specific class
pytest tests/unit/core/threading/test_thread_safety_stress.py::TestEventBackendConcurrencyStress -v

# With coverage
pytest tests/unit/core/threading/test_thread_safety_stress.py --cov=victor.core.events.backends
```

## Key Test Scenarios

### 1. Concurrent Subscribe/Unsubscribe
- **Component:** `InMemoryEventBackend`
- **Tests:** Race conditions, lost subscriptions, orphaned subscriptions
- **Concurrency:** 10-100 threads, 20-100 operations

### 2. Embedding Cache Access
- **Component:** `LRUToolCache`
- **Tests:** Read/write contention, eviction, clear
- **Concurrency:** 10-100 threads, 50-200 cache size

### 3. Metrics Recording
- **Component:** `ExecutionMetrics`
- **Tests:** Counter updates, time tracking, reset
- **Concurrency:** 10-100 threads, 50-200 records

### 4. Service Container
- **Component:** `ServiceContainer`
- **Tests:** Registration, singleton/transient resolution
- **Concurrency:** 10-100 threads

## Invariants Verified

✓ **No Lost Updates** - All operations accounted for  
✓ **Consistent State** - Totals match sums  
✓ **No Corruption** - Data structures valid  
✓ **No Deadlocks** - Complete within timeout  
✓ **Atomicity** - Singletons remain single

## TDD Approach

1. Tests written **BEFORE** implementation
2. Tests document expected behavior
3. Tests **WILL FAIL** initially
4. Fix thread safety issues
5. Verify tests pass

## Expected Failures (Pre-Implementation)

- Race conditions in dict access
- Non-atomic counter updates
- OrderedDict corruption
- Multiple singleton instances
- Dictionary changed size during iteration
