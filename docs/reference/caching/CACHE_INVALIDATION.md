# Cache Invalidation Guide

## Overview

Cache invalidation is the process of removing stale or outdated entries from the cache. Victor AI's intelligent invalidation system provides multiple strategies to keep caches fresh while maximizing hit rates.

## Invalidation Strategies

### 1. Time-Based (TTL)

Entries expire after a fixed time period.

**Best for**: Data with known refresh intervals

```python
from victor.core.cache import CacheInvalidator, InvalidationStrategy

invalidator = CacheInvalidator(
    cache=cache,
    strategy=InvalidationStrategy.TTL,
    default_ttl=3600,  # 1 hour
)

# Entries auto-expire after TTL
```

### 2. Event-Based

Invalidate on specific events (file changes, config updates, etc.).

**Best for**: Data that changes unpredictably

```python
invalidator = CacheInvalidator(
    cache=cache,
    strategy=InvalidationStrategy.EVENT_BASED,
)

# Register event handler
async def on_file_change(file_path: str):
    await invalidator.invalidate_dependents(file_path)

# Trigger on event
await on_file_change("/src/main.py")
```

### 3. Manual

Explicit invalidation via API calls.

**Best for**: User-initiated or scheduled invalidation

```python
invalidator = CacheInvalidator(
    cache=cache,
    strategy=InvalidationStrategy.MANUAL,
)

# Manual invalidation
await invalidator.invalidate("key", namespace="tool")
```

### 4. Predictive

Proactively refresh hot entries before expiration.

**Best for**: High-traffic entries

```python
invalidator = CacheInvalidator(
    cache=cache,
    strategy=InvalidationStrategy.PREDICTIVE,
    predictive_refresh_threshold=10,  # Refresh after 10 accesses
)
```

### 5. Hybrid (Recommended)

Combines multiple strategies for optimal performance.

```python
invalidator = CacheInvalidator(
    cache=cache,
    strategy=InvalidationStrategy.HYBRID,
    default_ttl=3600,
    enable_tagging=True,
)
```

## Tag-Based Invalidation

Group cache entries by tags for bulk invalidation.

### Creating Tags

```python
# Tag entries when storing
await cache.set("result_123", value, namespace="tool")
invalidator.tag("result_123", "tool", ["python_files", "src"])

# Multiple tags
invalidator.tag("result_456", "tool", ["python_files", "tests", "unit"])
```

### Invalidating by Tag

```python
# Invalidate all Python file caches
count = await invalidator.invalidate_tag("python_files")

# Invalidate all test results
count = await invalidator.invalidate_tag("tests")
```

### Tag Hierarchies

```python
# Hierarchical tagging
invalidator.tag("key1", "tool", ["code", "python", "src"])
invalidator.tag("key2", "tool", ["code", "python", "tests"])
invalidator.tag("key3", "tool", ["code", "javascript", "src"])

# Invalidate by level
await invalidator.invalidate_tag("code")  # Invalidates all three
await invalidator.invalidate_tag("python")  # Invalidates key1, key2
await invalidator.invalidate_tag("src")  # Invalidates key1, key3
```

## Dependency-Based Invalidation

Cascade invalidation through dependency graphs.

### Setting Up Dependencies

```python
# Add dependency relationships
invalidator.add_dependency("analysis_result", "tool", "/src/main.py")
invalidator.add_dependency("test_result", "tool", "/tests/test_main.py")

# One key can depend on multiple resources
invalidator.add_dependency("full_report", "tool", "/src/main.py")
invalidator.add_dependency("full_report", "tool", "/src/utils.py")
```

### Cascading Invalidation

```python
# When file changes, invalidate dependent entries
async def on_file_modified(file_path: str):
    count = await invalidator.invalidate_dependents(file_path)
    logger.info(f"Invalidated {count} entries due to {file_path} change")

# On file edit
await on_file_modified("/src/main.py")
# Invalidates: analysis_result, full_report
```

### Transitive Dependencies

```python
# A depends on B, B depends on C
invalidator.add_dependency("result_a", "tool", "result_b")
invalidator.add_dependency("result_b", "tool", "result_c")

# When result_c changes
await invalidator.invalidate_dependents("result_c")
# Invalidates: result_b (direct), result_a (transitive)
```

## Event-Driven Invalidation

React to system events for automatic invalidation.

### File System Events

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CacheInvalidationHandler(FileSystemEventHandler):
    def __init__(self, invalidator: CacheInvalidator):
        self.invalidator = invalidator

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            asyncio.create_task(
                self.invalidator.invalidate_dependents(event.src_path)
            )

# Setup file watcher
handler = CacheInvalidationHandler(invalidator)
observer = Observer()
observer.schedule(handler, path='.', recursive=True)
observer.start()
```

### Config Changes

```python
async def watch_config_changes():
    while True:
        # Check for config changes
        if config_has_changed():
            await invalidator.invalidate_tag("config")

        await asyncio.sleep(10)
```

### Custom Events

```python
# Register event handler
async def on_user_logout(user_id: str):
    await invalidator.invalidate_namespace(f"user_{user_id}")

invalidator.register_event_handler("user_logout", on_user_logout)

# Trigger event
await invalidator.trigger_event("user_logout", user_id="user123")
```

## Manual Invalidation API

### Single Entry Invalidation

```python
# Invalidate specific entry
deleted = await invalidator.invalidate("result_123", namespace="tool")

if deleted:
    logger.info("Entry invalidated successfully")
else:
    logger.info("Entry not found in cache")
```

### Namespace Invalidation

```python
# Clear all entries in a namespace
count = await invalidator.invalidate_namespace("tool")

logger.info(f"Invalidated {count} entries in 'tool' namespace")
```

### Full Cache Clear

```python
# Clear all cache entries
await invalidator.invalidate_all()

logger.info("All cache entries invalidated")
```

## Predictive Refresh

Refresh hot entries before they expire.

### How It Works

```python
invalidator = CacheInvalidator(
    cache=cache,
    strategy=InvalidationStrategy.PREDICTIVE,
    predictive_refresh_threshold=10,  # Refresh after 10 accesses
)

# Entry accessed 10+ times → Refresh before expiration
# Reduces cache misses for hot data
```

### Integration with Analytics

```python
from victor.core.cache import CacheAnalytics

analytics = CacheAnalytics(cache=cache)

# Get hot keys
hot_keys = analytics.get_hot_keys(top_n=100)

# Proactively refresh hot entries
for hot_key in hot_keys:
    if hot_key.access_count >= invalidator.config.predictive_refresh_threshold:
        # Refresh entry
        new_value = await recompute_value(hot_key.key)
        await cache.set(hot_key.key, new_value, namespace=hot_key.namespace)
```

## Best Practices

### 1. Use Appropriate TTLs

```python
# Short TTL for frequently changing data
await cache.set("result", value, ttl=60)  # 1 minute

# Long TTL for stable data
await cache.set("config", value, ttl=86400)  # 1 day
```

### 2. Tag Strategically

```python
# Good: Specific tags
invalidator.tag("key1", "tool", ["python", "src", "api"])

# Avoid: Too generic
invalidator.tag("key1", "tool", ["data"])  # Not specific enough

# Avoid: Too specific
invalidator.tag("key1", "tool", ["python", "src", "api", "v1", "endpoint"])  # Too granular
```

### 3. Set Up Dependencies

```python
# Always add dependencies for file-based data
invalidator.add_dependency("analysis", "tool", "/src/main.py")

# Use file paths as resource identifiers
invalidator.add_dependency("result", "tool", "config:/app/config.yaml")
```

### 4. Monitor Invalidation Rate

```python
stats = invalidator.get_stats()

if stats['invalidations'] > 1000:
    logger.warning("High invalidation rate, consider TTL adjustments")

if stats['tag_invalidations'] > stats['invalidations'] * 0.5:
    logger.info("Tag-based invalidation working well")
```

### 5. Graceful Degradation

```python
try:
    await invalidator.invalidate(key, namespace)
except Exception as e:
    logger.error(f"Invalidation failed: {e}")
    # Continue with stale data rather than failing
```

## Invalidation Patterns

### Write-Through Invalidation

```python
async def write_and_invalidate(key: str, value: Any, namespace: str):
    # Write to source
    await write_to_source(key, value)

    # Invalidate cache
    await invalidator.invalidate(key, namespace)

    # Update cache with new value
    await cache.set(key, value, namespace)
```

### Read-Through Invalidation

```python
async def get_with_refresh(key: str, namespace: str):
    # Try cache
    value = await cache.get(key, namespace)

    if value is None:
        # Cache miss, fetch from source
        value = await fetch_from_source(key)

        # Check if should be cached
        if is_cacheable(value):
            await cache.set(key, value, namespace)

    return value
```

### Scheduled Invalidation

```python
async def scheduled_invalidation():
    while True:
        # Invalidate expired entries every hour
        await asyncio.sleep(3600)

        # Clear specific namespace
        await invalidator.invalidate_namespace("temp")

        # Invalidate by tag
        await invalidator.invalidate_tag("hourly_data")
```

## Monitoring and Analytics

### Invalidation Statistics

```python
stats = invalidator.get_stats()

print(f"Strategy: {stats['strategy']}")
print(f"Total Invalidation: {stats['invalidations']}")
print(f"Tag Invalidation: {stats['tag_invalidations']}")
print(f"Dependency Invalidation: {stats['dependency_invalidations']}")

if stats.get('tags'):
    tags = stats['tags']
    print(f"Total Tags: {tags['total_tags']}")
    print(f"Tagged Entries: {tags['tagged_entries']}")

if stats.get('dependencies'):
    deps = stats['dependencies']
    print(f"Resources: {deps['resources']}")
    print(f"Dependent Entries: {deps['dependent_entries']}")
```

### Invalidation Rate Analysis

```python
# Track invalidation rate over time
invalidations_per_hour = stats['invalidations'] / hours_running

if invalidations_per_hour > 100:
    logger.warning("High invalidation rate: "
                   f"{invalidations_per_hour:.1f}/hour")
```

## Integration Examples

### With Multi-Level Cache

```python
from victor.core.cache import MultiLevelCache, CacheInvalidator

cache = MultiLevelCache(...)
invalidator = CacheInvalidator(cache=cache)

# Invalidation removes from both L1 and L2
await invalidator.invalidate("key", namespace="tool")
```

### With File Watcher

```python
from watchdog.observers import Observer

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, invalidator):
        self.invalidator = invalidator

    def on_modified(self, event):
        # Invalidate dependent cache entries on file change
        asyncio.create_task(
            self.invalidator.invalidate_dependents(event.src_path)
        )

observer = Observer()
observer.schedule(FileChangeHandler(invalidator), path=".", recursive=True)
observer.start()
```

### With Cache Warming

```python
# Re-warm entries after invalidation
async def invalidate_and_rewarm(key: str, namespace: str):
    await invalidator.invalidate(key, namespace)

    # Re-warm if important
    if is_important_key(key):
        new_value = await recompute_value(key)
        await cache.set(key, new_value, namespace)
```

## Troubleshooting

### Stale Data in Cache

**Symptoms**: Returning outdated data

**Possible Causes**:
1. TTL too long
2. Invalidation not triggered
3. Missing dependency setup

**Solutions**:
1. Reduce TTL
2. Add event-driven invalidation
3. Set up dependency tracking

### Excessive Invalidation

**Symptoms**: High invalidation rate, low hit rate

**Possible Causes**:
1. TTL too short
2. Over-aggressive event handlers
3. Incorrect tag usage

**Solutions**:
1. Increase TTL
2. Debounce event handlers
3. Use more specific tags

### Memory Leak

**Symptoms**: Memory usage growing

**Possible Causes**:
1. Tags not cleaned up
2. Dependencies not removed
3. Invalidation not running

**Solutions**:
1. Periodic cleanup
2. Remove dependencies on entry deletion
3. Monitor invalidation task

## See Also

- [Multi-Level Cache](MULTI_LEVEL_CACHE.md)
- [Cache Warming](CACHE_WARMING.md)
- [Semantic Caching](SEMANTIC_CACHING.md)
- [Cache Analytics](CACHE_ANALYTICS.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
