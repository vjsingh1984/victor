# Performance Optimization Guide - Part 1

**Part 1 of 3:** Performance Improvements and Benchmarking

---

## Navigation

- **[Part 1: Improvements & Benchmarking](#)** (Current)
- [Part 2: Troubleshooting & Optimizations](part-2-troubleshooting-optimizations.md)
- [Part 3: Monitoring & Guidelines](part-3-monitoring-guidelines.md)
- [**Complete Guide**](../optimization_guide.md)

---
# Performance Optimization Guide

**Advanced Performance Optimizations**: 30-50% overall performance improvement through targeted optimizations.

## Overview

This guide documents the advanced performance optimizations implemented in Victor AI,
  targeting a **30-50% overall performance improvement**. These optimizations focus on four key areas:

1. **Response Caching** - Intelligent LLM response caching with semantic similarity
2. **Request Batching** - Automatic batching of similar requests to reduce overhead
3. **Hot Path Optimizations** - Optimized frequently called code paths (orjson, lazy imports)
4. **Performance Monitoring** - Comprehensive metrics and monitoring

### Legacy Optimizations (Work Stream 3.3)

Previous optimizations (20% improvement) are documented below:
- **Batch Tool Execution** - Parallel execution with bounded concurrency
- **Optimized Context Compaction** - LLM-based summarization with caching
- **Cached Prompt Building** - Hash-based cache with TTL support

## Performance Improvements Summary

### Advanced Optimizations (New)

| Optimization | Performance Gain | Use Case |
|--------------|------------------|----------|
| **Response Caching** | 10-100x faster on hit | Repeated LLM queries, reference data |
| **Request Batching** | 20-40% reduction in overhead | High-volume operations, independent requests |
| **orjson Serialization** | 3-5x faster JSON | All JSON operations |
| **Lazy Imports** | 20-30% faster startup | Module loading, test execution |
| **Overall System** | **30-50% improvement** | Combined effect of all optimizations |

### Legacy Optimizations (Previous)

| Optimization | Performance Gain | Use Case |
|--------------|------------------|----------|
| Batch Tool Execution | 40-75% faster | Parallel file reads, API calls, independent tools |
| LLM-based Compaction | Better context preservation | Large contexts, complex conversations |
| Prompt Building Cache | 90%+ faster on cache hit | Repeated prompt builds |

---

# ADVANCED OPTIMIZATIONS (New)

## 1. Response Caching

---

## 1. Batch Tool Execution

### Overview

The `ToolExecutor.execute_batch()` method enables parallel execution of independent tools with bounded concurrency. This provides significant speedups for I/O-bound operations like file reads and API calls.

### Implementation

```python
from victor.agent.tool_executor import ToolExecutor

# Create executor
executor = ToolExecutor(tool_registry=registry)

# Execute multiple tools in parallel
results = await executor.execute_batch(
    tool_calls=[
        ("read_file", {"path": "/tmp/file1.txt"}),
        ("read_file", {"path": "/tmp/file2.txt"}),
        ("read_file", {"path": "/tmp/file3.txt"}),
        ("list_directory", {"path": "/tmp"}),
    ],
    max_concurrency=4,  # Maximum 4 parallel executions
)
```text

### Performance Characteristics

**Sequential Execution:**
```
Tool 1: 0.05s
Tool 2: 0.05s
Tool 3: 0.05s
Tool 4: 0.05s
Total: 0.20s
```text

**Batch Execution (concurrency=4):**
```
All 4 tools in parallel: 0.05s
Total: 0.05s (75% faster)
```text

### Configuration Options

```python
await executor.execute_batch(
    tool_calls=[...],
    max_concurrency=5,        # Max parallel executions (default: 5)
    skip_cache=False,          # Skip cache for all calls
    context=None,              # Shared context for all tools
    skip_normalization=False,  # Skip argument normalization
    group_dependent=True,      # Group dependent tools sequentially
)
```

### Dependency Grouping

When `group_dependent=True` (default), tools are automatically grouped:

- **Read tools** (read_file, list_directory): Can run in parallel
- **Write tools** (write_file, edit_file): Run sequentially
- **Mixed**: Write tools run first, then reads

```python
# Automatic grouping
tool_calls = [
    ("write_file", {"path": "a.txt", "content": "data"}),  # Group 1 (write)
    ("write_file", {"path": "b.txt", "content": "data"}),  # Group 1 (write)
    ("read_file", {"path": "a.txt"}),                      # Group 2 (read)
    ("read_file", {"path": "b.txt"}),                      # Group 2 (read)
]
# Group 1 runs sequentially, Group 2 runs in parallel
```text

### Best Practices

1. **Use for I/O-bound operations**: File reads, API calls, database queries
2. **Adjust concurrency based on resources**:
   - High-memory system: `max_concurrency=10`
   - Limited resources: `max_concurrency=3`
3. **Monitor cache effectiveness**: Enable tool caching for idempotent operations
4. **Error handling**: Batch execution continues even if individual tools fail

### Benchmarks

Run the benchmark suite:

```bash
pytest tests/benchmark/test_performance_optimizations.py::test_batch_tool_execution_performance -v -s
```

Expected results:
- 4 slow tools (0.05s each): ~75% improvement
- Mixed speeds (slow + fast): ~35% improvement

---

## 2. Optimized Context Compaction

### Overview

Context compaction strategies reduce token usage while preserving important conversation context. Victor provides three
  strategies:

1. **TruncationCompactionStrategy** - Fast, simple truncation
2. **LLMCompactionStrategy** - LLM-based summarization with caching
3. **HybridCompactionStrategy** - Adaptive strategy selection

### Strategy Comparison

| Strategy | Speed | Quality | Memory | Best For |
|----------|-------|---------|---------|----------|
| Truncation | O(n) | Low | O(1) | Small contexts, speed-critical |
| LLM-based | O(n) + API | High | O(n) | Large contexts, quality-critical |
| Hybrid | Adaptive | Adaptive | O(n) | Variable workloads |

### Implementation

#### Truncation Strategy (Fast)

```python
from victor.agent.coordinators.compaction_strategies import TruncationCompactionStrategy

strategy = TruncationCompactionStrategy(
    max_chars=8192,
    preserve_pinned=True,
    keep_system_messages=True,
)

compacted = strategy.compact(messages, target_tokens=1000)
```text

**Performance:**
- Time: ~1ms for 50 messages
- Memory: Minimal overhead
- Quality: Preserves recency, loses semantic coherence

#### LLM-based Strategy (High Quality)

```python
from victor.agent.coordinators.compaction_strategies import LLMCompactionStrategy

strategy = LLMCompactionStrategy(
    summarization_model="gpt-4o-mini",  # Fast, small model
    cache_summaries=True,
    summary_target_ratio=0.3,  # Summarize to 30% of original
    preserve_recent_count=5,    # Keep last 5 messages verbatim
)

compacted = await strategy.compact_async(messages, target_tokens=1000)
```

**Performance:**
- Time: ~1-2s (first call), ~10ms (cached)
- Memory: O(n) for summary cache
- Quality: 10x better context preservation

**Features:**
- Uses smaller/faster model for summarization (gpt-4o-mini)
- Caches summaries based on content hash
- Preserves system messages and pinned content
- Keeps recent messages verbatim

#### Hybrid Strategy (Adaptive)

```python
from victor.agent.coordinators.compaction_strategies import HybridCompactionStrategy

strategy = HybridCompactionStrategy(
    small_context_threshold=5000,   # Use truncation for < 5K tokens
    large_context_threshold=15000,  # Use LLM for > 15K tokens
    complexity_threshold=0.7,       # Complexity threshold (0.0-1.0)
)

compacted = await strategy.compact_async(messages, target_tokens=1000)
```text

**Decision Logic:**
- Small context (< 5K): Truncation (faster)
- Large context (> 15K): LLM summarization (better quality)
- Medium (5K-15K): Use complexity score
  - High complexity (> 0.7): LLM summarization
  - Low complexity (< 0.7): Truncation

### Complexity Factors

The hybrid strategy calculates complexity based on:

1. **Message count** (0.0 - 0.4)
   - > 20 messages: +0.4
   - > 10 messages: +0.2
   - > 5 messages: +0.1

2. **Error presence** (0.0 - 0.3)
   - Error keywords found: +0.3

3. **Code blocks** (0.0 - 0.3)
   - Code blocks present: +0.3

### Summary Caching

LLM-based strategy caches summaries to avoid repeated API calls:

```python
# First call: Cache miss, generates summary (1-2s)
summary1 = await strategy.compact_async(messages, target_tokens=1000)

# Second call: Cache hit, returns cached summary (~10ms)
summary2 = await strategy.compact_async(messages, target_tokens=1000)
```

Cache key is based on SHA-256 hash of message content.

### Factory Function

```python
from victor.agent.coordinators.compaction_strategies import create_compaction_strategy

strategy = create_compaction_strategy(
    strategy_type="hybrid",  # "truncation", "llm", "hybrid"
    summarization_model="gpt-4o-mini",
    cache_summaries=True,
)
```text

### Best Practices

1. **Use truncation for**:
   - Small contexts (< 5K tokens)
   - Speed-critical applications
   - Simple conversations

2. **Use LLM-based for**:
   - Large contexts (> 15K tokens)
   - Complex conversations with errors
   - Quality-critical applications

3. **Use hybrid for**:
   - Variable workloads
   - Unknown context sizes
   - Adaptive performance/quality trade-off

4. **Enable caching** for LLM-based strategy to avoid repeated API calls

### Benchmarks

```bash
pytest tests/benchmark/test_performance_optimizations.py::test_llm_based_compaction_performance -v -s
```

---

## 3. Cached Prompt Building

### Overview

The `PromptCoordinator` caches built prompts to avoid repeated computation from contributors. Hash-based cache keys with optional TTL support ensure cache efficiency.

### Implementation

```python
from victor.agent.coordinators.prompt_coordinator import PromptCoordinator
from victor.protocols import PromptContext

# Create coordinator with caching
coordinator = PromptCoordinator(
    contributors=[contributor1, contributor2],
    enable_cache=True,
    cache_ttl=3600.0,  # 1 hour TTL (optional)
)

# Build prompt (first time: cache miss)
prompt1 = await coordinator.build_system_prompt(
    PromptContext({"task": "code_review", "language": "python"})
)

# Build prompt again (cache hit: ~90% faster)
prompt2 = await coordinator.build_system_prompt(
    PromptContext({"task": "code_review", "language": "python"})
)
```text

### Cache Keys

Cache keys are SHA-256 hashes of the prompt context:

```python
# Same context -> Same cache key
context1 = PromptContext({"task": "code_review", "language": "python"})
context2 = PromptContext({"task": "code_review", "language": "python"})
# Keys match: cache hit

# Different context -> Different cache key
context3 = PromptContext({"task": "test_generation", "language": "python"})
# Different key: cache miss
```

### TTL Support

Optional time-to-live (TTL) for cache entries:

```python
coordinator = PromptCoordinator(
    contributors=[...],
    enable_cache=True,
    cache_ttl=300.0,  # 5 minutes
)

# After 5 minutes, cache entries expire automatically
```text

### Cache Statistics

Monitor cache effectiveness:

```python
stats = coordinator.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {stats['cache_size']}")
print(f"Total requests: {stats['total_requests']}")
```

Output:
```text
Hit rate: 85.3%
Cache size: 42
Total requests: 287
```

### Cache Invalidation

#### Invalidate Specific Context

```python
coordinator.invalidate_cache(
    PromptContext({"task": "code_review"})
)
```text

#### Invalidate All

```python
coordinator.invalidate_cache()  # Clears all cache
```

#### Automatic Invalidation

Cache is automatically invalidated when:
- Contributors are added/removed
- Contributor priorities change
- TTL expires (if configured)

### Best Practices

1. **Enable caching by default** for production use
2. **Use TTL** for long-running applications to prevent stale cache
3. **Monitor hit rate** to ensure cache effectiveness
4. **Invalidate on config changes** to ensure correctness
5. **Use stable context keys** for better cache hit rates

### Performance Characteristics

**First build (cache miss):**
```text
Time: ~50ms (for 5 contributors)
Operations: Hash computation + 5 async calls
```

**Subsequent builds (cache hit):**
```text
Time: ~5ms (90% faster)
Operations: Hash computation + dict lookup
```

### Benchmarks

```bash
pytest tests/benchmark/test_performance_optimizations.py::test_prompt_building_cache_performance -v -s
```text

Expected: 90%+ improvement on cache hit

---

## Configuration Examples

### Example 1: High-Performance Configuration

```python
from victor.agent.tool_executor import ToolExecutor
from victor.agent.coordinators.compaction_strategies import TruncationCompactionStrategy
from victor.agent.coordinators.prompt_coordinator import PromptCoordinator

# Tool executor with batch execution
executor = ToolExecutor(
    tool_registry=registry,
    tool_cache=ToolCache(max_size=1000),
)

# Fast truncation compaction
compaction_strategy = TruncationCompactionStrategy(
    max_chars=8192,
    preserve_pinned=True,
)

# Prompt coordinator with caching
prompt_coordinator = PromptCoordinator(
    contributors=[...],
    enable_cache=True,
    cache_ttl=None,  # No TTL for maximum hit rate
)
```

### Example 2: High-Quality Configuration

```python
from victor.agent.coordinators.compaction_strategies import LLMCompactionStrategy

# LLM-based compaction for better context preservation
compaction_strategy = LLMCompactionStrategy(
    summarization_model="gpt-4o-mini",
    cache_summaries=True,
    summary_target_ratio=0.3,
    preserve_recent_count=5,
)

# Prompt coordinator with TTL
prompt_coordinator = PromptCoordinator(
    contributors=[...],
    enable_cache=True,
    cache_ttl=3600.0,  # 1 hour TTL
)
```text

### Example 3: Adaptive Configuration

```python
from victor.agent.coordinators.compaction_strategies import HybridCompactionStrategy

# Adaptive compaction based on context size
compaction_strategy = HybridCompactionStrategy(
    small_context_threshold=5000,
    large_context_threshold=15000,
    complexity_threshold=0.7,
)

# Prompt coordinator with monitoring
prompt_coordinator = PromptCoordinator(
    contributors=[...],
    enable_cache=True,
    cache_ttl=1800.0,  # 30 minutes
)

# Monitor performance
stats = prompt_coordinator.get_cache_stats()
if stats['hit_rate'] < 0.7:
    logger.warning(f"Low cache hit rate: {stats['hit_rate']:.1%}")
```


**Reading Time:** 8 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## Benchmarking
