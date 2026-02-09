# Semantic Caching Guide

## Overview

Semantic caching enhances traditional caching by using vector similarity to find cached responses that are semantically
  similar to the current query,
  even if they're not exact matches. This dramatically increases cache hit rates for natural language queries.

## How It Works

```
Query: "How do I parse JSON in Python?"
    ↓
Compute Embedding: [0.23, -0.45, 0.67, ...]  (1536-dim vector)
    ↓
Similarity Search: Find cached entries with similar embeddings
    ↓
Best Match: "Python JSON parsing example" (similarity: 0.92)
    ↓
Threshold Check: 0.92 > 0.85 ✓
    ↓
Return Cached Result
```

## Benefits

- **Higher Hit Rates**: 40-60% hit rate for similar queries (vs. 20-30% exact match)
- **Reduced API Costs**: Fewer redundant API calls for similar queries
- **Faster Responses**: Sub-millisecond retrieval vs. seconds for API calls
- **Better UX**: Users get relevant results faster

## Configuration

### Basic Setup

```python
from victor.core.cache import SemanticCache

cache = SemanticCache(
    similarity_threshold=0.85,  # Minimum similarity for match
    embedding_model="text-embedding-ada-002",
    max_size=1000,
    default_ttl=3600,
    enable_exact_match_fallback=True,
    batch_size=100,
)
```

### Similarity Threshold

The `similarity_threshold` determines how similar queries must be to return cached results:

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.90-0.95 | Very High | Low | Strict requirements, low tolerance for error |
| 0.85-0.90 | High | Medium | General purpose (recommended) |
| 0.75-0.85 | Medium | High | Maximum hit rate, some false positives |
| 0.65-0.75 | Low | Very High | Exploratory, tolerance for noise |

```python
# Strict matching (high precision)
cache = SemanticCache(similarity_threshold=0.92)

# Balanced (recommended)
cache = SemanticCache(similarity_threshold=0.85)

# Lenient (high recall)
cache = SemanticCache(similarity_threshold=0.75)
```

### Embedding Models

```python
# OpenAI embeddings (recommended)
cache = SemanticCache(embedding_model="text-embedding-ada-002")

# Custom embedding model
cache = SemanticCache(embedding_model="your-custom-model")
```

## Usage Examples

### Basic Usage

```python
from victor.providers.base import Message

# Store response
messages = [Message(role="user", content="How do I parse JSON?")]
response = CompletionResponse(content="Use json.loads()")

await cache.put(messages, response)

# Retrieve with exact match
result = await cache.get(messages)

# Retrieve with semantic similarity
result = await cache.get_similar(messages)
```

### Semantic Similarity Search

```python
# Original query
messages1 = [Message(role="user", content="Parse JSON in Python")]
await cache.put(messages1, response1)

# Similar query (different wording)
messages2 = [Message(role="user", content="How to read JSON file Python")]

# Will find semantically similar cached response
result = await cache.get_similar(messages2)

# Similarity score is tracked
stats = cache.get_stats()
print(f"Semantic hit rate: {stats['semantic_hit_rate']:.1%}")
```

### Exact Match Fallback

```python
cache = SemanticCache(
    enable_exact_match_fallback=True,  # Default
)

# First tries semantic match
# If no similar entry found, falls back to exact match
result = await cache.get_similar(messages)
```

### Custom Threshold Per Query

```python
# Use different threshold for specific query
result = await cache.get_similar(
    messages,
    threshold=0.90,  # Stricter than default
)
```

## Similarity Metrics

### Cosine Similarity

Victor AI uses cosine similarity to compare embeddings:

```python
similarity = (A · B) / (||A|| * ||B||)

# Where:
# A · B = dot product of vectors
# ||A|| = magnitude (L2 norm) of vector A
# ||B|| = magnitude of vector B
```

**Range**: -1 to 1
- **1.0**: Identical vectors (same direction)
- **0.0**: Orthogonal vectors (unrelated)
- **-1.0**: Opposite vectors (rare in practice)

For text embeddings, typical range is 0.3 to 0.95.

### Interpretation

| Similarity | Interpretation |
|------------|---------------|
| 0.90-1.00 | Very similar, likely same meaning |
| 0.80-0.90 | Similar, related concepts |
| 0.70-0.80 | Somewhat related, possible match |
| 0.60-0.70 | Weakly related, low confidence |
| 0.00-0.60 | Unrelated, unlikely match |

## Performance Tuning

### Batch Size

For efficient similarity search, process entries in batches:

```python
# Small batches (default): Better accuracy
cache = SemanticCache(batch_size=50)

# Large batches: Faster computation
cache = SemanticCache(batch_size=200)
```

**Trade-off**:
- Small batches: More accurate, slightly slower
- Large batches: Faster, minor accuracy loss
- Recommendation: 100 (default) provides good balance

### Cache Size

```python
# Small cache: Fast lookup, lower hit rate
cache = SemanticCache(max_size=500)

# Medium cache: Balanced (recommended)
cache = SemanticCache(max_size=1000)

# Large cache: Higher hit rate, slower lookup
cache = SemanticCache(max_size=5000)
```

### TTL Configuration

```python
# Short TTL: Fresh data, more API calls
cache = SemanticCache(default_ttl=300)  # 5 minutes

# Medium TTL: Balanced (recommended)
cache = SemanticCache(default_ttl=3600)  # 1 hour

# Long TTL: High hit rate, stale data risk
cache = SemanticCache(default_ttl=86400)  # 1 day
```

## Monitoring and Analytics

### Cache Statistics

```python
stats = cache.get_stats()

print(f"Size: {stats['size']}/{stats['max_size']}")
print(f"Hit Rate: {stats['hit_rate']:.1%}")
print(f"Semantic Hit Rate: {stats['semantic_hit_rate']:.1%}")
print(f"Exact Hits: {stats['exact_hits']}")
print(f"Semantic Hits: {stats['semantic_hits']}")
print(f"Misses: {stats['misses']}")

# Output:
# Size: 843/1000
# Hit Rate: 67.3%
# Semantic Hit Rate: 54.2%
# Exact Hits: 321
# Semantic Hits: 405
# Misses: 327
```

### Performance Metrics

```python
# Track semantic vs exact hit performance
semantic_hits = stats['semantic_hits']
exact_hits = stats['exact_hits']
total_hits = semantic_hits + exact_hits

semantic_ratio = semantic_hits / total_hits if total_hits > 0 else 0

print(f"Semantic hits: {semantic_ratio:.1%} of all hits")
print(f"Semantic hit rate advantage: {semantic_ratio - 0.5:.1%}")
```

## Best Practices

### 1. Set Appropriate Threshold

```python
# Too low: Many false positives
cache = SemanticCache(similarity_threshold=0.70)  # Not recommended

# Too high: Missing valid matches
cache = SemanticCache(similarity_threshold=0.95)  # Too strict

# Just right: Good balance
cache = SemanticCache(similarity_threshold=0.85)  # Recommended
```

### 2. Use Exact Match for Critical Queries

```python
# For critical queries, use exact match only
if is_critical_query(messages):
    result = await cache.get(messages)  # Exact match
else:
    result = await cache.get_similar(messages)  # Semantic match
```

### 3. Monitor Semantic Hit Rate

```python
stats = cache.get_stats()

if stats['semantic_hit_rate'] < 0.3:
    logger.warning("Low semantic hit rate, consider lowering threshold")

if stats['semantic_hit_rate'] > 0.8:
    logger.info("High semantic hit rate, consider raising threshold")
```

### 4. Handle Edge Cases

```python
try:
    result = await cache.get_similar(messages)
    if result is None:
        # Cache miss, compute result
        result = await compute_result(messages)
        await cache.put(messages, result)
except Exception as e:
    logger.error(f"Semantic cache error: {e}")
    # Fall back to computation
    result = await compute_result(messages)
```

## Use Cases

### 1. Code Q&A

```python
# User asks similar questions in different ways
q1 = "How do I parse JSON in Python?"
q2 = "Python JSON parsing example"
q3 = "Read JSON file Python code"

# All three can match the same cached response
```

### 2. Documentation Search

```python
# Similar documentation queries
q1 = "Set up authentication"
q2 = "How to configure auth"
q3 = "Authentication setup guide"

# Return same cached documentation
```

### 3. Error Resolution

```python
# Similar error descriptions
q1 = "Fix NullPointerException in Java"
q2 = "Java null pointer exception handling"
q3 = "How to avoid NPE in Java code"

# Return same solution
```

## Integration Examples

### With Multi-Level Cache

```python
from victor.core.cache import SemanticCache, MultiLevelCache

# Use semantic cache for L1
l1_cache = SemanticCache(max_size=1000)

# Use regular cache for L2
l2_cache = MultiLevelCache(...)

# Manual coordination
async def get_with_semantic(key: str, messages: list):
    # Try semantic L1 first
    result = await l1_cache.get_similar(messages)
    if result:
        return result

    # Fall back to L2
    result = await l2_cache.get(key, namespace="tool")
    if result:
        # Promote to L1
        await l1_cache.put(messages, result)
        return result

    # Compute and cache
    result = await compute_result(messages)
    await l1_cache.put(messages, result)
    await l2_cache.set(key, result, namespace="tool")
    return result
```

### With Analytics

```python
from victor.core.cache import SemanticCache, CacheAnalytics

cache = SemanticCache(...)
analytics = CacheAnalytics(cache=cache, track_hot_keys=True)

# Record semantic hits
async def get_and_track(messages: list):
    result = await cache.get_similar(messages)

    analytics.record_access(
        key=str(messages),
        namespace="semantic",
        hit=result is not None,
    )

    return result
```

## Performance Characteristics

### Latency Breakdown

| Operation | Latency |
|-----------|---------|
| Embedding computation | 50-200ms |
| Similarity search (1000 entries) | 1-5ms |
| Exact match lookup | <0.1ms |
| Total semantic hit | 50-205ms |

**Optimization**: Cache embeddings to avoid recomputation

### Scalability

| Cache Size | Search Time | Memory |
|------------|-------------|--------|
| 100 | <1ms | ~5MB |
| 1,000 | 1-5ms | ~50MB |
| 10,000 | 10-50ms | ~500MB |

**Recommendation**: Use 1,000-5,000 entries for optimal performance

## Troubleshooting

### Low Semantic Hit Rate

**Symptoms**: Semantic hit rate < 30%

**Possible Causes**:
1. Similarity threshold too high
2. Poor embedding quality
3. Insufficient cache size
4. Diverse queries (low similarity)

**Solutions**:
1. Lower similarity_threshold to 0.80
2. Use better embedding model
3. Increase max_size
4. Accept lower hit rate for diverse queries

### High False Positive Rate

**Symptoms**: Returning irrelevant cached results

**Possible Causes**:
1. Similarity threshold too low
2. Embedding model not capturing semantics

**Solutions**:
1. Increase similarity_threshold to 0.90
2. Use domain-specific embedding model
3. Add post-processing validation

### Slow Performance

**Symptoms**: Semantic search > 100ms

**Possible Causes**:
1. Large cache size
2. Slow embedding computation
3. Batch size too small

**Solutions**:
1. Reduce max_size
2. Cache embeddings
3. Increase batch_size

## See Also

- [Multi-Level Cache](MULTI_LEVEL_CACHE.md)
- [Cache Warming](CACHE_WARMING.md)
- [Cache Invalidation](CACHE_INVALIDATION.md)
- [Cache Analytics](CACHE_ANALYTICS.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
