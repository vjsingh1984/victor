# Enhanced Memory Guide - Part 3

**Part 3 of 3:** Troubleshooting and Examples

---

## Navigation

- [Part 1: Memory Architecture](part-1-memory-architecture.md)
- [Part 2: Retrieval & Best Practices](part-2-retrieval-best-practices.md)
- **[Part 3: Troubleshooting & Examples](#)** (Current)
- [**Complete Guide**](../ENHANCED_MEMORY.md)

---

## Troubleshooting

### Poor Retrieval Results

**Problem**: Retrieval returns irrelevant episodes/facts.

**Solutions**:
1. **Adjust thresholds**: Increase similarity thresholds
2. **Improve context**: Store more detailed context
3. **Better queries**: Use more specific query terms
4. **Clean data**: Remove low-quality episodes

```python
# Increase threshold
episodes = await orchestrator.episodic_memory.recall_relevant(
    query="specific query",
    threshold=0.5  # Higher threshold
)

# Clean low-quality episodes
await orchestrator.episodic_memory.clear_by_reward(
    max_reward=2.0  # Remove low-reward episodes
)
```text

### Memory Bloat

**Problem**: Memory usage growing too large.

**Solutions**:
1. **Set limits**: Configure max_episodes/max_facts
2. **Regular cleanup**: Clear old/low-quality entries
3. **Consolidate**: Convert episodes to facts
4. **Compress**: Use lower-dimensional embeddings

```python
# Configure limits
settings.episodic_memory_max_episodes = 1000
settings.semantic_memory_max_facts = 5000

# Regular cleanup
await orchestrator.episodic_memory.clear_old_episodes(days=30)
await orchestrator.semantic_memory.clear_old_facts(days=90)

# Consolidate to reduce episodes
await orchestrator.consolidate_memories()
```

### Slow Retrieval

**Problem**: Memory queries are slow.

**Solutions**:
1. **Use indexes**: Ensure embeddings are indexed
2. **Limit results**: Reduce k in queries
3. **Cache queries**: Cache frequent queries
4. **Batch operations**: Batch multiple queries

```python
# Limit results
episodes = await orchestrator.episodic_memory.recall_relevant(
    query="...",
    k=10  # Reduce from default
)

# Cache frequent queries
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(query):
    return orchestrator.episodic_memory.recall_relevant(query)
```text

---

## Examples

### Example 1: Learning from Bug Fixes

```python
async def learn_from_bugfix():
    # Store bugfix episode
    episode_id = await orchestrator.episodic_memory.store_episode(
        inputs={
            "query": "Fix authentication bug",
            "error": "Invalid token signature"
        },
        actions=[
            "read_file: src/auth/jwt_validator.py",
            "edit_file: Updated signature verification",
            "run_tests: All tests passing"
        ],
        outcomes={
            "success": True,
            "files_changed": 1,
            "bugs_fixed": 1
        },
        rewards=10.0,
        context={
            "task_type": "bugfix",
            "technologies": ["JWT", "Python"]
        }
    )

    # Later, retrieve when similar bug occurs
    relevant = await orchestrator.episodic_memory.recall_relevant(
        query="JWT token validation failed",
        k=3
    )

    for episode in relevant:
        print(f"Similar bug fix: {episode.actions}")
```

### Example 2: Building Knowledge Base

```python
async def build_knowledge_base():
    # Store security best practices
    await orchestrator.semantic_memory.store_knowledge(
        fact="JWT tokens should have short expiry times (15-30 minutes)",
        category="security",
        confidence=0.95
    )

    await orchestrator.semantic_memory.store_knowledge(
        fact="Database passwords must be hashed with bcrypt or argon2",
        category="security",
        confidence=0.99
    )

    # Query knowledge base
    facts = await orchestrator.semantic_memory.query_knowledge(
        query="security best practices",
        k=5
    )
```text

### Example 3: Continuous Learning

```python
async def continuous_learning_loop():
    # After each task
    episode = await orchestrator.episodic_memory.store_episode(...)

    # Consolidate daily
    if should_consolidate():
        await orchestrator.consolidate_memories()

    # Clean up weekly
    if should_cleanup():
        await orchestrator.episodic_memory.clear_old_episodes(days=30)
        await orchestrator.semantic_memory.clear_old_facts(days=90)
```

---

## Additional Resources

### API Documentation

- `EpisodicMemory` class: `/victor/agent/memory/episodic_memory.py`
- `SemanticMemory` class: `/victor/agent/memory/semantic_memory.py`
- `MemoryConsolidator` class: `/victor/agent/memory/consolidator.py`

### Related Guides

- [Knowledge Management](../KNOWLEDGE_MANAGEMENT.md)
- [Learning from Experience](../LEARNING_SYSTEMS.md)
- [Performance Optimization](./PERFORMANCE_TUNING.md)

### Configuration

Enable memory systems in settings:

```python
# config/settings.py
enable_episodic_memory = True
enable_semantic_memory = True
episodic_memory_max_episodes = 1000
semantic_memory_max_facts = 5000
```text

---

**Last Updated:** February 01, 2026
**Reading Time:** 12 min (all parts)
