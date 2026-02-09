# Enhanced Memory Guide - Part 2

**Part 2 of 3:** Retrieval Strategies and Best Practices

---

## Navigation

- [Part 1: Memory Architecture](part-1-memory-architecture.md)
- **[Part 2: Retrieval & Best Practices](#)** (Current)
- [Part 3: Troubleshooting & Examples](part-3-troubleshooting-examples.md)
- [**Complete Guide**](../ENHANCED_MEMORY.md)

---

## Retrieval Strategies

Different retrieval strategies for different use cases.

### 1. Semantic Similarity Retrieval

Best for: Finding conceptually similar experiences/knowledge

```python
# Episodic memory
episodes = await orchestrator.episodic_memory.recall_relevant(
    query="Fix authentication bug",
    k=5,
    threshold=0.3
)

# Semantic memory
facts = await orchestrator.semantic_memory.query_knowledge(
    query="JWT best practices",
    k=5,
    threshold=0.25
)
```text

### 2. Temporal Retrieval

Best for: Finding recent experiences or knowledge trends

```python
# Recent episodes (last N)
recent = await orchestrator.episodic_memory.recall_recent(
    n=10,
    task_type="bugfix"
)

# Recent knowledge
recent_facts = await orchestrator.semantic_memory.get_recent_facts(
    n=10,
    category="security"
)

# Time-range queries
from datetime import datetime, timedelta

time_range = (
    datetime.now() - timedelta(days=7),
    datetime.now()
)
episodes = await orchestrator.episodic_memory.recall_by_time_range(
    start_time=time_range[0],
    end_time=time_range[1]
)
```

### 3. Outcome-Based Retrieval

Best for: Learning from successes/failures

```python
# Successful episodes
successes = await orchestrator.episodic_memory.recall_by_outcome(
    outcome_key="success",
    outcome_value=True,
    n=10
)

# Failed episodes
failures = await orchestrator.episodic_memory.recall_by_outcome(
    outcome_key="success",
    outcome_value=False,
    n=5
)

# High-reward episodes
best = await orchestrator.episodic_memory.recall_by_reward(
    min_reward=8.0,
    n=10
)
```text

### 4. Hybrid Retrieval

Best for: Comprehensive context gathering

```python
async def get_context_for_task(query):
    """Gather comprehensive context from both memories."""

    # Get similar past experiences
    episodes = await orchestrator.episodic_memory.recall_relevant(
        query=query,
        k=5
    )

    # Get relevant facts
    facts = await orchestrator.semantic_memory.query_knowledge(
        query=query,
        k=5
    )

    # Get recent context
    recent = await orchestrator.episodic_memory.recall_recent(n=3)

    return {
        "episodes": episodes,
        "facts": facts,
        "recent": recent
    }
```

---

## Best Practices

### 1. Episode Design

**DO:**
- Include detailed context (task_type, technologies, complexity)
- Store structured outcomes (success metrics, errors)
- Use meaningful reward values

**DON'T:**
- Store vague episodes ("fixed a bug")
- Ignore negative outcomes (failures are valuable)
- Forget to tag with metadata

```python
# Good episode
episode_id = await orchestrator.episodic_memory.store_episode(
    inputs={"query": "Fix JWT authentication", "error_code": 401},
    actions=[
        "read_file: src/auth/jwt.py",
        "edit_file: Fixed signature verification",
        "run_tests: 15/15 passed"
    ],
    outcomes={
        "success": True,
        "bugs_fixed": 1,
        "tests_passed": 15
    },
    rewards=10.0,
    context={
        "task_type": "bugfix",
        "complexity": "medium",
        "technologies": ["JWT", "Python"]
    }
)
```text

### 2. Semantic Knowledge

**DO:**
- Store decontextualized facts
- Use clear, generalizable statements
- Organize by category

**DON'T:**
- Store specific experiences as facts
- Use vague statements
- Forget to cite sources

```python
# Good fact
fact_id = await orchestrator.semantic_memory.store_knowledge(
    fact="JWT tokens should have short expiry times (15-30 minutes) for security",
    category="security",
    confidence=0.95,
    source="OWASP best practices"
)
```

### 3. Memory Consolidation

**DO:**
- Consolidate regularly (daily/weekly)
- Use high-reward episodes for consolidation
- Validate consolidated facts

**DON'T:**
- Consolidate too frequently (wasteful)
- Consolidate low-quality episodes
- Forget to review consolidated facts

```python
# Trigger consolidation
await orchestrator.consolidate_memories()
```text

### 4. Performance Optimization

**DO:**
- Set appropriate limits (max_episodes, max_facts)
- Use thresholds in queries
- Cache frequent queries

**DON'T:**
- Store unlimited episodes (memory bloat)
- Use very low thresholds (slow retrieval)
- Query too frequently

```python
# Configure limits
settings.episodic_memory_max_episodes = 1000
settings.semantic_memory_max_facts = 5000

# Use appropriate thresholds
episodes = await orchestrator.episodic_memory.recall_relevant(
    query="...",
    k=10,
    threshold=0.3  # Not too low
)
```


**Reading Time:** 2 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 3: Troubleshooting & Examples](part-3-troubleshooting-examples.md)**
