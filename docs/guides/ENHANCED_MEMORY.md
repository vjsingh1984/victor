# Enhanced Memory Guide

## Overview

Victor AI's enhanced memory system provides a dual-memory architecture combining episodic memory (experiences) and semantic memory (knowledge) for intelligent knowledge retention and retrieval. This guide explains how to use the memory systems effectively.

## Table of Contents

- [What is Enhanced Memory?](#what-is-enhanced-memory)
- [Episodic Memory](#episodic-memory)
- [Semantic Memory](#semantic-memory)
- [Memory Consolidation](#memory-consolidation)
- [Retrieval Strategies](#retrieval-strategies)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## What is Enhanced Memory?

Victor's memory system is inspired by human cognitive architecture:

### Episodic Memory
Stores **experiences** - what happened, when, and in what context.
- "Fixed authentication bug by updating JWT validation logic"
- "Refactored database layer, improved query performance by 40%"
- "Attempted microservice migration but hit issues with state management"

### Semantic Memory
Stores **facts and knowledge** - generalized, decontextualized information.
- "JWT tokens should have short expiry times (15-30 minutes)"
- "Database indexes improve read performance but slow down writes"
- "Microservices require inter-service communication mechanisms"

### Key Benefits

1. **Learning from Experience**: Store and retrieve past solutions
2. **Knowledge Accumulation**: Build a knowledge base over time
3. **Context-Aware Retrieval**: Find relevant information using semantic search
4. **Automatic Consolidation**: Convert experiences into generalized knowledge
5. **Efficient Storage**: Vector embeddings for fast similarity search

## Episodic Memory

Episodic memory stores agent experiences as episodes with context, actions, and outcomes.

### Basic Usage

```python
from victor.agent import AgentOrchestrator
from victor.config.settings import Settings

# Enable episodic memory
settings = Settings()
settings.enable_episodic_memory = True

orchestrator = AgentOrchestrator(settings=settings, ...)

# Store an episode
episode_id = await orchestrator.episodic_memory.store_episode(
    inputs={
        "query": "Fix authentication bug",
        "context": {"error": "Invalid token signature"}
    },
    actions=[
        "read_file: src/auth/jwt_validator.py",
        "edit_file: Updated signature verification",
        "run_tests: All tests passing"
    ],
    outcomes={
        "success": True,
        "files_changed": 1,
        "bugs_fixed": 1,
        "tests_passed": 15
    },
    rewards=10.0,  # Positive reward for success
    context={
        "task_type": "bugfix",
        "complexity": "medium",
        "technologies": ["JWT", "Python"]
    }
)

print(f"Stored episode: {episode_id}")
```

### Retrieving Episodes

```python
# Recall relevant episodes by semantic similarity
relevant = await orchestrator.episodic_memory.recall_relevant(
    query="JWT token validation issue",
    k=5,  # Return top 5 most relevant
    threshold=0.3  # Minimum similarity score
)

for episode in relevant:
    print(f"Episode {episode.id}:")
    print(f"  Actions: {episode.actions}")
    print(f"  Success: {episode.outcomes.get('success')}")
    print(f"  Similarity: {episode.similarity}")
```

### Recall Strategies

```python
# 1. Semantic recall (similarity-based)
episodes = await orchestrator.episodic_memory.recall_relevant(
    "database optimization",
    k=10
)

# 2. Temporal recall (recent episodes)
recent = await orchestrator.episodic_memory.recall_recent(
    n=20,  # Last 20 episodes
    task_type="bugfix"  # Optional filter
)

# 3. Outcome-based recall
successful = await orchestrator.episodic_memory.recall_by_outcome(
    outcome_key="success",
    outcome_value=True,
    n=10
)

failed = await orchestrator.episodic_memory.recall_by_outcome(
    outcome_key="success",
    outcome_value=False,
    n=5
)

# 4. Reward-based recall (highest rewards first)
best = await orchestrator.episodic_memory.recall_by_reward(
    min_reward=5.0,
    n=10
)
```

### Memory Statistics

```python
stats = await orchestrator.episodic_memory.get_memory_statistics()

print(f"Total episodes: {stats['total_episodes']}")
print(f"Average reward: {stats['average_reward']:.2f}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Oldest episode: {stats['oldest_episode_timestamp']}")
print(f"Newest episode: {stats['newest_episode_timestamp']}")
```

### Episode Management

```python
# Clear old episodes
await orchestrator.episodic_memory.clear_old_episodes(
    days=30  # Delete episodes older than 30 days
)

# Clear specific episodes
await orchestrator.episodic_memory.clear_episode(episode_id)

# Consolidate into semantic memory
consolidated = await orchestrator.episodic_memory.consolidate_into_knowledge(
    min_episodes=5,  # Need at least 5 similar episodes
    similarity_threshold=0.7  # High similarity for consolidation
)

print(f"Consolidated {consolidated['facts_created']} new facts")
```

## Semantic Memory

Semantic memory stores facts and knowledge in a knowledge graph structure.

### Basic Usage

```python
# Enable semantic memory
settings.enable_semantic_memory = True

orchestrator = AgentOrchestrator(settings=settings, ...)

# Store knowledge
fact_id = await orchestrator.semantic_memory.store_knowledge(
    fact="Python uses asyncio for concurrent programming",
    metadata={
        "category": "programming",
        "language": "python",
        "topic": "concurrency"
    },
    confidence=1.0,  # Confidence level (0.0 - 1.0)
    source="documentation"  # Optional source
)

print(f"Stored fact: {fact_id}")
```

### Querying Knowledge

```python
# Query by semantic similarity
facts = await orchestrator.semantic_memory.query_knowledge(
    query="How to handle concurrent operations in Python?",
    k=5,
    threshold=0.25  # Minimum similarity score
)

for fact in facts:
    print(f"Fact: {fact.fact}")
    print(f"Similarity: {fact.similarity:.2f}")
    print(f"Confidence: {fact.confidence:.2f}")
    print(f"Metadata: {fact.metadata}")
```

### Knowledge Graph Operations

```python
# Link related facts
await orchestrator.semantic_memory.link_facts(
    fact_id_1=fact_id,
    fact_id_2=another_fact_id,
    link_type="related",
    strength=0.9,  # Link strength (0.0 - 1.0)
    metadata={"reason": "Both discuss async patterns"}
)

# Get knowledge graph
graph = await orchestrator.semantic_memory.get_knowledge_graph()

print(f"Total facts: {len(graph.facts)}")
print(f"Total links: {len(graph.links)}")

# Traverse from a fact
related_facts = await orchestrator.semantic_memory.get_related_facts(
    fact_id=fact_id,
    max_depth=2,  # Traverse up to 2 hops
    min_strength=0.5  # Only strong links
)

for related in related_facts:
    print(f"{related.fact} (strength: {related.link_strength})")
```

### Knowledge by Category

```python
# Get facts by metadata
python_facts = await orchestrator.semantic_memory.get_facts_by_metadata(
    key="language",
    value="python"
)

security_facts = await orchestrator.semantic_memory.get_facts_by_metadata(
    key="category",
    value="security"
)

# Get high-confidence facts
certain_facts = await orchestrator.semantic_memory.get_facts_by_confidence(
    min_confidence=0.9
)
```

### Knowledge Statistics

```python
stats = await orchestrator.semantic_memory.get_statistics()

print(f"Total facts: {stats['total_facts']}")
print(f"Total links: {stats['total_links']}")
print(f"Average confidence: {stats['average_confidence']:.2f}")
print(f"Categories: {stats['categories']}")
```

### Knowledge Management

```python
# Update a fact
await orchestrator.semantic_memory.update_fact(
    fact_id=fact_id,
    fact="Updated fact text",
    confidence=0.95
)

# Update fact metadata
await orchestrator.semantic_memory.update_fact_metadata(
    fact_id=fact_id,
    metadata={"category": "advanced", "verified": True}
)

# Delete a fact
await orchestrator.semantic_memory.delete_fact(fact_id)

# Clear old facts
await orchestrator.semantic_memory.clear_old_facts(
    days=90,  # Delete facts older than 90 days
    min_confidence=0.5  # Keep high-confidence facts
)
```

## Memory Consolidation

Memory consolidation converts episodic experiences into semantic knowledge.

### Automatic Consolidation

```python
# Enable automatic consolidation
settings.episodic_memory_consolidation_enabled = True
settings.episodic_memory_consolidation_interval = 100  # Every 100 episodes

# Consolidation runs automatically when threshold reached
```

### Manual Consolidation

```python
# Manually trigger consolidation
result = await orchestrator.consolidate_memories(
    strategy="frequency",  # or "similarity", "hybrid"
    min_episodes=3,
    similarity_threshold=0.6
)

print(f"Consolidated {result['episodes_processed']} episodes")
print(f"Created {result['facts_created']} new facts")
print(f"Updated {result['facts_updated']} existing facts")
```

### Consolidation Strategies

```python
# 1. Frequency-based: Extract patterns from repeated experiences
result = await orchestrator.consolidate_memories(
    strategy="frequency",
    min_occurrences=3,  # Must appear in 3+ episodes
    confidence_boost=0.1  # Boost confidence for repeated facts
)

# 2. Similarity-based: Cluster similar episodes
result = await orchestrator.consolidate_memories(
    strategy="similarity",
    similarity_threshold=0.7,
    min_cluster_size=3
)

# 3. Hybrid: Combine frequency and similarity
result = await orchestrator.consolidate_memories(
    strategy="hybrid",
    min_occurrences=2,
    similarity_threshold=0.6,
    min_cluster_size=3
)
```

### Custom Consolidation Logic

```python
# Define custom consolidation function
async def custom_consolidation(episodes):
    """Extract common patterns from episodes."""
    # Group by task type
    by_task = {}
    for ep in episodes:
        task = ep.context.get("task_type", "general")
        by_task.setdefault(task, []).append(ep)

    # Extract patterns for each task type
    facts = []
    for task, task_episodes in by_task.items():
        if len(task_episodes) >= 3:  # Minimum threshold
            # Calculate success rate
            successful = sum(1 for e in task_episodes if e.outcomes.get("success"))
            success_rate = successful / len(task_episodes)

            # Extract common actions
            all_actions = [a for ep in task_episodes for a in ep.actions]
            common_actions = set([a for a in all_actions if all_actions.count(a) >= 2])

            fact = f"For {task} tasks: {', '.join(common_actions)} (success rate: {success_rate:.1%})"
            facts.append({
                "fact": fact,
                "confidence": min(success_rate, 1.0),
                "metadata": {"task_type": task, "sample_size": len(task_episodes)}
            })

    return facts

# Apply custom consolidation
result = await orchestrator.consolidate_memories(
    consolidation_fn=custom_consolidation
)
```

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
```

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
```

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
    recent = await orchestrator.episodic_memory.recall_recent(n=5)

    # Combine into context
    context = {
        "query": query,
        "similar_experiences": [
            {
                "actions": ep.actions,
                "outcomes": ep.outcomes,
                "success": ep.outcomes.get("success")
            }
            for ep in episodes
        ],
        "relevant_facts": [
            {
                "fact": f.fact,
                "confidence": f.confidence
            }
            for f in facts
        ],
        "recent_activity": [
            {
                "actions": ep.actions,
                "timestamp": ep.timestamp
            }
            for ep in recent
        ]
    }

    return context

# Use context
context = await get_context_for_task("Implement OAuth2")
result = await orchestrator.chat(
    "Implement OAuth2 authentication",
    context=context
)
```

## Best Practices

### 1. Store Rich Context

```python
# Good: Rich context
episode_id = await orchestrator.episodic_memory.store_episode(
    inputs={
        "query": "Fix authentication bug",
        "error_message": "Invalid token signature",
        "stack_trace": "...",
        "environment": "production"
    },
    actions=[...],
    outcomes={...},
    context={
        "task_type": "bugfix",
        "complexity": "medium",
        "technologies": ["JWT", "Python"],
        "files_modified": ["src/auth/jwt_validator.py"],
        "root_cause": "Missing secret key validation"
    }
)

# Bad: Minimal context
episode_id = await orchestrator.episodic_memory.store_episode(
    inputs={"query": "Fix bug"},
    actions=["fixed it"],
    outcomes={"done": True}
)
```

### 2. Use Appropriate Rewards

```python
# Reward based on outcome quality
reward = 0.0

if outcome["success"]:
    base_reward = 10.0

    # Adjust for complexity
    if context["complexity"] == "high":
        base_reward *= 1.5

    # Adjust for impact
    if outcome.get("performance_improvement", 0) > 0.3:
        base_reward *= 1.2

    # Adjust for quality
    if outcome.get("tests_passed", 0) == outcome.get("total_tests", 0):
        base_reward *= 1.1

    reward = base_reward
else:
    # Small positive reward for learning from failures
    reward = 1.0

await orchestrator.episodic_memory.store_episode(
    ...,
    rewards=reward
)
```

### 3. Consolidate Regularly

```python
# Schedule regular consolidation
import asyncio

async def periodic_consolidation():
    while True:
        await asyncio.sleep(3600)  # Every hour
        result = await orchestrator.consolidate_memories()
        print(f"Consolidated: {result}")

# Run as background task
asyncio.create_task(periodic_consolidation())
```

### 4. Use Metadata Effectively

```python
# Good: Structured metadata
fact_id = await orchestrator.semantic_memory.store_knowledge(
    fact="JWT tokens should have short expiry times",
    metadata={
        "category": "security",
        "technology": "JWT",
        "best_practice": True,
        "verified": True,
        "source": "OWASP",
        "severity": "high"
    }
)

# Enables targeted queries
security_facts = await orchestrator.semantic_memory.get_facts_by_metadata(
    key="category",
    value="security"
)
```

### 5. Monitor Memory Health

```python
# Regular health checks
async def check_memory_health():
    # Check episodic memory
    episodic_stats = await orchestrator.episodic_memory.get_memory_statistics()
    print(f"Episodic: {episodic_stats['total_episodes']} episodes")
    print(f"Success rate: {episodic_stats['success_rate']:.1%}")

    # Check semantic memory
    semantic_stats = await orchestrator.semantic_memory.get_statistics()
    print(f"Semantic: {semantic_stats['total_facts']} facts")
    print(f"Avg confidence: {semantic_stats['average_confidence']:.2f}")

    # Check if consolidation needed
    if episodic_stats['total_episodes'] > 100:
        print("Consider consolidating episodes")

    # Check memory usage
    if episodic_stats['total_episodes'] > settings.episodic_memory_max_episodes * 0.9:
        print("Episodic memory near capacity, consider clearing old episodes")

# Run checks periodically
asyncio.create_task(check_memory_health())
```

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
```

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
```

## Examples

### Example 1: Learning from Bug Fixes

```python
async def learn_from_bugfix():
    # Store bugfix episode
    episode_id = await orchestrator.episodic_memory.store_episode(
        inputs={
            "bug_type": "authentication",
            "error": "Invalid token signature"
        },
        actions=[
            "Read JWT validator code",
            "Identify missing validation",
            "Add secret key check",
            "Run tests"
        ],
        outcomes={
            "success": True,
            "bugs_fixed": 1
        },
        rewards=10.0,
        context={
            "task_type": "bugfix",
            "root_cause": "Missing validation",
            "solution": "Add validation check"
        }
    )

    # Later, when similar bug occurs
    relevant = await orchestrator.episodic_memory.recall_relevant(
        query="JWT authentication issue",
        k=3
    )

    for ep in relevant:
        print(f"Previous solution: {ep.context.get('solution')}")
        print(f"Actions that worked: {ep.actions}")
```

### Example 2: Building Knowledge Base

```python
async def build_knowledge_base():
    # Learn from documentation
    await orchestrator.semantic_memory.store_knowledge(
        fact="JWT tokens should expire after 15-30 minutes",
        metadata={"category": "security", "technology": "JWT"},
        confidence=1.0,
        source="OWASP guidelines"
    )

    await orchestrator.semantic_memory.store_knowledge(
        fact="Use HTTPS for all JWT transmissions",
        metadata={"category": "security", "technology": "JWT"},
        confidence=1.0,
        source="OWASP guidelines"
    )

    # Link related facts
    jwt_facts = await orchestrator.semantic_memory.query_knowledge(
        query="JWT",
        k=10
    )

    for i in range(len(jwt_facts) - 1):
        await orchestrator.semantic_memory.link_facts(
            fact_id_1=jwt_facts[i].id,
            fact_id_2=jwt_facts[i + 1].id,
            link_type="related",
            strength=0.8
        )
```

### Example 3: Context-Aware Assistance

```python
async def context_aware_assistance(query):
    # Gather context from memories
    similar_experiences = await orchestrator.episodic_memory.recall_relevant(
        query=query,
        k=5
    )

    relevant_facts = await orchestrator.semantic_memory.query_knowledge(
        query=query,
        k=5
    )

    # Build context
    context = {
        "similar_tasks": [
            {
                "actions": ep.actions,
                "success": ep.outcomes.get("success"),
                "outcomes": ep.outcomes
            }
            for ep in similar_experiences
        ],
        "relevant_knowledge": [
            {
                "fact": f.fact,
                "confidence": f.confidence
            }
            for f in relevant_facts
        ]
    }

    # Use context to inform response
    response = await orchestrator.chat(
        query,
        context=context
    )

    return response
```

### Example 4: Memory Consolidation Workflow

```python
async def consolidation_workflow():
    # Check if consolidation needed
    stats = await orchestrator.episodic_memory.get_memory_statistics()

    if stats['total_episodes'] >= 100:
        print(f"Consolidating {stats['total_episodes']} episodes...")

        # Consolidate
        result = await orchestrator.consolidate_memories(
            strategy="hybrid",
            min_occurrences=3,
            similarity_threshold=0.6
        )

        print(f"Created {result['facts_created']} new facts")
        print(f"Updated {result['facts_updated']} existing facts")

        # Clean up consolidated episodes
        await orchestrator.episodic_memory.clear_old_episodes(days=30)

        # Verify consolidation
        new_stats = await orchestrator.episodic_memory.get_memory_statistics()
        print(f"Episodes after consolidation: {new_stats['total_episodes']}")
```

## Additional Resources

- [API Reference](../api/NEW_CAPABILITIES_API.md)
- [User Guide](../user-guide/index.md)
- [Hierarchical Planning Guide](HIERARCHICAL_PLANNING.md)
- [Dynamic Skills Guide](DYNAMIC_SKILLS.md)
- [Memory System Architecture](../architecture/overview.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
