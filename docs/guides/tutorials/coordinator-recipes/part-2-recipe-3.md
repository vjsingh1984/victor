# Coordinator Recipes - Part 2

**Part 2 of 4:** Recipe 3 (Analytics Export)

---

## Navigation

- [Part 1: Recipes 1-2](part-1-recipes-1-2.md)
- **[Part 2: Recipe 3](#)** (Current)
- [Part 3: Recipes 4-6](part-3-recipes-4-6.md)
- [Part 4: Recipes 7-9](part-4-recipes-7-9.md)
- [**Complete Guide**](../coordinator_recipes.md)

---
            batch_size: Number of events to batch per insert
        """
        self.connection_string = connection_string
        self.batch_size = batch_size
        self._pool = None

    async def _get_pool(self):
        """Lazy-load connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.connection_string)
        return self._pool

    async def export(self, events: List[AnalyticsEvent]) -> ExportResult:
        """Export events to PostgreSQL."""
        if not events:
            return ExportResult(success=True, exported_count=0)

        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                # Batch insert events
                await conn.executemany(
                    """
                    INSERT INTO analytics_events
                    (session_id, event_type, event_data, timestamp, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (session_id, timestamp) DO NOTHING
                    """,
                    [
                        (
                            e.session_id,
                            e.type,
                            json.dumps(e.data),
                            e.timestamp
                        )
                        for e in events
                    ]
                )

            return ExportResult(
                success=True,
                exported_count=len(events)
            )
        except Exception as e:
            return ExportResult(
                success=False,
                error=str(e),
                exported_count=0
            )

    async def shutdown(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
```

#### Step 2: Create Database Schema

```sql
-- migrations/001_create_analytics_table.sql

CREATE TABLE IF NOT EXISTS analytics_events (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_event UNIQUE (session_id, timestamp)
);

-- Indexes for common queries
CREATE INDEX idx_session_id ON analytics_events(session_id);
CREATE INDEX idx_event_type ON analytics_events(event_type);
CREATE INDEX idx_timestamp ON analytics_events(timestamp);

-- Partitioning for large datasets (optional)
-- CREATE TABLE analytics_events_2025_01 PARTITION OF analytics_events
-- FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

#### Step 3: Use with Orchestrator

```python
# main.py
import os
from victor.agent.coordinators import AnalyticsCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create database exporter
db_exporter = PostgreSQLAnalyticsExporter(
    connection_string=os.getenv("DATABASE_URL"),
    batch_size=100
)

# Create analytics coordinator
analytics_coordinator = AnalyticsCoordinator(exporters=[
    db_exporter,
    ConsoleAnalyticsExporter(),  # Also log to console
])

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(enable_analytics=True),
    provider=provider,
    model=model,
    _analytics_coordinator=analytics_coordinator
)
```

### Testing

```python
import pytest
import asyncpg
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_postgresql_analytics_exporter():
    exporter = PostgreSQLAnalyticsExporter(
        connection_string="postgresql://localhost/test"
    )

    events = [
        AnalyticsEvent(
            session_id="test-session",
            type="tool_call",
            data={"tool": "read"},
            timestamp="2025-01-13T00:00:00Z"
        )
    ]

    # Mock database connection
    with patch.object(exporter, '_get_pool') as mock_pool:
        mock_conn = AsyncMock()
        mock_pool.return_value.__aenter__.return_value = mock_conn

        result = await exporter.export(events)

        assert result.success
        assert result.exported_count == 1
        mock_conn.executemany.assert_called_once()
```

### Production Considerations

- **Connection pooling**: Use connection pools for performance
- **Batch inserts**: Insert events in batches for efficiency
- **Retry logic**: Implement retry logic for transient failures
- **Dead letter queue**: Store failed events for later retry
- **Partitioning**: Partition tables by date for large datasets
- **Retention policy**: Implement data retention policies
- **Monitoring**: Monitor database performance and connection pool health

---

## Recipe 4: Implement Smart Context Compaction

### Problem Statement

You want to intelligently compact conversation context to maintain the most important information while reducing token usage.

### Solution Overview

Create a custom `BaseCompactionStrategy` that uses semantic analysis to preserve important messages.

### Step-by-Step Instructions

#### Step 1: Create Semantic Compaction Strategy

```python
# compaction_strategies.py
from typing import List
from victor.protocols import Context
from victor.agent.coordinators.context_coordinator import BaseCompactionStrategy

class SemanticCompactionStrategy(BaseCompactionStrategy):
    """Use semantic similarity to preserve important messages."""

    def __init__(self, embedding_service, target_ratio: float = 0.5):
        """
        Args:
            embedding_service: Service for computing embeddings
            target_ratio: Target size ratio (0.5 = keep 50% of messages)
        """
        self.embedding_service = embedding_service
        self.target_ratio = target_ratio

    async def compact(self, context: Context) -> Context:
        """Compaction using semantic clustering."""
        messages = context.messages

        if len(messages) <= 10:
            return context  # No compaction needed

        # Always keep system messages
        system_messages = [m for m in messages if m.role == "system"]
        user_assistant_messages = [m for m in messages if m.role != "system"]

        # Calculate target count
        target_count = max(
            10,  # Keep at least 10 messages
            int(len(user_assistant_messages) * self.target_ratio)
        )

        # Compute embeddings
        embeddings = await self._compute_embeddings(user_assistant_messages)

        # Select diverse messages
        selected_indices = await self._select_diverse_messages(
            embeddings,
            target_count
        )

        selected_messages = [user_assistant_messages[i] for i in selected_indices]

        # Combine system + selected messages
        return Context(messages=system_messages + selected_messages)

    async def _compute_embeddings(self, messages: List) -> List[List[float]]:
        """Compute embeddings for messages."""
        texts = [m.content for m in messages]
        return await self.embedding_service.embed_batch(texts)

    async def _select_diverse_messages(
        self,
        embeddings: List[List[float]],
        target_count: int
    ) -> List[int]:
        """Select diverse messages using k-means clustering."""
        from sklearn.cluster import KMeans
        import numpy as np

        # Convert to numpy array
        embedding_array = np.array(embeddings)

        # Perform k-means clustering
        n_clusters = min(target_count, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embedding_array)

        # Select message closest to centroid for each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_embeddings = embedding_array[cluster_mask]

            # Find closest to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = np.argmin(distances)

            # Map back to original indices
            cluster_indices = np.where(cluster_mask)[0]
            selected_indices.append(cluster_indices[closest_idx])

        return sorted(selected_indices)

class RecentImportantCompactionStrategy(BaseCompactionStrategy):
    """Keep recent messages and important keywords."""

    def __init__(self, keep_last_n: int = 10, keywords: List[str] = None):
        self.keep_last_n = keep_last_n
        self.keywords = keywords or ["error", "bug", "issue", "fix", "important"]

    async def compact(self, context: Context) -> Context:
        """Keep recent messages + keyword matches."""
        messages = context.messages

        if len(messages) <= self.keep_last_n:
            return context

        # Always keep system messages
        system_messages = [m for m in messages if m.role == "system"]
        user_assistant_messages = [m for m in messages if m.role != "system"]

        # Keep messages with keywords
        keyword_messages = [
            m for m in user_assistant_messages
            if any(kw in m.content.lower() for kw in self.keywords)
        ]

        # Keep recent messages
        recent_messages = user_assistant_messages[-self.keep_last_n:]

        # Combine and deduplicate
        all_messages = keyword_messages + recent_messages
        unique_messages = list({id(m): m for m in all_messages}.values())

        # Sort by original order
        message_order = {m: i for i, m in enumerate(user_assistant_messages)}
        unique_messages.sort(key=lambda m: message_order.get(m, float('inf')))

        return Context(messages=system_messages + unique_messages)
```

#### Step 2: Use with Orchestrator

```python
# main.py
from victor.agent.coordinators import ContextCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create compaction strategy
compaction_strategy = SemanticCompactionStrategy(
    embedding_service=your_embedding_service,
    target_ratio=0.6  # Keep 60% of messages
)

# Create context coordinator
context_coordinator = ContextCoordinator(
    compaction_strategy=compaction_strategy
)

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(
        context_compaction_threshold=0.8,  # Trigger at 80% of context limit
    ),
    provider=provider,
    model=model,
    _context_coordinator=context_coordinator
)
```

### Testing

```python
@pytest.mark.asyncio
async def test_semantic_compaction():
    from unittest.mock import AsyncMock

    strategy = SemanticCompactionStrategy(
        embedding_service=AsyncMock(),
        target_ratio=0.5
    )

    # Create test context
    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content=f"Message {i}"),
        Message(role="assistant", content=f"Response {i}"),
        for i in range(20)
    ]
    context = Context(messages=messages)

    # Mock embeddings
    strategy.embedding_service.embed_batch = AsyncMock(
        return_value=[[0.1] * 1536 for _ in messages]
    )

    compacted = await strategy.compact(context)

    # Should have fewer messages
    assert len(compacted.messages) < len(messages)
    # Should still have system message
    assert any(m.role == "system" for m in compacted.messages)
```

### Production Considerations

- **Embedding cost**: Semantic compaction requires embedding computation (API costs)
- **Latency**: Clustering adds latency; consider caching
- **Fallback**: Have fallback to simpler strategies if semantic compaction fails
- **Tuning**: Tune `target_ratio` based on your use case
- **Performance**: For high-volume scenarios, consider approximate nearest neighbor algorithms

---

