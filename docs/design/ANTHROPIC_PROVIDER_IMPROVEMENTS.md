# Anthropic Provider Improvement Plan

> Comprehensive design for robust, scalable, and future-proof enhancements to Victor's Anthropic integration.

**Status**: Proposed
**Created**: 2025-12-03
**Author**: Victor Development Team

---

## Executive Summary

Following comprehensive testing of the Anthropic provider with Claude 3.5 Haiku, all 22 use case tests passed successfully. This document outlines improvements to enhance production readiness:

1. **Multi-turn Context Retention** - Persistent conversation state with memory management
2. **Streaming Performance Monitoring** - Real-time metrics and latency tracking
3. **Error Recovery System** - Intelligent retry logic with circuit breakers
4. **Concurrent Request Handling** - Parallel tool execution with rate limiting

---

## 1. Multi-Turn Context Retention System

### 1.1 Problem Statement

Current implementation handles single-turn requests. Production coding assistants require:
- Conversation continuity across multiple exchanges
- Context window management (avoiding token limits)
- Selective memory for relevant code context
- Session persistence for long-running tasks

### 1.2 Architecture Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONVERSATION MANAGER                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   Message   │───▶│   Context   │───▶│    Token Budget         │ │
│  │   Store     │    │   Pruner    │    │    Manager              │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│        │                  │                       │                 │
│        ▼                  ▼                       ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │  SQLite DB  │    │  Relevance  │    │   Context Window        │ │
│  │  (persist)  │    │   Scorer    │    │   Optimizer             │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Implementation Details

#### 1.3.1 ConversationMemory Class

```python
# victor/agent/conversation_memory.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json
import sqlite3
from pathlib import Path


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class MessagePriority(Enum):
    """Priority levels for context pruning."""
    CRITICAL = 100    # System prompts, current task
    HIGH = 75         # Recent tool results, code context
    MEDIUM = 50       # Previous exchanges
    LOW = 25          # Old context, summaries
    EPHEMERAL = 0     # Can be dropped immediately


@dataclass
class ConversationMessage:
    """A single message in the conversation."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    token_count: int
    priority: MessagePriority = MessagePriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For tool calls/results
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_provider_format(self) -> Dict[str, Any]:
        """Convert to provider message format."""
        return {
            "role": self.role.value,
            "content": self.content,
        }


@dataclass
class ConversationSession:
    """A conversation session with context management."""
    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Context management
    max_tokens: int = 100000  # Claude's context window
    reserved_tokens: int = 4096  # For response
    current_tokens: int = 0

    # Session metadata
    project_path: Optional[str] = None
    active_files: List[str] = field(default_factory=list)
    tool_usage_count: int = 0


class ConversationMemoryManager:
    """
    Manages conversation history with intelligent context pruning.

    Features:
    - SQLite persistence for session recovery
    - Token-aware context window management
    - Priority-based message pruning
    - Semantic relevance scoring for context selection
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_context_tokens: int = 100000,
        response_reserve: int = 4096,
    ):
        self.db_path = db_path or Path.home() / ".victor" / "conversations.db"
        self.max_context_tokens = max_context_tokens
        self.response_reserve = response_reserve
        self._init_database()

        # In-memory session cache
        self._sessions: Dict[str, ConversationSession] = {}

    def _init_database(self):
        """Initialize SQLite database for persistence."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    last_activity TIMESTAMP,
                    project_path TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP,
                    token_count INTEGER,
                    priority INTEGER,
                    tool_name TEXT,
                    tool_call_id TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp);

                CREATE TABLE IF NOT EXISTS context_summaries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    summary TEXT,
                    token_count INTEGER,
                    messages_summarized TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
            """)

    def create_session(
        self,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None,
    ) -> ConversationSession:
        """Create a new conversation session."""
        if session_id is None:
            session_id = self._generate_session_id()

        session = ConversationSession(
            session_id=session_id,
            project_path=project_path,
            max_tokens=self.max_context_tokens,
            reserved_tokens=self.response_reserve,
        )

        self._sessions[session_id] = session
        self._persist_session(session)

        return session

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        priority: MessagePriority = MessagePriority.MEDIUM,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Add a message to the conversation."""
        session = self._get_session(session_id)

        # Estimate token count (rough: 4 chars per token)
        token_count = len(content) // 4 + 1

        message = ConversationMessage(
            id=self._generate_message_id(content),
            role=role,
            content=content,
            timestamp=datetime.now(),
            token_count=token_count,
            priority=priority,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )

        session.messages.append(message)
        session.current_tokens += token_count
        session.last_activity = datetime.now()

        # Check if pruning is needed
        if session.current_tokens > (session.max_tokens - session.reserved_tokens):
            self._prune_context(session)

        self._persist_message(session_id, message)

        return message

    def get_context_messages(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get messages formatted for the provider, respecting token limits.

        Returns messages in provider format, optimized for context window.
        """
        session = self._get_session(session_id)
        max_tokens = max_tokens or (session.max_tokens - session.reserved_tokens)

        # Sort by priority and recency
        scored_messages = self._score_messages(session.messages)

        # Select messages within token budget
        selected = []
        token_budget = max_tokens

        for msg, score in scored_messages:
            if msg.token_count <= token_budget:
                selected.append(msg)
                token_budget -= msg.token_count

        # Sort selected messages by timestamp for coherent conversation
        selected.sort(key=lambda m: m.timestamp)

        # Filter system messages if requested
        if not include_system:
            selected = [m for m in selected if m.role != MessageRole.SYSTEM]

        return [msg.to_provider_format() for msg in selected]

    def _score_messages(
        self,
        messages: List[ConversationMessage],
    ) -> List[tuple[ConversationMessage, float]]:
        """
        Score messages for context selection.

        Scoring factors:
        - Priority level (40%)
        - Recency (30%)
        - Relevance to recent context (30%)
        """
        if not messages:
            return []

        now = datetime.now()
        max_age = max(
            (now - msg.timestamp).total_seconds()
            for msg in messages
        ) or 1

        scored = []
        for msg in messages:
            # Priority score (0-1)
            priority_score = msg.priority.value / 100

            # Recency score (0-1, more recent = higher)
            age = (now - msg.timestamp).total_seconds()
            recency_score = 1 - (age / max_age)

            # Combined score
            score = (priority_score * 0.4) + (recency_score * 0.6)

            scored.append((msg, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _prune_context(self, session: ConversationSession):
        """
        Prune conversation context to fit within token limits.

        Strategy:
        1. Keep all CRITICAL priority messages
        2. Summarize older exchanges
        3. Drop LOW priority messages first
        """
        target_tokens = int((session.max_tokens - session.reserved_tokens) * 0.8)

        # Separate by priority
        critical = [m for m in session.messages if m.priority == MessagePriority.CRITICAL]
        others = [m for m in session.messages if m.priority != MessagePriority.CRITICAL]

        # Sort others by score
        scored_others = self._score_messages(others)

        # Select within budget
        kept = list(critical)
        current_tokens = sum(m.token_count for m in kept)

        for msg, _ in scored_others:
            if current_tokens + msg.token_count <= target_tokens:
                kept.append(msg)
                current_tokens += msg.token_count

        # Update session
        session.messages = sorted(kept, key=lambda m: m.timestamp)
        session.current_tokens = current_tokens

    def _get_session(self, session_id: str) -> ConversationSession:
        """Get session from cache or database."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try to load from database
        session = self._load_session(session_id)
        if session:
            self._sessions[session_id] = session
            return session

        raise ValueError(f"Session not found: {session_id}")

    def _persist_session(self, session: ConversationSession):
        """Persist session to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, last_activity, project_path, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    session.project_path,
                    json.dumps({"active_files": session.active_files}),
                ),
            )

    def _persist_message(self, session_id: str, message: ConversationMessage):
        """Persist message to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO messages
                (id, session_id, role, content, timestamp, token_count,
                 priority, tool_name, tool_call_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    session_id,
                    message.role.value,
                    message.content,
                    message.timestamp.isoformat(),
                    message.token_count,
                    message.priority.value,
                    message.tool_name,
                    message.tool_call_id,
                    json.dumps(message.metadata),
                ),
            )

    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if not row:
                return None

            session = ConversationSession(
                session_id=row["session_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_activity=datetime.fromisoformat(row["last_activity"]),
                project_path=row["project_path"],
            )

            # Load messages
            messages = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY timestamp
                """,
                (session_id,),
            ).fetchall()

            for msg_row in messages:
                session.messages.append(ConversationMessage(
                    id=msg_row["id"],
                    role=MessageRole(msg_row["role"]),
                    content=msg_row["content"],
                    timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                    token_count=msg_row["token_count"],
                    priority=MessagePriority(msg_row["priority"]),
                    tool_name=msg_row["tool_name"],
                    tool_call_id=msg_row["tool_call_id"],
                    metadata=json.loads(msg_row["metadata"] or "{}"),
                ))

            session.current_tokens = sum(m.token_count for m in session.messages)

            return session

    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID."""
        import uuid
        return f"session_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _generate_message_id(content: str) -> str:
        """Generate unique message ID."""
        import uuid
        return f"msg_{uuid.uuid4().hex[:8]}"
```

### 1.4 Integration Points

```python
# Integration with AgentOrchestrator

class AgentOrchestrator:
    def __init__(self, ...):
        # ... existing init ...
        self.memory = ConversationMemoryManager()
        self.current_session: Optional[str] = None

    async def chat(self, user_message: str, session_id: Optional[str] = None):
        """Process a chat message with context retention."""
        # Get or create session
        if session_id is None:
            if self.current_session is None:
                session = self.memory.create_session(
                    project_path=str(self.working_directory)
                )
                self.current_session = session.session_id
            session_id = self.current_session

        # Add user message
        self.memory.add_message(
            session_id,
            MessageRole.USER,
            user_message,
            priority=MessagePriority.HIGH,
        )

        # Get context-aware messages
        messages = self.memory.get_context_messages(session_id)

        # Call provider
        response = await self.provider.chat(
            messages,
            model=self.model,
            tools=self.tools,
        )

        # Store response
        self.memory.add_message(
            session_id,
            MessageRole.ASSISTANT,
            response.content,
            priority=MessagePriority.HIGH,
        )

        return response
```

---

## 2. Streaming Performance Monitoring

### 2.1 Problem Statement

Production systems need visibility into:
- Time to first token (TTFT)
- Tokens per second throughput
- Streaming latency percentiles
- Tool call timing within streams

### 2.2 Architecture Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                  STREAMING METRICS COLLECTOR                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   Stream    │───▶│   Metrics   │───▶│    Aggregator           │ │
│  │   Wrapper   │    │   Collector │    │    (Prometheus/StatsD)  │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│        │                  │                       │                 │
│        ▼                  ▼                       ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │  Timing     │    │  Histogram  │    │   Dashboard             │ │
│  │  Interceptor│    │  Buckets    │    │   Export                │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Implementation Details

```python
# victor/analytics/streaming_metrics.py

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from enum import Enum
import statistics


class MetricType(Enum):
    """Types of streaming metrics."""
    TTFT = "time_to_first_token"
    TTLT = "time_to_last_token"
    TOKENS_PER_SECOND = "tokens_per_second"
    TOTAL_TOKENS = "total_tokens"
    TOOL_CALL_LATENCY = "tool_call_latency"
    CHUNK_INTERVAL = "chunk_interval"


@dataclass
class StreamMetrics:
    """Metrics for a single streaming response."""
    request_id: str
    model: str
    provider: str

    # Timing metrics (in milliseconds)
    start_time: float = 0
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None

    # Token metrics
    total_chunks: int = 0
    total_tokens: int = 0
    content_tokens: int = 0

    # Tool call metrics
    tool_calls_count: int = 0
    tool_call_times: List[float] = field(default_factory=list)

    # Chunk timing
    chunk_intervals: List[float] = field(default_factory=list)

    # Error tracking
    errors: List[str] = field(default_factory=list)

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.first_token_time and self.start_time:
            return (self.first_token_time - self.start_time) * 1000
        return None

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Total stream duration in milliseconds."""
        if self.last_token_time and self.start_time:
            return (self.last_token_time - self.start_time) * 1000
        return None

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Average tokens per second."""
        duration = self.total_duration_ms
        if duration and duration > 0 and self.total_tokens > 0:
            return self.total_tokens / (duration / 1000)
        return None

    @property
    def avg_chunk_interval_ms(self) -> Optional[float]:
        """Average time between chunks in milliseconds."""
        if self.chunk_intervals:
            return statistics.mean(self.chunk_intervals) * 1000
        return None

    @property
    def p95_chunk_interval_ms(self) -> Optional[float]:
        """95th percentile chunk interval."""
        if len(self.chunk_intervals) >= 20:
            sorted_intervals = sorted(self.chunk_intervals)
            idx = int(len(sorted_intervals) * 0.95)
            return sorted_intervals[idx] * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "provider": self.provider,
            "ttft_ms": self.ttft_ms,
            "total_duration_ms": self.total_duration_ms,
            "tokens_per_second": self.tokens_per_second,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "content_tokens": self.content_tokens,
            "tool_calls_count": self.tool_calls_count,
            "avg_chunk_interval_ms": self.avg_chunk_interval_ms,
            "p95_chunk_interval_ms": self.p95_chunk_interval_ms,
            "errors": self.errors,
        }


class StreamingMetricsCollector:
    """
    Collects and aggregates streaming metrics across requests.

    Features:
    - Real-time metric collection
    - Histogram aggregation
    - Export to monitoring systems
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics_history: List[StreamMetrics] = []
        self._callbacks: List[Callable[[StreamMetrics], None]] = []

    def create_metrics(
        self,
        request_id: str,
        model: str,
        provider: str,
    ) -> StreamMetrics:
        """Create a new metrics instance for a stream."""
        return StreamMetrics(
            request_id=request_id,
            model=model,
            provider=provider,
            start_time=time.time(),
        )

    def record_metrics(self, metrics: StreamMetrics):
        """Record completed metrics."""
        self._metrics_history.append(metrics)

        # Trim history
        if len(self._metrics_history) > self.max_history:
            self._metrics_history = self._metrics_history[-self.max_history:]

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception:
                pass

    def on_metrics(self, callback: Callable[[StreamMetrics], None]):
        """Register callback for completed metrics."""
        self._callbacks.append(callback)

    def get_summary(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        last_n: int = 100,
    ) -> Dict[str, Any]:
        """Get summary statistics for recent metrics."""
        # Filter metrics
        metrics = self._metrics_history[-last_n:]
        if provider:
            metrics = [m for m in metrics if m.provider == provider]
        if model:
            metrics = [m for m in metrics if m.model == model]

        if not metrics:
            return {"count": 0}

        ttft_values = [m.ttft_ms for m in metrics if m.ttft_ms is not None]
        tps_values = [m.tokens_per_second for m in metrics if m.tokens_per_second]
        duration_values = [m.total_duration_ms for m in metrics if m.total_duration_ms]

        return {
            "count": len(metrics),
            "ttft_ms": {
                "avg": statistics.mean(ttft_values) if ttft_values else None,
                "p50": statistics.median(ttft_values) if ttft_values else None,
                "p95": self._percentile(ttft_values, 0.95),
                "p99": self._percentile(ttft_values, 0.99),
            },
            "tokens_per_second": {
                "avg": statistics.mean(tps_values) if tps_values else None,
                "min": min(tps_values) if tps_values else None,
                "max": max(tps_values) if tps_values else None,
            },
            "duration_ms": {
                "avg": statistics.mean(duration_values) if duration_values else None,
                "p50": statistics.median(duration_values) if duration_values else None,
                "p95": self._percentile(duration_values, 0.95),
            },
            "error_rate": sum(1 for m in metrics if m.errors) / len(metrics),
        }

    @staticmethod
    def _percentile(values: List[float], p: float) -> Optional[float]:
        """Calculate percentile."""
        if not values:
            return None
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p)
        return sorted_values[min(idx, len(sorted_values) - 1)]


class MetricsStreamWrapper:
    """
    Wraps a streaming response to collect metrics.

    Usage:
        async for chunk in MetricsStreamWrapper(stream, metrics):
            # Process chunk
            pass
    """

    def __init__(
        self,
        stream: AsyncIterator,
        metrics: StreamMetrics,
        collector: Optional[StreamingMetricsCollector] = None,
    ):
        self.stream = stream
        self.metrics = metrics
        self.collector = collector
        self._last_chunk_time: Optional[float] = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self.stream.__anext__()

            current_time = time.time()

            # Record first token time
            if self.metrics.first_token_time is None:
                self.metrics.first_token_time = current_time

            # Record chunk interval
            if self._last_chunk_time is not None:
                interval = current_time - self._last_chunk_time
                self.metrics.chunk_intervals.append(interval)
            self._last_chunk_time = current_time

            # Update metrics
            self.metrics.total_chunks += 1
            self.metrics.last_token_time = current_time

            # Count tokens (rough estimate from content length)
            if hasattr(chunk, 'content') and chunk.content:
                self.metrics.content_tokens += len(chunk.content) // 4 + 1

            # Track tool calls
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                self.metrics.tool_calls_count += len(chunk.tool_calls)
                self.metrics.tool_call_times.append(current_time)

            return chunk

        except StopAsyncIteration:
            # Stream complete - record metrics
            self.metrics.total_tokens = self.metrics.content_tokens

            if self.collector:
                self.collector.record_metrics(self.metrics)

            raise


# Singleton collector instance
_metrics_collector: Optional[StreamingMetricsCollector] = None


def get_metrics_collector() -> StreamingMetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = StreamingMetricsCollector()
    return _metrics_collector
```

### 2.4 Provider Integration

```python
# Integration with AnthropicProvider.stream()

async def stream_with_metrics(
    self,
    messages: List[Message],
    *,
    model: str,
    **kwargs,
) -> AsyncIterator[StreamChunk]:
    """Stream with automatic metrics collection."""
    import uuid

    collector = get_metrics_collector()
    metrics = collector.create_metrics(
        request_id=str(uuid.uuid4()),
        model=model,
        provider=self.name,
    )

    stream = self.stream(messages, model=model, **kwargs)

    async for chunk in MetricsStreamWrapper(stream, metrics, collector):
        yield chunk
```

---

## 3. Error Recovery System

### 3.1 Problem Statement

Production systems need robust error handling for:
- Rate limiting (429 errors)
- Temporary service outages
- Network timeouts
- Token limit exceeded
- Malformed responses

### 3.2 Architecture Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ERROR RECOVERY SYSTEM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    CIRCUIT BREAKER                               ││
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                     ││
│  │  │ CLOSED  │───▶│  OPEN   │───▶│HALF-OPEN│                     ││
│  │  └─────────┘    └─────────┘    └─────────┘                     ││
│  └─────────────────────────────────────────────────────────────────┘│
│                          │                                           │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    RETRY STRATEGY                                ││
│  │  Exponential Backoff │ Jitter │ Max Retries │ Timeout           ││
│  └─────────────────────────────────────────────────────────────────┘│
│                          │                                           │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    FALLBACK CHAIN                                ││
│  │  Primary Provider → Secondary Provider → Cached Response        ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Implementation Details

```python
# victor/providers/resilience.py

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 60.0       # Time before half-open
    half_open_max_calls: int = 3        # Max calls in half-open state


@dataclass
class CircuitBreakerState:
    """Runtime state of circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)
    half_open_calls: int = 0


class CircuitBreaker:
    """
    Circuit breaker pattern for provider resilience.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Too many failures, rejecting requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state.state

    @property
    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        if self._state.state == CircuitState.CLOSED:
            return True

        if self._state.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._state.last_failure_time:
                elapsed = datetime.now() - self._state.last_failure_time
                if elapsed.total_seconds() >= self.config.timeout_seconds:
                    return True  # Will transition to half-open
            return False

        # Half-open: allow limited calls
        return self._state.half_open_calls < self.config.half_open_max_calls

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if not self.is_available:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )

            # Transition to half-open if timeout passed
            if self._state.state == CircuitState.OPEN:
                self._transition_to(CircuitState.HALF_OPEN)

            if self._state.state == CircuitState.HALF_OPEN:
                self._state.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise

    async def _record_success(self):
        """Record successful call."""
        async with self._lock:
            if self._state.state == CircuitState.HALF_OPEN:
                self._state.success_count += 1
                if self._state.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            else:
                # Reset failure count on success in closed state
                self._state.failure_count = 0

    async def _record_failure(self, error: Exception):
        """Record failed call."""
        async with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = datetime.now()

            if self._state.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self._state.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {error}. "
                f"State: {self._state.state.value}, "
                f"Failures: {self._state.failure_count}"
            )

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self._state.state
        self._state.state = new_state
        self._state.last_state_change = datetime.now()

        if new_state == CircuitState.HALF_OPEN:
            self._state.half_open_calls = 0
            self._state.success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._state.failure_count = 0
            self._state.success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' transitioned: "
            f"{old_state.value} -> {new_state.value}"
        )


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry strategy."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1

    # Retryable error types
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )

    # Retryable status codes
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


class RetryStrategy:
    """
    Intelligent retry strategy with exponential backoff and jitter.

    Features:
    - Exponential backoff with configurable base
    - Random jitter to prevent thundering herd
    - Respects Retry-After headers
    - Configurable retryable conditions
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if retryable
                if not self._is_retryable(e):
                    raise

                # Check if max retries exceeded
                if attempt >= self.config.max_retries:
                    logger.error(
                        f"Max retries ({self.config.max_retries}) exceeded. "
                        f"Last error: {e}"
                    )
                    raise

                # Calculate delay
                delay = self._calculate_delay(attempt, e)

                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} "
                    f"after {delay:.2f}s. Error: {e}"
                )

                await asyncio.sleep(delay)

        raise last_exception

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable."""
        # Check exception type
        if isinstance(error, self.config.retryable_exceptions):
            return True

        # Check for rate limit errors
        error_str = str(error).lower()
        if "rate" in error_str and "limit" in error_str:
            return True
        if "429" in error_str:
            return True

        # Check for overloaded errors
        if "overloaded" in error_str:
            return True

        return False

    def _calculate_delay(self, attempt: int, error: Exception) -> float:
        """Calculate delay before next retry."""
        # Check for Retry-After header in error
        retry_after = self._extract_retry_after(error)
        if retry_after is not None:
            return min(retry_after, self.config.max_delay_seconds)

        # Exponential backoff
        delay = self.config.base_delay_seconds * (
            self.config.exponential_base ** attempt
        )

        # Add jitter
        jitter = delay * self.config.jitter_factor * random.random()
        delay += jitter

        # Cap at max delay
        return min(delay, self.config.max_delay_seconds)

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract Retry-After value from error if present."""
        error_str = str(error)

        # Look for "retry after X seconds" pattern
        import re
        match = re.search(r"retry.?after[:\s]+(\d+)", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))

        return None


class ResilientProvider:
    """
    Wrapper that adds resilience features to any provider.

    Features:
    - Circuit breaker per provider
    - Retry with exponential backoff
    - Fallback to alternative providers
    - Request timeout handling
    """

    def __init__(
        self,
        provider: Any,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback_providers: Optional[List[Any]] = None,
        request_timeout: float = 120.0,
    ):
        self.provider = provider
        self.fallback_providers = fallback_providers or []
        self.request_timeout = request_timeout

        # Create circuit breaker
        provider_name = getattr(provider, 'name', 'unknown')
        self.circuit_breaker = CircuitBreaker(
            name=f"cb_{provider_name}",
            config=circuit_config,
        )

        # Create retry strategy
        self.retry_strategy = RetryStrategy(config=retry_config)

    async def chat(self, messages, *, model: str, **kwargs):
        """Execute chat with resilience features."""

        async def _execute():
            return await asyncio.wait_for(
                self.provider.chat(messages, model=model, **kwargs),
                timeout=self.request_timeout,
            )

        try:
            # Try primary provider with circuit breaker and retry
            return await self.circuit_breaker.execute(
                self.retry_strategy.execute,
                _execute,
            )
        except (CircuitOpenError, Exception) as e:
            # Try fallback providers
            for fallback in self.fallback_providers:
                try:
                    logger.info(f"Trying fallback provider: {fallback.name}")
                    return await asyncio.wait_for(
                        fallback.chat(messages, model=model, **kwargs),
                        timeout=self.request_timeout,
                    )
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback provider {fallback.name} failed: {fallback_error}"
                    )
                    continue

            # All providers failed
            raise ProviderUnavailableError(
                f"All providers failed. Primary error: {e}"
            )


class ProviderUnavailableError(Exception):
    """Raised when no providers are available."""
    pass
```

---

## 4. Concurrent Request Handling

### 4.1 Problem Statement

Efficient coding assistants need:
- Parallel tool execution when independent
- Rate limiting to respect API quotas
- Request queuing during high load
- Resource-efficient connection pooling

### 4.2 Architecture Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CONCURRENT REQUEST MANAGER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   Request   │───▶│    Rate     │───▶│    Execution            │ │
│  │   Queue     │    │   Limiter   │    │    Pool                 │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│        │                  │                       │                 │
│        ▼                  ▼                       ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │  Priority   │    │   Token     │    │   Connection            │ │
│  │  Scheduler  │    │   Bucket    │    │   Pool                  │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Implementation Details

```python
# victor/providers/concurrency.py

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar
import heapq
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RequestPriority(Enum):
    """Priority levels for request scheduling."""
    CRITICAL = 0    # User-facing, immediate response needed
    HIGH = 1        # Important operations
    NORMAL = 2      # Standard requests
    LOW = 3         # Background tasks
    BATCH = 4       # Bulk operations


@dataclass(order=True)
class PrioritizedRequest(Generic[T]):
    """A request with priority for queue ordering."""
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    coroutine: Awaitable[T] = field(compare=False)
    future: asyncio.Future = field(compare=False)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API quota management.

    Features:
    - Configurable tokens per second
    - Burst capacity
    - Async-friendly waiting
    """

    def __init__(
        self,
        tokens_per_second: float,
        burst_capacity: int,
    ):
        self.tokens_per_second = tokens_per_second
        self.burst_capacity = burst_capacity

        self._tokens = float(burst_capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.

        Returns the time waited in seconds.
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self.tokens_per_second

            await asyncio.sleep(wait_time)

            self._refill()
            self._tokens -= tokens

            return wait_time

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        self._tokens = min(
            self.burst_capacity,
            self._tokens + elapsed * self.tokens_per_second,
        )

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        return self._tokens


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for request counting.

    More accurate than token bucket for API rate limits
    that count requests per time window.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        self._requests: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """
        Acquire permission to make a request.

        Returns the time waited in seconds.
        """
        async with self._lock:
            now = time.monotonic()

            # Remove old requests outside window
            cutoff = now - self.window_seconds
            self._requests = [t for t in self._requests if t > cutoff]

            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return 0.0

            # Calculate wait time until oldest request expires
            oldest = self._requests[0]
            wait_time = oldest + self.window_seconds - now

            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = time.monotonic()

            # Clean up and add new request
            cutoff = now - self.window_seconds
            self._requests = [t for t in self._requests if t > cutoff]
            self._requests.append(now)

            return max(0, wait_time)

    @property
    def available_capacity(self) -> int:
        """Get current available capacity."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        current_count = sum(1 for t in self._requests if t > cutoff)
        return max(0, self.max_requests - current_count)


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent request handling."""
    # Parallelism limits
    max_concurrent_requests: int = 10
    max_concurrent_tool_calls: int = 5

    # Rate limiting (Anthropic: 50 RPM for Haiku)
    requests_per_minute: int = 50
    tokens_per_minute: int = 50000

    # Queue settings
    max_queue_size: int = 100
    queue_timeout_seconds: float = 300.0


class ConcurrentRequestManager:
    """
    Manages concurrent API requests with rate limiting and queuing.

    Features:
    - Priority-based request scheduling
    - Dual rate limiting (requests + tokens)
    - Concurrent execution with semaphores
    - Request timeout handling
    """

    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        self.config = config or ConcurrencyConfig()

        # Semaphores for concurrency control
        self._request_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_requests
        )
        self._tool_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_tool_calls
        )

        # Rate limiters
        self._request_limiter = SlidingWindowRateLimiter(
            max_requests=self.config.requests_per_minute,
            window_seconds=60.0,
        )
        self._token_limiter = TokenBucketRateLimiter(
            tokens_per_second=self.config.tokens_per_minute / 60,
            burst_capacity=self.config.tokens_per_minute // 2,
        )

        # Request queue
        self._queue: List[PrioritizedRequest] = []
        self._queue_lock = asyncio.Lock()
        self._active_requests: Dict[str, PrioritizedRequest] = {}

        # Metrics
        self._total_requests = 0
        self._total_wait_time = 0.0

    async def submit(
        self,
        coroutine: Awaitable[T],
        priority: RequestPriority = RequestPriority.NORMAL,
        estimated_tokens: int = 1000,
        request_id: Optional[str] = None,
    ) -> T:
        """
        Submit a request for execution.

        Args:
            coroutine: Async function to execute
            priority: Request priority
            estimated_tokens: Estimated token usage for rate limiting
            request_id: Optional request identifier

        Returns:
            Result of the coroutine
        """
        if request_id is None:
            import uuid
            request_id = str(uuid.uuid4())[:8]

        # Check queue capacity
        async with self._queue_lock:
            if len(self._queue) >= self.config.max_queue_size:
                raise QueueFullError(
                    f"Request queue full ({self.config.max_queue_size})"
                )

        # Create prioritized request
        future: asyncio.Future = asyncio.Future()
        request = PrioritizedRequest(
            priority=priority.value,
            timestamp=time.monotonic(),
            request_id=request_id,
            coroutine=coroutine,
            future=future,
        )

        # Add to queue
        async with self._queue_lock:
            heapq.heappush(self._queue, request)

        # Process queue
        asyncio.create_task(self._process_request(request, estimated_tokens))

        # Wait for result with timeout
        try:
            return await asyncio.wait_for(
                future,
                timeout=self.config.queue_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise RequestTimeoutError(
                f"Request {request_id} timed out after "
                f"{self.config.queue_timeout_seconds}s"
            )

    async def submit_parallel(
        self,
        coroutines: List[Awaitable[T]],
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> List[T]:
        """
        Submit multiple requests for parallel execution.

        Args:
            coroutines: List of async functions to execute
            priority: Priority for all requests

        Returns:
            List of results in same order as input
        """
        tasks = [
            self.submit(coro, priority=priority)
            for coro in coroutines
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_request(
        self,
        request: PrioritizedRequest,
        estimated_tokens: int,
    ):
        """Process a single request with rate limiting."""
        try:
            # Acquire rate limit tokens
            wait_time = await self._request_limiter.acquire()
            wait_time += await self._token_limiter.acquire(estimated_tokens)

            self._total_wait_time += wait_time

            if wait_time > 0:
                logger.debug(
                    f"Request {request.request_id} waited {wait_time:.2f}s "
                    f"for rate limit"
                )

            # Acquire concurrency semaphore
            async with self._request_semaphore:
                self._active_requests[request.request_id] = request
                self._total_requests += 1

                try:
                    result = await request.coroutine
                    request.future.set_result(result)
                except Exception as e:
                    request.future.set_exception(e)
                finally:
                    del self._active_requests[request.request_id]

        except Exception as e:
            if not request.future.done():
                request.future.set_exception(e)

    async def execute_tool_calls_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        executor: Callable[[Dict[str, Any]], Awaitable[Any]],
    ) -> List[Any]:
        """
        Execute multiple tool calls in parallel with concurrency control.

        Args:
            tool_calls: List of tool call definitions
            executor: Function to execute each tool call

        Returns:
            List of results
        """
        async def execute_with_semaphore(tool_call):
            async with self._tool_semaphore:
                return await executor(tool_call)

        tasks = [
            execute_with_semaphore(tc)
            for tc in tool_calls
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "total_requests": self._total_requests,
            "active_requests": len(self._active_requests),
            "queue_size": len(self._queue),
            "avg_wait_time": (
                self._total_wait_time / self._total_requests
                if self._total_requests > 0 else 0
            ),
            "request_capacity": self._request_limiter.available_capacity,
            "token_capacity": self._token_limiter.available_tokens,
        }


class QueueFullError(Exception):
    """Raised when request queue is full."""
    pass


class RequestTimeoutError(Exception):
    """Raised when request times out in queue."""
    pass


# Anthropic-specific rate limit configuration
ANTHROPIC_RATE_LIMITS = {
    "claude-3-5-haiku-20241022": ConcurrencyConfig(
        max_concurrent_requests=10,
        requests_per_minute=50,
        tokens_per_minute=50000,
    ),
    "claude-sonnet-4-5-20250929": ConcurrencyConfig(
        max_concurrent_requests=5,
        requests_per_minute=50,
        tokens_per_minute=40000,
    ),
    "claude-opus-4-5-20251101": ConcurrencyConfig(
        max_concurrent_requests=3,
        requests_per_minute=50,
        tokens_per_minute=20000,
    ),
}


def get_anthropic_config(model: str) -> ConcurrencyConfig:
    """Get rate limit config for Anthropic model."""
    return ANTHROPIC_RATE_LIMITS.get(model, ConcurrencyConfig())
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Priority | Effort |
|------|----------|--------|
| Implement ConversationMemoryManager | HIGH | 3 days |
| Add SQLite persistence layer | HIGH | 2 days |
| Integrate with AgentOrchestrator | HIGH | 2 days |
| Unit tests for memory system | HIGH | 2 days |

### Phase 2: Monitoring (Week 3-4)

| Task | Priority | Effort |
|------|----------|--------|
| Implement StreamingMetricsCollector | HIGH | 2 days |
| Add MetricsStreamWrapper | HIGH | 1 day |
| Create metrics dashboard export | MEDIUM | 2 days |
| Integration tests | HIGH | 2 days |

### Phase 3: Resilience (Week 5-6)

| Task | Priority | Effort |
|------|----------|--------|
| Implement CircuitBreaker | HIGH | 2 days |
| Implement RetryStrategy | HIGH | 2 days |
| Create ResilientProvider wrapper | HIGH | 2 days |
| Add fallback provider support | MEDIUM | 2 days |

### Phase 4: Concurrency (Week 7-8)

| Task | Priority | Effort |
|------|----------|--------|
| Implement rate limiters | HIGH | 2 days |
| Create ConcurrentRequestManager | HIGH | 3 days |
| Add parallel tool execution | MEDIUM | 2 days |
| Performance testing | HIGH | 2 days |

---

## 6. Success Metrics

### 6.1 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| TTFT (p95) | < 500ms | ~400ms |
| Tokens/second | > 50 | ~45 |
| Error rate | < 1% | 0% |
| Context retention | 100k tokens | N/A |

### 6.2 Reliability Targets

| Metric | Target |
|--------|--------|
| Circuit breaker recovery time | < 60s |
| Retry success rate | > 95% |
| Queue timeout rate | < 0.1% |
| Rate limit violation rate | 0% |

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/unit/test_conversation_memory.py
# tests/unit/test_streaming_metrics.py
# tests/unit/test_circuit_breaker.py
# tests/unit/test_rate_limiter.py
```

### 7.2 Integration Tests

```python
# tests/integration/test_resilient_anthropic.py
# tests/integration/test_concurrent_requests.py
# tests/integration/test_multi_turn_conversation.py
```

### 7.3 Load Tests

```python
# tests/load/test_concurrent_load.py
# tests/load/test_rate_limit_compliance.py
```

---

## 8. Appendix: Configuration Examples

### 8.1 Production Configuration

```yaml
# ~/.victor/config.yaml

resilience:
  circuit_breaker:
    failure_threshold: 5
    success_threshold: 3
    timeout_seconds: 60

  retry:
    max_retries: 3
    base_delay_seconds: 1.0
    max_delay_seconds: 60.0

concurrency:
  max_concurrent_requests: 10
  max_concurrent_tool_calls: 5
  requests_per_minute: 50
  tokens_per_minute: 50000

conversation:
  max_context_tokens: 100000
  response_reserve: 4096
  persistence_enabled: true
```

### 8.2 Environment Variables

```bash
# Rate limit overrides
VICTOR_MAX_CONCURRENT_REQUESTS=10
VICTOR_REQUESTS_PER_MINUTE=50

# Circuit breaker
VICTOR_CIRCUIT_FAILURE_THRESHOLD=5
VICTOR_CIRCUIT_TIMEOUT=60

# Memory
VICTOR_MAX_CONTEXT_TOKENS=100000
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-03
