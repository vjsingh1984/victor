# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Intelligent system prompt builder with embedding-based context selection.

Architecture (Strategy + Observer + Builder patterns):
- Uses conversation embeddings to select relevant historical context
- Learns from user feedback via reinforcement learning signals
- Adapts prompts per-profile based on model capabilities
- Supports cold start (lazy) and warm cache (background) modes

Key Features:
1. Embedding-Based Context Selection:
   - Retrieves semantically similar past interactions
   - Filters by task type, success rate, and recency
   - Weighs context by relevance score

2. Profile-Specific Learning:
   - Tracks per-model performance metrics
   - Adjusts prompt style based on model strengths/weaknesses
   - Learns optimal tool budgets and mode transitions

3. Adaptive Prompt Generation:
   - Task-type-specific prompt templates
   - Dynamic grounding rules based on model reliability
   - Success-weighted example selection

4. Cold/Warm Cache Management:
   - Lazy evaluation on first use (cold start)
   - Background embedding refresh (warm cache)
   - Automatic cache invalidation on model switch

Usage:
    builder = await IntelligentPromptBuilder.create(
        provider_name="ollama",
        model="qwen2.5:32b",
        profile_name="local-qwen",
    )

    prompt = await builder.build(
        task="analyze the authentication module",
        task_type="analysis",
        conversation_history=messages,
    )
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
    from victor.storage.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Prompt generation strategies."""

    MINIMAL = "minimal"  # Cloud models - minimal guidance
    STRUCTURED = "structured"  # Capable local models - structured guidance
    STRICT = "strict"  # Less capable models - strict rules
    ADAPTIVE = "adaptive"  # Dynamic based on learned performance


class CacheState(Enum):
    """Embedding cache states."""

    COLD = "cold"  # No embeddings loaded, lazy on demand
    WARMING = "warming"  # Background loading in progress
    WARM = "warm"  # Embeddings ready for fast retrieval
    STALE = "stale"  # Cache needs refresh


@dataclass
class ProfileMetrics:
    """Performance metrics for a model profile.

    Used for reinforcement learning to optimize prompts.
    """

    profile_name: str
    provider: str
    model: str

    # Success metrics
    total_requests: int = 0
    successful_completions: int = 0
    tool_call_success_rate: float = 0.0
    grounding_accuracy: float = 0.0

    # Response quality
    avg_quality_score: float = 0.5
    avg_response_time_ms: float = 0.0
    avg_token_usage: int = 0

    # Tool usage patterns
    avg_tool_calls_per_request: float = 0.0
    tool_budget_adherence: float = 1.0  # 1.0 = stays within budget

    # Mode transition efficiency
    mode_transition_success: float = 0.0
    optimal_tool_budget: int = 10

    # Learned preferences
    prefers_structured_prompts: bool = False
    needs_strict_grounding: bool = True
    supports_parallel_tools: bool = False

    # Temporal tracking
    last_updated: datetime = field(default_factory=datetime.now)

    def update_from_interaction(
        self,
        success: bool,
        quality_score: float,
        response_time_ms: float,
        tool_calls: int,
        tool_budget: int,
        grounded: bool,
    ) -> None:
        """Update metrics from an interaction using exponential moving average."""
        alpha = 0.1
        beta = 1 - alpha  # Pre-calculate for efficiency

        self.total_requests += 1
        if success:
            self.successful_completions += 1

        # Batch EMA updates
        current_rate = self.successful_completions / self.total_requests
        adherence = 1.0 if tool_calls <= tool_budget else tool_budget / tool_calls
        grounding_score = float(grounded)

        self.tool_call_success_rate = beta * self.tool_call_success_rate + alpha * current_rate
        self.avg_quality_score = beta * self.avg_quality_score + alpha * quality_score
        self.avg_response_time_ms = beta * self.avg_response_time_ms + alpha * response_time_ms
        self.avg_tool_calls_per_request = (
            beta * self.avg_tool_calls_per_request + alpha * tool_calls
        )
        self.tool_budget_adherence = beta * self.tool_budget_adherence + alpha * adherence
        self.grounding_accuracy = beta * self.grounding_accuracy + alpha * grounding_score

        # Update derived metrics
        if success and tool_calls > 0:
            self.optimal_tool_budget = int(beta * self.optimal_tool_budget + alpha * tool_calls)
        if success and quality_score > 0.7:
            self.prefers_structured_prompts = tool_calls > 3
        self.needs_strict_grounding = self.grounding_accuracy < 0.8
        self.last_updated = datetime.now()

    def get_recommended_strategy(self) -> PromptStrategy:
        """Get recommended prompt strategy based on learned metrics."""
        if self.grounding_accuracy > 0.9 and self.tool_call_success_rate > 0.9:
            return PromptStrategy.MINIMAL
        elif self.grounding_accuracy > 0.7 and self.tool_call_success_rate > 0.7:
            return PromptStrategy.STRUCTURED
        elif self.total_requests < 10:
            return PromptStrategy.ADAPTIVE  # Not enough data
        else:
            return PromptStrategy.STRICT


@dataclass
class ContextFragment:
    """A fragment of relevant context from conversation history."""

    content: str
    similarity: float
    task_type: str
    was_successful: bool
    timestamp: datetime
    source: str  # "conversation" | "example" | "documentation"

    @property
    def relevance_score(self) -> float:
        """Calculate weighted relevance score."""
        # Combine similarity, success, and recency
        recency_weight = 1.0 / (1.0 + (datetime.now() - self.timestamp).days / 7)
        success_weight = 1.2 if self.was_successful else 0.8
        return self.similarity * success_weight * recency_weight


@dataclass
class PromptContext:
    """Context for prompt generation."""

    task: str
    task_type: str
    profile_name: str
    provider: str
    model: str

    # Historical context
    relevant_fragments: List[ContextFragment] = field(default_factory=list)

    # Tool context
    available_tools: List[str] = field(default_factory=list)
    recommended_tool_budget: int = 10

    # Mode context
    current_mode: str = "explore"
    iteration_budget: int = 20
    continuation_context: Optional[str] = None

    # Profile metrics
    profile_metrics: Optional[ProfileMetrics] = None


class ProfileLearningStore:
    """SQLite-backed store for profile learning metrics.

    Persists learned profile behaviors for reinforcement learning.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the learning store."""
        if db_path is None:
            from victor.config.settings import get_project_paths

            paths = get_project_paths()
            db_path = paths.project_victor_dir / "profile_learning.db"

        self.db_path = db_path
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure database tables exist."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_metrics (
                    profile_name TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    response_time_ms REAL NOT NULL,
                    tool_calls INTEGER NOT NULL,
                    tool_budget INTEGER NOT NULL,
                    grounded INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interaction_profile
                ON interaction_history(profile_name, timestamp)
            """
            )

        self._initialized = True

    def save_metrics(self, metrics: ProfileMetrics) -> None:
        """Save profile metrics to database."""
        self._ensure_initialized()

        # Use __dict__ and filter out non-serializable fields for efficiency
        metrics_dict = {
            k: v
            for k, v in metrics.__dict__.items()
            if k not in ("profile_name", "provider", "model", "last_updated")
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO profile_metrics "
                "(profile_name, provider, model, metrics_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    metrics.profile_name,
                    metrics.provider,
                    metrics.model,
                    json.dumps(metrics_dict, default=str),
                    datetime.now().isoformat(),
                ),
            )

    def load_metrics(self, profile_name: str, provider: str, model: str) -> ProfileMetrics:
        """Load profile metrics from database."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT metrics_json, updated_at FROM profile_metrics
                WHERE profile_name = ?
            """,
                (profile_name,),
            ).fetchone()

            if row:
                metrics_dict = json.loads(row["metrics_json"])
                return ProfileMetrics(
                    profile_name=profile_name,
                    provider=provider,
                    model=model,
                    total_requests=metrics_dict.get("total_requests", 0),
                    successful_completions=metrics_dict.get("successful_completions", 0),
                    tool_call_success_rate=metrics_dict.get("tool_call_success_rate", 0.0),
                    grounding_accuracy=metrics_dict.get("grounding_accuracy", 0.0),
                    avg_quality_score=metrics_dict.get("avg_quality_score", 0.5),
                    avg_response_time_ms=metrics_dict.get("avg_response_time_ms", 0.0),
                    avg_token_usage=metrics_dict.get("avg_token_usage", 0),
                    avg_tool_calls_per_request=metrics_dict.get("avg_tool_calls_per_request", 0.0),
                    tool_budget_adherence=metrics_dict.get("tool_budget_adherence", 1.0),
                    mode_transition_success=metrics_dict.get("mode_transition_success", 0.0),
                    optimal_tool_budget=metrics_dict.get("optimal_tool_budget", 10),
                    prefers_structured_prompts=metrics_dict.get(
                        "prefers_structured_prompts", False
                    ),
                    needs_strict_grounding=metrics_dict.get("needs_strict_grounding", True),
                    supports_parallel_tools=metrics_dict.get("supports_parallel_tools", False),
                    last_updated=datetime.fromisoformat(row["updated_at"]),
                )

        # Return new metrics if not found
        return ProfileMetrics(
            profile_name=profile_name,
            provider=provider,
            model=model,
        )

    def record_interaction(
        self,
        profile_name: str,
        task_type: str,
        success: bool,
        quality_score: float,
        response_time_ms: float,
        tool_calls: int,
        tool_budget: int,
        grounded: bool,
    ) -> None:
        """Record an interaction for learning."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO interaction_history
                (profile_name, task_type, success, quality_score, response_time_ms,
                 tool_calls, tool_budget, grounded, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    profile_name,
                    task_type,
                    1 if success else 0,
                    quality_score,
                    response_time_ms,
                    tool_calls,
                    tool_budget,
                    1 if grounded else 0,
                    datetime.now().isoformat(),
                ),
            )

    def get_recent_interactions(
        self,
        profile_name: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent interactions for a profile."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM interaction_history
                WHERE profile_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (profile_name, limit),
            ).fetchall()

            return [dict(row) for row in rows]


class EmbeddingScheduler:
    """Manages cold/warm embedding cache for intelligent prompting.

    Strategies:
    - Cold: Lazy load on first access (good for quick tasks)
    - Warm: Background pre-load (good for interactive sessions)
    - On-demand: Targeted loading for specific task types
    """

    def __init__(
        self,
        embedding_store: Optional["ConversationEmbeddingStore"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ):
        """Initialize the scheduler."""
        self._store = embedding_store
        self._service = embedding_service
        self._state = CacheState.COLD
        self._last_refresh: Optional[datetime] = None
        self._background_task: Optional[asyncio.Task] = None
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_ttl = timedelta(minutes=30)

    @property
    def state(self) -> CacheState:
        """Get current cache state."""
        if (
            self._state == CacheState.WARM
            and self._last_refresh
            and datetime.now() - self._last_refresh > self._cache_ttl
        ):
            return CacheState.STALE
        return self._state

    async def ensure_warm(self, session_id: Optional[str] = None) -> None:
        """Ensure cache is warm for the session."""
        if self._state == CacheState.WARM:
            return

        if self._state == CacheState.WARMING:
            # Wait for background task
            if self._background_task:
                await self._background_task
            return

        await self._warm_cache(session_id)

    async def _warm_cache(self, session_id: Optional[str] = None) -> None:
        """Load embeddings into warm cache."""
        self._state = CacheState.WARMING

        try:
            if self._store:
                # Trigger lazy embedding if needed
                await self._store._ensure_embeddings(session_id)

            self._state = CacheState.WARM
            self._last_refresh = datetime.now()
            logger.info("[EmbeddingScheduler] Cache warmed successfully")

        except Exception as e:
            logger.warning(f"[EmbeddingScheduler] Cache warming failed: {e}")
            self._state = CacheState.COLD

    def start_background_refresh(self, session_id: Optional[str] = None) -> None:
        """Start background cache refresh."""
        if self._background_task and not self._background_task.done():
            return

        self._background_task = asyncio.create_task(self._background_refresh_loop(session_id))

    async def _background_refresh_loop(self, session_id: Optional[str] = None) -> None:
        """Background loop to keep cache fresh."""
        sleep_duration = self._cache_ttl.total_seconds() / 2
        while True:
            try:
                await asyncio.sleep(sleep_duration)
                await self._warm_cache(session_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[EmbeddingScheduler] Background refresh failed: {e}")

    def stop_background_refresh(self) -> None:
        """Stop background refresh."""
        if self._background_task:
            self._background_task.cancel()
            self._background_task = None

    def invalidate(self) -> None:
        """Invalidate the cache."""
        self._cache.clear()
        self._state = CacheState.COLD
        self._last_refresh = None


class IntelligentPromptBuilder:
    """Intelligent system prompt builder with learning capabilities.

    Uses:
    - Conversation embeddings for relevant context retrieval
    - Profile-specific learning for adaptive prompts
    - Cold/warm cache management for performance
    - Reinforcement signals from user feedback
    """

    # Cloud providers with robust native tool calling
    CLOUD_PROVIDERS = {"anthropic", "openai", "google", "xai", "moonshot", "kimi", "deepseek"}
    LOCAL_PROVIDERS = {"ollama", "lmstudio", "vllm"}

    # Grounding rules
    GROUNDING_RULES_MINIMAL = """
GROUNDING: Base ALL responses on tool output only. Never invent file paths or content.
Quote code exactly from tool output. If more info needed, call another tool.
""".strip()

    GROUNDING_RULES_STRICT = """
CRITICAL - TOOL OUTPUT GROUNDING:
When you receive tool output in <TOOL_OUTPUT> tags:
1. The content between ═══ markers is ACTUAL file/command output - NEVER ignore it
2. You MUST base your analysis ONLY on this actual content
3. NEVER fabricate, invent, or imagine file contents that differ from tool output
4. If you need more information, call another tool - do NOT guess
5. When citing code, quote EXACTLY from the tool output
6. If tool output is empty or truncated, acknowledge this limitation

VIOLATION OF THESE RULES WILL RESULT IN INCORRECT ANALYSIS.
""".strip()

    def __init__(
        self,
        provider_name: str,
        model: str,
        profile_name: str,
        embedding_store: Optional["ConversationEmbeddingStore"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        learning_store: Optional[ProfileLearningStore] = None,
    ):
        """Initialize the intelligent prompt builder.

        Args:
            provider_name: Provider name (e.g., "ollama", "anthropic")
            model: Model name/identifier
            profile_name: Profile name for learning tracking
            embedding_store: Optional conversation embedding store
            embedding_service: Optional embedding service
            learning_store: Optional profile learning store
        """
        self.provider_name = (str(provider_name) if provider_name else "").lower()
        self.model = str(model) if model else ""
        self.model_lower = self.model.lower()
        self.profile_name = profile_name or f"{self.provider_name}:{self.model}"

        self._embedding_store = embedding_store
        self._embedding_service = embedding_service
        self._learning_store = learning_store or ProfileLearningStore()
        self._scheduler = EmbeddingScheduler(embedding_store, embedding_service)

        # Load profile metrics
        self._metrics = self._learning_store.load_metrics(
            self.profile_name, self.provider_name, self.model
        )

        # Observers for feedback
        self._observers: List[Callable[[str, float, bool], None]] = []

    @classmethod
    async def create(
        cls,
        provider_name: str,
        model: str,
        profile_name: Optional[str] = None,
    ) -> "IntelligentPromptBuilder":
        """Factory method to create an initialized builder.

        Args:
            provider_name: Provider name
            model: Model name
            profile_name: Optional profile name

        Returns:
            Initialized IntelligentPromptBuilder
        """
        # Get embedding services if available
        embedding_store = None
        embedding_service = None

        try:
            from victor.agent.conversation_embedding_store import (
                get_conversation_embedding_store,
            )
            from victor.storage.embeddings.service import EmbeddingService

            embedding_service = EmbeddingService.get_instance()
            embedding_store = await get_conversation_embedding_store(embedding_service)
        except ImportError:
            logger.debug("[IntelligentPromptBuilder] Embedding services not available")
        except Exception as e:
            logger.warning(f"[IntelligentPromptBuilder] Failed to init embeddings: {e}")

        return cls(
            provider_name=provider_name,
            model=model,
            profile_name=profile_name or f"{provider_name}:{model}",
            embedding_store=embedding_store,
            embedding_service=embedding_service,
        )

    def add_observer(self, observer: Callable[[str, float, bool], None]) -> None:
        """Add observer for feedback notifications."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[str, float, bool], None]) -> None:
        """Remove observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    async def build(
        self,
        task: str,
        task_type: str = "general",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[str]] = None,
        current_mode: str = "explore",
        tool_budget: int = 10,
        iteration_budget: int = 20,
        session_id: Optional[str] = None,
        continuation_context: Optional[str] = None,
    ) -> str:
        """Build an intelligent system prompt.

        Args:
            task: Current task/query
            task_type: Detected task type
            conversation_history: Recent conversation messages
            available_tools: List of available tool names
            current_mode: Current agent mode (explore/build/plan)
            tool_budget: Remaining tool call budget
            iteration_budget: Remaining iteration budget
            session_id: Session ID for context retrieval
            continuation_context: Context from previous continuation

        Returns:
            Optimized system prompt
        """
        # Ensure embeddings are ready if in interactive mode
        if self._scheduler.state == CacheState.COLD and session_id:
            # Don't block - warm in background
            self._scheduler.start_background_refresh(session_id)

        # Build context
        context = await self._build_context(
            task=task,
            task_type=task_type,
            conversation_history=conversation_history,
            available_tools=available_tools or [],
            current_mode=current_mode,
            tool_budget=tool_budget,
            iteration_budget=iteration_budget,
            session_id=session_id,
            continuation_context=continuation_context,
        )

        # Determine strategy
        strategy = self._determine_strategy(context)

        # Generate prompt
        prompt = self._generate_prompt(context, strategy)

        logger.debug(
            f"[IntelligentPromptBuilder] Generated {strategy.value} prompt "
            f"for {self.profile_name} ({len(prompt)} chars)"
        )

        return prompt

    async def _build_context(
        self,
        task: str,
        task_type: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        available_tools: List[str],
        current_mode: str,
        tool_budget: int,
        iteration_budget: int,
        session_id: Optional[str],
        continuation_context: Optional[str],
    ) -> PromptContext:
        """Build context for prompt generation."""
        context = PromptContext(
            task=task,
            task_type=task_type,
            profile_name=self.profile_name,
            provider=self.provider_name,
            model=self.model,
            available_tools=available_tools,
            recommended_tool_budget=self._metrics.optimal_tool_budget,
            current_mode=current_mode,
            iteration_budget=iteration_budget,
            continuation_context=continuation_context,
            profile_metrics=self._metrics,
        )

        # Retrieve relevant context fragments
        if self._embedding_store and session_id:
            context.relevant_fragments = await self._retrieve_relevant_context(
                task, task_type, session_id
            )

        return context

    async def _retrieve_relevant_context(
        self,
        task: str,
        task_type: str,
        session_id: str,
        limit: int = 5,
    ) -> List[ContextFragment]:
        """Retrieve relevant context fragments from conversation history."""
        if not self._embedding_store:
            return []

        try:
            results = await self._embedding_store.search_similar(
                query=task,
                session_id=session_id,
                limit=limit * 2,  # Fetch more for filtering
                min_similarity=0.4,
            )

            fragments = []
            for result in results[:limit]:
                fragment = ContextFragment(
                    content=f"[Previous: {result.message_id}]",  # Content from SQLite
                    similarity=result.similarity,
                    task_type=task_type,
                    was_successful=True,  # Would need to track this
                    timestamp=result.timestamp or datetime.now(),
                    source="conversation",
                )
                fragments.append(fragment)

            return sorted(fragments, key=lambda f: f.relevance_score, reverse=True)

        except Exception as e:
            logger.warning(f"[IntelligentPromptBuilder] Context retrieval failed: {e}")
            return []

    def _determine_strategy(self, context: PromptContext) -> PromptStrategy:
        """Determine the best prompt strategy based on context and learning."""
        # Cloud providers with good native tool support
        if self.provider_name in self.CLOUD_PROVIDERS:
            return PromptStrategy.MINIMAL

        # Use learned strategy if we have enough data
        if self._metrics.total_requests >= 10:
            return self._metrics.get_recommended_strategy()

        # Default based on model capabilities
        if self._has_native_tool_support():
            return PromptStrategy.STRUCTURED

        return PromptStrategy.STRICT

    def _has_native_tool_support(self) -> bool:
        """Check if model has native tool calling support."""
        # Use set for O(1) lookup instead of list iteration
        patterns = {
            "qwen2.5",
            "qwen-2.5",
            "qwen3",
            "qwen-3",
            "llama-3.1",
            "llama3.1",
            "llama-3.2",
            "llama3.2",
            "llama-3.3",
            "llama3.3",
            "ministral",
            "mistral",
            "mixtral",
            "command-r",
            "firefunction",
            "hermes",
            "functionary",
        }
        return any(pattern in self.model_lower for pattern in patterns)

    def _generate_prompt(self, context: PromptContext, strategy: PromptStrategy) -> str:
        """Generate the system prompt based on strategy."""
        # Build parts list efficiently
        parts = [self._get_base_identity(strategy)]

        # Add optional parts only if they exist
        optional_parts = [
            self._get_task_hint(context.task_type),
            self._get_mode_hint(context.current_mode, context.iteration_budget),
            self._get_tool_guidance(context, strategy),
            (
                f"\nCONTINUATION CONTEXT:\n{context.continuation_context}"
                if context.continuation_context
                else None
            ),
            (
                self._format_context_fragments(context.relevant_fragments)
                if context.relevant_fragments
                else None
            ),
        ]

        parts.extend(part for part in optional_parts if part)
        parts.append(self._get_grounding_rules(strategy, context.profile_metrics))

        return "\n\n".join(parts)

    def _get_base_identity(self, strategy: PromptStrategy) -> str:
        """Get base identity prompt based on strategy."""
        if strategy == PromptStrategy.MINIMAL:
            return (
                "You are an expert code analyst with access to tools for exploring "
                "and modifying code. Use them effectively."
            )
        elif strategy == PromptStrategy.STRUCTURED:
            return (
                "You are an expert coding assistant. You can analyze, explain, and generate code.\n"
                "When asked to write or complete code, provide working implementations directly.\n"
                "When asked to explore or analyze code, use the available tools."
            )
        else:  # STRICT or ADAPTIVE
            return (
                "You are a code analyst. Follow the rules below EXACTLY.\n"
                "Your primary job is to help the user understand and modify code."
            )

    def _get_task_hint(self, task_type: str) -> str:
        """Get task-specific hint."""
        hints = {
            "code_generation": "[GENERATE] Write code directly. No exploration needed. Complete implementation.",
            "create_simple": "[CREATE] Write file immediately. Skip codebase exploration. One tool call max.",
            "create": "[CREATE+CONTEXT] Read 1-2 relevant files, then create. Follow existing patterns.",
            "edit": "[EDIT] Read target file first, then modify. Focused changes only.",
            "search": "[SEARCH] Use code_search/list_directory. Summarize after 2-4 calls.",
            "action": "[ACTION] Execute git/test/build operations. Continue until complete.",
            "analysis_deep": "[ANALYSIS] Thorough codebase exploration. Read all relevant modules.",
            "analyze": "[ANALYZE] Examine code carefully. Read related files. Structured findings.",
            "general": "[GENERAL] Moderate exploration. 3-6 tool calls. Answer concisely.",
        }
        # Defensive: ensure task_type is a string before calling .lower()
        task_type_str = str(task_type).lower() if task_type else "general"
        return hints.get(task_type_str, "")

    def _get_mode_hint(self, mode: str, iteration_budget: int) -> str:
        """Get mode-specific hint."""
        mode_hints = {
            "explore": f"MODE: Explore - Focus on understanding code. Budget: {iteration_budget} iterations.",
            "build": f"MODE: Build - Focus on implementation. Budget: {iteration_budget} iterations.",
            "plan": f"MODE: Plan - Create detailed plan before implementing. Budget: {iteration_budget} iterations.",
        }
        # Defensive: ensure mode is a string before calling .lower()
        mode_str = str(mode).lower() if mode else "explore"
        return mode_hints.get(mode_str, "")

    def _get_tool_guidance(
        self,
        context: PromptContext,
        strategy: PromptStrategy,
    ) -> str:
        """Get tool usage guidance based on strategy and learned patterns."""
        if strategy == PromptStrategy.MINIMAL:
            return (
                "Tool usage:\n"
                "- Call tools when needed for information\n"
                "- Parallel calls allowed for independent operations\n"
                f"- Tool budget: {context.recommended_tool_budget}"
            )

        elif strategy == PromptStrategy.STRUCTURED:
            return (
                "TOOL USAGE:\n"
                "- Use list_directory and read_file to inspect code\n"
                "- Call tools one at a time, waiting for results\n"
                f"- Budget: {context.recommended_tool_budget} tool calls\n"
                "- After gathering info, provide a clear answer\n"
                "- Do NOT repeat identical tool calls"
            )

        else:  # STRICT
            return (
                "CRITICAL TOOL RULES:\n"
                "1. Call tools ONE AT A TIME. Wait for each result.\n"
                f"2. Budget: {context.recommended_tool_budget} tool calls maximum.\n"
                "3. After reading 2-3 files, STOP and provide your answer.\n"
                "4. Do NOT repeat the same tool call.\n\n"
                "OUTPUT FORMAT:\n"
                "1. Write your answer in plain English text.\n"
                "2. Do NOT output JSON objects in your response.\n"
                "3. Do NOT output XML tags or function call syntax."
            )

    def _format_context_fragments(self, fragments: List[ContextFragment]) -> str:
        """Format relevant context fragments."""
        if not fragments:
            return ""

        lines = ["RELEVANT CONTEXT (from previous interactions):"]
        for i, frag in enumerate(fragments[:3], 1):
            lines.append(f"{i}. [{frag.source}] (relevance: {frag.relevance_score:.2f})")

        return "\n".join(lines)

    def _get_grounding_rules(
        self,
        strategy: PromptStrategy,
        metrics: Optional[ProfileMetrics],
    ) -> str:
        """Get grounding rules based on strategy and learned needs."""
        # If we've learned this model needs strict grounding
        if metrics and metrics.needs_strict_grounding:
            return self.GROUNDING_RULES_STRICT

        # Otherwise based on strategy
        if strategy in (PromptStrategy.STRICT, PromptStrategy.ADAPTIVE):
            return self.GROUNDING_RULES_STRICT

        return self.GROUNDING_RULES_MINIMAL

    def record_feedback(
        self,
        task_type: str,
        success: bool,
        quality_score: float,
        response_time_ms: float,
        tool_calls: int,
        tool_budget: int,
        grounded: bool,
    ) -> None:
        """Record feedback for reinforcement learning.

        Call this after each interaction to improve future prompts.
        """
        # Update metrics
        self._metrics.update_from_interaction(
            success=success,
            quality_score=quality_score,
            response_time_ms=response_time_ms,
            tool_calls=tool_calls,
            tool_budget=tool_budget,
            grounded=grounded,
        )

        # Persist learning
        self._learning_store.save_metrics(self._metrics)
        self._learning_store.record_interaction(
            profile_name=self.profile_name,
            task_type=task_type,
            success=success,
            quality_score=quality_score,
            response_time_ms=response_time_ms,
            tool_calls=tool_calls,
            tool_budget=tool_budget,
            grounded=grounded,
        )

        # Notify observers
        for observer in self._observers:
            try:
                observer(task_type, quality_score, success)
            except Exception as e:
                logger.warning(f"[IntelligentPromptBuilder] Observer error: {e}")

        logger.debug(
            f"[IntelligentPromptBuilder] Recorded feedback: "
            f"success={success}, quality={quality_score:.2f}, "
            f"tools={tool_calls}/{tool_budget}"
        )

    def get_profile_stats(self) -> Dict[str, Any]:
        """Get profile statistics."""
        return {
            "profile_name": self.profile_name,
            "provider": self.provider_name,
            "model": self.model,
            "total_requests": self._metrics.total_requests,
            "success_rate": self._metrics.tool_call_success_rate,
            "avg_quality": self._metrics.avg_quality_score,
            "grounding_accuracy": self._metrics.grounding_accuracy,
            "optimal_tool_budget": self._metrics.optimal_tool_budget,
            "recommended_strategy": self._metrics.get_recommended_strategy().value,
            "cache_state": self._scheduler.state.value,
        }

    def reset_learning(self) -> None:
        """Reset learned profile metrics."""
        self._metrics = ProfileMetrics(
            profile_name=self.profile_name,
            provider=self.provider_name,
            model=self.model,
        )
        self._learning_store.save_metrics(self._metrics)
        logger.info(f"[IntelligentPromptBuilder] Reset learning for {self.profile_name}")


# Convenience function for backward compatibility
async def build_intelligent_prompt(
    provider_name: str,
    model: str,
    task: str,
    task_type: str = "general",
    profile_name: Optional[str] = None,
    **kwargs,
) -> str:
    """Build an intelligent system prompt (convenience function).

    Args:
        provider_name: Provider name
        model: Model name
        task: Current task
        task_type: Task type
        profile_name: Optional profile name
        **kwargs: Additional arguments for build()

    Returns:
        System prompt string
    """
    builder = await IntelligentPromptBuilder.create(
        provider_name=provider_name,
        model=model,
        profile_name=profile_name,
    )
    return await builder.build(task=task, task_type=task_type, **kwargs)
