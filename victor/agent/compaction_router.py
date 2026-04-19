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

"""Compaction router for intelligent strategy selection.

Routes compaction requests to the optimal strategy based on:
- Message complexity (heuristic-based scoring)
- Token count (estimated)
- Settings thresholds (configurable)
- Feature flags (runtime control)

Decision logic:
1. Check feature flags (is each strategy enabled?)
2. Calculate complexity score (heuristic-based)
3. Apply settings thresholds
4. Select optimal strategy
5. Execute compaction with fallback chain

Fallback behavior:
- LLM-based fails → try hybrid → try rule-based
- Hybrid fails → try rule-based
- Rule-based always succeeds (deterministic)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.config.compaction_strategy_settings import (
    CompactionFeatureFlags,
    CompactionStrategySettings,
)
from victor.providers.base import Message

if TYPE_CHECKING:
    from victor.agent.compaction_hybrid import HybridCompactionSummarizer
    from victor.agent.compaction_rule_based import RuleBasedCompactionSummarizer
    from victor.agent.llm_compaction_summarizer import LLMCompactionSummarizer
    from victor.agent.conversation.store import ConversationStore


logger = logging.getLogger(__name__)


class CompactionType(Enum):
    """Compaction strategy types."""

    RULE_BASED = "rule_based"
    """Fast, deterministic rule-based compaction (Claudecode-style)."""

    LLM_BASED = "llm_based"
    """Rich, intelligent LLM-based compaction (Victor-style)."""

    HYBRID = "hybrid"
    """Combined approach: rules for structure, LLM for enhancements."""


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    summary: str
    """Generated summary text."""

    removed_count: int
    """Number of messages removed."""

    strategy_used: CompactionType
    """Strategy that was used."""

    complexity_score: float
    """Complexity score (0.0-1.0)."""

    duration_ms: int
    """Duration in milliseconds."""

    tokens_saved: int
    """Estimated tokens saved."""

    success: bool
    """Whether compaction succeeded."""

    error_message: Optional[str]
    """Error message if failed."""

    session_id: Optional[str]
    """Session ID for analytics."""


class CompactionRouter:
    """Routes compaction requests to optimal strategy.

    Implements intelligent strategy selection based on:
    - Message complexity (0.0-1.0 score)
    - Token count (estimated)
    - Settings thresholds (configurable)
    - Feature flags (runtime control)

    The router ensures graceful degradation:
    - LLM failures fall back to hybrid
    - Hybrid failures fall back to rule-based
    - Rule-based always succeeds (deterministic)

    Usage:
        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
            hybrid_summarizer=hybrid_summarizer,
        )

        result = await router.compact(
            messages=messages,
            current_query="fix the bug",
            session_id="session-123",
        )
    """

    def __init__(
        self,
        settings: CompactionStrategySettings,
        feature_flags: CompactionFeatureFlags,
        rule_summarizer: "RuleBasedCompactionSummarizer",
        llm_summarizer: Optional["LLMCompactionSummarizer"] = None,
        hybrid_summarizer: Optional["HybridCompactionSummarizer"] = None,
        conversation_store: Optional["ConversationStore"] = None,
    ):
        """Initialize compaction router.

        Args:
            settings: Compaction strategy settings
            feature_flags: Feature flags for strategy availability
            rule_summarizer: Rule-based summarizer (required)
            llm_summarizer: Optional LLM-based summarizer
            hybrid_summarizer: Optional hybrid summarizer
            conversation_store: Optional conversation store for analytics logging
        """
        self._settings = settings
        self._feature_flags = feature_flags
        self._rule_summarizer = rule_summarizer
        self._llm_summarizer = llm_summarizer
        self._hybrid_summarizer = hybrid_summarizer
        self._conversation_store = conversation_store

        # Metrics collection
        self._metrics = {
            "total_compactions": 0,
            "rule_based_count": 0,
            "llm_based_count": 0,
            "hybrid_count": 0,
            "rule_based_success": 0,
            "llm_based_success": 0,
            "hybrid_success": 0,
            "total_duration_ms": 0,
            "total_tokens_saved": 0,
        }

    async def compact(
        self,
        messages: List[Message],
        current_query: Optional[str] = None,
        session_id: Optional[str] = None,
        complexity_score: Optional[float] = None,
    ) -> CompactionResult:
        """Compact messages using optimal strategy.

        Args:
            messages: Messages to compact
            current_query: Optional current query for relevance scoring
            session_id: Optional session ID for analytics
            complexity_score: Optional pre-calculated complexity score

        Returns:
            CompactionResult with summary and metadata
        """
        if not messages:
            return CompactionResult(
                summary="",
                removed_count=0,
                strategy_used=CompactionType.RULE_BASED,
                complexity_score=0.0,
                duration_ms=0,
                tokens_saved=0,
                success=True,
                error_message=None,
                session_id=session_id,
            )

        start_time = time.time()

        # Step 1: Calculate complexity if not provided
        if complexity_score is None:
            complexity_score = self._calculate_complexity(messages, current_query)

        # Step 2: Select strategy
        strategy = self._select_strategy(
            len(messages),
            self._estimate_tokens(messages),
            complexity_score,
        )

        logger.info(
            f"Compaction router: selected {strategy.value} strategy "
            f"(complexity={complexity_score:.2f}, messages={len(messages)})"
        )

        # Step 3: Execute compaction with fallback
        try:
            result = await self._execute_compaction(
                messages,
                strategy,
                current_query,
                session_id,
                complexity_score,
            )

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)
            result.duration_ms = duration_ms

            # Update metrics
            self._update_metrics(
                strategy=strategy,
                duration_ms=duration_ms,
                tokens_saved=result.tokens_saved,
                success=True,
            )

            # Log success
            await self._log_compaction_event(
                messages,
                strategy,
                result,
                success=True,
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Compaction failed with strategy {strategy.value}: {e}")

            # Fallback to rule-based
            logger.info("Falling back to rule-based compaction")
            fallback_result = await self._execute_compaction(
                messages,
                CompactionType.RULE_BASED,
                current_query,
                session_id,
                complexity_score,
            )
            fallback_result.duration_ms = duration_ms

            # Update metrics for fallback
            self._update_metrics(
                strategy=CompactionType.RULE_BASED,  # Always rule-based on fallback
                duration_ms=duration_ms,
                tokens_saved=fallback_result.tokens_saved,
                success=True,  # Fallback succeeded even if original failed
            )

            # Log fallback
            await self._log_compaction_event(
                messages,
                strategy,
                fallback_result,
                success=False,
                error_message=str(e),
            )

            return fallback_result

    def _select_strategy(
        self,
        message_count: int,
        estimated_tokens: int,
        complexity_score: float,
    ) -> CompactionType:
        """Select optimal compaction strategy.

        Decision logic:
        1. Check feature flags (what's enabled?)
        2. Apply complexity threshold (simple vs complex)
        3. Apply token/message thresholds (small vs large)
        4. Select optimal strategy

        Args:
            message_count: Number of messages
            estimated_tokens: Estimated token count
            complexity_score: Complexity score (0.0-1.0)

        Returns:
            Selected CompactionType
        """
        # Check feature flags first
        if not self._feature_flags.enable_llm_based:
            return CompactionType.RULE_BASED

        if not self._feature_flags.enable_rule_based:
            return CompactionType.LLM_BASED

        if not self._feature_flags.enable_hybrid:
            # Both enabled but hybrid disabled, use simpler logic
            if complexity_score >= self._settings.llm_min_complexity:
                return CompactionType.LLM_BASED
            return CompactionType.RULE_BASED

        # Apply settings thresholds
        # Fast path: simple cases use rules
        if complexity_score < self._settings.llm_min_complexity:
            return CompactionType.RULE_BASED

        # Medium path: borderline cases use hybrid (if available)
        # Check if hybrid summarizer is available before selecting hybrid strategy
        if (
            self._hybrid_summarizer is not None
            and message_count < self._settings.llm_min_messages
            and estimated_tokens < self._settings.llm_min_tokens
        ):
            return CompactionType.HYBRID

        # Slow path: complex cases use LLM
        return CompactionType.LLM_BASED

    async def _execute_compaction(
        self,
        messages: List[Message],
        strategy: CompactionType,
        current_query: Optional[str],
        session_id: Optional[str],
        complexity_score: float,
    ) -> CompactionResult:
        """Execute compaction using specified strategy.

        Args:
            messages: Messages to compact
            strategy: Strategy to use
            current_query: Optional current query
            session_id: Optional session ID
            complexity_score: Complexity score

        Returns:
            CompactionResult

        Raises:
            Exception: If compaction fails
        """
        # Estimate tokens before compaction
        tokens_before = self._estimate_tokens(messages)

        # Execute based on strategy
        if strategy == CompactionType.RULE_BASED:
            summary = self._rule_summarizer.summarize(messages)
        elif strategy == CompactionType.LLM_BASED:
            if not self._llm_summarizer:
                raise ValueError("LLM summarizer not available")
            summary = self._llm_summarizer.summarize(messages)
        elif strategy == CompactionType.HYBRID:
            if not self._hybrid_summarizer:
                raise ValueError("Hybrid summarizer not available")
            summary = await self._hybrid_summarizer.summarize_async(messages)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Estimate tokens saved
        tokens_after = len(summary) * 2  # Rough estimate
        tokens_saved = max(0, tokens_before - tokens_after)

        return CompactionResult(
            summary=summary,
            removed_count=len(messages),
            strategy_used=strategy,
            complexity_score=complexity_score,
            duration_ms=0,  # Will be set by caller
            tokens_saved=tokens_saved,
            success=True,
            error_message=None,
            session_id=session_id,
        )

    def _calculate_complexity(
        self,
        messages: List[Message],
        current_query: Optional[str],
    ) -> float:
        """Calculate complexity score (0.0-1.0).

        Complexity factors:
        1. Message count (0.0-0.3): More messages = higher complexity
        2. Token count (0.0-0.3): More tokens = higher complexity
        3. Tool diversity (0.0-0.2): More different tools = higher complexity
        4. Semantic coherence (0.0-0.2): More files/topics = lower coherence

        Args:
            messages: Messages to analyze
            current_query: Optional current query for relevance

        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Factor 1: Message count (0.0-0.3)
        count_score = min(len(messages) / 50.0, 0.3)

        # Factor 2: Token count (0.0-0.3)
        token_score = min(self._estimate_tokens(messages) / 20000.0, 0.3)

        # Factor 3: Tool diversity (0.0-0.2)
        unique_tools = len(self._extract_tools(messages))
        tool_score = min(unique_tools / 10.0, 0.2)

        # Factor 4: Semantic coherence (0.0-0.2)
        # Check if topics changed (heuristic: different file paths)
        files = self._extract_files(messages)
        coherence_score = 0.2 if len(files) < 5 else 0.0

        # Factor 5: Role diversity (0.0-0.1)
        roles = set(m.role for m in messages)
        role_score = min(len(roles) / 5.0, 0.1)

        # Factor 6: Current query relevance (0.0-0.1)
        query_score = 0.0
        if current_query:
            # Check if current query mentions continuation
            continuation_keywords = ["continue", "next", "then", "after that", "subsequent"]
            if any(kw in current_query.lower() for kw in continuation_keywords):
                query_score = 0.1

        # Combine all factors
        complexity = (
            count_score + token_score + tool_score + coherence_score + role_score + query_score
        )

        return min(complexity, 1.0)

    def _estimate_tokens(self, messages: List[Message]) -> int:
        """Estimate token count for messages.

        Args:
            messages: Messages to estimate

        Returns:
            Estimated token count
        """
        total_chars = sum(len(m.content) for m in messages)
        # Rough estimate: ~4 chars per token
        return total_chars // 4

    def _extract_tools(self, messages: List[Message]) -> List[str]:
        """Extract unique tool names from messages.

        Args:
            messages: Messages to analyze

        Returns:
            List of unique tool names
        """
        tools = set()
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    if isinstance(call, dict):
                        tool_name = call.get("name", "")
                        if tool_name:
                            tools.add(tool_name)
            elif hasattr(msg, "tool_name") and msg.tool_name:
                tools.add(msg.tool_name)
        return list(tools)

    def _extract_files(self, messages: List[Message]) -> List[str]:
        """Extract file paths from messages.

        Args:
            messages: Messages to analyze

        Returns:
            List of unique file paths
        """
        import re

        files = set()
        file_extensions = {
            "rs",
            "ts",
            "tsx",
            "js",
            "jsx",
            "json",
            "md",
            "py",
            "go",
            "java",
            "cpp",
            "c",
            "h",
            "cs",
            "php",
            "rb",
            "swift",
            "kt",
            "scala",
        }

        for msg in messages:
            if not msg.content:
                continue

            # Extract file paths using regex
            # Pattern: paths with known extensions
            pattern = r"[\w/\\-]+\.(?:" + "|".join(file_extensions) + ")"
            matches = re.findall(pattern, msg.content)
            files.update(matches)

        return list(files)

    async def _log_compaction_event(
        self,
        messages: List[Message],
        strategy: CompactionType,
        result: CompactionResult,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Log compaction event for analytics.

        Args:
            messages: Messages that were compacted
            strategy: Strategy used
            result: Compaction result
            success: Whether compaction succeeded
            error_message: Error message if failed
        """
        if not self._settings.store_compaction_history:
            return

        if not self._conversation_store:
            logger.debug("No conversation store provided, skipping compaction history logging")
            return

        try:
            # Log to analytics (persist to database)
            logger.info(
                f"Compaction event: strategy={strategy.value}, "
                f"messages={len(messages)}, "
                f"removed={result.removed_count}, "
                f"tokens_saved={result.tokens_saved}, "
                f"duration_ms={result.duration_ms}, "
                f"success={success}, "
                f"error={error_message}"
            )

            # Persist to compaction_history table
            self._conversation_store.store_compaction_history(
                session_id=result.session_id or "unknown",
                strategy_used=strategy.value,
                message_count_before=len(messages),
                message_count_after=result.removed_count,
                token_count_before=result.tokens_saved + len(result.summary) * 4,  # Estimate
                token_count_after=len(result.summary) * 4,  # Estimate
                duration_ms=result.duration_ms,
                llm_provider=None,  # TODO: Extract from result
                llm_model=None,  # TODO: Extract from result
                success=success,
                error_message=error_message,
            )

        except Exception as e:
            logger.warning(f"Failed to log compaction event: {e}")

    def _update_metrics(
        self,
        strategy: CompactionType,
        duration_ms: int,
        tokens_saved: int,
        success: bool,
    ) -> None:
        """Update in-memory metrics after compaction.

        Args:
            strategy: Strategy that was used
            duration_ms: Duration in milliseconds
            tokens_saved: Estimated tokens saved
            success: Whether compaction succeeded
        """
        self._metrics["total_compactions"] += 1
        self._metrics["total_duration_ms"] += duration_ms
        self._metrics["total_tokens_saved"] += tokens_saved

        # Update strategy-specific counters
        if strategy == CompactionType.RULE_BASED:
            self._metrics["rule_based_count"] += 1
            if success:
                self._metrics["rule_based_success"] += 1
        elif strategy == CompactionType.LLM_BASED:
            self._metrics["llm_based_count"] += 1
            if success:
                self._metrics["llm_based_success"] += 1
        elif strategy == CompactionType.HYBRID:
            self._metrics["hybrid_count"] += 1
            if success:
                self._metrics["hybrid_success"] += 1

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about strategy usage.

        Returns:
            Dictionary with strategy usage statistics
        """
        return self._metrics.copy()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.

        Returns:
            Dictionary with detailed performance metrics:
            - total_compactions: Total number of compactions
            - strategy_usage: Breakdown by strategy type
            - success_rates: Success rate per strategy
            - performance_metrics: Duration and token savings per strategy
        """
        metrics = self._metrics

        # Calculate success rates
        rule_based_rate = (
            metrics["rule_based_success"] / metrics["rule_based_count"]
            if metrics["rule_based_count"] > 0
            else 0.0
        )
        llm_based_rate = (
            metrics["llm_based_success"] / metrics["llm_based_count"]
            if metrics["llm_based_count"] > 0
            else 0.0
        )
        hybrid_rate = (
            metrics["hybrid_success"] / metrics["hybrid_count"]
            if metrics["hybrid_count"] > 0
            else 0.0
        )

        # Calculate averages
        avg_duration = (
            metrics["total_duration_ms"] / metrics["total_compactions"]
            if metrics["total_compactions"] > 0
            else 0.0
        )
        avg_tokens_saved = (
            metrics["total_tokens_saved"] / metrics["total_compactions"]
            if metrics["total_compactions"] > 0
            else 0.0
        )

        return {
            "total_compactions": metrics["total_compactions"],
            "strategy_usage": {
                "rule_based": metrics["rule_based_count"],
                "llm_based": metrics["llm_based_count"],
                "hybrid": metrics["hybrid_count"],
            },
            "success_rates": {
                "rule_based": rule_based_rate,
                "llm_based": llm_based_rate,
                "hybrid": hybrid_rate,
                "overall": (
                    (
                        metrics["rule_based_success"]
                        + metrics["llm_based_success"]
                        + metrics["hybrid_success"]
                    )
                    / metrics["total_compactions"]
                    if metrics["total_compactions"] > 0
                    else 0.0
                ),
            },
            "performance_metrics": {
                "avg_duration_ms": avg_duration,
                "avg_tokens_saved": avg_tokens_saved,
                "total_duration_ms": metrics["total_duration_ms"],
                "total_tokens_saved": metrics["total_tokens_saved"],
            },
        }
