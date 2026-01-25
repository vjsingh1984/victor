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


from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.agent.message_history import MessageHistory
from victor.agent.conversation_state import ConversationStateMachine, ConversationStage
from victor.agent.prompt_normalizer import PromptNormalizer, get_prompt_normalizer
from victor.config.orchestrator_constants import (
    COMPACTION_CONFIG,
    CONTEXT_LIMITS,
    SEMANTIC_THRESHOLDS,
)
from victor.providers.base import Message

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService
    from victor.agent.conversation_memory import ConversationStore


class CompactionStrategy(Enum):
    SIMPLE = "simple"
    TIERED = "tiered"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class MessageImportance:
    message: Message
    index: int
    score: float
    reason: str = ""


logger = logging.getLogger(__name__)


# Patterns for pinned output requirements (never compacted)
# These patterns indicate user-specified deliverable format requirements
# that must survive compaction to prevent the agent from losing focus
PINNED_REQUIREMENT_PATTERNS = [
    re.compile(r"\bmust\s+output\b", re.IGNORECASE),
    re.compile(r"\brequired\s+format\b", re.IGNORECASE),
    re.compile(r"\bfindings\s+table\b", re.IGNORECASE),
    re.compile(r"\btop[-\s]?\d+\s+fixes\b", re.IGNORECASE),
    re.compile(r"\bdeliverables?\s*:", re.IGNORECASE),
    re.compile(r"\bcreate\s+a\s+findings\s+table\b", re.IGNORECASE),
    re.compile(r"\bprovide\s+top[-\s]?\d+\b", re.IGNORECASE),
    re.compile(r"\boutput\s+must\s+include\b", re.IGNORECASE),
    re.compile(r"\brequired\s+outputs?\s*:", re.IGNORECASE),
    re.compile(r"\b\d+[-–]\d+\s+findings\b", re.IGNORECASE),  # "6-10 findings"
    re.compile(r"\bprioritize\s+\d+[-–]\d+\s+findings\b", re.IGNORECASE),
]


def _is_pinned_content(content: str) -> bool:
    """Check if content contains pinned output requirement patterns.

    Args:
        content: Message content to check

    Returns:
        True if content contains pinned requirement patterns
    """
    for pattern in PINNED_REQUIREMENT_PATTERNS:
        if pattern.search(content):
            return True
    return False


@dataclass
class ContextMetrics:
    char_count: int
    estimated_tokens: int
    message_count: int
    is_overflow_risk: bool = False
    max_context_chars: int = 200000

    @property
    def total_chars(self) -> int:
        """Backward compatibility alias for char_count."""
        return self.char_count

    @property
    def utilization(self) -> float:
        return min(1.0, self.char_count / self.max_context_chars) if self.max_context_chars else 0.0

    @property
    def max_tokens(self) -> int:
        return int(self.max_context_chars / 2.8)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.estimated_tokens)


@dataclass
class ConversationConfig:
    max_context_chars: int = CONTEXT_LIMITS.max_context_chars
    chars_per_token_estimate: int = CONTEXT_LIMITS.chars_per_token_estimate
    enable_stage_tracking: bool = True
    enable_context_monitoring: bool = True
    compaction_strategy: CompactionStrategy = CompactionStrategy.TIERED
    min_messages_to_keep: int = 6
    tool_result_retention_weight: float = COMPACTION_CONFIG.tool_result_retention_weight
    recent_message_weight: float = COMPACTION_CONFIG.recent_message_weight
    semantic_relevance_threshold: float = SEMANTIC_THRESHOLDS.compaction_relevance


class ConversationController:

    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        message_history: Optional[MessageHistory] = None,
        state_machine: Optional[ConversationStateMachine] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        conversation_store: Optional["ConversationStore"] = None,
        session_id: Optional[str] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
    ):
        self.config = config or ConversationConfig()
        self._history = message_history or MessageHistory()
        self._state_machine = state_machine or ConversationStateMachine()
        self._embedding_service = embedding_service
        self._conversation_store = conversation_store
        self._session_id = session_id
        self._system_prompt: Optional[str] = None
        self._system_added = False
        self._context_callbacks: List[Callable[[ContextMetrics], None]] = []
        self._compaction_summaries: List[str] = []
        self._current_plan: Optional[Any] = None
        # PromptNormalizer for input deduplication (DIP compliance)
        self._normalizer = prompt_normalizer or get_prompt_normalizer()

    @property
    def messages(self) -> List[Message]:
        return self._history.messages

    @property
    def message_count(self) -> int:
        return len(self._history.messages)

    @property
    def stage(self) -> ConversationStage:
        return self._state_machine.get_stage()

    @property
    def system_prompt(self) -> Optional[str]:
        return self._system_prompt

    @property
    def current_plan(self) -> Optional[Any]:
        """Get the current execution plan.

        Returns:
            The current ExecutionPlan or None if no plan is active.
        """
        return self._current_plan

    def set_current_plan(self, plan: Optional[Any]) -> None:
        """Set the current execution plan.

        Args:
            plan: The ExecutionPlan to set, or None to clear.
        """
        self._current_plan = plan

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        self._system_added = False

    def ensure_system_message(self) -> None:
        if self._system_added or not self._system_prompt:
            return
        if self._history.messages and self._history.messages[0].role == "system":
            self._system_added = True
            return
        self._history._messages.insert(0, Message(role="system", content=self._system_prompt))
        self._system_added = True

    def add_user_message(self, content: str, **kwargs: Any) -> Message:
        """Add a user message with optional normalization.

        Uses PromptNormalizer to:
        1. Normalize action verbs (view→read, check→read)
        2. Detect and log duplicate messages
        3. Collapse repeated continuation requests

        Args:
            content: User message content
            **kwargs: Additional metadata (ignored for user messages)

        Returns:
            The added Message
        """
        self.ensure_system_message()

        # Normalize input using PromptNormalizer
        result = self._normalizer.normalize(content)
        normalized_content = result.normalized

        # Log normalizations if any were made
        if result.changes:
            logger.debug(f"Normalized prompt: {result.changes}")
        if result.is_duplicate:
            logger.debug("Detected duplicate message (will still add for conversational flow)")

        message = self._history.add_user_message(normalized_content)
        if self.config.enable_stage_tracking:
            self._state_machine.record_message(normalized_content, is_user=True)
        if (
            self.config.enable_context_monitoring
            and (metrics := self.get_context_metrics()).is_overflow_risk
        ):
            self._notify_context_callbacks(metrics)
        return message

    def add_assistant_message(
        self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None, **kwargs: Any
    ) -> Message:
        message = self._history.add_assistant_message(content, tool_calls=tool_calls)
        if self.config.enable_stage_tracking:
            self._state_machine.record_message(content, is_user=False)
        if (
            self.config.enable_context_monitoring
            and (metrics := self.get_context_metrics()).is_overflow_risk
        ):
            self._notify_context_callbacks(metrics)
        return message

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> Message:
        """Add a tool result message to history.

        Args:
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            result: Tool execution result

        Returns:
            The created message
        """
        message = self._history.add_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=result,
        )
        return message

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        """Add a message with specified role (backward compatibility).

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **kwargs: Additional message metadata (e.g., tool_calls for assistant)

        Returns:
            The created message
        """
        if role == "user":
            return self.add_user_message(content)
        elif role == "assistant":
            return self.add_assistant_message(content, **kwargs)
        elif role == "system":
            self.set_system_prompt(content)
            self.ensure_system_message()
            return self.messages[0]
        else:
            # For other roles, use add_message from history
            return self._history.add_message(role, content, **kwargs)

    def get_context_metrics(self) -> ContextMetrics:
        """Calculate current context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        total_chars = sum(len(m.content) for m in self.messages)
        estimated_tokens = total_chars // self.config.chars_per_token_estimate

        return ContextMetrics(
            char_count=total_chars,
            estimated_tokens=estimated_tokens,
            message_count=len(self.messages),
            is_overflow_risk=total_chars
            > self.config.max_context_chars * CONTEXT_LIMITS.overflow_threshold,
            max_context_chars=self.config.max_context_chars,
        )

    def check_context_overflow(self) -> bool:
        """Check if context is at risk of overflow.

        Returns:
            True if context is dangerously large
        """
        metrics = self.get_context_metrics()
        if metrics.is_overflow_risk:
            logger.warning(
                "Context overflow risk: %d chars (~%d tokens). Max: %d chars",
                metrics.char_count,
                metrics.estimated_tokens,
                metrics.max_context_chars,
            )
        return metrics.is_overflow_risk

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for current conversation stage.

        Returns:
            Set of recommended tool names
        """
        return self._state_machine.get_stage_tools()

    def reset(self) -> None:
        """Reset conversation to initial state."""
        self._history.clear()
        self._state_machine.reset()
        self._system_added = False
        logger.info("Conversation reset")

    def compact_history(self, keep_recent: int = 10) -> int:
        """Compact history by removing old messages (simple strategy).

        Keeps system message and most recent messages.
        For smarter compaction, use smart_compact_history().

        Args:
            keep_recent: Number of recent messages to keep

        Returns:
            Number of messages removed
        """
        if len(self.messages) <= keep_recent + 1:  # +1 for system message
            return 0

        # Keep system message and recent messages
        system_msg = None
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]

        recent = self.messages[-keep_recent:]
        removed = len(self.messages) - keep_recent - (1 if system_msg else 0)

        self._history.clear()
        if system_msg:
            self._history._messages.append(system_msg)
        for msg in recent:
            self._history._messages.append(msg)

        logger.info(f"Compacted history: removed {removed} messages")
        return removed

    def smart_compact_history(
        self,
        target_messages: Optional[int] = None,
        current_query: Optional[str] = None,
    ) -> int:
        """Smart context compaction using configured strategy.

        Uses the configured compaction strategy to intelligently reduce
        context size while preserving the most important information.

        Strategies:
        - SIMPLE: Same as compact_history (keep N recent)
        - TIERED: Prioritize tool results over discussion messages
        - SEMANTIC: Use embeddings to keep task-relevant messages
        - HYBRID: Combine tiered and semantic approaches

        Args:
            target_messages: Target number of messages to keep (default: min_messages_to_keep)
            current_query: Current user query for semantic relevance scoring

        Returns:
            Number of messages removed
        """
        target = target_messages or self.config.min_messages_to_keep
        strategy = self.config.compaction_strategy

        if len(self.messages) <= target + 1:  # +1 for system message
            return 0

        logger.info(f"Smart compacting with strategy: {strategy.value}")

        if strategy == CompactionStrategy.SIMPLE:
            return self.compact_history(keep_recent=target)

        # Score all messages for importance
        scored_messages = self._score_messages(current_query)

        # Sort by score (descending) and keep top N
        scored_messages.sort(key=lambda x: x.score, reverse=True)

        # Always keep system message
        system_msg = None
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]
            # Remove system from scored list (we'll add it back)
            scored_messages = [sm for sm in scored_messages if sm.index != 0]

        # Keep top messages (up to target)
        messages_to_keep = scored_messages[:target]

        # Sort by original index to maintain conversation order
        messages_to_keep.sort(key=lambda x: x.index)

        # Generate summary of removed messages
        removed_indices = {sm.index for sm in scored_messages[target:]}
        removed_messages = [m for i, m in enumerate(self.messages) if i in removed_indices]
        if removed_messages:
            summary = self._generate_compaction_summary(removed_messages)
            if summary:
                self._compaction_summaries.append(summary)
                # Persist to SQLite for future semantic retrieval
                message_ids = [f"msg_{i}" for i in removed_indices]
                self.persist_compaction_summary(summary, message_ids)

        # Rebuild message list
        removed_count = len(self.messages) - len(messages_to_keep) - (1 if system_msg else 0)
        self._history.clear()

        if system_msg:
            self._history._messages.append(system_msg)

        for scored_msg in messages_to_keep:
            self._history._messages.append(scored_msg.message)

        logger.info(
            f"Smart compacted: removed {removed_count} messages, "
            f"kept {len(messages_to_keep)} (strategy: {strategy.value})"
        )
        return removed_count

    def _score_messages(self, current_query: Optional[str] = None) -> List[MessageImportance]:
        """Score messages for importance during compaction.

        Scoring factors:
        - Role: Tool results score higher than discussion
        - Recency: Recent messages get a boost
        - Semantic: Messages similar to current query score higher
        - Content length: Substantive messages score higher

        Args:
            current_query: Optional query for semantic relevance

        Returns:
            List of MessageImportance objects
        """
        scored: List[MessageImportance] = []
        total_messages = len(self.messages)

        for i, msg in enumerate(self.messages):
            score = 0.0
            reason_parts = []

            # Skip system message in scoring (always kept)
            if msg.role == "system":
                scored.append(MessageImportance(msg, i, 1000.0, "system"))
                continue

            # Pinned output requirements (never compacted)
            # These contain user's deliverable format specifications
            if _is_pinned_content(msg.content):
                scored.append(MessageImportance(msg, i, 999.0, "pinned_requirement"))
                logger.debug(f"Message {i} marked as pinned (output requirement)")
                continue

            # Role-based scoring
            if msg.role == "tool":
                score += 3.0 * self.config.tool_result_retention_weight
                reason_parts.append("tool_result")
            elif msg.role == "assistant":
                score += 2.0
                reason_parts.append("assistant")
            elif msg.role == "user":
                score += 2.5  # User messages slightly more important
                reason_parts.append("user")

            # Recency scoring (exponential decay)
            recency_position = (i + 1) / total_messages  # 0 to 1
            recency_boost = recency_position**2 * self.config.recent_message_weight
            score += recency_boost
            if recency_position > 0.7:
                reason_parts.append("recent")

            # Content length scoring (prefer substantive messages)
            content_len = len(msg.content)
            if content_len > 500:
                score += 1.0
                reason_parts.append("substantive")
            elif content_len > 100:
                score += 0.5

            # Semantic relevance (if enabled and embedding service available)
            if (
                current_query
                and self._embedding_service
                and self.config.compaction_strategy
                in (CompactionStrategy.SEMANTIC, CompactionStrategy.HYBRID)
            ):
                similarity = self._compute_semantic_similarity(msg.content, current_query)
                if similarity > self.config.semantic_relevance_threshold:
                    score += similarity * 3.0
                    reason_parts.append(f"semantic:{similarity:.2f}")

            scored.append(MessageImportance(msg, i, score, "+".join(reason_parts)))

        return scored

    async def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not self._embedding_service:
            return 0.0

        try:
            # Truncate long texts to avoid embedding issues
            t1 = text1[:2000] if len(text1) > 2000 else text1
            t2 = text2[:2000] if len(text2) > 2000 else text2

            embeddings = await self._embedding_service.embed_batch([t1, t2])
            if len(embeddings) == 2:
                from victor.processing.native import cosine_similarity

                emb1, emb2 = embeddings[0], embeddings[1]
                # Use Rust-accelerated cosine similarity (with NumPy fallback)
                return cosine_similarity(list(emb1), list(emb2))
        except ImportError:
            logger.debug("cosine_similarity not available, returning 0.0")
        except (ValueError, TypeError) as e:
            logger.warning(f"Semantic similarity computation failed (bad embeddings): {e}")

        return 0.0

    def _generate_compaction_summary(self, removed_messages: List[Message]) -> str:
        """Generate a summary of removed messages for context preservation.

        Creates a brief summary of what was discussed in removed messages
        so the AI retains awareness of earlier conversation topics.

        Args:
            removed_messages: Messages being removed

        Returns:
            Summary string, or empty string if nothing noteworthy
        """
        if not removed_messages:
            return ""

        # Count message types
        user_msgs = [m for m in removed_messages if m.role == "user"]
        tool_msgs = [m for m in removed_messages if m.role == "tool"]

        # Extract key topics (simple keyword extraction)
        all_content = " ".join(m.content[:200] for m in removed_messages[:10])
        topics = self._extract_key_topics(all_content)

        parts = []
        if user_msgs:
            parts.append(f"{len(user_msgs)} user messages")
        if tool_msgs:
            parts.append(f"{len(tool_msgs)} tool results")
        if topics:
            parts.append(f"topics: {', '.join(topics[:3])}")

        if parts:
            return f"[Earlier conversation: {'; '.join(parts)}]"
        return ""

    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text using simple heuristics.

        Args:
            text: Text to extract topics from

        Returns:
            List of key topic words/phrases
        """
        import re

        # Simple keyword extraction - find capitalized words and technical terms
        words = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", text)  # CamelCase
        words += re.findall(r"\b[a-z]+_[a-z_]+\b", text)  # snake_case
        words += re.findall(r"\b(?:def|class|function|file|error|test|api)\s+\w+", text.lower())

        # Deduplicate and limit
        seen = set()
        topics = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in seen and len(w) > 3:
                seen.add(w_lower)
                topics.append(w)
                if len(topics) >= 5:
                    break

        return topics

    def get_compaction_summaries(self) -> List[str]:
        """Get summaries from previous compaction operations.

        Returns:
            List of summary strings
        """
        return self._compaction_summaries.copy()

    def inject_compaction_context(self) -> bool:
        """Inject compaction summaries as context reminder.

        Adds a message summarizing compacted content if summaries exist.

        Returns:
            True if context was injected
        """
        if not self._compaction_summaries:
            return False

        combined = " | ".join(self._compaction_summaries[-3:])  # Last 3 summaries
        reminder = f"[Context reminder: {combined}]"

        # Insert as assistant message after system prompt
        if len(self.messages) > 1:
            reminder_msg = Message(role="assistant", content=reminder)
            self._history._messages.insert(1, reminder_msg)
            logger.debug(f"Injected compaction context: {reminder[:100]}...")
            return True

        return False

    def set_embedding_service(self, service: "EmbeddingService") -> None:
        """Set the embedding service for semantic compaction.

        Args:
            service: EmbeddingService instance
        """
        self._embedding_service = service
        # Also set on conversation store if available
        if self._conversation_store:
            self._conversation_store.set_embedding_service(service)

    def set_conversation_store(
        self,
        store: "ConversationStore",
        session_id: str,
    ) -> None:
        """Set the persistent conversation store for enhanced compaction.

        Enables semantic retrieval from full conversation history stored in SQLite.

        Args:
            store: ConversationStore instance
            session_id: Session ID for store operations
        """
        self._conversation_store = store
        self._session_id = session_id
        # Share embedding service with store
        if self._embedding_service:
            store.set_embedding_service(self._embedding_service)

    def retrieve_relevant_history(
        self,
        query: str,
        limit: int = 5,
    ) -> List[str]:
        """Retrieve relevant historical context from persistent store.

        Uses semantic similarity to find relevant messages and summaries
        from the full conversation history in SQLite.

        Args:
            query: Current query to find relevant history for
            limit: Maximum items to retrieve

        Returns:
            List of relevant historical context strings
        """
        if not self._conversation_store or not self._session_id:
            return []

        relevant_context: List[str] = []

        try:
            # Get semantically similar historical messages
            relevant_msgs = self._conversation_store.get_semantically_relevant_messages(
                self._session_id,
                query,
                limit=limit,
                min_similarity=self.config.semantic_relevance_threshold,
            )

            for msg, score in relevant_msgs:
                # Format as context reminder
                role_label = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                context = f"[Historical {role_label} (relevance: {score:.2f})]: {msg.content[:500]}"
                relevant_context.append(context)

            # Get relevant summaries
            relevant_summaries = self._conversation_store.get_relevant_summaries(
                self._session_id,
                query,
                limit=2,
            )

            for summary, score in relevant_summaries:
                context = f"[Prior context summary (relevance: {score:.2f})]: {summary}"
                relevant_context.append(context)

        except (OSError, IOError) as e:
            logger.warning(f"Failed to retrieve relevant history (I/O error): {e}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to retrieve relevant history (data error): {e}")

        return relevant_context[:limit]

    def persist_compaction_summary(self, summary: str, message_ids: List[str]) -> None:
        """Persist a compaction summary to the SQLite store.

        Args:
            summary: Summary of compacted content
            message_ids: IDs of messages that were summarized
        """
        if not self._conversation_store or not self._session_id:
            return

        try:
            self._conversation_store.store_compaction_summary(
                self._session_id,
                summary,
                message_ids,
            )
        except (OSError, IOError) as e:
            logger.warning(f"Failed to persist compaction summary (I/O error): {e}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to persist compaction summary (data error): {e}")

    def on_context_overflow(self, callback: Callable[[ContextMetrics], None]) -> None:
        """Register callback for context overflow events.

        Args:
            callback: Function to call when overflow is detected
        """
        self._context_callbacks.append(callback)

    def _notify_context_callbacks(self, metrics: ContextMetrics) -> None:
        """Notify registered callbacks about context state."""
        for callback in self._context_callbacks:
            try:
                callback(metrics)
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning(f"Context callback failed (invalid callback): {e}")
            except Exception as e:
                # Log with traceback for unexpected callback failures
                logger.exception(f"Context callback failed unexpectedly: {e}")

    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message.

        Returns:
            Last user message content or None
        """
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message.

        Returns:
            Last assistant message content or None
        """
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Export conversation state as dictionary.

        Returns:
            Dictionary representation of conversation
        """
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "stage": self.stage.value,
            "metrics": {
                "char_count": self.get_context_metrics().char_count,
                "message_count": self.message_count,
            },
        }

    @classmethod
    def from_messages(
        cls,
        messages: List[Message],
        config: Optional[ConversationConfig] = None,
    ) -> "ConversationController":
        """Create controller from existing messages.

        Args:
            messages: List of messages to initialize with
            config: Optional configuration

        Returns:
            New ConversationController with messages
        """
        controller = cls(config=config)
        for msg in messages:
            controller._history._messages.append(msg)
        if messages and messages[0].role == "system":
            controller._system_prompt = messages[0].content
            controller._system_added = True
        return controller
