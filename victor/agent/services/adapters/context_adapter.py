"""Context service adapter that wraps ConversationController and ContextCompactor.

Implements ContextServiceProtocol by delegating to the existing
conversation controller and context compactor components.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.context_compactor import ContextCompactor

logger = logging.getLogger(__name__)


class ContextServiceAdapter:
    """Adapts ConversationController + ContextCompactor to ContextServiceProtocol.

    This adapter bridges existing conversation management components
    to the service protocol interface.
    """

    def __init__(
        self,
        conversation_controller: "ConversationController",
        context_compactor: Optional["ContextCompactor"] = None,
    ) -> None:
        self._conversation_controller = conversation_controller
        self._context_compactor = context_compactor
        self._last_compaction_saved_tokens: int = 0

    def get_context_metrics(self) -> Any:
        """Get current context metrics from the conversation controller."""
        return self._conversation_controller.get_context_metrics()

    async def check_context_overflow(self) -> bool:
        """Check whether the underlying conversation is at overflow risk."""
        return self._conversation_controller.check_context_overflow()

    async def compact_context(
        self,
        strategy: str = "tiered",
        min_messages: int = 6,
    ) -> int:
        """Compact context via the context compactor."""
        if self._context_compactor is None:
            return 0
        result = self._context_compactor.check_and_compact(force=True)
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "messages_removed"):
            self._last_compaction_saved_tokens = int(getattr(result, "tokens_freed", 0) or 0)
            return int(getattr(result, "messages_removed", 0) or 0)
        return int(result) if isinstance(result, int) else 0

    async def prepare_for_tool_output_injection(
        self,
        estimated_output_tokens: int,
        *,
        provider_name: str = "",
        model_name: str = "",
        task_type: str = "",
        min_messages: int = 6,
        default_strategy: str = "tiered",
    ) -> Dict[str, Any]:
        """Use the canonical service policy for pre-tool-output compaction."""
        from victor.agent.services.context_service import _build_tool_output_compaction_decision

        decision = _build_tool_output_compaction_decision(
            estimated_output_tokens=estimated_output_tokens,
            remaining_tokens=self.get_remaining_tokens(),
            utilization_percent=self.get_context_utilization(),
            recommendation=self.get_compaction_recommendation(),
            provider_name=provider_name,
            model_name=model_name,
            task_type=task_type,
            default_strategy=default_strategy,
        )
        if not decision["should_compact"]:
            return decision

        removed = await self.compact_context(
            strategy=str(decision["strategy"] or default_strategy),
            min_messages=min_messages,
        )
        decision["messages_removed"] = removed
        decision["compacted"] = removed > 0
        if removed > 0:
            decision["saved_tokens"] = self._last_compaction_saved_tokens
        return decision

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the conversation."""
        self._conversation_controller.add_message(role, content, **metadata)

    def add_messages(self, messages: List[Any]) -> None:
        """Add multiple messages to the conversation."""
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
                metadata = {
                    key: value for key, value in message.items() if key not in {"role", "content"}
                }
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
                metadata = {}
                for key in ("name", "tool_calls", "tool_call_id", "metadata"):
                    value = getattr(message, key, None)
                    if value is not None:
                        metadata[key] = value

            if role is None or content is None:
                continue
            self.add_message(role, content, **metadata)

    def get_messages(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List[Any]:
        """Get messages from conversation history."""
        messages = self._conversation_controller.messages
        if role:
            messages = [m for m in messages if getattr(m, "role", None) == role]
        if limit:
            messages = messages[-limit:]
        return messages

    def clear_messages(self, retain_system: bool = True) -> None:
        """Clear conversation messages while optionally preserving the system prompt."""
        self._conversation_controller.reset()
        if retain_system and hasattr(self._conversation_controller, "ensure_system_message"):
            self._conversation_controller.ensure_system_message()

    def manages_conversation_controller(self, controller: Any) -> bool:
        """Return whether this adapter owns the provided conversation controller."""
        return self._conversation_controller is controller

    def get_max_tokens(self) -> int:
        """Return the approximate max token budget from controller configuration."""
        config = getattr(self._conversation_controller, "config", None)
        if config is None:
            return 0
        max_chars = getattr(config, "max_context_chars", 0)
        chars_per_token = max(getattr(config, "chars_per_token_estimate", 4), 1)
        return max_chars // chars_per_token

    def get_context_utilization(self) -> float:
        """Expose utilization for compatibility paths that still use the adapter."""
        max_tokens = self.get_max_tokens()
        if max_tokens <= 0 or self._conversation_controller is None:
            return 0.0
        metrics = self._conversation_controller.get_context_metrics()
        current_tokens = int(getattr(metrics, "estimated_tokens", 0) or 0)
        return (current_tokens / max_tokens) * 100 if max_tokens > 0 else 0.0

    def get_remaining_tokens(self) -> int:
        """Expose remaining token capacity for compatibility paths."""
        max_tokens = self.get_max_tokens()
        if max_tokens <= 0 or self._conversation_controller is None:
            return 0
        metrics = self._conversation_controller.get_context_metrics()
        current_tokens = int(getattr(metrics, "estimated_tokens", 0) or 0)
        return max(0, max_tokens - current_tokens)

    def get_compaction_recommendation(self) -> Dict[str, Any]:
        """Expose a recommendation surface matching the canonical context service."""
        current_tokens = max(0, self.get_max_tokens() - self.get_remaining_tokens())
        utilization = self.get_context_utilization()
        threshold_percent = self._get_compaction_threshold_percent()
        should_compact = utilization >= threshold_percent
        message_count = len(self.get_messages())
        recommended_removal = 0
        if should_compact and message_count > 0:
            target_tokens = max(0.0, ((threshold_percent - 10.0) / 100.0) * self.get_max_tokens())
            excess_tokens = max(0.0, float(current_tokens) - target_tokens)
            avg_tokens = max(current_tokens / message_count, 1)
            recommended_removal = int(excess_tokens / avg_tokens) + 1 if excess_tokens > 0 else 0
        return {
            "should_compact": should_compact,
            "current_tokens": current_tokens,
            "max_tokens": self.get_max_tokens(),
            "utilization_percent": utilization,
            "threshold_percent": threshold_percent,
            "recommended_removal": recommended_removal,
            "message_count": message_count,
        }

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the approximate max token budget on controller configuration."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")

        config = getattr(self._conversation_controller, "config", None)
        if config is None:
            return
        chars_per_token = max(getattr(config, "chars_per_token_estimate", 4), 1)
        config.max_context_chars = max_tokens * chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using the controller configuration heuristic."""
        config = getattr(self._conversation_controller, "config", None)
        chars_per_token = max(getattr(config, "chars_per_token_estimate", 4), 1)
        return len(text) // chars_per_token

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Expose compaction savings metrics when backed by a real compactor."""
        stats = self._get_compactor_statistics()
        return {
            "average_compaction_time": 0.0,
            "compaction_efficiency": 0.0,
            "operation_count": int(stats.get("compaction_count", 0) or 0),
            "cache_hit_rate": 0.0,
            "last_compaction_saved_tokens": self._last_compaction_saved_tokens,
            "total_tokens_saved": int(stats.get("total_tokens_freed", 0) or 0),
        }

    def is_healthy(self) -> bool:
        """Check if the context service is healthy."""
        return self._conversation_controller is not None

    def _get_compaction_threshold_percent(self) -> float:
        """Return the best available compaction threshold for this adapter path."""
        compactor_config = getattr(self._context_compactor, "config", None)
        threshold = getattr(compactor_config, "proactive_threshold", None)
        if threshold is None:
            return 90.0
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            return 90.0
        return threshold_value * 100.0 if threshold_value <= 1.0 else threshold_value

    def _get_compactor_statistics(self) -> Dict[str, Any]:
        """Return compactor statistics when available."""
        getter = getattr(self._context_compactor, "get_statistics", None)
        if not callable(getter):
            return {}
        try:
            stats = getter() or {}
        except Exception:
            logger.debug("Failed to read context-compactor statistics", exc_info=True)
            return {}
        return stats if isinstance(stats, dict) else {}
