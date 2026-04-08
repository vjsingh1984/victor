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

"""Turn-boundary context assembler.

Selects and arranges messages for each LLM call, respecting token budgets
while preserving the session ledger, recent turns, and high-value older messages.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from victor.config.orchestrator_constants import (
    ContextAssemblerConfig,
    CONTEXT_ASSEMBLER_CONFIG,
)
from victor.providers.base import Message

logger = logging.getLogger(__name__)


class TurnBoundaryContextAssembler:
    """Assembles the message list for each provider.chat() call.

    Behavior:
    1. Always keeps system prompt + rendered session ledger
    2. Keeps last `full_turn_count` turns in full
    3. For older messages: scores using score_fn, selects top within budget
    4. Returns new List[Message] — does NOT modify input
    5. If no ledger/score_fn, returns messages unchanged (graceful degradation)
    """

    def __init__(
        self,
        config: Optional[ContextAssemblerConfig] = None,
        session_ledger: Optional[object] = None,
        score_fn: Optional[Callable] = None,
        conversation_controller: Optional[object] = None,
    ):
        self._config = config or CONTEXT_ASSEMBLER_CONFIG
        self._session_ledger = session_ledger
        self._score_fn = score_fn
        self._conversation_controller = conversation_controller

    @property
    def config(self) -> ContextAssemblerConfig:
        return self._config

    def assemble(
        self,
        messages: List[Message],
        max_context_chars: int,
        current_query: Optional[str] = None,
    ) -> List[Message]:
        """Assemble messages for an LLM call within token budget.

        Args:
            messages: Full conversation history
            max_context_chars: Maximum context size in characters
            current_query: Current user query for relevance scoring

        Returns:
            New list of messages fitting within budget
        """
        if not messages:
            return []

        # Graceful degradation: if no ledger and no score_fn, pass through
        if self._session_ledger is None and self._score_fn is None:
            return list(messages)

        # Budget allocation
        history_budget = int(max_context_chars * self._config.history_budget_pct)
        ledger_budget = int(max_context_chars * self._config.ledger_budget_pct)

        result: List[Message] = []
        chars_used = 0

        # 1. Keep system prompt (first message if role=system)
        start_idx = 0
        if messages and messages[0].role == "system":
            result.append(messages[0])
            chars_used += len(messages[0].content)
            start_idx = 1

        # 2. Inject ledger as early assistant message
        if self._session_ledger is not None:
            render_fn = getattr(self._session_ledger, "render", None)
            if render_fn:
                ledger_text = render_fn(max_chars=ledger_budget)
                if ledger_text:
                    ledger_msg = Message(role="assistant", content=ledger_text)
                    result.append(ledger_msg)
                    chars_used += len(ledger_text)

        remaining = messages[start_idx:]
        if not remaining:
            return result

        # 3. Identify recent turns to keep in full
        full_turn_count = self._config.full_turn_count
        # Count turns (a turn = user message + subsequent assistant/tool messages)
        turn_boundaries = []
        for i, msg in enumerate(remaining):
            if msg.role == "user":
                turn_boundaries.append(i)

        if len(turn_boundaries) <= full_turn_count:
            # All messages fit as recent turns
            for msg in remaining:
                result.append(msg)
                chars_used += len(msg.content)
            return result

        # Split: older messages vs recent turns
        recent_start = turn_boundaries[-full_turn_count]
        older_messages = remaining[:recent_start]
        recent_messages = remaining[recent_start:]

        # Calculate budget for older messages
        recent_chars = sum(len(m.content) for m in recent_messages)
        older_budget = max(0, history_budget - chars_used - recent_chars)

        # 4. Score and select older messages
        older_chars = 0
        if older_messages and self._score_fn and older_budget > 0:
            scored = self._score_fn(older_messages, current_query)
            # scored should be list of (message, score) or similar
            if scored and isinstance(scored[0], tuple):
                scored.sort(key=lambda x: x[1], reverse=True)
                selected_older = []
                older_chars = 0
                for msg, score in scored:
                    msg_len = len(msg.content)
                    if older_chars + msg_len <= older_budget:
                        selected_older.append((msg, score, older_messages.index(msg)))
                        older_chars += msg_len
                # Sort by original order
                selected_older.sort(key=lambda x: x[2])
                for msg, _, _ in selected_older:
                    result.append(msg)
            else:
                # score_fn returned non-tuple, include what fits
                older_chars = 0
                for msg in older_messages:
                    if older_chars + len(msg.content) <= older_budget:
                        result.append(msg)
                        older_chars += len(msg.content)
        elif older_messages and older_budget > 0:
            # No score_fn, include what fits from the end (most recent older msgs)
            older_chars = 0
            for msg in reversed(older_messages):
                if older_chars + len(msg.content) <= older_budget:
                    result.insert(len(result), msg)
                    older_chars += len(msg.content)

        # 5. Semantic augmentation: retrieve relevant compacted context
        if self._conversation_controller and current_query:
            retrieve_fn = getattr(
                self._conversation_controller, "retrieve_relevant_history", None
            )
            if retrieve_fn:
                try:
                    remaining_budget = older_budget - older_chars
                    if remaining_budget > 500:
                        relevant = retrieve_fn(query=current_query, limit=3)
                        semantic_chars = 0
                        for ctx_str in relevant:
                            if semantic_chars + len(ctx_str) <= remaining_budget:
                                result.append(
                                    Message(
                                        role="assistant",
                                        content=f"[Historical context: {ctx_str}]",
                                    )
                                )
                                semantic_chars += len(ctx_str)
                except Exception as e:
                    logger.debug(f"Semantic retrieval failed gracefully: {e}")

        # 6. Append recent turns (always kept)
        for msg in recent_messages:
            result.append(msg)

        return result
