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

"""Cache key generation for tool selection caching.

This module provides functions to generate consistent, collision-resistant cache keys
for tool selection operations. Keys are based on:

1. Query text (user prompt)
2. Tools registry state (hash of all tool definitions)
3. Selector configuration (weights, thresholds, etc.)
4. Conversation context (last N messages for context-aware caching)
5. Pending actions (for context-aware caching)

The key generation strategy ensures:
- Cache invalidation when tools are added/removed/modified
- Cache invalidation when configuration changes
- Context-aware keys for context-dependent selections
- RL-aware keys for time-bounded RL boost caching

Example:
    from victor.tools.caches import CacheKeyGenerator

    key_gen = CacheKeyGenerator()

    # Query selection cache key
    query_key = key_gen.generate_query_key(
        query="read the file",
        tools_hash="abc123",
        config_hash="def456"
    )

    # Context-aware selection cache key
    context_key = key_gen.generate_context_key(
        query="read the file",
        tools_hash="abc123",
        conversation_history=[...],
        pending_actions=["edit", "commit"]
    )
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class CacheKeyGenerator:
    """Generates consistent cache keys for tool selection operations.

    Cache keys are SHA256 hashes truncated to 16 characters for efficiency
    while maintaining collision resistance.

    Key Types:
        1. query_key: For simple query-based selection (query + tools + config)
        2. context_key: For context-aware selection (includes history)
        3. rl_key: For RL ranking (task_type + tools + hour_bucket)
    """

    # Number of characters to truncate hash to (16 hex chars = 64 bits)
    HASH_TRUNCATE_LENGTH: int = 16

    # Maximum history messages to include in context key
    MAX_HISTORY_MESSAGES: int = 3

    def __init__(self) -> None:
        """Initialize cache key generator."""
        self._tools_hash_cache: Optional[str] = None
        self._tools_hash_registry_id: Optional[int] = None

    def generate_query_key(
        self,
        query: str,
        tools_hash: str,
        config_hash: str,
    ) -> str:
        """Generate cache key for query-based tool selection.

        Args:
            query: User query/prompt text
            tools_hash: Hash of current tools registry state
            config_hash: Hash of selector configuration

        Returns:
            Truncated SHA256 hash as cache key

        Example:
            key = gen.generate_query_key(
                query="read the file",
                tools_hash="abc123...",
                config_hash="def456..."
            )
        """
        # Normalize query (lowercase, strip)
        normalized_query = query.strip().lower()

        # Combine components
        combined = f"{normalized_query}|{tools_hash}|{config_hash}"
        return self._hash(combined)

    def generate_context_key(
        self,
        query: str,
        tools_hash: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        pending_actions: Optional[List[str]] = None,
    ) -> str:
        """Generate cache key for context-aware tool selection.

        Includes last N messages from history to capture conversation context.

        Args:
            query: User query/prompt text
            tools_hash: Hash of current tools registry state
            conversation_history: Optional conversation history
            pending_actions: Optional list of pending action types

        Returns:
            Truncated SHA256 hash as cache key

        Example:
            key = gen.generate_context_key(
                query="and now edit it",
                tools_hash="abc123...",
                conversation_history=[
                    {"role": "user", "content": "read the file"},
                    {"role": "assistant", "content": "..."}
                ],
                pending_actions=["edit"]
            )
        """
        # Normalize query
        normalized_query = query.strip().lower()

        # Extract last N messages for context
        history_hash = self._hash_history(conversation_history or [])

        # Hash pending actions
        pending_hash = self._hash_pending_actions(pending_actions or [])

        # Combine components
        combined = f"{normalized_query}|{tools_hash}|{history_hash}|{pending_hash}"
        return self._hash(combined)

    def generate_rl_key(
        self,
        task_type: str,
        tools_hash: str,
        hour_bucket: int,
    ) -> str:
        """Generate cache key for RL-based tool ranking.

        RL rankings change over time as Q-values are updated, so we include
        an hour bucket to limit cache lifetime.

        Args:
            task_type: Task type for RL ranking
            tools_hash: Hash of current tools registry state
            hour_bucket: Hour of day (0-23) for time-bounded caching

        Returns:
            Truncated SHA256 hash as cache key

        Example:
            import time
            hour_bucket = int(time.time()) // 3600
            key = gen.generate_rl_key(
                task_type="analysis",
                tools_hash="abc123...",
                hour_bucket=hour_bucket
            )
        """
        combined = f"{task_type}|{tools_hash}|hour:{hour_bucket}"
        return self._hash(combined)

    def calculate_tools_hash(self, tools: "ToolRegistry") -> str:
        """Calculate hash of tools registry state for cache invalidation.

        Caches the result to avoid redundant calculations.

        Args:
            tools: Tool registry to hash

        Returns:
            SHA256 hash truncated to 16 characters

        Example:
            tools_hash = gen.calculate_tools_hash(tools_registry)
        """
        # Check cache using registry id as proxy
        registry_id = id(tools)
        if self._tools_hash_registry_id == registry_id and self._tools_hash_cache:
            return self._tools_hash_cache

        # Calculate hash from tool definitions
        tool_list = sorted(tools.list_tools(), key=lambda t: t.name)
        tool_names = sorted([t.name for t in tool_list])

        # Build hash string
        parts = [f"count:{len(tool_list)}", f"names:{','.join(tool_names)}"]

        for tool in tool_list:
            # Include name, description, and parameters in hash
            tool_str = f"{tool.name}:{tool.description}:{tool.parameters}"
            parts.append(tool_str)

        combined = "|".join(parts)
        self._tools_hash_cache = self._hash(combined)
        self._tools_hash_registry_id = registry_id

        return self._tools_hash_cache

    def calculate_config_hash(
        self,
        semantic_weight: float,
        keyword_weight: float,
        max_tools: int,
        similarity_threshold: float,
    ) -> str:
        """Calculate hash of selector configuration for cache invalidation.

        Args:
            semantic_weight: Weight for semantic results
            keyword_weight: Weight for keyword results
            max_tools: Maximum tools to return
            similarity_threshold: Minimum similarity threshold

        Returns:
            SHA256 hash truncated to 16 characters

        Example:
            config_hash = gen.calculate_config_hash(
                semantic_weight=0.7,
                keyword_weight=0.3,
                max_tools=10,
                similarity_threshold=0.18
            )
        """
        combined = (
            f"semantic:{semantic_weight}|"
            f"keyword:{keyword_weight}|"
            f"max:{max_tools}|"
            f"threshold:{similarity_threshold}"
        )
        return self._hash(combined)

    def invalidate_tools_cache(self) -> None:
        """Invalidate cached tools hash.

        Call this when tools are added/removed/modified.
        """
        self._tools_hash_cache = None
        self._tools_hash_registry_id = None
        logger.debug("Tools hash cache invalidated")

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _hash(self, data: str) -> str:
        """Generate truncated SHA256 hash.

        Args:
            data: String to hash

        Returns:
            Truncated hash (16 hex chars)
        """
        full_hash = hashlib.sha256(data.encode()).hexdigest()
        return full_hash[: self.HASH_TRUNCATE_LENGTH]

    def _hash_history(self, history: List[Dict[str, Any]]) -> str:
        """Hash conversation history for context key.

        Only includes last N messages to keep cache effective while
        capturing recent context.

        Args:
            history: Conversation history

        Returns:
            Truncated hash of history
        """
        if not history:
            return "none"

        # Take last N messages
        recent = history[-self.MAX_HISTORY_MESSAGES :]

        # Build simplified representation
        parts = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:100]  # Truncate long content
            parts.append(f"{role}:{content}")

        combined = "|".join(parts)
        return self._hash(combined)

    def _hash_pending_actions(self, actions: List[str]) -> str:
        """Hash pending actions for context key.

        Args:
            actions: List of pending action types

        Returns:
            Truncated hash of actions
        """
        if not actions:
            return "none"

        # Sort for consistency
        sorted_actions = sorted(actions)
        combined = ",".join(sorted_actions)
        return self._hash(combined)


# Global singleton instance for convenience
_global_key_generator: Optional[CacheKeyGenerator] = None


def get_cache_key_generator() -> CacheKeyGenerator:
    """Get global cache key generator instance.

    Returns:
        Shared CacheKeyGenerator instance
    """
    global _global_key_generator
    if _global_key_generator is None:
        _global_key_generator = CacheKeyGenerator()
    return _global_key_generator


def generate_query_key(
    query: str,
    tools_hash: str,
    config_hash: str,
) -> str:
    """Convenience function to generate query cache key.

    Args:
        query: User query/prompt text
        tools_hash: Hash of current tools registry state
        config_hash: Hash of selector configuration

    Returns:
        Truncated SHA256 hash as cache key
    """
    return get_cache_key_generator().generate_query_key(query, tools_hash, config_hash)


def generate_context_key(
    query: str,
    tools_hash: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    pending_actions: Optional[List[str]] = None,
) -> str:
    """Convenience function to generate context cache key.

    Args:
        query: User query/prompt text
        tools_hash: Hash of current tools registry state
        conversation_history: Optional conversation history
        pending_actions: Optional list of pending action types

    Returns:
        Truncated SHA256 hash as cache key
    """
    return get_cache_key_generator().generate_context_key(
        query, tools_hash, conversation_history, pending_actions
    )


def calculate_tools_hash(tools: "ToolRegistry") -> str:
    """Convenience function to calculate tools registry hash.

    Args:
        tools: Tool registry to hash

    Returns:
        SHA256 hash truncated to 16 characters
    """
    return get_cache_key_generator().calculate_tools_hash(tools)


__all__ = [
    "CacheKeyGenerator",
    "get_cache_key_generator",
    "generate_query_key",
    "generate_context_key",
    "calculate_tools_hash",
]
