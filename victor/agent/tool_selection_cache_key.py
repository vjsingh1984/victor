from __future__ import annotations

from typing import Any


class SemanticToolSelectionCacheKeyBuilder:
    """Canonical cache-key builder for semantic tool selection."""

    def build(
        self,
        *,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None,
        conversation_depth: int,
        stage: Any,
    ) -> str:
        from victor.storage.cache.generic_result_cache import _create_tool_selection_cache_key

        return _create_tool_selection_cache_key(
            user_message=user_message,
            conversation_history=conversation_history,
            conversation_depth=conversation_depth,
            stage=stage.value if stage is not None else None,
            use_semantic=True,
        )
