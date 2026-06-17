"""Removed streaming pipeline compatibility surface.

This module intentionally fails on import. The canonical streaming runtime is
``victor.agent.services.chat_stream_executor.StreamingChatExecutor``.
"""

raise ImportError(
    "victor.agent.streaming.pipeline has been removed. "
    "Use victor.agent.services.chat_stream_executor.StreamingChatExecutor or "
    "ChatService.stream_chat() instead."
)
