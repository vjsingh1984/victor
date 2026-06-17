"""Shared runtime chat resolution helpers.

This module owns the lower-level chat-runtime contract used by framework and
integration surfaces:
- resolve the canonical chat service when one exists
- fall back to direct orchestrator chat/stream_chat only through a wrapped
  runtime surface
- suppress direct-orchestrator deprecation warnings at the compatibility edge
"""

from __future__ import annotations

import warnings
from typing import Any, AsyncIterator

from victor.runtime.context import resolve_runtime_services


class _OrchestratorChatRuntimeAdapter:
    """Internal adapter preserving a chat-runtime shape on orchestrator fallback."""

    def __init__(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator

    async def chat(self, message: str) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Direct orchestrator\\.chat\\(\\) access is deprecated\\..*",
                category=DeprecationWarning,
            )
            return await self._orchestrator.chat(message)

    async def stream_chat(self, message: str) -> AsyncIterator[Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Direct orchestrator\\.stream_chat\\(\\) access is deprecated\\..*",
                category=DeprecationWarning,
            )
            async for chunk in self._orchestrator.stream_chat(message):
                yield chunk


def resolve_chat_service(runtime_owner: Any, execution_context: Any = None) -> Any:
    """Resolve the canonical chat service instance when available."""
    return resolve_runtime_services(runtime_owner, execution_context).chat


def resolve_chat_runtime(runtime_owner: Any, execution_context: Any = None) -> Any:
    """Resolve the canonical chat runtime for framework/integration callers."""
    services = resolve_runtime_services(runtime_owner, execution_context)
    if services.chat is not None:
        return services.chat
    return _OrchestratorChatRuntimeAdapter(runtime_owner)


__all__ = [
    "resolve_chat_runtime",
    "resolve_chat_service",
]
