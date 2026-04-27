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

"""Service-owned compatibility shim for deprecated unified chat coordination.

The canonical chat entry point is `ChatService`. This module only keeps the
older sync/streaming facade shape available while callers migrate away from
coordinator-first orchestration, but the implementation now lives under
`victor.agent.services`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, AsyncIterator, TYPE_CHECKING

from victor.agent.services.chat_compat_telemetry import (
    record_deprecated_chat_shim_access,
)
from victor.providers.base import CompletionResponse, StreamChunk
from victor.agent.services.protocols.chat_runtime import ExecutionMode

if TYPE_CHECKING:
    from victor.agent.services.sync_chat_compat import SyncChatCoordinator
    from victor.agent.services.streaming_chat_compat import (
        StreamingChatCoordinator,
    )

logger = logging.getLogger(__name__)


class UnifiedChatCoordinator:
    """Deprecated facade adapter for sync/streaming compatibility.

    This shim preserves the older sync/streaming facade shape while the
    canonical runtime remains service-first. When a ``ChatService`` is bound,
    both sync and streaming paths forward directly to that service.

    Args:
        sync_coordinator: Deprecated legacy compatibility dependency. Unused by
            the canonical service-bound shim path.
        streaming_coordinator: Deprecated legacy compatibility dependency.
            Unused by the canonical service-bound shim path.
        default_mode: Default execution mode (AUTO, SYNC, or STREAMING)
    """

    def __init__(
        self,
        sync_coordinator: "SyncChatCoordinator | None" = None,
        streaming_coordinator: "StreamingChatCoordinator | None" = None,
        default_mode: ExecutionMode = ExecutionMode.AUTO,
        chat_service: Any = None,
    ) -> None:
        """Initialize the UnifiedChatCoordinator.

        Args:
            sync_coordinator: Deprecated compatibility dependency retained for
                constructor stability. No longer required by the canonical shim path.
            streaming_coordinator: Deprecated compatibility dependency retained
                for constructor stability. No longer required by the canonical shim path.
            default_mode: Default execution mode
        """
        self._default_mode = default_mode
        self._chat_service = chat_service

        if self._chat_service is None:
            warnings.warn(
                "UnifiedChatCoordinator without a bound ChatService is deprecated "
                "compatibility construction. Prefer ChatService instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    def bind_chat_service(self, chat_service: Any) -> None:
        """Bind the canonical ChatService for backward-compatible delegation."""
        self._chat_service = chat_service

    # =====================================================================
    # Public API
    # =====================================================================

    async def chat(
        self,
        user_message: str,
        mode: ExecutionMode = ExecutionMode.AUTO,
        use_planning: bool = False,
    ) -> CompletionResponse:
        """Execute chat in specified mode.

        Compatibility entry point for legacy callers. When a ``ChatService`` is
        bound, this method delegates there directly. Deprecated compatibility
        shims no longer own fallback execution.

        Args:
            user_message: User's message
            mode: Execution mode (SYNC, STREAMING, or AUTO)
            use_planning: Whether to use structured planning for complex tasks

        Returns:
            CompletionResponse with complete response

        Note:
            - SYNC mode: prefers ``ChatService.chat()``
            - STREAMING mode: prefers ``ChatService.chat(..., stream=True)``
            - AUTO mode: resolves to the configured default mode for service delegation
        """
        warnings.warn(
            "UnifiedChatCoordinator.chat() is deprecated compatibility surface. "
            "Use ChatService.chat() or AgentOrchestrator.chat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        execution_mode = self._resolve_execution_mode(mode)

        if self._chat_service is not None:
            if execution_mode == ExecutionMode.STREAMING:
                record_deprecated_chat_shim_access(
                    "unified_chat_coordinator", "chat", "chat_service_streaming"
                )
                return await self._chat_service.chat(
                    user_message,
                    stream=True,
                    use_planning=use_planning,
                )
            record_deprecated_chat_shim_access(
                "unified_chat_coordinator", "chat", "chat_service_sync"
            )
            return await self._chat_service.chat(
                user_message,
                use_planning=use_planning,
            )

        record_deprecated_chat_shim_access(
            "unified_chat_coordinator", "chat", "missing_runtime"
        )
        raise RuntimeError(
            "UnifiedChatCoordinator requires a bound ChatService. "
            "Bind ChatService before using deprecated compatibility shims."
        )

    async def stream_chat(
        self,
        user_message: str,
    ) -> AsyncIterator[StreamChunk]:
        """Execute chat with streaming response.

        Compatibility convenience wrapper that always requests streaming. When a
        ``ChatService`` is bound, it forwards there directly.

        Args:
            user_message: User's message

        Yields:
            StreamChunk objects with incremental response
        """
        warnings.warn(
            "UnifiedChatCoordinator.stream_chat() is deprecated compatibility surface. "
            "Use ChatService.stream_chat() or AgentOrchestrator.stream_chat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if self._chat_service is not None:
            record_deprecated_chat_shim_access(
                "unified_chat_coordinator", "stream_chat", "chat_service"
            )
            async for chunk in self._chat_service.stream_chat(user_message):
                yield chunk
            return

        record_deprecated_chat_shim_access(
            "unified_chat_coordinator", "stream_chat", "missing_runtime"
        )
        raise RuntimeError(
            "UnifiedChatCoordinator requires a bound ChatService. "
            "Bind ChatService before using deprecated compatibility shims."
        )

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _resolve_execution_mode(self, mode: ExecutionMode) -> ExecutionMode:
        """Resolve the execution mode.

        If mode is AUTO, resolve to SYNC or STREAMING based on heuristics.

        Args:
            mode: Requested execution mode

        Returns:
            Resolved execution mode (SYNC or STREAMING)
        """
        if mode != ExecutionMode.AUTO:
            return mode

        # AUTO mode: use default mode
        return self._default_mode


__all__ = [
    "UnifiedChatCoordinator",
]
