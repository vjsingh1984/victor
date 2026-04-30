# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned session runtime helper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class SessionRuntime:
    """Bridge orchestrator runtime state to the canonical session service."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    def sync_runtime_state(self) -> None:
        """Keep SessionService bound to the current live runtime state."""
        runtime = self._runtime
        if runtime._session_service is None or not hasattr(
            runtime._session_service,
            "bind_runtime_components",
        ):
            return

        runtime._session_service.bind_runtime_components(
            lifecycle_manager=runtime._lifecycle_manager,
            memory_manager=runtime.memory_manager,
            checkpoint_manager=runtime._checkpoint_manager,
            cost_tracker=runtime._session_cost_tracker,
            memory_session_id=runtime._memory_session_id,
        )

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions for recovery."""
        self.sync_runtime_state()
        return self._runtime._session_service.get_recent_sessions(limit)

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session."""
        runtime = self._runtime
        self.sync_runtime_state()
        success = runtime._session_service.recover_session(session_id)
        if success:
            runtime._memory_session_id = session_id
        return success

    def get_memory_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get token-aware context messages from the canonical session service."""
        runtime = self._runtime
        self.sync_runtime_state()
        return runtime._session_service.get_memory_context(
            max_tokens=max_tokens,
            messages=runtime.messages,
        )

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current memory session."""
        self.sync_runtime_state()
        return self._runtime._session_service.get_session_stats()
