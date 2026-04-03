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

"""Resilience domain facade for orchestrator decomposition.

Groups error recovery, context management, RL coordination, code execution,
background task management, and streaming cancellation components behind
a single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Set

logger = logging.getLogger(__name__)


class ResilienceFacade:
    """Groups resilience, recovery, and runtime management components.

    Satisfies ``ResilienceFacadeProtocol`` structurally.  The orchestrator
    creates this facade after all resilience-domain components are initialized,
    passing references to the already-created instances.

    Components managed:
        - resilience_runtime: Resilience runtime boundary components
        - recovery_handler: Optional recovery handler for model failures
        - recovery_integration: Recovery integration submodule
        - recovery_coordinator: StreamingRecoveryCoordinator
        - chunk_generator: ChunkGenerator for streaming output
        - context_manager: Centralized context window management
        - rl_coordinator: RL coordinator for framework-level learning
        - code_manager: Code execution manager (Docker-based)
        - background_tasks: Set of background asyncio tasks
        - cancel_event: Cancellation event for streaming
        - is_streaming: Whether currently streaming
    """

    def __init__(
        self,
        *,
        resilience_runtime: Optional[Any] = None,
        recovery_handler: Optional[Any] = None,
        recovery_integration: Optional[Any] = None,
        recovery_coordinator: Optional[Any] = None,
        chunk_generator: Optional[Any] = None,
        context_manager: Optional[Any] = None,
        rl_coordinator: Optional[Any] = None,
        code_manager: Optional[Any] = None,
        background_tasks: Optional[Set[asyncio.Task]] = None,
        cancel_event: Optional[asyncio.Event] = None,
        is_streaming: bool = False,
    ) -> None:
        self._resilience_runtime = resilience_runtime
        self._recovery_handler = recovery_handler
        self._recovery_integration = recovery_integration
        self._recovery_coordinator = recovery_coordinator
        self._chunk_generator = chunk_generator
        self._context_manager = context_manager
        self._rl_coordinator = rl_coordinator
        self._code_manager = code_manager
        self._background_tasks = background_tasks if background_tasks is not None else set()
        self._cancel_event = cancel_event
        self._is_streaming = is_streaming

        logger.debug(
            "ResilienceFacade initialized (recovery=%s, rl=%s, code=%s)",
            recovery_handler is not None,
            rl_coordinator is not None,
            code_manager is not None,
        )

    # ------------------------------------------------------------------
    # Properties (satisfy ResilienceFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def resilience_runtime(self) -> Optional[Any]:
        """Resilience runtime boundary components."""
        return self._resilience_runtime

    @property
    def recovery_handler(self) -> Optional[Any]:
        """Optional recovery handler for model failure recovery."""
        return self._recovery_handler

    @recovery_handler.setter
    def recovery_handler(self, value: Any) -> None:
        """Update the recovery handler."""
        self._recovery_handler = value

    @property
    def recovery_integration(self) -> Optional[Any]:
        """Recovery integration submodule."""
        return self._recovery_integration

    @recovery_integration.setter
    def recovery_integration(self, value: Any) -> None:
        """Update the recovery integration."""
        self._recovery_integration = value

    @property
    def recovery_coordinator(self) -> Optional[Any]:
        """StreamingRecoveryCoordinator for centralized recovery logic."""
        return self._recovery_coordinator

    @property
    def chunk_generator(self) -> Optional[Any]:
        """ChunkGenerator for streaming output."""
        return self._chunk_generator

    @property
    def context_manager(self) -> Optional[Any]:
        """Centralized context window management."""
        return self._context_manager

    @property
    def rl_coordinator(self) -> Optional[Any]:
        """RL coordinator for framework-level learning."""
        return self._rl_coordinator

    @property
    def code_manager(self) -> Optional[Any]:
        """Code execution manager (Docker-based)."""
        return self._code_manager

    @property
    def background_tasks(self) -> Set[asyncio.Task]:
        """Set of background asyncio tasks."""
        return self._background_tasks

    @property
    def cancel_event(self) -> Optional[asyncio.Event]:
        """Cancellation event for streaming."""
        return self._cancel_event

    @cancel_event.setter
    def cancel_event(self, value: Optional[asyncio.Event]) -> None:
        """Update the cancellation event."""
        self._cancel_event = value

    @property
    def is_streaming(self) -> bool:
        """Whether currently streaming."""
        return self._is_streaming

    @is_streaming.setter
    def is_streaming(self, value: bool) -> None:
        """Update the streaming flag."""
        self._is_streaming = value
