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

"""Session domain facade for orchestrator decomposition.

Groups session state, ledger, persistence, checkpoint, and lifecycle
coordination components behind a single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SessionFacade:
    """Groups session state, ledger, and lifecycle components.

    Satisfies ``SessionFacadeProtocol`` structurally.  The orchestrator creates
    this facade after all session-domain components are initialized, passing
    references to the already-created instances.

    Components managed:
        - session_state: SessionStateManager for tracking tool calls, budget, etc.
        - session_accessor: SessionStateAccessor for consolidated delegation
        - session_ledger: SessionLedger for append-only event log
        - lifecycle_manager: LifecycleManager for session boundaries
        - active_session_id: Active session identifier
        - memory_session_id: Session identifier for memory operations
        - profile_name: Optional profile name for session tracking
        - checkpoint_manager: ConversationCheckpointManager for time-travel
    """

    def __init__(
        self,
        *,
        session_state: Any,
        session_accessor: Any,
        session_ledger: Any,
        lifecycle_manager: Optional[Any] = None,
        active_session_id: Optional[str] = None,
        memory_session_id: Optional[str] = None,
        profile_name: Optional[str] = None,
        checkpoint_manager: Optional[Any] = None,
    ) -> None:
        self._session_state = session_state
        self._session_accessor = session_accessor
        self._session_ledger = session_ledger
        self._lifecycle_manager = lifecycle_manager
        self._active_session_id = active_session_id
        self._memory_session_id = memory_session_id
        self._profile_name = profile_name
        self._checkpoint_manager = checkpoint_manager

        logger.debug(
            "SessionFacade initialized (session_id=%s, profile=%s, checkpoint=%s)",
            active_session_id,
            profile_name,
            checkpoint_manager is not None,
        )

    # ------------------------------------------------------------------
    # Properties (satisfy SessionFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def session_state(self) -> Any:
        """SessionStateManager for tracking tool calls, budget, etc."""
        return self._session_state

    @property
    def session_accessor(self) -> Any:
        """SessionStateAccessor for consolidated state delegation."""
        return self._session_accessor

    @property
    def session_ledger(self) -> Any:
        """SessionLedger for append-only event log."""
        return self._session_ledger

    @session_ledger.setter
    def session_ledger(self, value: Any) -> None:
        """Update the session ledger."""
        self._session_ledger = value

    @property
    def lifecycle_manager(self) -> Optional[Any]:
        """Optional lifecycle manager for session boundaries."""
        return self._lifecycle_manager

    @property
    def active_session_id(self) -> Optional[str]:
        """Active session identifier."""
        return self._active_session_id

    @active_session_id.setter
    def active_session_id(self, value: Optional[str]) -> None:
        """Update the active session identifier."""
        self._active_session_id = value

    @property
    def memory_session_id(self) -> Optional[str]:
        """Session identifier for memory operations."""
        return self._memory_session_id

    @memory_session_id.setter
    def memory_session_id(self, value: Optional[str]) -> None:
        """Update the memory session identifier."""
        self._memory_session_id = value

    @property
    def profile_name(self) -> Optional[str]:
        """Optional profile name for session tracking."""
        return self._profile_name

    @property
    def checkpoint_manager(self) -> Optional[Any]:
        """ConversationCheckpointManager for time-travel debugging."""
        return self._checkpoint_manager
