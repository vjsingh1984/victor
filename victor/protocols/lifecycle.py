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

"""Lifecycle manager protocol for dependency inversion.

This module defines the ILifecycleManager protocol that enables
dependency injection for session lifecycle management, following the
Dependency Inversion Principle (DIP).

Design Principles:
    - DIP: High-level modules depend on this protocol, not concrete implementations
    - OCP: New lifecycle management strategies can be added without modification
    - SRP: Protocol contains only lifecycle-related methods

Usage:
    class SessionLifecycleManager(ILifecycleManager):
        async def initialize_session(self, session_id: str, config: SessionConfig) -> SessionContext:
            # Initialize new session with resources
            ...

        async def cleanup_session(self, session_id: str) -> CleanupResult:
            # Cleanup session resources
            ...

        async def recover_session(self, session_id: str) -> RecoveryResult:
            # Recover failed session
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class SessionMetadata:
    """Metadata for a session (renamed from SessionContext to avoid conflicts).

    This is distinct from SessionContext in victor/framework/agent_components.py
    which is used for AgentSession components. This type is specifically
    for lifecycle manager session tracking.

    Attributes:
        session_id: Unique session identifier
        created_at: When session was created
        config: Session configuration
        resources: Resources allocated for this session
        metadata: Additional session metadata
    """

    session_id: str
    created_at: str
    config: "SessionConfig"
    resources: Dict[str, Any]
    metadata: Dict[str, Any] | None = None


@dataclass
class SessionConfig:
    """Configuration for a session.

    Attributes:
        provider: LLM provider name
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tools: Tool configuration
        vertical: Optional vertical to use
        metadata: Additional configuration
    """

    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: Any = None
    vertical: Optional[str] = None
    metadata: Dict[str, Any] | None = None


@dataclass
class CleanupResult:
    """Result from session cleanup operation.

    Attributes:
        success: Whether cleanup succeeded
        session_id: Session that was cleaned up
        resources_freed: List of resources freed
        error_message: Error message if cleanup failed
        metadata: Additional cleanup metadata
    """

    success: bool
    session_id: str
    resources_freed: List[str]
    error_message: str | None = None
    metadata: Dict[str, Any] | None = None


@dataclass
class RecoveryResult:
    """Result from session recovery operation.

    Attributes:
        success: Whether recovery succeeded
        session_id: Session that was recovered
        state_recovered: Whether session state was restored
        error_message: Error message if recovery failed
        metadata: Additional recovery metadata
    """

    success: bool
    session_id: str
    state_recovered: bool
    error_message: str | None = None
    metadata: Dict[str, Any] | None = None


@runtime_checkable
class ILifecycleManager(Protocol):
    """Protocol for session lifecycle management.

    Implementations manage the complete lifecycle of sessions:
    - Initialization: Create new sessions with required resources
    - Cleanup: Free resources when sessions end
    - Recovery: Restore failed sessions to last known good state
    - Monitoring: Track session health and resource usage

    Responsibilities:
    - Session resource allocation and deallocation
    - State persistence and restoration
    - Graceful shutdown handling
    - Background task management
    """

    async def initialize_session(
        self,
        session_id: str,
        config: SessionConfig,
    ) -> SessionMetadata:
        """Initialize a new session.

        Allocates resources for the session, initializes state,
        and prepares the session for use.

        Args:
            session_id: Unique session identifier
            config: Session configuration

        Returns:
            SessionMetadata with allocated resources

        Raises:
            SessionInitializationError: If initialization fails

        Example:
            metadata = await lifecycle.initialize_session(
                session_id="abc123",
                config=SessionConfig(provider="anthropic", model="claude-sonnet-4-20250514")
            )
        """
        ...

    async def cleanup_session(self, session_id: str) -> CleanupResult:
        """Cleanup session resources.

        Frees all resources allocated for the session, persists
        state if needed, and handles graceful shutdown.

        Args:
            session_id: Session to cleanup

        Returns:
            CleanupResult with cleanup status and freed resources

        Example:
            result = await lifecycle.cleanup_session("abc123")
            if result.success:
                print(f"Freed resources: {result.resources_freed}")
        """
        ...

    async def recover_session(self, session_id: str) -> RecoveryResult:
        """Recover a failed session.

        Attempts to restore a failed session to its last known
        good state, using persisted state and recovery strategies.

        Args:
            session_id: Session to recover

        Returns:
            RecoveryResult with recovery status and metadata

        Example:
            result = await lifecycle.recover_session("abc123")
            if result.success and result.state_recovered:
                print("Session recovered successfully")
        """
        ...

    async def get_session_status(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get status of a session.

        Returns health and resource usage information for
        monitoring and debugging.

        Args:
            session_id: Session to query

        Returns:
            Dictionary with session status information

        Example:
            status = await lifecycle.get_session_status("abc123")
            # {
            #     "healthy": True,
            #     "uptime_seconds": 1234,
            #     "resource_usage": {...},
            # }
        """
        ...


__all__ = [
    "SessionMetadata",
    "SessionConfig",
    "CleanupResult",
    "RecoveryResult",
    "ILifecycleManager",
]
