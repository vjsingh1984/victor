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

"""State manager implementations for each scope.

This module provides concrete implementations of IStateManager protocol
for each StateScope. These managers directly replace legacy state systems.

SOLID Principles:
- SRP: Each manager handles one scope only
- OCP: Extensible via protocol implementation
- LSP: All managers are substitutable via IStateManager
- ISP: Implements focused IStateManager interface
- DIP: High-level modules depend on IStateManager abstraction

Replacements:
- WorkflowStateManager replaces ExecutionContext
- ConversationStateManager replaces ConversationStateMachine
- TeamStateManager replaces TeamContext
- GlobalStateManager provides application-wide state

Usage:
    from victor.state.managers import WorkflowStateManager

    manager = WorkflowStateManager()
    await manager.set("key", "value")
    value = await manager.get("key")
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from typing import Any, Dict, List

from victor.state.protocols import IStateManager, IStateObserver, StateScope

logger = logging.getLogger(__name__)


class WorkflowStateManager:
    """State manager for workflow scope.

    Replaces ExecutionContext directly (no adapter).
    Manages state for single workflow executions.

    SOLID: SRP (workflow state only), implements IStateManager

    Attributes:
        scope: StateScope.WORKFLOW

    Example:
        >>> manager = WorkflowStateManager()
        >>> await manager.set("task_id", "task-123")
        >>> task_id = await manager.get("task_id")
    """

    def __init__(self) -> None:
        """Initialize workflow state manager."""
        self.scope: StateScope = StateScope.WORKFLOW
        self._state: Dict[str, Any] = {}
        self._observers: List[IStateObserver] = []

        logger.debug("WorkflowStateManager initialized")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with key, or default if not found
        """
        return self._state.get(key, default)

    async def set(self, key: str, value: Any, notify: bool = True) -> None:
        """Set a value by key.

        Args:
            key: The key to set
            value: The value to store
            notify: Whether to notify observers (default: True)

        Performance: Only notifies observers if value actually changed.
        Notifications run in parallel for multiple observers.
        """
        old_value = self._state.get(key)

        # Skip notification if value hasn't changed (key exists and same value)
        if old_value == value and key in self._state:
            # Value unchanged, skip notification (but still update state)
            return

        self._state[key] = value

        # Notify observers in parallel if requested
        if notify and self._observers:
            await asyncio.gather(
                *[
                    observer.on_state_changed(
                        scope=self.scope,
                        key=key,
                        old_value=old_value,
                        new_value=value,
                    )
                    for observer in self._observers
                ],
                return_exceptions=True,  # Don't fail on observer errors
            )

        logger.debug(f"Workflow state set: {key} = {str(value)[:50]}")

    async def delete(self, key: str) -> None:
        """Delete a value by key.

        Args:
            key: The key to delete
        """
        if key in self._state:
            old_value = self._state[key]
            del self._state[key]

            # Notify observers
            for observer in self._observers:
                try:
                    await observer.on_state_changed(
                        scope=self.scope,
                        key=key,
                        old_value=old_value,
                        new_value=None,
                    )
                except Exception as e:
                    logger.warning(f"Observer notification failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self._state

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern.

        Args:
            pattern: Glob pattern to match keys (default: "*" for all)

        Returns:
            List of keys matching pattern
        """
        if pattern == "*":
            return list(self._state.keys())

        return [k for k in self._state.keys() if fnmatch.fnmatch(k, pattern)]

    async def get_all(self) -> Dict[str, Any]:
        """Get all state as dictionary.

        Returns:
            Dictionary of all key-value pairs
        """
        return dict(self._state)

    async def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys at once.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            await self.set(key, value)

    async def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
        logger.debug("Workflow state cleared")

    async def snapshot(self) -> Dict[str, Any]:
        """Create immutable snapshot for checkpointing.

        Returns:
            Dictionary snapshot of current state
        """
        return dict(self._state)

    async def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore from snapshot.

        Args:
            snapshot: Snapshot dictionary to restore from
        """
        self._state = dict(snapshot)
        logger.debug(f"Workflow state restored from snapshot ({len(snapshot)} keys)")

    def add_observer(self, observer: IStateObserver) -> None:
        """Add state change observer.

        Args:
            observer: Observer to notify of state changes
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: IStateObserver) -> None:
        """Remove state change observer.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)


class ConversationStateManager:
    """State manager for conversation scope.

    Replaces ConversationStateMachine directly (no adapter).
    Manages state for multi-turn conversations.

    SOLID: SRP (conversation state only), implements IStateManager

    Attributes:
        scope: StateScope.CONVERSATION

    Example:
        >>> manager = ConversationStateManager()
        >>> await manager.set("stage", "gathering")
        >>> stage = await manager.get("stage")
    """

    def __init__(self) -> None:
        """Initialize conversation state manager."""
        self.scope: StateScope = StateScope.CONVERSATION
        self._state: Dict[str, Any] = {}
        self._observers: List[IStateObserver] = []

        logger.debug("ConversationStateManager initialized")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with key, or default if not found
        """
        return self._state.get(key, default)

    async def set(self, key: str, value: Any, notify: bool = True) -> None:
        """Set a value by key.

        Args:
            key: The key to set
            value: The value to store
            notify: Whether to notify observers (default: True)

        Performance: Only notifies observers if value actually changed.
        Notifications run in parallel for multiple observers.
        """
        old_value = self._state.get(key)

        # Skip notification if value hasn't changed (key exists and same value)
        if old_value == value and key in self._state:
            return

        self._state[key] = value

        # Notify observers in parallel if requested
        if notify and self._observers:
            await asyncio.gather(
                *[
                    observer.on_state_changed(
                        scope=self.scope,
                        key=key,
                        old_value=old_value,
                        new_value=value,
                    )
                    for observer in self._observers
                ],
                return_exceptions=True,
            )

        logger.debug(f"Conversation state set: {key} = {str(value)[:50]}")

    async def delete(self, key: str) -> None:
        """Delete a value by key.

        Args:
            key: The key to delete
        """
        if key in self._state:
            old_value = self._state[key]
            del self._state[key]

            # Notify observers
            for observer in self._observers:
                try:
                    await observer.on_state_changed(
                        scope=self.scope,
                        key=key,
                        old_value=old_value,
                        new_value=None,
                    )
                except Exception as e:
                    logger.warning(f"Observer notification failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self._state

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern.

        Args:
            pattern: Glob pattern to match keys (default: "*" for all)

        Returns:
            List of keys matching pattern
        """
        if pattern == "*":
            return list(self._state.keys())

        return [k for k in self._state.keys() if fnmatch.fnmatch(k, pattern)]

    async def get_all(self) -> Dict[str, Any]:
        """Get all state as dictionary.

        Returns:
            Dictionary of all key-value pairs
        """
        return dict(self._state)

    async def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys at once.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            await self.set(key, value)

    async def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
        logger.debug("Conversation state cleared")

    async def snapshot(self) -> Dict[str, Any]:
        """Create immutable snapshot for checkpointing.

        Returns:
            Dictionary snapshot of current state
        """
        return dict(self._state)

    async def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore from snapshot.

        Args:
            snapshot: Snapshot dictionary to restore from
        """
        self._state = dict(snapshot)
        logger.debug(f"Conversation state restored from snapshot ({len(snapshot)} keys)")

    def add_observer(self, observer: IStateObserver) -> None:
        """Add state change observer.

        Args:
            observer: Observer to notify of state changes
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: IStateObserver) -> None:
        """Remove state change observer.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)


class TeamStateManager:
    """State manager for team scope.

    Replaces TeamContext directly (no adapter).
    Manages state for multi-agent team coordination.

    SOLID: SRP (team state only), implements IStateManager

    Attributes:
        scope: StateScope.TEAM

    Example:
        >>> manager = TeamStateManager()
        >>> await manager.set("coordinator", "agent-1")
        >>> coordinator = await manager.get("coordinator")
    """

    def __init__(self) -> None:
        """Initialize team state manager."""
        self.scope: StateScope = StateScope.TEAM
        self._state: Dict[str, Any] = {}
        self._observers: List[IStateObserver] = []

        logger.debug("TeamStateManager initialized")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with key, or default if not found
        """
        return self._state.get(key, default)

    async def set(self, key: str, value: Any, notify: bool = True) -> None:
        """Set a value by key.

        Args:
            key: The key to set
            value: The value to store
            notify: Whether to notify observers (default: True)

        Performance: Only notifies observers if value actually changed.
        Notifications run in parallel for multiple observers.
        """
        old_value = self._state.get(key)

        # Skip notification if value hasn't changed (key exists and same value)
        if old_value == value and key in self._state:
            return

        self._state[key] = value

        # Notify observers in parallel if requested
        if notify and self._observers:
            await asyncio.gather(
                *[
                    observer.on_state_changed(
                        scope=self.scope,
                        key=key,
                        old_value=old_value,
                        new_value=value,
                    )
                    for observer in self._observers
                ],
                return_exceptions=True,
            )

        logger.debug(f"Team state set: {key} = {str(value)[:50]}")

    async def delete(self, key: str) -> None:
        """Delete a value by key.

        Args:
            key: The key to delete
        """
        if key in self._state:
            old_value = self._state[key]
            del self._state[key]

            # Notify observers
            for observer in self._observers:
                try:
                    await observer.on_state_changed(
                        scope=self.scope,
                        key=key,
                        old_value=old_value,
                        new_value=None,
                    )
                except Exception as e:
                    logger.warning(f"Observer notification failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self._state

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern.

        Args:
            pattern: Glob pattern to match keys (default: "*" for all)

        Returns:
            List of keys matching pattern
        """
        if pattern == "*":
            return list(self._state.keys())

        return [k for k in self._state.keys() if fnmatch.fnmatch(k, pattern)]

    async def get_all(self) -> Dict[str, Any]:
        """Get all state as dictionary.

        Returns:
            Dictionary of all key-value pairs
        """
        return dict(self._state)

    async def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys at once.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            await self.set(key, value)

    async def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
        logger.debug("Team state cleared")

    async def snapshot(self) -> Dict[str, Any]:
        """Create immutable snapshot for checkpointing.

        Returns:
            Dictionary snapshot of current state
        """
        return dict(self._state)

    async def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore from snapshot.

        Args:
            snapshot: Snapshot dictionary to restore from
        """
        self._state = dict(snapshot)
        logger.debug(f"Team state restored from snapshot ({len(snapshot)} keys)")

    def add_observer(self, observer: IStateObserver) -> None:
        """Add state change observer.

        Args:
            observer: Observer to notify of state changes
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: IStateObserver) -> None:
        """Remove state change observer.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)


class GlobalStateManagerImpl:
    """State manager for global scope.

    Manages cross-cutting application state.
    Used for configuration, settings, and global state.

    SOLID: SRP (global state only), implements IStateManager

    Attributes:
        scope: StateScope.GLOBAL

    Example:
        >>> manager = GlobalStateManagerImpl()
        >>> await manager.set("config", {"debug": True})
        >>> config = await manager.get("config")
    """

    def __init__(self) -> None:
        """Initialize global state manager."""
        self.scope: StateScope = StateScope.GLOBAL
        self._state: Dict[str, Any] = {}
        self._observers: List[IStateObserver] = []

        logger.debug("GlobalStateManagerImpl initialized")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with key, or default if not found
        """
        return self._state.get(key, default)

    async def set(self, key: str, value: Any, notify: bool = True) -> None:
        """Set a value by key.

        Args:
            key: The key to set
            value: The value to store
            notify: Whether to notify observers (default: True)

        Performance: Only notifies observers if value actually changed.
        Notifications run in parallel for multiple observers.
        """
        old_value = self._state.get(key)

        # Skip notification if value hasn't changed (key exists and same value)
        if old_value == value and key in self._state:
            return

        self._state[key] = value

        # Notify observers in parallel if requested
        if notify and self._observers:
            await asyncio.gather(
                *[
                    observer.on_state_changed(
                        scope=self.scope,
                        key=key,
                        old_value=old_value,
                        new_value=value,
                    )
                    for observer in self._observers
                ],
                return_exceptions=True,
            )

        logger.debug(f"Global state set: {key} = {str(value)[:50]}")

    async def delete(self, key: str) -> None:
        """Delete a value by key.

        Args:
            key: The key to delete
        """
        if key in self._state:
            old_value = self._state[key]
            del self._state[key]

            # Notify observers
            for observer in self._observers:
                try:
                    await observer.on_state_changed(
                        scope=self.scope,
                        key=key,
                        old_value=old_value,
                        new_value=None,
                    )
                except Exception as e:
                    logger.warning(f"Observer notification failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self._state

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern.

        Args:
            pattern: Glob pattern to match keys (default: "*" for all)

        Returns:
            List of keys matching pattern
        """
        if pattern == "*":
            return list(self._state.keys())

        return [k for k in self._state.keys() if fnmatch.fnmatch(k, pattern)]

    async def get_all(self) -> Dict[str, Any]:
        """Get all state as dictionary.

        Returns:
            Dictionary of all key-value pairs
        """
        return dict(self._state)

    async def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys at once.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            await self.set(key, value)

    async def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
        logger.debug("Global state cleared")

    async def snapshot(self) -> Dict[str, Any]:
        """Create immutable snapshot for checkpointing.

        Returns:
            Dictionary snapshot of current state
        """
        return dict(self._state)

    async def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore from snapshot.

        Args:
            snapshot: Snapshot dictionary to restore from
        """
        self._state = dict(snapshot)
        logger.debug(f"Global state restored from snapshot ({len(snapshot)} keys)")

    def add_observer(self, observer: IStateObserver) -> None:
        """Add state change observer.

        Args:
            observer: Observer to notify of state changes
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: IStateObserver) -> None:
        """Remove state change observer.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)


__all__ = [
    "WorkflowStateManager",
    "ConversationStateManager",
    "TeamStateManager",
    "GlobalStateManagerImpl",
]
