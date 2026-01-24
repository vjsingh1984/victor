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

"""Mode controller protocol for agent mode management.

Phase 1.1: Integrate AgentModeController with protocol.

This module re-exports ModeControllerProtocol from the canonical location
(victor.agent.protocols) and defines ExtendedModeControllerProtocol for
full mode management capabilities.

Design Principles:
- ISP Compliant: Small, focused interface (~6 methods)
- DIP Compliant: Orchestrator depends on protocol, not concrete class
- Substitutable: Any implementation can be injected for testing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

# Re-export canonical ModeControllerProtocol from victor.agent.protocols
from victor.agent.protocols import ModeControllerProtocol

if TYPE_CHECKING:
    from victor.agent.mode_controller import AgentMode, OperationalModeConfig


@runtime_checkable
class ExtendedModeControllerProtocol(ModeControllerProtocol, Protocol):
    """Extended protocol with additional mode management features.

    This extends ModeControllerProtocol with optional features for
    full mode management capabilities. Use this when you need:
    - Mode history navigation
    - Callback registration
    - Status reporting
    - Sandbox settings access

    Implementations:
        - AgentModeController: Full implementation with history and callbacks
        - ModeControllerAdapter: DI adapter wrapping AgentModeController
    """

    def previous_mode(self) -> Optional["AgentMode"]:
        """Switch to the previous mode in history.

        Returns:
            The previous mode, or None if no history
        """
        ...

    def register_callback(
        self, callback: Callable[["AgentMode", "AgentMode"], None]
    ) -> None:
        """Register a callback for mode changes.

        Args:
            callback: Function called with (old_mode, new_mode) on transitions
        """
        ...

    def get_status(self) -> Dict[str, Any]:
        """Get current mode status information.

        Returns:
            Dictionary with mode information
        """
        ...

    def get_mode_list(self) -> List[Dict[str, str]]:
        """Get list of available modes.

        Returns:
            List of mode info dictionaries
        """
        ...

    @property
    def sandbox_dir(self) -> Optional[str]:
        """Get sandbox directory for restricted modes.

        Returns:
            Sandbox directory path or None
        """
        ...

    @property
    def allow_sandbox_edits(self) -> bool:
        """Check if sandbox edits are allowed.

        Returns:
            True if edits allowed in sandbox
        """
        ...

    @property
    def require_write_confirmation(self) -> bool:
        """Check if write confirmation is required.

        Returns:
            True if writes require confirmation
        """
        ...

    @property
    def max_files_per_operation(self) -> int:
        """Get maximum files allowed per operation.

        Returns:
            Max files limit (0 = unlimited)
        """
        ...


__all__ = [
    "ModeControllerProtocol",
    "ExtendedModeControllerProtocol",
]
