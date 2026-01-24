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

"""Middleware Coordinator for vertical middleware management.

This coordinator manages:
- Vertical middleware configuration
- Middleware chain construction
- Safety patterns management
- Code correction middleware access

Design Pattern: Coordinator Pattern
- Centralizes middleware management logic
- Provides clean API for vertical integration
- Implements VerticalStorageProtocol for DIP compliance

Phase 6 Refactoring: Extracted from AgentOrchestrator
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from victor.agent.middleware.base import MiddlewareChain

logger = logging.getLogger(__name__)


class MiddlewareCoordinator:
    """Coordinator for vertical middleware management.

    Manages middleware configuration, chain construction, and
    safety patterns for vertical integration.

    Attributes:
        _vertical_middleware: List of middleware instances
        _middleware_chain: MiddlewareChain instance
        _vertical_safety_patterns: List of safety patterns
        _code_correction_middleware: Code correction middleware instance

    Example:
        coordinator = MiddlewareCoordinator()
        coordinator.set_middleware([logging_middleware, metrics_middleware])
        chain = coordinator.get_middleware_chain()
        patterns = coordinator.get_safety_patterns()
    """

    def __init__(self) -> None:
        """Initialize MiddlewareCoordinator."""
        self._vertical_middleware: List[Any] = []
        self._middleware_chain: Optional[Any] = None
        self._vertical_safety_patterns: List[Any] = []
        self._code_correction_middleware: Optional[Any] = None

    # ========================================================================
    # Middleware Management (VerticalStorageProtocol Implementation)
    # ========================================================================

    def set_middleware(self, middleware: List[Any]) -> None:
        """Store middleware configuration.

        Implements VerticalStorageProtocol.set_middleware().
        Provides a clean public interface for setting vertical middleware,
        replacing direct private attribute access.

        Args:
            middleware: List of MiddlewareProtocol implementations
        """
        self._vertical_middleware = middleware
        logger.debug(f"Set {len(middleware)} middleware instances")

    def get_middleware(self) -> List[Any]:
        """Retrieve middleware configuration.

        Implements VerticalStorageProtocol.get_middleware().
        Returns the list of middleware instances configured by vertical integration.

        Returns:
            List of middleware instances, or empty list if not set
        """
        return self._vertical_middleware.copy()

    # ========================================================================
    # Safety Patterns Management (VerticalStorageProtocol Implementation)
    # ========================================================================

    def set_safety_patterns(self, patterns: List[Any]) -> None:
        """Store safety patterns.

        Implements VerticalStorageProtocol.set_safety_patterns().
        Provides a clean public interface for setting vertical safety patterns,
        replacing direct private attribute access.

        Args:
            patterns: List of SafetyPattern instances from vertical extensions
        """
        self._vertical_safety_patterns = patterns
        logger.debug(f"Set {len(patterns)} safety patterns")

    def get_safety_patterns(self) -> List[Any]:
        """Retrieve safety patterns.

        Implements VerticalStorageProtocol.get_safety_patterns().
        Returns the list of safety patterns configured by vertical integration.

        Returns:
            List of safety pattern instances, or empty list if not set
        """
        return self._vertical_safety_patterns.copy()

    # ========================================================================
    # Middleware Chain Management
    # ========================================================================

    def get_middleware_chain(self) -> Optional[Any]:
        """Get the middleware chain for tool execution.

        Returns:
            MiddlewareChain instance or None if not initialized.
        """
        return self._middleware_chain

    def set_middleware_chain(self, chain: Any) -> None:
        """Set middleware chain for tool execution.

        Args:
            chain: MiddlewareChain instance
        """
        self._middleware_chain = chain
        logger.debug("Set middleware chain")

    def build_middleware_chain(
        self,
        middleware_list: List[Any],
    ) -> Optional[Any]:
        """Build middleware chain from list of middleware.

        Args:
            middleware_list: List of middleware instances

        Returns:
            MiddlewareChain instance or None if no middleware
        """
        if not middleware_list:
            self._middleware_chain = None
            return None

        try:
            from victor.agent.middleware.base import MiddlewareChain

            chain = MiddlewareChain(middleware_list)
            self._middleware_chain = chain
            return chain
        except ImportError:
            logger.warning("MiddlewareChain not available")
            return None

    # ========================================================================
    # Code Correction Middleware
    # ========================================================================

    def set_code_correction_middleware(self, middleware: Any) -> None:
        """Set code correction middleware instance.

        Args:
            middleware: Code correction middleware instance
        """
        self._code_correction_middleware = middleware
        logger.debug("Set code correction middleware")

    def get_code_correction_middleware(self) -> Optional[Any]:
        """Get code correction middleware instance.

        Returns:
            Code correction middleware or None if not set
        """
        return self._code_correction_middleware

    # ========================================================================
    # Internal Storage Setters (DIP Compliance)
    # ========================================================================

    def _set_vertical_middleware_storage(self, middleware: List[Any]) -> None:
        """Internal: Set vertical middleware storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            middleware: List of middleware instances
        """
        self._vertical_middleware = middleware

    def _set_middleware_chain_storage(self, chain: Any) -> None:
        """Internal: Set middleware chain storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            chain: MiddlewareChain instance
        """
        self._middleware_chain = chain

    def _set_safety_patterns_storage(self, patterns: List[Any]) -> None:
        """Internal: Set safety patterns storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            patterns: List of safety pattern instances
        """
        self._vertical_safety_patterns = patterns

    # ========================================================================
    # State Management
    # ========================================================================

    def get_state(self) -> dict[str, Any]:
        """Get coordinator state for checkpointing.

        Returns:
            Dictionary with coordinator state (excluding non-serializable objects)
        """
        return {
            "vertical_middleware_count": len(self._vertical_middleware),
            "vertical_safety_patterns_count": len(self._vertical_safety_patterns),
            "has_middleware_chain": self._middleware_chain is not None,
            "has_code_correction": self._code_correction_middleware is not None,
        }

    def reset(self) -> None:
        """Reset all coordinator state."""
        self._vertical_middleware = []
        self._middleware_chain = None
        self._vertical_safety_patterns = []
        self._code_correction_middleware = None

    # ========================================================================
    # Computed Properties
    # ========================================================================

    def has_middleware(self) -> bool:
        """Check if any middleware is configured.

        Returns:
            True if middleware configured, False otherwise
        """
        return len(self._vertical_middleware) > 0

    def has_safety_patterns(self) -> bool:
        """Check if any safety patterns are configured.

        Returns:
            True if safety patterns configured, False otherwise
        """
        return len(self._vertical_safety_patterns) > 0

    def has_code_correction(self) -> bool:
        """Check if code correction middleware is configured.

        Returns:
            True if code correction configured, False otherwise
        """
        return self._code_correction_middleware is not None

    def get_middleware_summary(self) -> dict[str, Any]:
        """Get summary of middleware configuration.

        Returns:
            Dictionary with middleware summary:
            - total_middleware: Number of middleware instances
            - total_safety_patterns: Number of safety patterns
            - has_chain: Whether middleware chain is built
            - has_code_correction: Whether code correction is enabled
        """
        return {
            "total_middleware": len(self._vertical_middleware),
            "total_safety_patterns": len(self._vertical_safety_patterns),
            "has_chain": self._middleware_chain is not None,
            "has_code_correction": self._code_correction_middleware is not None,
        }


def create_middleware_coordinator() -> MiddlewareCoordinator:
    """Factory function to create MiddlewareCoordinator.

    Returns:
        Configured MiddlewareCoordinator instance
    """
    return MiddlewareCoordinator()


__all__ = [
    "MiddlewareCoordinator",
    "create_middleware_coordinator",
]
