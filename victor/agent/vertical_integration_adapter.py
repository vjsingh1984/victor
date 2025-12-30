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

"""Single implementation for vertical integration methods.

This module provides a unified adapter for vertical integration, eliminating
duplicate apply_vertical_* method implementations in AgentOrchestrator.

The adapter consolidates two distinct responsibilities:
1. Storing vertical state in VerticalContext for reference
2. Applying state to runtime components (middleware chain, safety checker)

Design Philosophy:
- Single Responsibility: One place for vertical integration logic
- Open/Closed: New integration points added via the adapter
- Dependency Inversion: Uses capability registry protocol, not private attributes

DIP Compliance (Phase 1 Foundation Fix):
- Uses has_capability/invoke_capability/get_capability_value instead of _private attrs
- Falls back to CAPABILITY_METHOD_MAPPINGS for method resolution
- Graceful degradation when capability registry not available

Usage:
    from victor.agent.vertical_integration_adapter import VerticalIntegrationAdapter

    # Create adapter with orchestrator reference
    adapter = VerticalIntegrationAdapter(orchestrator)

    # Apply middleware (single implementation)
    adapter.apply_middleware(middleware_list)

    # Apply safety patterns (single implementation)
    adapter.apply_safety_patterns(patterns)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from victor.agent.capability_registry import CAPABILITY_METHOD_MAPPINGS, get_method_for_capability

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class VerticalIntegrationAdapter:
    """Adapter providing single implementation for vertical integration.

    Eliminates duplicate apply_vertical_* methods in AgentOrchestrator by
    providing a unified implementation that handles:
    - Storage in VerticalContext
    - Application to runtime components (middleware chain, safety checker)

    DIP Compliance:
    - Uses capability registry protocol instead of private attribute access
    - Falls back to public method invocation via CAPABILITY_METHOD_MAPPINGS
    - Maintains backward compatibility with orchestrators without capability registry

    Attributes:
        _orchestrator: Reference to the parent orchestrator
    """

    def __init__(self, orchestrator: "AgentOrchestrator"):
        """Initialize the adapter with orchestrator reference.

        Args:
            orchestrator: Parent orchestrator instance
        """
        self._orchestrator = orchestrator

    # =========================================================================
    # DIP-Compliant Capability Access Helpers
    # =========================================================================

    def _has_capability(self, name: str) -> bool:
        """Check if orchestrator has a capability (DIP-compliant).

        Uses capability registry protocol if available, falls back to method check.
        """
        # Prefer capability registry protocol
        if hasattr(self._orchestrator, "has_capability"):
            return self._orchestrator.has_capability(name)

        # Fall back to checking if method exists
        method_name = get_method_for_capability(name)
        return hasattr(self._orchestrator, method_name)

    def _get_capability_value(self, name: str) -> Optional[Any]:
        """Get capability value (DIP-compliant).

        Uses capability registry protocol if available, falls back to public getter.
        """
        # Prefer capability registry protocol
        if hasattr(self._orchestrator, "get_capability_value"):
            try:
                return self._orchestrator.get_capability_value(name)
            except (KeyError, TypeError):
                pass

        # Fall back to public getter
        getter_name = f"get_{name}"
        if hasattr(self._orchestrator, getter_name):
            getter = getattr(self._orchestrator, getter_name)
            if callable(getter):
                return getter()

        return None

    def _invoke_capability(self, name: str, *args: Any, **kwargs: Any) -> bool:
        """Invoke a capability (DIP-compliant).

        Uses capability registry protocol if available, falls back to public method.

        Returns:
            True if capability was invoked successfully
        """
        # Prefer capability registry protocol
        if hasattr(self._orchestrator, "invoke_capability"):
            try:
                self._orchestrator.invoke_capability(name, *args, **kwargs)
                return True
            except (KeyError, TypeError) as e:
                logger.debug(f"Capability invocation failed for '{name}': {e}")

        # Fall back to public method
        method_name = get_method_for_capability(name)
        if hasattr(self._orchestrator, method_name):
            method = getattr(self._orchestrator, method_name)
            if callable(method):
                method(*args, **kwargs)
                return True

        return False

    def apply_middleware(self, middleware: List[Any]) -> None:
        """Apply middleware from vertical (single implementation).

        This method consolidates the two previous implementations:
        1. First (lines ~1300-1312): Stored in context, applied to chain if exists
        2. Second (lines ~1795-1825): Stored in attribute, initialized chain if needed

        DIP Compliance: Uses capability registry for reading, direct access for writing
        to avoid recursion (this method IS the capability implementation).

        The unified implementation:
        1. Returns early if middleware list is empty
        2. Stores middleware via vertical_context capability
        3. Stores reference directly (we ARE the capability implementation)
        4. Gets/creates middleware chain
        5. Adds all middleware to the chain

        Args:
            middleware: List of MiddlewareProtocol implementations
        """
        if not middleware:
            return

        # Store in vertical context via capability (DIP-compliant read)
        vertical_context = self._get_capability_value("vertical_context")
        if vertical_context is not None and hasattr(vertical_context, "apply_middleware"):
            vertical_context.apply_middleware(middleware)

        # Store reference directly - we ARE the implementation, don't recurse via capability
        # This is safe because we're setting the storage, not calling the capability
        # Note: We set directly to maintain backward compatibility with tests
        self._orchestrator._vertical_middleware = middleware

        # Get middleware chain via capability (DIP-compliant read)
        chain = self._get_capability_value("middleware_chain")
        if chain is None:
            # Try to get or create middleware chain
            try:
                from victor.agent.middleware_chain import MiddlewareChain

                chain = MiddlewareChain()
                # Store directly on orchestrator (we're the implementation)
                if hasattr(self._orchestrator, "_middleware_chain"):
                    self._orchestrator._middleware_chain = chain
            except ImportError:
                logger.warning("MiddlewareChain not available")
                return

        if chain is None:
            logger.debug("No middleware chain available")
            return

        # Add all middleware to chain via public interface
        for mw in middleware:
            if hasattr(chain, "add"):
                chain.add(mw)

        logger.debug(f"Applied {len(middleware)} middleware from vertical")

    def apply_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical (single implementation).

        This method consolidates the two previous implementations:
        1. First (lines ~1314-1328): Stored in context, used add_patterns or _custom_patterns
        2. Second (lines ~1827-1857): Stored in attribute, used add_custom_pattern with args

        DIP Compliance: Uses capability registry for reading, direct access for writing
        to avoid recursion (this method IS the capability implementation).

        The unified implementation:
        1. Returns early if patterns list is empty
        2. Stores patterns via vertical_context capability
        3. Stores reference directly (we ARE the capability implementation)
        4. Applies to safety checker via public interface

        Args:
            patterns: List of SafetyPattern instances
        """
        if not patterns:
            return

        # Store in vertical context via capability (DIP-compliant read)
        vertical_context = self._get_capability_value("vertical_context")
        if vertical_context is not None and hasattr(vertical_context, "apply_safety_patterns"):
            vertical_context.apply_safety_patterns(patterns)

        # Store reference directly - we ARE the implementation, don't recurse via capability
        # Note: We set directly to maintain backward compatibility with tests
        self._orchestrator._vertical_safety_patterns = patterns

        # Get safety checker via capability or public method
        checker = None
        if self._has_capability("safety_checker"):
            checker = self._get_capability_value("safety_checker")
        elif hasattr(self._orchestrator, "get_safety_checker"):
            checker = self._orchestrator.get_safety_checker()
        elif hasattr(self._orchestrator, "_safety_checker"):
            # Last resort fallback for testing
            checker = self._orchestrator._safety_checker

        if checker is None:
            logger.debug("No safety checker available for pattern application")
            return

        try:
            # Try add_custom_pattern with expanded arguments (newer interface)
            if hasattr(checker, "add_custom_pattern"):
                for pattern in patterns:
                    checker.add_custom_pattern(
                        pattern=pattern.pattern,
                        description=pattern.description,
                        risk_level=getattr(pattern, "risk_level", "medium"),
                        category=getattr(pattern, "category", "custom"),
                    )
            # Fall back to add_patterns (batch interface - public API)
            elif hasattr(checker, "add_patterns"):
                checker.add_patterns(patterns)
            else:
                logger.debug("SafetyChecker does not support pattern addition via public API")
                return

            logger.debug(f"Applied {len(patterns)} safety patterns from vertical")
        except AttributeError as e:
            logger.debug(f"SafetyChecker pattern application failed: {e}")


__all__ = ["VerticalIntegrationAdapter"]
