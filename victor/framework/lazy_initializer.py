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

"""Lazy initializer for eliminating import side-effects.

This module provides a LazyInitializer that enables lazy initialization
of vertical registration, eliminating import-time side effects while
maintaining thread safety.

Phase 5: Import Side-Effects - Lazy Registration
This implementation eliminates registration-on-import side effects by
deferring initialization until first use, following the Open/Closed Principle.

Design Philosophy:
- Lazy initialization triggered on first use, not import
- Thread-safe with double-checked locking pattern
- Singleton pattern per vertical
- Supports multiple initializers (run in sequence)
- Graceful error handling with retry support
- Feature flag control (VICTOR_LAZY_INITIALIZATION)

OLD Pattern (import side-effects):
    # In victor/coding/__init__.py
    def _register_escape_hatches() -> None:
        registry = EscapeHatchRegistry.get_instance()
        registry.register_from_vertical("coding", ...)
    _register_escape_hatches()  # Side effect on import!

NEW Pattern (lazy initialization):
    # In victor/coding/__init__.py
    from victor.framework.lazy_initializer import get_initializer_for_vertical

    def _register_escape_hatches() -> None:
        registry = EscapeHatchRegistry.get_instance()
        registry.register_from_vertical("coding", ...)

    _lazy_init = get_initializer_for_vertical(
        "coding",
        _register_escape_hatches
    )
    # No side effect on import - registers on first use

Usage:
    from victor.framework.lazy_initializer import get_initializer_for_vertical

    # Create lazy initializer
    lazy_init = get_initializer_for_vertical(
        vertical_name="my_vertical",
        initializer=lambda: register_my_vertical()
    )

    # Triggers registration on first access
    lazy_init.get_or_initialize()
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")


# =============================================================================
# Lazy Initializer
# =============================================================================


class LazyInitializer:
    """Thread-safe lazy initializer for eliminating import side-effects.

    This class provides lazy initialization with thread safety, ensuring
    that initialization only happens once (on first access) regardless of
    concurrent access.

    Thread Safety:
        Uses double-checked locking pattern:
        1. Check if initialized (no lock) - fast path
        2. Acquire lock
        3. Check if initialized again (race condition check)
        4. Initialize if not initialized
        5. Release lock

    Attributes:
        vertical_name: Name of the vertical
        _initializers: List of initializer functions to run
        _initialized: Whether initialization has occurred
        _result: Cached initialization result
        _init_lock: Thread lock for safe lazy initialization
        _initializing: Flag to prevent recursive initialization
    """

    def __init__(
        self,
        vertical_name: str,
        initializer: Optional[Callable[[], T]] = None,
        initializers: Optional[List[Callable[[], Any]]] = None,
    ) -> None:
        """Initialize the lazy initializer.

        Args:
            vertical_name: Name of the vertical
            initializer: Single initializer function (optional)
            initializers: List of initializer functions (optional)
        """
        self.vertical_name = vertical_name

        # Support both single initializer and list of initializers
        if initializer is not None:
            self._initializers: List[Callable[[], Any]] = [initializer]
        else:
            self._initializers = list(initializers) if initializers else []

        self._initialized = False
        self._result: Any = None
        self._init_lock = threading.RLock()  # Reentrant lock for thread safety
        self._initializing = False

    def initialize(self) -> Any:
        """Initialize synchronously (eager initialization).

        This method forces immediate initialization, regardless of whether
        lazy initialization is enabled.

        Returns:
            Initialization result

        Raises:
            Exception: If initialization fails
        """
        if self._initialized:
            return self._result  # type: ignore[unreachable]

        with self._init_lock:
            if self._initialized:
                return self._result  # type: ignore[unreachable]

            # Check for recursive initialization
            if self._initializing:
                raise RuntimeError(
                    f"Recursive initialization detected for vertical '{self.vertical_name}'"
                )

            try:
                self._initializing = True
                logger.debug(f"Initializing vertical '{self.vertical_name}'")

                # Run all initializers in sequence
                result = None
                for init_fn in self._initializers:
                    result = init_fn()

                self._result = result
                self._initialized = True

                logger.debug(f"Successfully initialized vertical '{self.vertical_name}'")
                return self._result  # type: ignore[unreachable]
            except Exception as e:
                logger.error(f"Failed to initialize vertical '{self.vertical_name}': {e}")
                raise
            finally:
                self._initializing = False

    def get_or_initialize(self) -> Any:
        """Get initialization result, initializing if necessary.

        This is the main entry point for lazy initialization. It will
        initialize on first access and cache the result for subsequent calls.

        Returns:
            Initialization result

        Raises:
            Exception: If initialization fails
        """
        # Check if lazy initialization is disabled
        lazy_init_enabled = os.getenv("VICTOR_LAZY_INITIALIZATION", "true").lower() == "true"

        if not lazy_init_enabled:
            # Eager initialization - initialize immediately
            return self.initialize()

        # Fast path: already initialized (no lock)
        if self._initialized:
            return self._result

        # Slow path: need to initialize
        with self._init_lock:
            # Double-check: another thread may have initialized
            if self._initialized:
                return self._result

            # Check for recursive initialization
            if self._initializing:
                raise RuntimeError(
                    f"Recursive initialization detected for vertical '{self.vertical_name}'"
                )

            # Initialize
            return self.initialize()

    def is_initialized(self) -> bool:
        """Check if initialization has occurred.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized

    def reset(self) -> None:
        """Reset initialization state.

        This clears the cached initialization result, allowing it
        to be re-initialized. Useful for testing or recovery.

        Note:
            This method is not thread-safe. Use external synchronization
            if calling from multiple threads.
        """
        with self._init_lock:
            self._initialized = False
            self._result = None
            self._initializing = False
            logger.debug(f"Reset initialization state for vertical '{self.vertical_name}'")


# =============================================================================
# Factory Functions
# =============================================================================

_initializers: Dict[str, LazyInitializer] = {}
_initializers_lock = threading.Lock()


def get_initializer_for_vertical(
    vertical_name: str,
    initializer: Optional[Callable[[], T]] = None,
    initializers: Optional[List[Callable[[], Any]]] = None,
) -> LazyInitializer:
    """Get or create lazy initializer for a vertical.

    This function provides a singleton initializer per vertical, ensuring
    that each vertical only has one initializer instance.

    Args:
        vertical_name: Name of the vertical
        initializer: Single initializer function (optional)
        initializers: List of initializer functions (optional)

    Returns:
        LazyInitializer instance (singleton per vertical)

    Example:
        # Create lazy initializer
        lazy_init = get_initializer_for_vertical(
            "coding",
            lambda: register_coding_escape_hatches()
        )

        # Use later (triggers registration)
        lazy_init.get_or_initialize()
    """
    with _initializers_lock:
        if vertical_name not in _initializers:
            _initializers[vertical_name] = LazyInitializer(
                vertical_name=vertical_name,
                initializer=initializer,
                initializers=initializers,
            )
            logger.debug(f"Created lazy initializer for vertical '{vertical_name}'")

        return _initializers[vertical_name]


def clear_all_initializers() -> None:
    """Clear all cached initializers (mainly for testing).

    This function removes all cached initializer instances, allowing
    them to be recreated on next access.

    Example:
        clear_all_initializers()
        init = get_initializer_for_vertical("test", lambda: None)
        # init is a new instance
    """
    with _initializers_lock:
        _initializers.clear()
        logger.debug("Cleared all lazy initializers")


def get_all_initializers() -> Dict[str, LazyInitializer]:
    """Get all cached initializers.

    Returns:
        Dictionary mapping vertical names to their initializers

    Example:
        initializers = get_all_initializers()
        for name, init in initializers.items():
            if init.is_initialized():
                print(f"{name}: initialized")
            else:
                print(f"{name}: not initialized")
    """
    with _initializers_lock:
        return dict(_initializers)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "LazyInitializer",
    "get_initializer_for_vertical",
    "clear_all_initializers",
    "get_all_initializers",
]
