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

"""Optimized singleton reset for faster test execution.

This module provides a faster alternative to tests/singleton_reset.py by:
1. Only resetting critical singletons (48 → 3 resets)
2. Skipping expensive resets (observability, storage, processing)
3. Using lazy evaluation (only import if needed)

Performance: ~0.01s per reset vs 0.8s for full reset

Usage:
    # In conftest.py, for faster tests:
    from tests.singleton_reset_fast import reset_all_singletons_fast

    @pytest.fixture(autouse=True, scope="module")
    def reset_singletons_fast():
        reset_all_singletons_fast()
        yield
        reset_all_singletons_fast()
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class FastSingletonResetRegistry:
    """Optimized registry for fast test execution.

    Strategies:
    - Only 3 critical singletons (down from 48)
    - Lazy imports (only import what's needed)
    - Skip expensive resets (observability, storage)
    """

    def __init__(self) -> None:
        self._reset_functions: List[Callable[[], None]] = []
        self._initialized = False

    def register(self, reset_fn: Callable[[], None]) -> None:
        """Register a reset function."""
        self._reset_functions.append(reset_fn)

    def reset_all(self) -> None:
        """Execute all registered reset functions."""
        for reset_fn in self._reset_functions:
            try:
                reset_fn()
            except Exception as e:
                logger.debug(f"Singleton reset warning: {e}")

    def initialize(self) -> None:
        """Initialize with only critical singletons."""
        if self._initialized:
            return

        # Only 3 critical singletons that cause test pollution
        self._register_critical_singletons()

        self._initialized = True

    def _register_critical_singletons(self) -> None:
        """Register only critical singletons for fast reset.

        Critical singletons are those that:
        - Hold state that bleeds between tests
        - Are used frequently in tests
        - Are quick to reset (< 10ms each)
        """

        # Only 3 critical singletons that cause test pollution:
        # 1. DI Container - CRITICAL (holds service instances)
        # 2. Mode Config - CRITICAL (affects agent behavior)
        # 3. Vertical Config Cache - CRITICAL (fast, prevents pollution)

        # DI Container - CRITICAL
        def _reset_container():
            try:
                from victor.core.container import ServiceContainer

                if hasattr(ServiceContainer, "_instance"):
                    ServiceContainer._instance = None
            except ImportError:
                pass

        self.register(_reset_container)

        # Mode Config - CRITICAL
        def _reset_mode_config():
            try:
                from victor.core.mode_config import ModeConfigRegistry

                ModeConfigRegistry._instance = None
            except ImportError:
                pass

        self.register(_reset_mode_config)

        # Vertical Config Cache - CRITICAL
        def _reset_vertical_cache():
            try:
                from victor.core.verticals.base import VerticalBase

                VerticalBase._config_cache.clear()
            except ImportError:
                pass

        self.register(_reset_vertical_cache)


_global_fast_registry: Optional[FastSingletonResetRegistry] = None


def get_fast_reset_registry() -> FastSingletonResetRegistry:
    """Get or create the fast singleton reset registry."""
    global _global_fast_registry
    if _global_fast_registry is None:
        _global_fast_registry = FastSingletonResetRegistry()
        _global_fast_registry.initialize()
    return _global_fast_registry


def reset_all_singletons_fast() -> None:
    """Fast reset of only critical singletons.

    This is optimized for test execution speed and only resets singletons
    that are known to cause test pollution. Use this instead of the full
    reset_all_singletons() for faster test execution.
    """
    registry = get_fast_reset_registry()
    registry.reset_all()


def reset_if_necessary() -> None:
    """Reset singletons only if they've been accessed.

    This is the most efficient approach - only reset what was used.
    """
    try:
        # Check if DI container has instances
        from victor.core.container import ServiceContainer

        if hasattr(ServiceContainer, "_instance") and ServiceContainer._instance is not None:
            reset_all_singletons_fast()
    except ImportError:
        pass


__all__ = [
    "FastSingletonResetRegistry",
    "get_fast_reset_registry",
    "reset_all_singletons_fast",
    "reset_if_necessary",
]
