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

"""Lazy vertical loading system for improved startup performance.

This module provides lazy loading for verticals, loading them on-demand
rather than eagerly at startup. This significantly improves startup time
and reduces initial memory footprint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Set, Type

logger = logging.getLogger(__name__)


class LoadTrigger(str, Enum):
    """When to load vertical.

    EAGER: Load immediately at startup
    ON_DEMAND: Load when first accessed
    LAZY: Load asynchronously when needed
    """

    EAGER = "eager"
    ON_DEMAND = "on_demand"
    LAZY = "lazy"


@dataclass
class LazyVerticalProxy:
    """Proxy for lazy-loaded vertical.

    The proxy defers vertical loading until first access, at which point
    it delegates to the loader function.

    Attributes:
        vertical_name: Name of the vertical
        loader: Callable that loads the vertical
        _loaded: Whether the vertical has been loaded
        _instance: Cached loaded vertical instance
    """

    vertical_name: str
    loader: Callable[[], Type]
    _loaded: bool = False
    _instance: Optional[Type] = None

    def load(self) -> Type:
        """Load vertical on first access.

        Returns:
            Loaded vertical class
        """
        if not self._loaded:
            logger.debug(f"Lazy loading vertical: {self.vertical_name}")
            self._instance = self.loader()
            self._loaded = True
        return self._instance

    def unload(self) -> None:
        """Unload vertical to free memory.

        This clears the cached instance, allowing it to be garbage collected.
        The next call to load() will reload the vertical.
        """
        self._instance = None
        self._loaded = False

    def is_loaded(self) -> bool:
        """Check if vertical is currently loaded.

        Returns:
            True if vertical has been loaded
        """
        return self._loaded


class LazyVerticalLoader:
    """Manager for lazy vertical loading.

    This manager provides a centralized way to register and load verticals
    with different loading strategies. It tracks which verticals are currently
    loaded and provides methods for unloading to free memory.

    Example:
        loader = LazyVerticalLoader(LoadTrigger.ON_DEMAND)
        loader.register_vertical("coding", lambda: load_coding_vertical())
        loader.register_vertical("research", lambda: load_research_vertical())

        # Access verticals - loads on first access
        coding = loader.get_vertical("coding")
        research = loader.get_vertical("research")

        # Unload to free memory
        loader.unload_vertical("coding")
        loader.unload_all()
    """

    def __init__(self, load_trigger: LoadTrigger = LoadTrigger.ON_DEMAND):
        """Initialize lazy vertical loader.

        Args:
            load_trigger: When to load verticals
        """
        self._load_trigger = load_trigger
        self._proxies: Dict[str, LazyVerticalProxy] = {}
        self._loaded_verticals: Set[str] = set()

    def register_vertical(
        self, vertical_name: str, loader: Callable[[], Type], load_immediately: bool = False
    ) -> None:
        """Register vertical for lazy loading.

        Args:
            vertical_name: Name of the vertical
            loader: Callable that loads the vertical
            load_immediately: If True, load immediately regardless of trigger
        """
        proxy = LazyVerticalProxy(vertical_name=vertical_name, loader=loader)
        self._proxies[vertical_name] = proxy

        if self._load_trigger == LoadTrigger.EAGER or load_immediately:
            proxy.load()
            self._loaded_verticals.add(vertical_name)
            logger.info(f"Eagerly loaded vertical: {vertical_name}")

    def get_vertical(self, vertical_name: str) -> Optional[Type]:
        """Get vertical, loading if needed.

        Args:
            vertical_name: Name of the vertical

        Returns:
            Vertical class or None if not registered
        """
        proxy = self._proxies.get(vertical_name)
        if not proxy:
            logger.warning(f"Vertical not registered: {vertical_name}")
            return None

        vertical = proxy.load()
        self._loaded_verticals.add(vertical_name)
        return vertical

    def unload_vertical(self, vertical_name: str) -> bool:
        """Unload vertical to free memory.

        Args:
            vertical_name: Name of the vertical

        Returns:
            True if vertical was unloaded, False if not loaded
        """
        proxy = self._proxies.get(vertical_name)
        if proxy and proxy._loaded:
            proxy.unload()
            self._loaded_verticals.discard(vertical_name)
            logger.info(f"Unloaded vertical: {vertical_name}")
            return True
        return False

    def unload_all(self) -> int:
        """Unload all verticals to free memory.

        Returns:
            Number of verticals unloaded
        """
        count = 0
        for proxy in self._proxies.values():
            if proxy._loaded:
                proxy.unload()
                count += 1
        self._loaded_verticals.clear()
        logger.info(f"Unloaded {count} vertical(s)")
        return count

    def is_loaded(self, vertical_name: str) -> bool:
        """Check if vertical is currently loaded.

        Args:
            vertical_name: Name of the vertical

        Returns:
            True if vertical is loaded
        """
        return vertical_name in self._loaded_verticals

    def list_loaded(self) -> Set[str]:
        """List all currently loaded verticals.

        Returns:
            Set of loaded vertical names
        """
        return self._loaded_verticals.copy()

    def list_registered(self) -> list:
        """List all registered verticals (loaded or not).

        Returns:
            List of registered vertical names
        """
        return list(self._proxies.keys())

    def get_loaded_count(self) -> int:
        """Get count of currently loaded verticals.

        Returns:
            Number of loaded verticals
        """
        return len(self._loaded_verticals)

    def get_registered_count(self) -> int:
        """Get count of registered verticals.

        Returns:
            Number of registered verticals
        """
        return len(self._proxies)


# Global singleton instance for framework-wide use
_global_lazy_loader: Optional[LazyVerticalLoader] = None


def get_lazy_loader(load_trigger: LoadTrigger = LoadTrigger.ON_DEMAND) -> LazyVerticalLoader:
    """Get global lazy vertical loader instance.

    Args:
        load_trigger: When to load verticals (only used on first call)

    Returns:
        LazyVerticalLoader singleton instance
    """
    global _global_lazy_loader
    if _global_lazy_loader is None:
        _global_lazy_loader = LazyVerticalLoader(load_trigger)
        logger.info(f"Created global lazy loader with trigger: {load_trigger.value}")
    return _global_lazy_loader


__all__ = [
    "LoadTrigger",
    "LazyVerticalProxy",
    "LazyVerticalLoader",
    "get_lazy_loader",
]
