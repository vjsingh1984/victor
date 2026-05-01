"""Compatibility facade for singleton registry base classes.

The canonical import path is now ``victor.core.registry``:

    from victor.core.registry import ItemRegistry, SingletonRegistry
"""

from victor.core.registry.base import ItemRegistry, SingletonRegistry

__all__ = [
    "SingletonRegistry",
    "ItemRegistry",
]
