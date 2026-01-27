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

"""Decorators for the Victor agent framework.

This module provides reusable decorators for marking deprecated APIs,
managing deprecation warnings, and enforcing API lifecycle.
"""

import functools
import inspect
import logging
import warnings
from typing import Any, Callable, Optional, TypeVar, cast

# Type variables for generic decorator support
F = TypeVar("F", bound=Callable[..., Any])
P = TypeVar("P")

logger = logging.getLogger(__name__)


def deprecated(
    version: str,
    replacement: Optional[str] = None,
    remove_version: Optional[str] = None,
    reason: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to mark functions or methods as deprecated.

    This decorator issues a DeprecationWarning when the decorated function
    or method is called, providing guidance on migration paths.

    Args:
        version: The version when this API was deprecated (e.g., "0.5.0")
        replacement: The new API to use instead (e.g., "use new_method() instead")
        remove_version: The version when this API will be removed (e.g., "0.7.0")
        reason: Additional context for why this was deprecated

    Returns:
        Decorated function that issues a deprecation warning

    Example:
        ```python
        @deprecated(version="0.5.0", replacement="new_method()", remove_version="0.7.0")
        def old_method():
            pass

        @deprecated(
            version="0.5.0",
            replacement="coordinator.get_stats()",
            reason="Direct access violates encapsulation"
        )
        def get_legacy_stats():
            pass
        ```

    Migration Timeline:
    - Deprecated in: specified by `version` parameter
    - Removal target: specified by `remove_version` parameter (typically 2 major versions later)
    - Grace period: Users should migrate during this period
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build deprecation message
            parts = []

            # Base message
            if inspect.ismethod(func):
                # For methods, show class name
                if hasattr(args[0], "__class__"):
                    class_name = args[0].__class__.__name__
                    parts.append(f"{class_name}.{func.__name__} is deprecated")
                else:
                    parts.append(f"{func.__name__} is deprecated")
            else:
                parts.append(f"{func.__name__} is deprecated")

            # Version info
            if version:
                parts.append(f"(since {version})")

            # Replacement guidance
            if replacement:
                parts.append(f"— {replacement}")

            # Removal timeline
            if remove_version:
                parts.append(f"— will be removed in {remove_version}")

            # Additional reason
            if reason:
                parts.append(f"— {reason}")

            message = " ".join(parts)

            # Issue warning with proper stack level
            # stacklevel=2 ensures the warning points to the caller, not the wrapper
            warnings.warn(
                message,
                DeprecationWarning,
                stacklevel=3,  # Wrapper -> decorator func -> caller
            )

            # Log the deprecation for debugging
            logger.debug(f"Deprecated API called: {message}")

            # Call the original function
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_property(
    version: str,
    replacement: Optional[str] = None,
    remove_version: Optional[str] = None,
    reason: Optional[str] = None,
) -> Callable[[F], property]:
    """Decorator to mark properties as deprecated.

    Similar to @deprecated but specifically designed for properties,
    ensuring the warning is issued when the property is accessed (not just defined).

    Args:
        version: The version when this property was deprecated
        replacement: The new property or method to use instead
        remove_version: The version when this property will be removed
        reason: Additional context for the deprecation

    Returns:
        Decorated property that issues a deprecation warning

    Example:
        ```python
        class MyClass:
            @deprecated_property(
                version="0.5.0",
                replacement="new_attribute",
                remove_version="0.7.0"
            )
            def old_attribute(self) -> Any:
                return self._old_value
        ```
    """

    def decorator(func: F) -> property:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Build deprecation message
            parts = [f"Property '{func.__name__}' is deprecated"]

            if version:
                parts.append(f"(since {version})")

            if replacement:
                parts.append(f"— use '{replacement}' instead")

            if remove_version:
                parts.append(f"— will be removed in {remove_version}")

            if reason:
                parts.append(f"— {reason}")

            message = " ".join(parts)

            warnings.warn(
                message,
                DeprecationWarning,
                stacklevel=2,
            )

            logger.debug(f"Deprecated property accessed: {message}")

            # Call the original property getter
            return func(self)

        # Convert to property
        # Type: ignore needed because wrapper signature doesn't match property's expected Callable[[Any], Any]
        return property(wrapper)  # type: ignore[arg-type]

    return decorator


def deprecated_class(
    version: str,
    replacement: Optional[str] = None,
    remove_version: Optional[str] = None,
    reason: Optional[str] = None,
) -> Callable[[Any], Any]:
    """Decorator to mark classes as deprecated.

    Issues a warning when the class is instantiated, not just imported.
    This allows for gradual migration without import-time warnings.

    Args:
        version: The version when this class was deprecated
        replacement: The new class to use instead
        remove_version: The version when this class will be removed
        reason: Additional context for the deprecation

    Returns:
        Decorated class that issues a deprecation warning on instantiation

    Example:
        ```python
        @deprecated_class(
            version="0.5.0",
            replacement="NewClass",
            remove_version="0.7.0",
            reason="Replaced with more efficient implementation"
        )
        class OldClass:
            def __init__(self) -> None:
                pass
        ```
    """

    def decorator(cls: Any) -> Any:
        original_init = cls.__init__

        @functools.wraps(original_init)
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Build deprecation message
            parts = [f"Class '{cls.__name__}' is deprecated"]

            if version:
                parts.append(f"(since {version})")

            if replacement:
                parts.append(f"— use '{replacement}' instead")

            if remove_version:
                parts.append(f"— will be removed in {remove_version}")

            if reason:
                parts.append(f"— {reason}")

            message = " ".join(parts)

            warnings.warn(
                message,
                DeprecationWarning,
                stacklevel=2,
            )

            logger.debug(f"Deprecated class instantiated: {message}")

            # Call original __init__
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return decorator


__all__ = [
    "deprecated",
    "deprecated_property",
    "deprecated_class",
]
