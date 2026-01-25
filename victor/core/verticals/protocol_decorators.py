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

"""Protocol registration decorators for verticals.

Phase 2.1: Create @register_protocols decorator.

This module provides a class decorator that auto-detects which protocols
a vertical class implements and registers them with the protocol loader.

Usage:
    @register_protocols
    class MyVertical(VerticalBase):
        def get_tools(self) -> List[str]:
            return [...]

        def get_prompt_contributors(self) -> List[Any]:
            return [...]

    # No manual registration needed! The decorator auto-detects
    # ToolProvider and PromptContributorProvider implementations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Type

if TYPE_CHECKING:
    from typing import Protocol as TypingProtocol
else:
    # For runtime, we need a placeholder
    TypingProtocol = Type[object]  # type: ignore

logger = logging.getLogger(__name__)

# Known vertical protocols that can be auto-detected
# These are the protocols commonly implemented by verticals
KNOWN_VERTICAL_PROTOCOLS: List[Type[object]] = []


def _load_known_protocols() -> None:
    """Lazy load known vertical protocols to avoid circular imports."""
    global KNOWN_VERTICAL_PROTOCOLS

    if KNOWN_VERTICAL_PROTOCOLS:
        return  # Already loaded

    try:
        from victor.core.verticals.protocols import (
            ToolProvider,
            PromptContributorProvider,
            MiddlewareProvider,
            ToolDependencyProvider,
            HandlerProvider,
            CapabilityProvider,
            ModeConfigProvider,
            ServiceProvider,
            TieredToolConfigProvider,
            SafetyProvider,
        )

        KNOWN_VERTICAL_PROTOCOLS = [
            ToolProvider,
            PromptContributorProvider,
            MiddlewareProvider,
            ToolDependencyProvider,
            HandlerProvider,
            CapabilityProvider,
            ModeConfigProvider,
            ServiceProvider,
            TieredToolConfigProvider,
            SafetyProvider,
        ]
        logger.debug(f"Loaded {len(KNOWN_VERTICAL_PROTOCOLS)} known vertical protocols")
    except ImportError as e:
        logger.warning(f"Could not load vertical protocols: {e}")
        KNOWN_VERTICAL_PROTOCOLS = []


def get_implemented_protocols(
    cls: Type[Any],
    protocols: List[Type[object]],
) -> List[Type[object]]:
    """Detect which protocols a class implements.

    Uses isinstance checking with runtime_checkable protocols to
    determine which protocols a class implements.

    Args:
        cls: Class to check
        protocols: List of protocols to check against

    Returns:
        List of protocols that the class implements
    """
    implemented = []

    for protocol in protocols:
        try:
            # Check if the class has the required methods/attributes
            # by checking if an instance would pass isinstance
            if _class_implements_protocol(cls, protocol):
                implemented.append(protocol)
        except Exception as e:
            logger.debug(f"Error checking protocol {protocol.__name__}: {e}")

    return implemented


def _class_implements_protocol(cls: Type[Any], protocol: Type[object]) -> bool:
    """Check if a class implements a protocol.

    This performs a structural check to see if the class has the
    required methods and attributes of the protocol.

    Args:
        cls: Class to check
        protocol: Protocol to check against

    Returns:
        True if the class implements the protocol
    """
    # Get protocol methods/attributes
    protocol_attrs = getattr(protocol, "__protocol_attrs__", None)

    if protocol_attrs is None:
        # Fallback for older protocol style
        protocol_attrs = set()
        for name in dir(protocol):
            if not name.startswith("_"):
                protocol_attrs.add(name)

    # Check if class has all required attributes
    for attr in protocol_attrs:
        if not hasattr(cls, attr):
            return False

    return True


def register_protocols(
    cls: Optional[Type[Any]] = None,
    *,
    protocols: Optional[List[Type[object]]] = None,
    auto_detect: bool = True,
) -> Any:  # Returns the class decorator or the decorated class
    """Class decorator for automatic protocol registration.

    This decorator auto-detects which protocols a vertical class
    implements and registers them with the protocol loader.

    Usage:
        # Auto-detect protocols (recommended)
        @register_protocols
        class MyVertical(VerticalBase):
            ...

        # Explicit protocols
        @register_protocols(protocols=[ToolProvider, PromptContributorProvider])
        class MyVertical(VerticalBase):
            ...

        # Disable auto-detection
        @register_protocols(protocols=[ToolProvider], auto_detect=False)
        class MyVertical(VerticalBase):
            ...

    Args:
        cls: Class being decorated (when used without parentheses)
        protocols: Explicit list of protocols to register
        auto_detect: If True, auto-detect additional protocols

    Returns:
        Decorated class (unchanged)
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        # Lazy load known protocols
        _load_known_protocols()

        # Determine protocols to register
        protocols_to_register: List[Type[object]] = []

        # Add explicit protocols
        if protocols:
            protocols_to_register.extend(protocols)

        # Auto-detect additional protocols if enabled
        if auto_detect and KNOWN_VERTICAL_PROTOCOLS:
            detected = get_implemented_protocols(cls, KNOWN_VERTICAL_PROTOCOLS)
            for p in detected:
                if p not in protocols_to_register:
                    protocols_to_register.append(p)

        # Register with protocol loader
        if protocols_to_register:
            try:
                from victor.core.verticals.protocol_loader import (
                    ProtocolBasedExtensionLoader,
                )

                for protocol in protocols_to_register:
                    try:
                        ProtocolBasedExtensionLoader.register_protocol(protocol, cls)
                        logger.debug(f"Registered {cls.__name__} for protocol {protocol.__name__}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to register {cls.__name__} for {protocol.__name__}: {e}"
                        )
            except ImportError as e:
                logger.warning(f"Could not import protocol loader: {e}")

        return cls

    # Handle both @register_protocols and @register_protocols()
    if cls is not None:
        # Called without parentheses: @register_protocols
        return decorator(cls)
    else:
        # Called with parentheses: @register_protocols() or @register_protocols(...)
        return decorator


__all__ = [
    "register_protocols",
    "get_implemented_protocols",
    "KNOWN_VERTICAL_PROTOCOLS",
]
