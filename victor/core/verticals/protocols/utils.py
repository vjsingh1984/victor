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

"""Protocol checking utilities (Phase 11.1).

Provides standardized functions for protocol checking, replacing
inconsistent isinstance/hasattr patterns across the codebase.

Design Philosophy:
- Protocol-first: Use isinstance() with @runtime_checkable protocols
- No hasattr fallback: Explicit protocol conformance required
- Type safety: Returns typed values when conformant, None otherwise
- Consistent: Single pattern for all protocol checks

Usage:
    from victor.core.verticals.protocols.utils import check_protocol

    # Check if object conforms to protocol
    workflow_provider = check_protocol(vertical, WorkflowProviderProtocol)
    if workflow_provider:
        workflows = workflow_provider.get_workflows()

    # Optional provider pattern
    rl_provider = check_protocol_optional(vertical, "get_rl_config_provider", RLConfigProviderProtocol)
    if rl_provider:
        config = rl_provider.get_rl_config()
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)


# Type variable for protocol types
T = TypeVar("T")


def check_protocol(obj: Any, protocol: type[T]) -> Optional[T]:
    """Check if object conforms to protocol using isinstance.

    This function provides consistent protocol checking without hasattr fallbacks.
    It uses Python's isinstance() with @runtime_checkable protocols.

    Args:
        obj: Object to check for protocol conformance
        protocol: Protocol type (must be @runtime_checkable)

    Returns:
        The object cast to protocol type if conformant, None otherwise

    Example:
        workflow_provider = check_protocol(vertical, WorkflowProviderProtocol)
        if workflow_provider:
            workflows = workflow_provider.get_workflows()
    """
    if obj is None:
        return None

    try:
        if isinstance(obj, protocol):
            return obj
    except TypeError:
        # Protocol is not runtime_checkable
        logger.warning(f"Protocol {protocol.__name__} is not runtime_checkable")

    return None


def check_protocol_optional(
    obj: Any,
    method_name: str,
    protocol: type[T],
) -> Optional[T]:
    """Check optional provider pattern: obj.method_name() -> protocol conformant.

    This handles the common pattern where a vertical has an optional provider method
    that returns an object conforming to a protocol.

    Args:
        obj: Object that may have the method
        method_name: Name of method to call (e.g., "get_rl_config_provider")
        protocol: Protocol the returned object should conform to

    Returns:
        The provider cast to protocol type if available and conformant, None otherwise

    Example:
        rl_provider = check_protocol_optional(
            vertical,
            "get_rl_config_provider",
            RLConfigProviderProtocol
        )
        if rl_provider:
            config = rl_provider.get_rl_config()
    """
    if obj is None:
        return None

    # Check if method exists
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return None

    # Call method to get provider
    try:
        provider = method()
    except Exception:
        return None

    # Check protocol conformance
    return check_protocol(provider, protocol)


def is_protocol_conformant(obj: Any, protocol: type[T]) -> bool:
    """Check if object conforms to protocol (boolean result).

    Simple helper that returns True/False instead of the object.

    Args:
        obj: Object to check
        protocol: Protocol to check against

    Returns:
        True if object conforms to protocol, False otherwise
    """
    return check_protocol(obj, protocol) is not None


def get_protocol_methods(protocol: type[Any]) -> set[str]:
    """Get the method names defined by a protocol.

    Args:
        protocol: Protocol class

    Returns:
        Set of method names from the protocol

    Note:
        This only returns methods explicitly defined in the protocol,
        not inherited methods from Protocol base.
    """
    methods = set()

    # Get methods from protocol annotations
    if hasattr(protocol, "__protocol_attrs__"):
        methods.update(protocol.__protocol_attrs__)

    # Fallback: look at __annotations__
    if hasattr(protocol, "__annotations__"):
        methods.update(protocol.__annotations__.keys())

    # Also check for method definitions
    for name in dir(protocol):
        if name.startswith("_"):
            continue
        attr = getattr(protocol, name, None)
        if callable(attr) and not name.startswith("_"):
            methods.add(name)

    # Remove common Protocol methods
    methods -= {"__init__", "__subclasshook__", "__class_getitem__"}

    return methods


def protocol_error_message(obj: Any, protocol: type[Any]) -> str:
    """Generate an error message for protocol non-conformance.

    Args:
        obj: Object that does not conform
        protocol: Protocol it should conform to

    Returns:
        Descriptive error message
    """
    obj_type = type(obj).__name__
    protocol_name = protocol.__name__

    methods = get_protocol_methods(protocol)
    missing = []

    for method in methods:
        if not hasattr(obj, method):
            missing.append(method)

    if missing:
        return (
            f"{obj_type} does not conform to {protocol_name}. "
            f"Missing: {', '.join(sorted(missing))}"
        )

    return f"{obj_type} does not conform to {protocol_name}"


def require_protocol(obj: Any, protocol: type[T], context: str = "") -> T:
    """Require that object conforms to protocol, raise if not.

    Use this when protocol conformance is mandatory.

    Args:
        obj: Object that must conform
        protocol: Protocol to require
        context: Optional context for error message

    Returns:
        Object cast to protocol type

    Raises:
        TypeError: If object does not conform to protocol
    """
    result = check_protocol(obj, protocol)
    if result is None:
        msg = protocol_error_message(obj, protocol)
        if context:
            msg = f"{context}: {msg}"
        raise TypeError(msg)
    return result


__all__ = [
    "check_protocol",
    "check_protocol_optional",
    "is_protocol_conformant",
    "get_protocol_methods",
    "protocol_error_message",
    "require_protocol",
]
