"""Protocol verification utilities."""

from __future__ import annotations

from typing import Any


def verify_protocol_conformance(obj: Any, protocol: type) -> tuple[bool, list[str]]:
    """Verify that an object conforms to a protocol.

    Checks that all required methods/properties are present.
    Useful for debugging protocol conformance issues.

    Args:
        obj: Object to check
        protocol: Protocol class to check against

    Returns:
        Tuple of (conforms: bool, missing: List[str])

    Example:
        conforms, missing = verify_protocol_conformance(orch, OrchestratorProtocol)
        if not conforms:
            raise TypeError(f"Missing protocol methods: {missing}")
    """
    missing = []

    # Get protocol's __protocol_attrs__ if available (runtime_checkable)
    if hasattr(protocol, "__protocol_attrs__"):
        for attr in protocol.__protocol_attrs__:
            if not hasattr(obj, attr):
                missing.append(attr)
    else:
        # Fall back to checking annotations
        hints = getattr(protocol, "__annotations__", {})
        for attr in hints:
            if not hasattr(obj, attr):
                missing.append(attr)

    return len(missing) == 0, missing
