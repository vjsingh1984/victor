"""Interop tests for framework capability loader with SDK-defined entries."""

from __future__ import annotations

import tempfile
from pathlib import Path

from victor.framework.capability_loader import CapabilityLoader
from victor.framework.protocols import CapabilityType


def test_loader_accepts_sdk_capability_entries_from_module_path() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write("""
from victor_sdk.capabilities import (
    CapabilityEntry,
    CapabilityType,
    OrchestratorCapability,
)

def apply_sdk_capability(payload):
    return payload

CAPABILITIES = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="sdk_loaded_cap",
            capability_type=CapabilityType.SAFETY,
            setter="apply_sdk_capability",
        ),
        handler=apply_sdk_capability,
    )
]
""".strip())
        temp_path = handle.name

    try:
        loader = CapabilityLoader()
        loaded = loader.load_from_path(temp_path)

        assert "sdk_loaded_cap" in loaded
        entry = loader.get_capability("sdk_loaded_cap")
        assert entry is not None
        assert entry.capability_type == CapabilityType.SAFETY
        assert entry.handler is not None
    finally:
        Path(temp_path).unlink()


def test_loader_accepts_sdk_capability_decorator_metadata() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write("""
from victor_sdk.capabilities import CapabilityType, capability

@capability(
    name="sdk_decorated_cap",
    capability_type=CapabilityType.TOOL,
    setter="sdk_decorated_cap",
)
def sdk_decorated_cap(payload):
    return payload
""".strip())
        temp_path = handle.name

    try:
        loader = CapabilityLoader()
        loaded = loader.load_from_path(temp_path)

        assert "sdk_decorated_cap" in loaded
        entry = loader.get_capability("sdk_decorated_cap")
        assert entry is not None
        assert entry.capability_type == CapabilityType.TOOL
    finally:
        Path(temp_path).unlink()
