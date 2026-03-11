"""Tests for SDK capability ID runtime resolution."""

from __future__ import annotations

from victor.framework.sdk_capability_registry import (
    get_runtime_capability_registry,
    resolve_capability_requirement,
    resolve_capability_requirements,
)
from victor_sdk.constants import CapabilityIds


class LspCapableOrchestrator:
    """Simple object exposing the public LSP setter fallback."""

    def set_lsp(self, value):  # pragma: no cover - fallback surface only
        self.value = value


def test_runtime_capability_registry_contains_core_bindings():
    """The runtime registry should expose the initial SDK capability map."""

    registry = get_runtime_capability_registry(reset=True)

    assert registry.get(CapabilityIds.FILE_OPS) is not None
    assert registry.get(CapabilityIds.LSP) is not None
    assert registry.get(CapabilityIds.PRIVACY) is not None


def test_resolve_file_ops_requirement_from_tools():
    """File ops should resolve from the expected filesystem tool bundle."""

    resolution = resolve_capability_requirement(
        CapabilityIds.FILE_OPS,
        available_tools={"read", "write", "edit", "grep"},
    )

    assert resolution.available is True
    assert resolution.known is True
    assert resolution.source == "tools:read,write,edit,grep"


def test_resolve_lsp_requirement_from_orchestrator_capability():
    """LSP should resolve via the orchestrator capability bridge when present."""

    resolution = resolve_capability_requirement(
        CapabilityIds.LSP,
        orchestrator=LspCapableOrchestrator(),
    )

    assert resolution.available is True
    assert resolution.known is True
    assert resolution.source == "orchestrator:lsp"


def test_resolve_builtin_privacy_requirement():
    """Framework-owned builtins should resolve without external providers."""

    resolution = resolve_capability_requirement(CapabilityIds.PRIVACY)

    assert resolution.available is True
    assert resolution.known is True
    assert resolution.source is not None
    assert resolution.source.startswith("builtin:")


def test_unknown_capability_requirement_is_reported_explicitly():
    """Unknown SDK capability IDs should produce a missing-binding diagnostic."""

    resolution = resolve_capability_requirement("not_registered_capability")

    assert resolution.available is False
    assert resolution.known is False
    assert "No runtime binding" in (resolution.reason or "")


def test_resolve_capability_requirements_preserves_optionality():
    """Batch resolution should carry through required vs optional semantics."""

    resolutions = resolve_capability_requirements(
        [
            CapabilityIds.FILE_OPS,
            {"capability_id": "not_registered_capability", "optional": True},
        ],
        available_tools={"read", "write", "edit", "grep"},
    )

    assert [resolution.capability_id for resolution in resolutions] == [
        CapabilityIds.FILE_OPS,
        "not_registered_capability",
    ]
    assert resolutions[0].available is True
    assert resolutions[1].available is False
    assert resolutions[1].optional is True
