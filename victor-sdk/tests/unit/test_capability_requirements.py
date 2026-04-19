"""Unit tests for SDK capability identifiers and requirement helpers."""

from victor_sdk import (
    CapabilityIds,
    CapabilityRequirement,
    get_all_capability_ids,
    is_known_capability_id,
    normalize_capability_requirement,
    normalize_capability_requirements,
)
from victor_sdk.verticals.metadata import VerticalMetadata
from victor_sdk.verticals.protocols import CapabilityProvider


def test_capability_ids_are_exported_from_sdk() -> None:
    """The SDK exports a stable host capability registry."""

    assert CapabilityIds.FILE_OPS == "file_ops"
    assert CapabilityIds.LSP == "lsp"
    assert CapabilityIds.SOURCE_VERIFICATION == "source_verification"
    assert is_known_capability_id(CapabilityIds.PRIVACY) is True
    assert "file_ops" in get_all_capability_ids()


def test_normalize_capability_requirement_supports_strings_and_objects() -> None:
    """Legacy strings and typed requirements normalize to the same shape."""

    string_requirement = normalize_capability_requirement(CapabilityIds.FILE_OPS)
    typed_requirement = normalize_capability_requirement(
        CapabilityRequirement(
            capability_id=CapabilityIds.GIT,
            min_version="2.0",
            purpose="commit and branch management",
        )
    )

    assert string_requirement.capability_id == CapabilityIds.FILE_OPS
    assert string_requirement.optional is False
    assert typed_requirement.min_version == "2.0"
    assert typed_requirement.purpose == "commit and branch management"
    assert typed_requirement.to_dict()["capability_id"] == CapabilityIds.GIT


def test_normalize_capability_requirements_preserves_order() -> None:
    """Batch normalization should preserve declaration order."""

    requirements = normalize_capability_requirements(
        [
            CapabilityIds.FILE_OPS,
            CapabilityRequirement(capability_id=CapabilityIds.LSP, optional=True),
        ]
    )

    assert [requirement.capability_id for requirement in requirements] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.LSP,
    ]
    assert requirements[1].optional is True


def test_normalize_capability_requirement_supports_serialized_dicts() -> None:
    """Serialized capability payloads should round-trip through normalization."""

    requirement = normalize_capability_requirement(
        {
            "capability_id": CapabilityIds.GIT,
            "min_version": "2.1",
            "optional": True,
            "purpose": "optional repository automation",
            "metadata": {"mode": "readonly"},
        }
    )

    assert requirement.capability_id == CapabilityIds.GIT
    assert requirement.min_version == "2.1"
    assert requirement.optional is True
    assert requirement.metadata["mode"] == "readonly"


def test_vertical_metadata_tracks_typed_and_legacy_requirements() -> None:
    """Vertical metadata can bridge legacy and typed requirement declarations."""

    metadata = VerticalMetadata(
        name="coding", description="Coding vertical"
    ).with_requirement(
        CapabilityRequirement(
            capability_id=CapabilityIds.FILE_OPS,
            purpose="read and write project files",
        )
    )

    assert metadata.requirements == [CapabilityIds.FILE_OPS]
    assert metadata.get_requirement_names() == [CapabilityIds.FILE_OPS]
    assert metadata.capability_requirements[0].purpose == "read and write project files"
    assert (
        metadata.get_all_metadata()["capability_requirements"][0]["capability_id"]
        == "file_ops"
    )


def test_capability_provider_protocol_accepts_typed_requirements() -> None:
    """CapabilityProvider implementations can return typed requirements."""

    class ExampleCapabilityProvider:
        def get_capabilities(self) -> dict[str, object]:
            return {"file_ops": object()}

        def has_capability(self, capability_name: str) -> bool:
            return capability_name == CapabilityIds.FILE_OPS

        def get_capability_requirements(self) -> list[CapabilityRequirement]:
            return [CapabilityRequirement(capability_id=CapabilityIds.FILE_OPS)]

    assert isinstance(ExampleCapabilityProvider(), CapabilityProvider)
