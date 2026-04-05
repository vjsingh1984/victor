"""Unit tests for SDK tool requirement helpers."""

from victor_sdk import (
    ToolRequirement,
    normalize_tool_requirement,
    normalize_tool_requirements,
)


def test_normalize_tool_requirement_supports_strings_and_objects() -> None:
    """Legacy strings and typed tool requirements normalize to the same shape."""

    string_requirement = normalize_tool_requirement("read")
    typed_requirement = normalize_tool_requirement(
        ToolRequirement(
            tool_name="git",
            required=False,
            purpose="optional repository workflows",
        )
    )

    assert string_requirement.tool_name == "read"
    assert string_requirement.required is True
    assert typed_requirement.tool_name == "git"
    assert typed_requirement.required is False
    assert typed_requirement.to_dict()["purpose"] == "optional repository workflows"


def test_normalize_tool_requirements_preserves_order() -> None:
    """Batch normalization should preserve declaration order."""

    requirements = normalize_tool_requirements(
        [
            "read",
            ToolRequirement(tool_name="shell", required=False),
        ]
    )

    assert [requirement.tool_name for requirement in requirements] == ["read", "shell"]
    assert requirements[1].required is False


def test_normalize_tool_requirement_supports_serialized_dicts() -> None:
    """Serialized tool requirement payloads should round-trip through normalization."""

    requirement = normalize_tool_requirement(
        {
            "tool_name": "search",
            "required": False,
            "purpose": "optional discovery",
            "metadata": {"scope": "repo"},
        }
    )

    assert requirement.tool_name == "search"
    assert requirement.required is False
    assert requirement.metadata["scope"] == "repo"
