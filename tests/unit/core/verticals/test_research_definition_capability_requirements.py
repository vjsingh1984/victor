"""Regression coverage for Research capability requirements on the SDK definition."""

from victor.framework.sdk_capability_registry import resolve_capability_requirements
from victor.verticals.contrib.research.assistant import ResearchAssistant
from victor_sdk import CapabilityIds


def test_research_definition_declares_sdk_capability_requirements() -> None:
    """Research should declare runtime capabilities through SDK identifiers."""

    requirements = ResearchAssistant.get_capability_requirements()

    assert [requirement.capability_id for requirement in requirements] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.WEB_ACCESS,
        CapabilityIds.SOURCE_VERIFICATION,
        CapabilityIds.VALIDATION,
    ]


def test_research_capability_requirements_resolve_against_current_tool_bundle() -> None:
    """Current Research tooling should satisfy the new SDK capability declarations."""

    resolutions = resolve_capability_requirements(
        ResearchAssistant.get_capability_requirements(),
        available_tools=ResearchAssistant.get_tools(),
    )

    availability = {resolution.capability_id: resolution.available for resolution in resolutions}

    assert availability[CapabilityIds.FILE_OPS] is True
    assert availability[CapabilityIds.WEB_ACCESS] is True
    assert availability[CapabilityIds.SOURCE_VERIFICATION] is True
    assert availability[CapabilityIds.VALIDATION] is True
