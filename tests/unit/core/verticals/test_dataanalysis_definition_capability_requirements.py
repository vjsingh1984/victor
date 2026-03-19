"""Regression coverage for Data Analysis capability requirements on the SDK definition."""

from victor.framework.sdk_capability_registry import resolve_capability_requirements
from victor.verticals.contrib.dataanalysis.assistant import DataAnalysisAssistant
from victor_sdk import CapabilityIds


def test_dataanalysis_definition_declares_sdk_capability_requirements() -> None:
    """Data Analysis should declare runtime capabilities through SDK identifiers."""

    requirements = DataAnalysisAssistant.get_capability_requirements()

    assert [requirement.capability_id for requirement in requirements] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.SHELL_ACCESS,
        CapabilityIds.VALIDATION,
        CapabilityIds.WEB_ACCESS,
    ]
    assert requirements[-1].optional is True


def test_dataanalysis_capability_requirements_resolve_against_current_tool_bundle() -> None:
    """Current Data Analysis tooling should satisfy the new SDK capability declarations."""

    resolutions = resolve_capability_requirements(
        DataAnalysisAssistant.get_capability_requirements(),
        available_tools=DataAnalysisAssistant.get_tools(),
    )

    availability = {resolution.capability_id: resolution.available for resolution in resolutions}

    assert availability[CapabilityIds.FILE_OPS] is True
    assert availability[CapabilityIds.SHELL_ACCESS] is True
    assert availability[CapabilityIds.VALIDATION] is True
    assert availability[CapabilityIds.WEB_ACCESS] is True
