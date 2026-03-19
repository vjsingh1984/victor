"""Regression coverage for DevOps capability requirements on the SDK definition."""

from victor.framework.sdk_capability_registry import resolve_capability_requirements
from victor.verticals.contrib.devops.assistant import DevOpsAssistant
from victor_sdk import CapabilityIds


def test_devops_definition_declares_sdk_capability_requirements() -> None:
    """DevOps should declare runtime capabilities through SDK identifiers."""

    requirements = DevOpsAssistant.get_capability_requirements()

    assert [requirement.capability_id for requirement in requirements] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.SHELL_ACCESS,
        CapabilityIds.GIT,
        CapabilityIds.CONTAINER_RUNTIME,
        CapabilityIds.VALIDATION,
        CapabilityIds.WEB_ACCESS,
    ]
    assert requirements[-1].optional is True


def test_devops_capability_requirements_resolve_against_current_tool_bundle() -> None:
    """Current DevOps tooling should satisfy the new SDK capability declarations."""

    resolutions = resolve_capability_requirements(
        DevOpsAssistant.get_capability_requirements(),
        available_tools=DevOpsAssistant.get_tools(),
    )

    availability = {resolution.capability_id: resolution.available for resolution in resolutions}

    assert availability[CapabilityIds.FILE_OPS] is True
    assert availability[CapabilityIds.SHELL_ACCESS] is True
    assert availability[CapabilityIds.GIT] is True
    assert availability[CapabilityIds.CONTAINER_RUNTIME] is True
    assert availability[CapabilityIds.VALIDATION] is True
    assert availability[CapabilityIds.WEB_ACCESS] is True
