"""Regression coverage for DevOps prompt metadata definition/runtime parity."""

from victor.core.verticals.base import VerticalBase as RuntimeVerticalBase
from victor_sdk import StageDefinition

from victor.verticals.contrib.devops import DevOpsAssistant as RuntimeDevOpsAssistant
from victor.verticals.contrib.devops.assistant import DevOpsAssistant
from victor.verticals.contrib.devops.prompts import DevOpsPromptContributor


def test_devops_definition_exposes_serializable_prompt_metadata() -> None:
    """DevOps should expose prompt metadata through the SDK definition contract."""

    definition = DevOpsAssistant.get_definition()
    templates = {
        template.task_type: template.template for template in definition.prompt_metadata.templates
    }
    hints = {hint.task_type: hint for hint in definition.prompt_metadata.task_type_hints}

    assert templates["devops_operations"].startswith("## Security Checklist")
    assert {"infrastructure", "ci_cd", "dockerfile", "terraform", "general"}.issubset(hints)
    assert hints["terraform"].priority_tools == ["read", "write", "edit", "ls", "shell"]
    assert definition.prompt_metadata.metadata["priority"] == 5
    assert "Never invent file paths or content" in definition.prompt_metadata.metadata[
        "grounding_rules"
    ]


def test_devops_prompt_contributor_wraps_shared_prompt_metadata() -> None:
    """Runtime prompt contributor should derive from the shared metadata payload."""

    contributor = DevOpsPromptContributor()
    hints = contributor.get_task_type_hints()

    assert contributor.get_system_prompt_section().startswith("## Security Checklist")
    assert contributor.get_priority() == 5
    assert "terraform" in hints
    assert hints["docker_compose"].priority_tools == ["read", "write", "edit", "ls"]


def test_devops_assistant_uses_sdk_stage_contracts() -> None:
    """DevOps assistant should use SDK-owned stage definitions."""

    stages = DevOpsAssistant.get_stages()

    assert isinstance(stages["INITIAL"], StageDefinition)
    assert stages["IMPLEMENTATION"].tools == {
        "write",
        "edit",
        "shell",
        "docker",
    }
    assert stages["DEPLOYMENT"].next_stages == {"MONITORING", "COMPLETION"}


def test_devops_package_exports_runtime_compatible_shim() -> None:
    """The package boundary should preserve runtime helper behavior for DevOps."""

    assert DevOpsAssistant is not RuntimeDevOpsAssistant
    assert RuntimeDevOpsAssistant.__victor_sdk_source__ is DevOpsAssistant
    assert issubclass(RuntimeDevOpsAssistant, RuntimeVerticalBase)
    assert RuntimeDevOpsAssistant.get_definition().name == "devops"
