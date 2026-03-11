"""Regression coverage for Research prompt metadata definition/runtime parity."""

from victor.core.verticals.base import VerticalBase as RuntimeVerticalBase
from victor_sdk import StageDefinition

from victor.verticals.contrib.research import ResearchAssistant as RuntimeResearchAssistant
from victor.verticals.contrib.research.assistant import ResearchAssistant
from victor.verticals.contrib.research.prompts import ResearchPromptContributor


def test_research_definition_exposes_serializable_prompt_metadata() -> None:
    """Research should expose prompt metadata through the SDK definition contract."""

    definition = ResearchAssistant.get_definition()
    templates = {
        template.task_type: template.template for template in definition.prompt_metadata.templates
    }
    hints = {hint.task_type: hint for hint in definition.prompt_metadata.task_type_hints}

    assert templates["research_operations"].startswith("## Research Quality Checklist")
    assert {
        "fact_check",
        "literature_review",
        "competitive_analysis",
        "general",
    }.issubset(hints)
    assert hints["technical_research"].priority_tools == [
        "web_search",
        "web_fetch",
        "code_search",
        "read",
    ]
    assert definition.prompt_metadata.metadata["priority"] == 5
    assert (
        "Never fabricate sources or statistics"
        in definition.prompt_metadata.metadata["grounding_rules"]
    )


def test_research_prompt_contributor_wraps_shared_prompt_metadata() -> None:
    """Runtime prompt contributor should derive from the shared metadata payload."""

    contributor = ResearchPromptContributor()
    hints = contributor.get_task_type_hints()

    assert contributor.get_system_prompt_section().startswith("## Research Quality Checklist")
    assert contributor.get_priority() == 5
    assert "trend_research" in hints
    assert hints["general"].priority_tools == ["web_search", "web_fetch", "read"]


def test_research_assistant_uses_sdk_stage_contracts() -> None:
    """Research assistant should use SDK-owned stage definitions."""

    stages = ResearchAssistant.get_stages()

    assert isinstance(stages["INITIAL"], StageDefinition)
    assert stages["SEARCHING"].tools == {"web_search", "web_fetch", "grep"}
    assert stages["WRITING"].next_stages == {"VERIFICATION", "SYNTHESIZING"}


def test_research_package_exports_runtime_compatible_shim() -> None:
    """The package boundary should preserve runtime helper behavior for Research."""

    assert ResearchAssistant is not RuntimeResearchAssistant
    assert RuntimeResearchAssistant.__victor_sdk_source__ is ResearchAssistant
    assert issubclass(RuntimeResearchAssistant, RuntimeVerticalBase)
    assert RuntimeResearchAssistant.get_definition().name == "research"
