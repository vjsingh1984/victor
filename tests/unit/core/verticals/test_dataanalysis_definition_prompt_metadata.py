"""Regression coverage for Data Analysis prompt metadata definition/runtime parity."""

from victor.core.verticals.base import VerticalBase as RuntimeVerticalBase
from victor_sdk import StageDefinition

from victor.verticals.contrib.dataanalysis import (
    DataAnalysisAssistant as RuntimeDataAnalysisAssistant,
)
from victor.verticals.contrib.dataanalysis.assistant import DataAnalysisAssistant
from victor.verticals.contrib.dataanalysis.prompts import DataAnalysisPromptContributor


def test_dataanalysis_definition_exposes_serializable_prompt_metadata() -> None:
    """Data Analysis should expose prompt metadata through the SDK definition contract."""

    definition = DataAnalysisAssistant.get_definition()
    templates = {
        template.task_type: template.template for template in definition.prompt_metadata.templates
    }
    hints = {hint.task_type: hint for hint in definition.prompt_metadata.task_type_hints}

    assert templates["data_analysis_operations"].startswith("## Python Libraries Reference")
    assert {
        "data_analysis",
        "visualization",
        "data_profiling",
        "statistical_analysis",
        "general",
    }.issubset(hints)
    assert hints["regression"].priority_tools == ["shell", "read", "write", "edit"]
    assert definition.prompt_metadata.metadata["priority"] == 5
    assert (
        "Never fabricate data or statistics"
        in definition.prompt_metadata.metadata["grounding_rules"]
    )


def test_dataanalysis_prompt_contributor_wraps_shared_prompt_metadata() -> None:
    """Runtime prompt contributor should derive from the shared metadata payload."""

    contributor = DataAnalysisPromptContributor()
    hints = contributor.get_task_type_hints()

    assert contributor.get_system_prompt_section().startswith("## Python Libraries Reference")
    assert contributor.get_priority() == 5
    assert "visualization" in hints
    assert hints["time_series"].priority_tools == ["shell", "read", "write", "edit"]


def test_dataanalysis_assistant_uses_sdk_stage_contracts() -> None:
    """Data Analysis assistant should use SDK-owned stage definitions."""

    stages = DataAnalysisAssistant.get_stages()

    assert isinstance(stages["INITIAL"], StageDefinition)
    assert stages["DATA_LOADING"].tools == {"read", "shell", "write"}
    assert stages["ANALYSIS"].next_stages == {"VISUALIZATION", "REPORTING"}


def test_dataanalysis_package_exports_runtime_compatible_shim() -> None:
    """The package boundary should preserve runtime helper behavior for Data Analysis."""

    assert DataAnalysisAssistant is not RuntimeDataAnalysisAssistant
    assert RuntimeDataAnalysisAssistant.__victor_sdk_source__ is DataAnalysisAssistant
    assert issubclass(RuntimeDataAnalysisAssistant, RuntimeVerticalBase)
    assert RuntimeDataAnalysisAssistant.get_definition().name == "dataanalysis"
