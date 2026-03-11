"""Regression coverage for shared runtime-helper defaults on vertical assistants."""

from victor_sdk import ToolNames

from victor.verticals.contrib.coding.assistant import CodingAssistant
from victor.verticals.contrib.coding.capabilities import (
    get_capability_configs as get_coding_capability_configs,
)
from victor.verticals.contrib.dataanalysis.assistant import DataAnalysisAssistant
from victor.verticals.contrib.dataanalysis.capabilities import (
    DataAnalysisCapabilityProvider,
)
from victor.verticals.contrib.devops.assistant import DevOpsAssistant
from victor.verticals.contrib.devops.capabilities import DevOpsCapabilityProvider
from victor.verticals.contrib.rag.assistant import RAGAssistant
from victor.verticals.contrib.rag.capabilities import (
    get_capability_configs as get_rag_capability_configs,
)
from victor.verticals.contrib.rag.prompts import RAGPromptContributor
from victor.verticals.contrib.rag.rl import RAGRLConfig
from victor.verticals.contrib.rag.safety import RAGSafetyExtension
from victor.verticals.contrib.rag.teams import RAGTeamSpecProvider
from victor.verticals.contrib.research.assistant import ResearchAssistant
from victor.verticals.contrib.research.capabilities import (
    get_capability_configs as get_research_capability_configs,
)


def test_sdk_file_operation_group_is_used_by_definition_entrypoints() -> None:
    """Definition-facing assistants should use the SDK tool grouping for file ops."""

    file_ops = set(ToolNames.file_operations())

    assert file_ops.issubset(set(CodingAssistant.get_tools()))
    assert file_ops.issubset(set(ResearchAssistant.get_tools()))
    assert file_ops.issubset(set(DataAnalysisAssistant.get_tools()))


def test_capability_configs_autoload_from_vertical_capabilities_modules() -> None:
    """Shared metadata defaults should resolve capability configs without assistant wrappers."""

    assert CodingAssistant.get_capability_configs() == get_coding_capability_configs()
    assert RAGAssistant.get_capability_configs() == get_rag_capability_configs()
    assert ResearchAssistant.get_capability_configs() == get_research_capability_configs()


def test_capability_provider_autoloads_for_verticals_using_default_loader() -> None:
    """Assistants should rely on the shared capability-provider loader when possible."""

    assert isinstance(DevOpsAssistant.get_capability_provider(), DevOpsCapabilityProvider)
    assert isinstance(
        DataAnalysisAssistant.get_capability_provider(),
        DataAnalysisCapabilityProvider,
    )


def test_rag_runtime_extensions_resolve_via_shared_loader_defaults() -> None:
    """RAG should use the inherited runtime extension loader for common optional modules."""

    assert isinstance(RAGAssistant.get_safety_extension(), RAGSafetyExtension)
    assert isinstance(RAGAssistant.get_prompt_contributor(), RAGPromptContributor)
    assert isinstance(RAGAssistant.get_rl_config_provider(), RAGRLConfig)
    assert isinstance(RAGAssistant.get_team_spec_provider(), RAGTeamSpecProvider)
