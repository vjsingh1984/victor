"""Regression coverage for shared runtime-helper defaults on vertical assistants."""

from victor_sdk import ToolNames

from victor.verticals.contrib.coding.assistant import CodingAssistant
from victor.verticals.contrib.coding.capabilities import (
    get_capability_configs as get_coding_capability_configs,
)
from victor.verticals.contrib.coding.middleware import (
    CodeCorrectionMiddleware,
    GitSafetyMiddleware,
)
from victor.verticals.contrib.coding.service_provider import CodingServiceProvider
from victor.verticals.contrib.coding.composed_chains import CODING_CHAINS
from victor.verticals.contrib.coding.teams import CODING_PERSONAS
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


def test_coding_runtime_extensions_resolve_via_shared_loader_defaults() -> None:
    """Coding should inherit middleware and service-provider runtime hooks."""

    assert "get_middleware" not in CodingAssistant.__dict__
    assert "get_service_provider" not in CodingAssistant.__dict__
    assert "get_composed_chains" not in CodingAssistant.__dict__
    assert "get_personas" not in CodingAssistant.__dict__

    middleware = CodingAssistant.get_middleware()
    assert len(middleware) == 2
    assert isinstance(middleware[0], CodeCorrectionMiddleware)
    assert isinstance(middleware[1], GitSafetyMiddleware)
    assert isinstance(CodingAssistant.get_service_provider(), CodingServiceProvider)
    assert CodingAssistant.get_composed_chains() == CODING_CHAINS
    assert CodingAssistant.get_personas() == CODING_PERSONAS


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
