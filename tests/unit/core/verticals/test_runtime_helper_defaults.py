"""Regression coverage for shared runtime-helper defaults on vertical assistants."""

from victor_sdk import ToolNames

from victor.verticals.contrib.coding import CodingAssistant
from victor.verticals.contrib.coding.assistant import (
    CodingAssistant as CodingAssistantDefinition,
)
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
from victor.verticals.contrib.dataanalysis import DataAnalysisAssistant
from victor.verticals.contrib.dataanalysis.capabilities import (
    DataAnalysisCapabilityProvider,
)
from victor.verticals.contrib.dataanalysis.runtime.capabilities import (
    DataAnalysisCapabilityProvider as RuntimeDataAnalysisCapabilityProvider,
)
from victor.verticals.contrib.dataanalysis.mode_config import DataAnalysisModeConfigProvider
from victor.verticals.contrib.dataanalysis.rl import DataAnalysisRLConfig
from victor.verticals.contrib.dataanalysis.runtime.rl import (
    DataAnalysisRLConfig as RuntimeDataAnalysisRLConfig,
)
from victor.verticals.contrib.dataanalysis.runtime.teams import (
    DataAnalysisTeamSpecProvider as RuntimeDataAnalysisTeamSpecProvider,
)
from victor.verticals.contrib.dataanalysis.runtime.mode_config import (
    DataAnalysisModeConfigProvider as RuntimeDataAnalysisModeConfigProvider,
)
from victor.verticals.contrib.dataanalysis.runtime.safety import (
    DataAnalysisSafetyExtension as RuntimeDataAnalysisSafetyExtension,
)
from victor.verticals.contrib.dataanalysis.runtime.tool_dependencies import (
    get_provider as get_runtime_dataanalysis_tool_dependency_provider,
)
from victor.verticals.contrib.dataanalysis.runtime.workflows import (
    DataAnalysisWorkflowProvider as RuntimeDataAnalysisWorkflowProvider,
)
from victor.verticals.contrib.dataanalysis.runtime.safety_enhanced import (
    DataAnalysisSafetyRules as RuntimeDataAnalysisSafetyRules,
    EnhancedDataAnalysisSafetyExtension as RuntimeEnhancedDataAnalysisSafetyExtension,
)
from victor.verticals.contrib.dataanalysis.safety import DataAnalysisSafetyExtension
from victor.verticals.contrib.dataanalysis.safety_enhanced import (
    DataAnalysisSafetyRules,
    EnhancedDataAnalysisSafetyExtension,
)
from victor.verticals.contrib.dataanalysis.tool_dependencies import (
    get_provider as get_dataanalysis_tool_dependency_provider,
)
from victor.verticals.contrib.dataanalysis.teams import DataAnalysisTeamSpecProvider
from victor.verticals.contrib.dataanalysis.workflows import DataAnalysisWorkflowProvider
from victor.verticals.contrib.devops import DevOpsAssistant
from victor.verticals.contrib.devops.capabilities import DevOpsCapabilityProvider
from victor.verticals.contrib.devops.prompts import DevOpsPromptContributor
from victor.framework.middleware import (
    GitSafetyMiddleware as FrameworkGitSafetyMiddleware,
    LoggingMiddleware,
    SecretMaskingMiddleware,
)
from victor.verticals.contrib.rag import RAGAssistant
from victor.verticals.contrib.rag.capabilities import (
    RAGCapabilityProvider,
    get_capability_configs as get_rag_capability_configs,
)
from victor.verticals.contrib.rag.enrichment import (
    RAGEnrichmentConfig,
    RAGEnrichmentStrategy,
    get_rag_enrichment_strategy,
    reset_rag_enrichment_strategy,
)
from victor.verticals.contrib.rag.mode_config import RAGModeConfigProvider
from victor.verticals.contrib.rag.prompts import RAGPromptContributor
from victor.verticals.contrib.rag.rl import RAGRLConfig
from victor.verticals.contrib.rag.runtime.rl import RAGRLConfig as RuntimeRAGRLConfig
from victor.verticals.contrib.rag.safety import RAGSafetyExtension
from victor.verticals.contrib.rag.safety import (
    create_all_rag_safety_rules as create_root_rag_safety_rules,
)
from victor.verticals.contrib.rag.safety_enhanced import (
    EnhancedRAGSafetyExtension,
    RAGSafetyRules,
)
from victor.verticals.contrib.rag.runtime.safety import (
    RAGSafetyExtension as RuntimeRAGSafetyExtension,
    create_all_rag_safety_rules as create_runtime_rag_safety_rules,
)
from victor.verticals.contrib.rag.runtime.mode_config import (
    RAGModeConfigProvider as RuntimeRAGModeConfigProvider,
)
from victor.verticals.contrib.rag.runtime.capabilities import (
    RAGCapabilityProvider as RuntimeRAGCapabilityProvider,
    get_capability_configs as get_runtime_rag_capability_configs,
)
from victor.verticals.contrib.rag.runtime.enrichment import (
    RAGEnrichmentStrategy as RuntimeRAGEnrichmentStrategy,
    get_rag_enrichment_strategy as get_runtime_rag_enrichment_strategy,
    reset_rag_enrichment_strategy as reset_runtime_rag_enrichment_strategy,
)
from victor.verticals.contrib.rag.runtime.safety_enhanced import (
    EnhancedRAGSafetyExtension as RuntimeEnhancedRAGSafetyExtension,
    RAGSafetyRules as RuntimeRAGSafetyRules,
)
from victor.verticals.contrib.rag.teams import RAGTeamSpecProvider
from victor.verticals.contrib.rag.runtime.teams import (
    RAGTeamSpecProvider as RuntimeRAGTeamSpecProvider,
)
from victor.verticals.contrib.rag.runtime.workflows import (
    RAGWorkflowProvider as RuntimeRAGWorkflowProvider,
)
from victor.verticals.contrib.rag.workflows import RAGWorkflowProvider
from victor.verticals.contrib.research import ResearchAssistant
from victor.verticals.contrib.research.capabilities import (
    ResearchCapabilityProvider,
    get_capability_configs as get_research_capability_configs,
)
from victor.verticals.contrib.research.mode_config import ResearchModeConfigProvider
from victor.verticals.contrib.research.rl import ResearchRLConfig
from victor.verticals.contrib.research.runtime.capabilities import (
    ResearchCapabilityProvider as RuntimeResearchCapabilityProvider,
)
from victor.verticals.contrib.research.runtime.mode_config import (
    ResearchModeConfigProvider as RuntimeResearchModeConfigProvider,
)
from victor.verticals.contrib.research.runtime.team_personas import (
    list_personas as list_runtime_research_personas,
)
from victor.verticals.contrib.research.runtime.rl import (
    ResearchRLConfig as RuntimeResearchRLConfig,
)
from victor.verticals.contrib.research.runtime.safety import (
    ResearchSafetyExtension as RuntimeResearchSafetyExtension,
)
from victor.verticals.contrib.research.runtime.teams import (
    ResearchTeamSpecProvider as RuntimeResearchTeamSpecProvider,
)
from victor.verticals.contrib.research.runtime.tool_dependencies import (
    get_provider as get_runtime_research_tool_dependency_provider,
)
from victor.verticals.contrib.research.runtime.workflows import (
    ResearchWorkflowProvider as RuntimeResearchWorkflowProvider,
)
from victor.verticals.contrib.research.safety import ResearchSafetyExtension
from victor.verticals.contrib.research.teams import ResearchTeamSpecProvider
from victor.verticals.contrib.research.teams.personas import (
    list_personas as list_root_research_personas,
)
from victor.verticals.contrib.research.tool_dependencies import (
    get_provider as get_research_tool_dependency_provider,
)
from victor.verticals.contrib.research.workflows import ResearchWorkflowProvider


def _is_instance_by_name(obj, cls) -> bool:
    """Check isinstance by class name, tolerating cross-package identity splits.

    When contrib shims and external packages define the same class, Python
    treats them as distinct types. This helper matches by name in the MRO
    so tests pass regardless of which module loaded the class.
    """
    target_name = cls.__name__
    return any(c.__name__ == target_name for c in type(obj).__mro__)


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

    assert "get_middleware" not in CodingAssistantDefinition.__dict__
    assert "get_service_provider" not in CodingAssistantDefinition.__dict__
    assert "get_composed_chains" not in CodingAssistantDefinition.__dict__
    assert "get_personas" not in CodingAssistantDefinition.__dict__

    middleware = CodingAssistant.get_middleware()
    assert len(middleware) == 2
    assert isinstance(middleware[0], CodeCorrectionMiddleware)
    assert isinstance(middleware[1], GitSafetyMiddleware)
    assert _is_instance_by_name(CodingAssistant.get_service_provider(), CodingServiceProvider)
    assert set(CodingAssistant.get_composed_chains().keys()) == set(CODING_CHAINS.keys())
    assert set(CodingAssistant.get_personas().keys()) == set(CODING_PERSONAS.keys())


def test_capability_provider_autoloads_for_verticals_using_default_loader() -> None:
    """Assistants should rely on the shared capability-provider loader when possible."""

    assert _is_instance_by_name(DevOpsAssistant.get_capability_provider(), DevOpsCapabilityProvider)
    assert _is_instance_by_name(
        DataAnalysisAssistant.get_capability_provider(),
        DataAnalysisCapabilityProvider,
    )


def test_dataanalysis_root_runtime_shims_delegate_to_runtime_modules() -> None:
    """Data Analysis root runtime helpers should re-export runtime-owned modules."""

    root_tool_dependency_provider = get_dataanalysis_tool_dependency_provider()
    runtime_tool_dependency_provider = get_runtime_dataanalysis_tool_dependency_provider()

    assert DataAnalysisCapabilityProvider is RuntimeDataAnalysisCapabilityProvider
    assert DataAnalysisModeConfigProvider is RuntimeDataAnalysisModeConfigProvider
    assert DataAnalysisRLConfig is RuntimeDataAnalysisRLConfig
    assert DataAnalysisSafetyExtension is RuntimeDataAnalysisSafetyExtension
    assert DataAnalysisSafetyRules is RuntimeDataAnalysisSafetyRules
    assert EnhancedDataAnalysisSafetyExtension is RuntimeEnhancedDataAnalysisSafetyExtension
    assert DataAnalysisTeamSpecProvider is RuntimeDataAnalysisTeamSpecProvider
    assert DataAnalysisWorkflowProvider is RuntimeDataAnalysisWorkflowProvider
    assert (
        get_dataanalysis_tool_dependency_provider
        is get_runtime_dataanalysis_tool_dependency_provider
    )
    assert type(root_tool_dependency_provider) is type(runtime_tool_dependency_provider)
    assert root_tool_dependency_provider.yaml_path == runtime_tool_dependency_provider.yaml_path
    assert _is_instance_by_name(
        DataAnalysisAssistant.get_capability_provider(),
        RuntimeDataAnalysisCapabilityProvider,
    )
    assert _is_instance_by_name(
        DataAnalysisAssistant.get_mode_config_provider(),
        RuntimeDataAnalysisModeConfigProvider,
    )
    assert _is_instance_by_name(
        DataAnalysisAssistant.get_rl_config_provider(),
        RuntimeDataAnalysisRLConfig,
    )
    assert _is_instance_by_name(
        DataAnalysisAssistant.get_safety_extension(),
        RuntimeDataAnalysisSafetyExtension,
    )
    assert _is_instance_by_name(
        DataAnalysisAssistant.get_workflow_provider(),
        RuntimeDataAnalysisWorkflowProvider,
    )
    assert _is_instance_by_name(
        DataAnalysisAssistant.get_team_spec_provider(),
        RuntimeDataAnalysisTeamSpecProvider,
    )
    assert _is_instance_by_name(
        DataAnalysisAssistant.get_tool_dependency_provider(),
        type(runtime_tool_dependency_provider),
    )
    assert (
        DataAnalysisAssistant.get_tool_dependency_provider().yaml_path.name
        == runtime_tool_dependency_provider.yaml_path.name
    )


def test_research_root_runtime_shims_delegate_to_runtime_modules() -> None:
    """Research root runtime helpers should re-export runtime-owned modules."""

    root_tool_dependency_provider = get_research_tool_dependency_provider()
    runtime_tool_dependency_provider = get_runtime_research_tool_dependency_provider()

    assert ResearchCapabilityProvider is RuntimeResearchCapabilityProvider
    assert ResearchModeConfigProvider is RuntimeResearchModeConfigProvider
    assert ResearchRLConfig is RuntimeResearchRLConfig
    assert ResearchSafetyExtension is RuntimeResearchSafetyExtension
    assert ResearchTeamSpecProvider is RuntimeResearchTeamSpecProvider
    assert ResearchWorkflowProvider is RuntimeResearchWorkflowProvider
    assert list_root_research_personas is list_runtime_research_personas
    assert get_research_tool_dependency_provider is get_runtime_research_tool_dependency_provider
    assert type(root_tool_dependency_provider) is type(runtime_tool_dependency_provider)
    assert root_tool_dependency_provider.yaml_path == runtime_tool_dependency_provider.yaml_path
    assert _is_instance_by_name(
        ResearchAssistant.get_capability_provider(),
        RuntimeResearchCapabilityProvider,
    )
    assert _is_instance_by_name(
        ResearchAssistant.get_mode_config_provider(),
        RuntimeResearchModeConfigProvider,
    )
    assert _is_instance_by_name(
        ResearchAssistant.get_rl_config_provider(),
        RuntimeResearchRLConfig,
    )
    assert _is_instance_by_name(
        ResearchAssistant.get_safety_extension(),
        RuntimeResearchSafetyExtension,
    )
    assert _is_instance_by_name(
        ResearchAssistant.get_workflow_provider(),
        RuntimeResearchWorkflowProvider,
    )
    assert _is_instance_by_name(
        ResearchAssistant.get_team_spec_provider(),
        RuntimeResearchTeamSpecProvider,
    )
    assert _is_instance_by_name(
        ResearchAssistant.get_tool_dependency_provider(),
        type(runtime_tool_dependency_provider),
    )
    assert (
        ResearchAssistant.get_tool_dependency_provider().yaml_path.name
        == runtime_tool_dependency_provider.yaml_path.name
    )


def test_devops_runtime_extensions_resolve_via_shared_loader_defaults() -> None:
    """DevOps should inherit middleware and prompt runtime hooks from shared defaults."""

    assert "get_middleware" not in DevOpsAssistant.__victor_sdk_source__.__dict__

    middleware = DevOpsAssistant.get_middleware()

    assert len(middleware) == 3
    assert isinstance(middleware[0], FrameworkGitSafetyMiddleware)
    assert isinstance(middleware[1], SecretMaskingMiddleware)
    assert isinstance(middleware[2], LoggingMiddleware)
    assert _is_instance_by_name(DevOpsAssistant.get_prompt_contributor(), DevOpsPromptContributor)


def test_rag_runtime_extensions_resolve_via_shared_loader_defaults() -> None:
    """RAG should use the inherited runtime extension loader for common optional modules."""

    assert _is_instance_by_name(RAGAssistant.get_safety_extension(), RAGSafetyExtension)
    assert _is_instance_by_name(RAGAssistant.get_safety_extension(), RuntimeRAGSafetyExtension)
    assert _is_instance_by_name(RAGAssistant.get_prompt_contributor(), RAGPromptContributor)
    assert _is_instance_by_name(RAGAssistant.get_rl_config_provider(), RuntimeRAGRLConfig)
    assert _is_instance_by_name(RAGAssistant.get_team_spec_provider(), RuntimeRAGTeamSpecProvider)
    assert _is_instance_by_name(RAGAssistant.get_workflow_provider(), RuntimeRAGWorkflowProvider)


def test_rag_mode_config_and_enhanced_safety_root_shims_delegate_to_runtime_modules() -> None:
    """Root compatibility shims should re-export runtime-owned RAG helpers."""

    assert RAGModeConfigProvider is RuntimeRAGModeConfigProvider
    assert RAGCapabilityProvider is RuntimeRAGCapabilityProvider
    assert RAGRLConfig is RuntimeRAGRLConfig
    assert RAGSafetyExtension is RuntimeRAGSafetyExtension
    assert EnhancedRAGSafetyExtension is RuntimeEnhancedRAGSafetyExtension
    assert RAGSafetyRules is RuntimeRAGSafetyRules
    assert RAGTeamSpecProvider is RuntimeRAGTeamSpecProvider
    assert RAGWorkflowProvider is RuntimeRAGWorkflowProvider
    assert create_root_rag_safety_rules is create_runtime_rag_safety_rules
    assert get_rag_capability_configs is get_runtime_rag_capability_configs
    assert _is_instance_by_name(RAGAssistant.get_capability_provider(), RuntimeRAGCapabilityProvider)
    assert _is_instance_by_name(RAGAssistant.get_mode_config_provider(), RuntimeRAGModeConfigProvider)


def test_rag_enrichment_root_shim_delegates_to_runtime_module() -> None:
    """Root enrichment imports should delegate to the runtime-owned module."""

    assert RAGEnrichmentStrategy is RuntimeRAGEnrichmentStrategy
    assert get_rag_enrichment_strategy is get_runtime_rag_enrichment_strategy
    assert reset_rag_enrichment_strategy is reset_runtime_rag_enrichment_strategy

    reset_rag_enrichment_strategy()
    strategy = get_rag_enrichment_strategy(
        config=RAGEnrichmentConfig(use_llm_enhancement=False),
    )
    assert isinstance(strategy, RuntimeRAGEnrichmentStrategy)
    assert get_runtime_rag_enrichment_strategy() is strategy
    reset_runtime_rag_enrichment_strategy()
