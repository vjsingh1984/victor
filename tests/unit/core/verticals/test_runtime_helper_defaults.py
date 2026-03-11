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


def test_devops_runtime_extensions_resolve_via_shared_loader_defaults() -> None:
    """DevOps should inherit middleware and prompt runtime hooks from shared defaults."""

    assert "get_middleware" not in DevOpsAssistant.__victor_sdk_source__.__dict__

    middleware = DevOpsAssistant.get_middleware()

    assert len(middleware) == 3
    assert isinstance(middleware[0], FrameworkGitSafetyMiddleware)
    assert isinstance(middleware[1], SecretMaskingMiddleware)
    assert isinstance(middleware[2], LoggingMiddleware)
    assert isinstance(DevOpsAssistant.get_prompt_contributor(), DevOpsPromptContributor)


def test_rag_runtime_extensions_resolve_via_shared_loader_defaults() -> None:
    """RAG should use the inherited runtime extension loader for common optional modules."""

    assert isinstance(RAGAssistant.get_safety_extension(), RAGSafetyExtension)
    assert isinstance(RAGAssistant.get_safety_extension(), RuntimeRAGSafetyExtension)
    assert isinstance(RAGAssistant.get_prompt_contributor(), RAGPromptContributor)
    assert isinstance(RAGAssistant.get_rl_config_provider(), RuntimeRAGRLConfig)
    assert isinstance(RAGAssistant.get_team_spec_provider(), RuntimeRAGTeamSpecProvider)
    assert isinstance(RAGAssistant.get_workflow_provider(), RuntimeRAGWorkflowProvider)


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
    assert isinstance(RAGAssistant.get_capability_provider(), RuntimeRAGCapabilityProvider)
    assert isinstance(RAGAssistant.get_mode_config_provider(), RuntimeRAGModeConfigProvider)


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
