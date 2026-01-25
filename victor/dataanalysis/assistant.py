"""Data Analysis Assistant - Complete vertical for data exploration and insights.

Competitive positioning: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.framework.prompt_builder import PromptBuilder

from victor.core.verticals.base import VerticalBase
from victor.core.verticals.protocols import ToolDependencyProviderProtocol
from victor.core.vertical_types import StageDefinition

# Phase 3: Import framework capabilities
from victor.framework.capabilities import FileOperationsCapability

# Import ISP-compliant provider protocols
from victor.core.verticals.protocols.providers import (
    HandlerProvider,
    ToolDependencyProvider,
    ToolProvider,
)

# Phase 2.1: Protocol auto-registration decorator
from victor.core.verticals.protocol_decorators import register_protocols


@register_protocols
class DataAnalysisAssistant(VerticalBase):
    """Data analysis assistant for exploration, visualization, and insights.

    Competitive with: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.

    ISP Compliance:
        This vertical explicitly declares which protocols it implements through
        protocol registration, rather than inheriting from all possible protocol
        interfaces. This follows the Interface Segregation Principle (ISP) by
        implementing only needed protocols.

        Implemented Protocols:
        - ToolProvider: Provides tools optimized for data analysis tasks
        - ToolDependencyProvider: Provides tool dependency patterns
        - HandlerProvider: Provides workflow compute handlers

        Note: Other protocols (PromptContributorProvider, ModeConfigProvider,
        SafetyProvider, TieredToolConfigProvider) are inherited from VerticalBase
        and do not require explicit registration.
    """

    name = "data_analysis"
    description = "Data exploration, statistical analysis, visualization, and ML insights"
    version = "0.5.0"

    @classmethod
    def get_tools(cls) -> list[str]:
        """Get the list of tools for data analysis tasks.

        Uses framework FileOperationsCapability via DI for common file operations
        to reduce code duplication and maintain consistency across verticals.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames
        from victor.core.verticals.capability_injector import get_capability_injector

        # Get file operations from shared capability injector (DI singleton)
        file_ops = get_capability_injector().get_file_operations_capability()
        tools = file_ops.get_tool_list()

        # Add data analysis-specific tools
        tools.extend(
            [
                # Directory listing for data file exploration
                ToolNames.LS,  # list_directory → ls
                # Python/Shell execution for analysis
                ToolNames.SHELL,  # bash → shell (for running Python scripts)
                # Code generation and search
                ToolNames.CODE_SEARCH,  # Semantic code search
                ToolNames.OVERVIEW,  # codebase_overview → overview
                ToolNames.GRAPH,  # Code graph analysis (PageRank, dependencies)
                # Web for datasets and documentation
                ToolNames.WEB_SEARCH,  # Web search (internet search)
                ToolNames.WEB_FETCH,  # Fetch URL content
            ]
        )

        return tools

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for data analysis tasks."""
        return cls._get_system_prompt()

    # =========================================================================
    # PromptBuilder Support
    # =========================================================================

    @classmethod
    def get_prompt_builder(cls) -> "PromptBuilder":
        """Get configured PromptBuilder for data analysis vertical.

        Returns:
            PromptBuilder with data analysis-specific configuration

        Note:
            Uses DataAnalysisPromptTemplate for consistent prompt structure
            following the Template Method pattern.
        """
        from victor.dataanalysis.dataanalysis_prompt_template import DataAnalysisPromptTemplate

        template = DataAnalysisPromptTemplate()
        return template.get_prompt_builder()

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get Data Analysis-specific stage definitions.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the data and analysis goals",
                tools={ToolNames.READ, ToolNames.LS, ToolNames.OVERVIEW},
                keywords=["what", "data", "analyze", "understand", "explore"],
                next_stages={"DATA_LOADING", "EXPLORATION"},
            ),
            "DATA_LOADING": StageDefinition(
                name="DATA_LOADING",
                description="Loading and validating data files",
                tools={ToolNames.READ, ToolNames.SHELL, ToolNames.WRITE},
                keywords=["load", "import", "read", "open", "fetch"],
                next_stages={"EXPLORATION", "CLEANING"},
            ),
            "EXPLORATION": StageDefinition(
                name="EXPLORATION",
                description="Exploratory data analysis and profiling",
                tools={ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE},
                keywords=["explore", "profile", "describe", "summary", "statistics"],
                next_stages={"CLEANING", "ANALYSIS"},
            ),
            "CLEANING": StageDefinition(
                name="CLEANING",
                description="Data cleaning and transformation",
                tools={ToolNames.SHELL, ToolNames.WRITE, ToolNames.EDIT},
                keywords=["clean", "transform", "fix", "handle", "remove"],
                next_stages={"ANALYSIS", "EXPLORATION"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Statistical analysis and modeling",
                tools={ToolNames.SHELL, ToolNames.WRITE, ToolNames.READ},
                keywords=["analyze", "model", "correlate", "test", "predict"],
                next_stages={"VISUALIZATION", "REPORTING"},
            ),
            "VISUALIZATION": StageDefinition(
                name="VISUALIZATION",
                description="Creating charts and visualizations",
                tools={ToolNames.SHELL, ToolNames.WRITE},
                keywords=["plot", "chart", "visualize", "graph", "figure"],
                next_stages={"REPORTING", "ANALYSIS"},
            ),
            "REPORTING": StageDefinition(
                name="REPORTING",
                description="Generating insights and reports",
                tools={ToolNames.WRITE, ToolNames.EDIT, ToolNames.READ},
                keywords=["report", "summarize", "document", "present"],
                next_stages={"COMPLETION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Finalizing analysis deliverables",
                tools={ToolNames.WRITE, ToolNames.READ},
                keywords=["done", "complete", "finish", "final"],
                next_stages=set(),
            ),
        }

    @classmethod
    def _get_system_prompt(cls) -> str:
        from victor.dataanalysis.dataanalysis_prompt_template import DataAnalysisPromptTemplate

        return DataAnalysisPromptTemplate().build()

    # =========================================================================
    # Extension Protocol Methods
    # =========================================================================
    # Most extension getters are auto-generated by VerticalExtensionLoaderMeta
    # to eliminate ~800 lines of duplication. Only override for custom logic.

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        """Get tool dependency provider using centralized factory.

        Phase 4: Migrated from deprecated DataAnalysisToolDependencyProvider
        wrapper to centralized create_vertical_tool_dependency_provider().

        Returns:
            Tool dependency provider
        """

        def _create() -> ToolDependencyProviderProtocol:
            from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

            return create_vertical_tool_dependency_provider(cls.name)  # type: ignore[no-any-return]

        return cls._get_cached_extension("tool_dependency_provider", _create)

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for DataAnalysis workflows.

        Returns handlers from victor.dataanalysis.handlers for workflow execution.
        This replaces the previous import-side-effect registration pattern.

        Returns:
            Dict mapping handler names to handler instances
        """
        from victor.framework.handler_registry import HandlerRegistry

        registry = HandlerRegistry.get_instance()

        # Auto-discover handlers if not already registered
        dataanalysis_handlers = registry.list_by_vertical("dataanalysis")
        if not dataanalysis_handlers:
            registry.discover_from_vertical("dataanalysis")

        handlers = {}
        for handler_name in registry.list_by_vertical("dataanalysis"):
            entry = registry.get_entry(handler_name)
            if entry:
                handlers[handler_name] = entry.handler
        return handlers

    # NOTE: The following getters are auto-generated by VerticalExtensionLoaderMeta:
    # - get_safety_extension()
    # - get_prompt_contributor()
    # - get_mode_config_provider()
    # - get_tiered_tools()
    # - get_workflow_provider()
    # - get_rl_config_provider()
    # - get_rl_hooks()
    # - get_team_spec_provider()
    # - get_capability_provider()
    #
    # get_extensions() is inherited from VerticalBase with full caching support.
    # To clear all caches, use cls.clear_config_cache().


# Protocol registration is now handled by @register_protocols decorator
# which auto-detects implemented protocols:
# - ToolProvider (get_tools)
# - ToolDependencyProvider (get_tool_dependency_provider)
# - HandlerProvider (get_handlers)
#
# ISP Compliance Note:
# This vertical implements only the protocols it needs. The @register_protocols
# decorator auto-detects and registers these protocols at class decoration time.
