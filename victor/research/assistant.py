"""Research Assistant - Complete vertical for web research and synthesis.

Competitive positioning: Perplexity AI, Google Gemini Deep Research, ChatGPT Browse.

This vertical demonstrates ISP-compliant protocol registration, where only the
protocols actually implemented by the vertical are registered, rather than
inheriting from all possible protocol interfaces.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.framework.prompt_builder import PromptBuilder

from victor.core.verticals.base import VerticalBase
from victor.core.vertical_types import StageDefinition
from victor.core.verticals.protocols import (
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    TieredToolConfig,
    ToolDependencyProviderProtocol,
)

# Phase 3: Import framework capabilities
from victor.framework.capabilities import FileOperationsCapability

# Import ISP-compliant provider protocols
from victor.core.verticals.protocols.providers import (
    HandlerProvider,
    PromptContributorProvider,
    ToolDependencyProvider,
    ToolProvider,
)

# Phase 2.1: Protocol auto-registration decorator
from victor.core.verticals.protocol_decorators import register_protocols


@register_protocols
class ResearchAssistant(VerticalBase):
    """Research assistant for web research, fact-checking, and synthesis.

    Competitive with: Perplexity AI, Google Gemini Deep Research.

    ISP Compliance:
        This vertical explicitly declares which protocols it implements through
        protocol registration, rather than inheriting from all possible protocol
        interfaces. This follows the Interface Segregation Principle (ISP) by
        implementing only needed protocols.

        Implemented Protocols:
        - ToolProvider: Provides tool list for research tasks
        - PromptContributorProvider: Provides research-specific task hints
        - ToolDependencyProvider: Provides tool dependency patterns
        - HandlerProvider: Provides workflow compute handlers
    """

    name = "research"
    description = "Web research, fact-checking, literature synthesis, and report generation"
    version = "0.5.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for research tasks.

        Uses framework FileOperationsCapability via DI for common file operations
        to reduce code duplication and maintain consistency across verticals.

        Uses canonical tool names from victor.tools.tool_names.

        This method is part of the ToolProvider protocol.
        """
        from victor.tools.tool_names import ToolNames
        from victor.core.verticals.capability_injector import get_capability_injector

        # Get file operations from shared capability injector (DI singleton)
        file_ops = get_capability_injector().get_file_operations_capability()
        tools = file_ops.get_tool_list()

        # Add research-specific tools
        tools.extend(
            [
                # Core research tools
                ToolNames.WEB_SEARCH,  # Web search (internet search)
                ToolNames.WEB_FETCH,  # Fetch URL content
                # Directory listing for file exploration
                ToolNames.LS,  # list_directory → ls
                # Code search for technical research
                ToolNames.CODE_SEARCH,  # Semantic code search
                ToolNames.OVERVIEW,  # codebase_overview → overview
            ]
        )

        return list(tools)

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for research tasks."""
        return cls._get_system_prompt()

    # =========================================================================
    # PromptBuilder Support (Phase 7)
    # =========================================================================

    @classmethod
    def _get_vertical_prompt(cls) -> str:
        """Get research-specific prompt content for PromptBuilder.

        Returns:
            Research-specific vertical prompt content
        """
        from victor.research.research_prompt_template import ResearchPromptTemplate

        return ResearchPromptTemplate().get_vertical_prompt()

    @classmethod
    def get_prompt_builder(cls) -> "PromptBuilder":
        """Get configured PromptBuilder for research vertical.

        Returns:
            PromptBuilder with research-specific configuration

        Note:
            Uses ResearchPromptTemplate for consistent prompt structure
            following the Template Method pattern.
        """
        from victor.research.research_prompt_template import ResearchPromptTemplate

        # Use template for consistent structure
        template = ResearchPromptTemplate()
        builder = template.get_prompt_builder()

        return builder

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get research-specific stage definitions.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the research question",
                tools={ToolNames.WEB_SEARCH, ToolNames.READ, ToolNames.LS},
                keywords=["research", "find", "search", "look up"],
                next_stages={"SEARCHING"},
            ),
            "SEARCHING": StageDefinition(
                name="SEARCHING",
                description="Gathering sources and information",
                tools={ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.GREP},
                keywords=["search", "find", "gather", "discover"],
                next_stages={"READING", "SEARCHING"},
            ),
            "READING": StageDefinition(
                name="READING",
                description="Deep reading and extraction from sources",
                tools={ToolNames.WEB_FETCH, ToolNames.READ, ToolNames.CODE_SEARCH},
                keywords=["read", "extract", "analyze", "understand"],
                next_stages={"SYNTHESIZING", "SEARCHING"},
            ),
            "SYNTHESIZING": StageDefinition(
                name="SYNTHESIZING",
                description="Combining and analyzing information",
                tools={ToolNames.READ, ToolNames.OVERVIEW},
                keywords=["combine", "synthesize", "integrate", "compare"],
                next_stages={"WRITING", "READING"},
            ),
            "WRITING": StageDefinition(
                name="WRITING",
                description="Producing the research output",
                tools={ToolNames.WRITE, ToolNames.EDIT},
                keywords=["write", "document", "report", "summarize"],
                next_stages={"VERIFICATION", "SYNTHESIZING"},
            ),
            "VERIFICATION": StageDefinition(
                name="VERIFICATION",
                description="Fact-checking and source verification",
                tools={ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH},
                keywords=["verify", "check", "confirm", "validate"],
                next_stages={"COMPLETION", "WRITING"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Research complete with citations",
                tools=set(),
                keywords=["done", "complete", "finished"],
                next_stages=set(),
            ),
        }

    @classmethod
    def _get_system_prompt(cls) -> str:
        from victor.research.research_prompt_template import ResearchPromptTemplate

        return ResearchPromptTemplate().build()

    # =========================================================================
    # Extension Protocol Methods
    # =========================================================================
    # Most extension getters are auto-generated by VerticalExtensionLoaderMeta
    # to eliminate ~800 lines of duplication. Only override for custom logic.

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        """Get Research tool dependency provider (cached).

        Custom implementation using create_vertical_tool_dependency_provider.
        Auto-generated getter would try to import from victor.research.tool_dependencies.

        This method is part of the ToolDependencyProvider protocol.

        Returns:
            Tool dependency provider
        """

        from typing import cast

        def _create() -> ToolDependencyProviderProtocol:
            from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
            from victor.core.verticals.protocols import ToolDependencyProviderProtocol

            return cast(
                ToolDependencyProviderProtocol, create_vertical_tool_dependency_provider("research")
            )

        return cast(
            Optional[ToolDependencyProviderProtocol],
            cls._get_cached_extension("tool_dependency_provider", _create),
        )

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for research workflows.

        Returns handlers from victor.research.handlers for workflow execution.
        This replaces the previous import-side-effect registration pattern.

        This method is part of the HandlerProvider protocol.

        Returns:
            Dict mapping handler names to handler instances
        """
        from victor.framework.handler_registry import HandlerRegistry

        registry = HandlerRegistry.get_instance()

        # Auto-discover handlers if not already registered
        research_handlers = registry.list_by_vertical("research")
        if not research_handlers:
            registry.discover_from_vertical("research")

        handlers = {}
        for handler_name in registry.list_by_vertical("research"):
            entry = registry.get_entry(handler_name)
            if entry:
                handlers[handler_name] = entry.handler
        return handlers

    @classmethod
    def get_capability_configs(cls) -> Dict[str, Any]:
        """Get research capability configurations for centralized storage.

        Returns default research configuration for VerticalContext storage.
        This replaces direct orchestrator attribute assignments for research configs.

        Returns:
            Dict with default research capability configurations
        """
        from victor.research.capabilities import get_capability_configs

        return get_capability_configs()

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
# - PromptContributorProvider (get_prompt_contributor)
# - ToolDependencyProvider (get_tool_dependency_provider)
# - HandlerProvider (get_handlers)
#
# ISP Compliance Note:
# This vertical implements only the protocols it needs. The @register_protocols
# decorator auto-detects and registers these protocols at class decoration time.
