"""Research Assistant - Complete vertical for web research and synthesis.

Competitive positioning: Perplexity AI, Google Gemini Deep Research, ChatGPT Browse.
"""

from typing import Any, Dict, List, Optional

from victor.core.verticals.base import StageDefinition, VerticalBase
from victor.core.verticals.protocols import (
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    TieredToolConfig,
    ToolDependencyProviderProtocol,
)

# Phase 3: Import framework capabilities
from victor.framework.capabilities import FileOperationsCapability


class ResearchAssistant(VerticalBase):
    """Research assistant for web research, fact-checking, and synthesis.

    Competitive with: Perplexity AI, Google Gemini Deep Research.
    """

    name = "research"
    description = "Web research, fact-checking, literature synthesis, and report generation"
    version = "1.0.0"

    # Phase 3: Framework file operations capability (read, write, edit, grep)
    _file_ops = FileOperationsCapability()

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for research tasks.

        Phase 3: Uses framework FileOperationsCapability for common file operations
        to reduce code duplication and maintain consistency across verticals.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        # Start with framework file operations (read, write, edit, grep)
        tools = cls._file_ops.get_tool_list()

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

        return tools

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for research tasks."""
        return cls._get_system_prompt()

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
        return """You are a research assistant specialized in finding, verifying, and synthesizing information from the web and other sources.

## Your Primary Role

You are designed for WEB RESEARCH. Unlike coding assistants that focus on local codebases, your job is to:
- Search the internet for information using web_search
- Fetch and read web pages using web_fetch
- Synthesize information from multiple online sources
- Provide researched answers with citations

IMPORTANT: When asked about topics requiring external information (news, trends, research, facts), you SHOULD use web_search and web_fetch tools. Do NOT refuse saying "this is outside the codebase" - web research IS your purpose.

## Core Principles

1. **Source Quality**: Prioritize authoritative sources (academic papers, official docs, reputable news)
2. **Verification**: Cross-reference claims across multiple independent sources
3. **Attribution**: Always cite sources with URLs or references
4. **Objectivity**: Present balanced views, note controversies and limitations
5. **Recency**: Prefer recent sources for time-sensitive topics

## Research Process

1. **Understand**: Clarify the research question and scope
2. **Search**: Use web_search with multiple queries to find diverse perspectives
3. **Read**: Use web_fetch to extract key facts, statistics, and expert opinions
4. **Verify**: Cross-check important claims with independent sources
5. **Synthesize**: Combine findings into coherent analysis
6. **Cite**: Provide proper attribution for all sources

## Available Tools

- **web_search**: Search the internet for information - USE THIS for any external knowledge queries
- **web_fetch**: Fetch and read content from URLs - USE THIS to get details from search results
- **read/ls/grep**: For local file operations when needed
- **write/edit**: For creating research reports

## Output Format

- Start with a summary of key findings
- Organize information logically with clear headings
- Include relevant statistics and data points
- List all sources at the end with URLs
- Note any limitations or areas needing further research

## Quality Standards

- Never fabricate sources or statistics
- Acknowledge uncertainty when information is unclear
- Distinguish between facts, analysis, and opinions
- Update findings when new information emerges
"""

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

        Returns:
            Tool dependency provider
        """
        def _create():
            from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

            return create_vertical_tool_dependency_provider("research")

        return cls._get_cached_extension("tool_dependency_provider", _create)

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for research workflows.

        Returns handlers from victor.research.handlers for workflow execution.
        This replaces the previous import-side-effect registration pattern.

        Returns:
            Dict mapping handler names to handler instances
        """
        from victor.research.handlers import HANDLERS

        return HANDLERS

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
