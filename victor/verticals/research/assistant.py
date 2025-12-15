"""Research Assistant - Complete vertical for web research and synthesis.

Competitive positioning: Perplexity AI, Google Gemini Deep Research, ChatGPT Browse.
"""

from typing import Any, Dict, List, Optional

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.verticals.protocols import (
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    TieredToolConfig,
    ToolDependencyProviderProtocol,
)


class ResearchAssistant(VerticalBase):
    """Research assistant for web research, fact-checking, and synthesis.

    Competitive with: Perplexity AI, Google Gemini Deep Research.
    """

    name = "research"
    description = "Web research, fact-checking, literature synthesis, and report generation"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for research tasks.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        return [
            # Core research tools
            ToolNames.WEB,  # web_search → web
            ToolNames.FETCH,  # web_fetch → fetch
            # File operations for reading/writing reports
            ToolNames.READ,  # read_file → read
            ToolNames.WRITE,  # write_file → write
            ToolNames.EDIT,  # edit_files → edit
            ToolNames.LS,  # list_directory → ls
            # Code search for technical research
            ToolNames.GREP,  # code_search → grep
            ToolNames.SEARCH,  # semantic_code_search → search
            ToolNames.OVERVIEW,  # codebase_overview → overview
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for research tasks."""
        return cls._get_system_prompt()

    @classmethod
    def get_config(cls) -> VerticalConfig:
        from victor.framework.tools import ToolSet
        return VerticalConfig(
            tools=ToolSet.from_tools(cls.get_tools()),
            system_prompt=cls._get_system_prompt(),
            stages=cls.get_stages(),
            provider_hints={
                "preferred_providers": ["anthropic", "openai", "google"],
                "min_context_window": 100000,
                "features": ["web_search", "large_context"],
            },
            evaluation_criteria=[
                "accuracy",
                "source_quality",
                "comprehensiveness",
                "clarity",
                "attribution",
                "objectivity",
                "timeliness",
            ],
            metadata={
                "vertical_name": cls.name,
                "vertical_description": cls.description,
            },
        )

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
                tools={ToolNames.WEB, ToolNames.READ, ToolNames.LS},
                keywords=["research", "find", "search", "look up"],
                next_stages={"SEARCHING"},
            ),
            "SEARCHING": StageDefinition(
                name="SEARCHING",
                description="Gathering sources and information",
                tools={ToolNames.WEB, ToolNames.FETCH, ToolNames.GREP},
                keywords=["search", "find", "gather", "discover"],
                next_stages={"READING", "SEARCHING"},
            ),
            "READING": StageDefinition(
                name="READING",
                description="Deep reading and extraction from sources",
                tools={ToolNames.FETCH, ToolNames.READ, ToolNames.SEARCH},
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
                tools={ToolNames.WEB, ToolNames.FETCH},
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
        return """You are a research assistant specialized in finding, verifying, and synthesizing information.

## Core Principles

1. **Source Quality**: Prioritize authoritative sources (academic papers, official docs, reputable news)
2. **Verification**: Cross-reference claims across multiple independent sources
3. **Attribution**: Always cite sources with URLs or references
4. **Objectivity**: Present balanced views, note controversies and limitations
5. **Recency**: Prefer recent sources for time-sensitive topics

## Research Process

1. **Understand**: Clarify the research question and scope
2. **Search**: Use multiple search queries to find diverse perspectives
3. **Read**: Extract key facts, statistics, and expert opinions
4. **Verify**: Cross-check important claims with independent sources
5. **Synthesize**: Combine findings into coherent analysis
6. **Cite**: Provide proper attribution for all sources

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

    @classmethod
    def get_prompt_contributor(cls) -> Optional[PromptContributorProtocol]:
        from victor.verticals.research.prompts import ResearchPromptContributor
        return ResearchPromptContributor()

    @classmethod
    def get_mode_config_provider(cls) -> Optional[ModeConfigProviderProtocol]:
        from victor.verticals.research.mode_config import ResearchModeConfigProvider
        return ResearchModeConfigProvider()

    @classmethod
    def get_safety_extension(cls) -> Optional[SafetyExtensionProtocol]:
        from victor.verticals.research.safety import ResearchSafetyExtension
        return ResearchSafetyExtension()

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        from victor.verticals.research.tool_dependencies import ResearchToolDependencyProvider
        return ResearchToolDependencyProvider()

    @classmethod
    def get_tiered_tools(cls) -> Optional[TieredToolConfig]:
        """Get tiered tool configuration for research.

        Simplified configuration using consolidated tool metadata:
        - Mandatory: Core tools always included for any task
        - Vertical Core: Essential tools for research tasks
        - semantic_pool: Derived from ToolMetadataRegistry.get_all_tool_names()
        - stage_tools: Derived from @tool(stages=[...]) decorator metadata

        Returns:
            TieredToolConfig for research vertical
        """
        from victor.tools.tool_names import ToolNames

        return TieredToolConfig(
            # Tier 1: Mandatory - always included for any task
            mandatory={
                ToolNames.READ,  # Read files - essential
                ToolNames.LS,    # List directory - essential
                ToolNames.GREP,  # Search - essential for research
            },
            # Tier 2: Vertical Core - essential for research tasks
            vertical_core={
                ToolNames.WEB,       # Web search is core to research
                ToolNames.FETCH,     # Fetching content is core to research
                ToolNames.OVERVIEW,  # Codebase overview - core for understanding
            },
            # semantic_pool and stage_tools are now derived from @tool decorator metadata
            # Use get_effective_semantic_pool() and get_tools_for_stage_from_registry()
            # Research can benefit from readonly_only_for_analysis since it's primarily reading
            readonly_only_for_analysis=True,
        )
