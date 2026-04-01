"""Research Assistant - Complete vertical for web research and synthesis.

Competitive positioning: Perplexity AI, Google Gemini Deep Research, ChatGPT Browse.
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor_sdk import (
    CapabilityIds,
    CapabilityRequirement,
    ExtensionType,
    PromptMetadata,
    StageDefinition,
    ToolNames,
    VerticalBase,
)

from victor.core.verticals.registration import register_vertical
from victor.verticals.contrib.research.prompt_metadata import (
    RESEARCH_GROUNDING_RULES,
    RESEARCH_PROMPT_PRIORITY,
    RESEARCH_PROMPT_TEMPLATES,
    RESEARCH_SYSTEM_PROMPT_SECTION,
    RESEARCH_TASK_TYPE_HINTS,
)


@register_vertical(
    name="research",
    version="1.0.0",
    api_version=1,
    min_framework_version=">=0.5.0",
    provides={ExtensionType.TOOLS, ExtensionType.WORKFLOWS},
    canonicalize_tool_names=False,  # Preserves original tool names
    tool_dependency_strategy="entry_point",
    strict_mode=False,
    load_priority=50,
)
class ResearchAssistant(VerticalBase):
    """Research assistant for web research, fact-checking, and synthesis.

    Competitive with: Perplexity AI, Google Gemini Deep Research.
    """

    name = "research"
    description = "Web research, fact-checking, literature synthesis, and report generation"
    version = "1.0.0"

    @classmethod
    def get_name(cls) -> str:
        """Return the stable identifier for this vertical."""

        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """Return the human-readable vertical description."""

        return cls.description

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for research tasks.

        Uses SDK-owned canonical tool identifiers, including the shared file-operation
        tool group.
        """
        tools = list(ToolNames.file_operations())

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
    def get_prompt_templates(cls) -> Dict[str, str]:
        """Return serializable prompt templates for the research definition."""

        return dict(RESEARCH_PROMPT_TEMPLATES)

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Dict[str, Any]]:
        """Return serializable task-type hints for the research definition."""

        return {task_type: dict(config) for task_type, config in RESEARCH_TASK_TYPE_HINTS.items()}

    @classmethod
    def get_prompt_metadata(cls) -> PromptMetadata:
        """Return full prompt metadata, including runtime adapter hints."""

        metadata = super().get_prompt_metadata()
        return PromptMetadata(
            templates=metadata.templates,
            task_type_hints=metadata.task_type_hints,
            metadata={
                "system_prompt_section": RESEARCH_SYSTEM_PROMPT_SECTION,
                "grounding_rules": RESEARCH_GROUNDING_RULES,
                "priority": RESEARCH_PROMPT_PRIORITY,
            },
        )

    @classmethod
    def get_capability_requirements(cls) -> List[CapabilityRequirement]:
        """Declare runtime capabilities required by the research definition."""

        return [
            CapabilityRequirement(
                capability_id=CapabilityIds.FILE_OPS,
                purpose="Inspect local notes, saved sources, and generated research artifacts.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.WEB_ACCESS,
                purpose="Search the web and fetch source material required for research tasks.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.SOURCE_VERIFICATION,
                purpose="Cross-check claims and preserve source attribution for fact-based outputs.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.VALIDATION,
                purpose="Validate synthesized findings before presenting final research conclusions.",
            ),
        ]

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get research-specific stage definitions.

        Uses SDK-owned canonical tool identifiers.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the research question",
                optional_tools=[ToolNames.WEB_SEARCH, ToolNames.READ, ToolNames.LS],
                keywords=["research", "find", "search", "look up"],
                next_stages={"SEARCHING"},
            ),
            "SEARCHING": StageDefinition(
                name="SEARCHING",
                description="Gathering sources and information",
                optional_tools=[ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.GREP],
                keywords=["search", "find", "gather", "discover"],
                next_stages={"READING", "SEARCHING"},
            ),
            "READING": StageDefinition(
                name="READING",
                description="Deep reading and extraction from sources",
                optional_tools=[ToolNames.WEB_FETCH, ToolNames.READ, ToolNames.CODE_SEARCH],
                keywords=["read", "extract", "analyze", "understand"],
                next_stages={"SYNTHESIZING", "SEARCHING"},
            ),
            "SYNTHESIZING": StageDefinition(
                name="SYNTHESIZING",
                description="Combining and analyzing information",
                optional_tools=[ToolNames.READ, ToolNames.OVERVIEW],
                keywords=["combine", "synthesize", "integrate", "compare"],
                next_stages={"WRITING", "READING"},
            ),
            "WRITING": StageDefinition(
                name="WRITING",
                description="Producing the research output",
                optional_tools=[ToolNames.WRITE, ToolNames.EDIT],
                keywords=["write", "document", "report", "summarize"],
                next_stages={"VERIFICATION", "SYNTHESIZING"},
            ),
            "VERIFICATION": StageDefinition(
                name="VERIFICATION",
                description="Fact-checking and source verification",
                optional_tools=[ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH],
                keywords=["verify", "check", "confirm", "validate"],
                next_stages={"COMPLETION", "WRITING"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Research complete with citations",
                optional_tools=[],
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
