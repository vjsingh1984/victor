# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResearchAssistant - Vertical for web research and document analysis.

This vertical is optimized for:
- Web search and information gathering
- Document analysis and summarization
- Fact verification and cross-referencing
- Report generation

Example:
    from victor.verticals import ResearchAssistant

    config = ResearchAssistant.get_config()
    agent = await Agent.create(tools=config.tools)
    result = await agent.run("Research the latest AI safety developments")
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig


class ResearchAssistant(VerticalBase):
    """Research and document analysis assistant vertical.

    Optimized for:
    - Web research and information gathering
    - Document reading and analysis
    - Summarization and synthesis
    - Report writing

    Unlike CodingAssistant, this vertical focuses on:
    - Web tools over filesystem tools
    - Reading over writing
    - Analysis over execution

    Example:
        from victor.verticals import ResearchAssistant

        agent = await ResearchAssistant.create_agent()
        result = await agent.run("Summarize recent developments in quantum computing")
    """

    name = "research"
    description = "Research assistant for web search, document analysis, and report generation"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools optimized for research tasks.

        Returns:
            List of tool names focused on search, reading, and writing.
        """
        return [
            # Web research
            "web_search",
            "web_fetch",
            # Document reading
            "read",
            "ls",
            "overview",
            # Search
            "search",
            "semantic_code_search",  # Can search any text
            # Writing
            "write",
            "edit",
            # Note: Intentionally limited tool set
            # No shell, git, docker, etc.
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get research-focused system prompt.

        Returns:
            System prompt optimized for research tasks.
        """
        return """You are a research assistant specializing in information gathering and analysis.

Your capabilities:
- Web search for current information
- Fetching and analyzing web pages
- Reading and summarizing documents
- Synthesizing information from multiple sources
- Writing clear, well-organized reports

Guidelines:
1. **Verify information**: Cross-reference facts from multiple sources when possible
2. **Cite sources**: Always note where information came from
3. **Be current**: Prefer recent sources for time-sensitive topics
4. **Summarize effectively**: Extract key points without losing important nuance
5. **Organize clearly**: Structure findings in a logical, easy-to-follow format
6. **Acknowledge uncertainty**: Be clear about what is established vs. speculative

Research workflow:
1. SEARCHING: Use web_search to find relevant sources
2. READING: Use web_fetch to retrieve and analyze pages
3. SYNTHESIZING: Combine information from multiple sources
4. WRITING: Generate clear summaries or reports

When researching:
- Start with broad searches, then narrow down
- Identify authoritative sources (official docs, academic papers, reputable news)
- Note publication dates for time-sensitive information
- Look for primary sources when possible

You have access to focused research tools. Use them efficiently to gather accurate information."""

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get research-specific stage definitions.

        Returns:
            Stage definitions optimized for research workflow.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the research question",
                tools={"read", "ls"},
                keywords=["what", "explain", "help", "research", "find out"],
                next_stages={"SEARCHING"},
            ),
            "SEARCHING": StageDefinition(
                name="SEARCHING",
                description="Searching for information",
                tools={"web_search", "search"},
                keywords=["search", "find", "look for", "discover"],
                next_stages={"READING", "SYNTHESIZING"},
            ),
            "READING": StageDefinition(
                name="READING",
                description="Reading and extracting information",
                tools={"web_fetch", "read", "overview"},
                keywords=["read", "fetch", "get", "analyze", "examine"],
                next_stages={"SYNTHESIZING", "SEARCHING"},
            ),
            "SYNTHESIZING": StageDefinition(
                name="SYNTHESIZING",
                description="Combining and analyzing findings",
                tools={"read"},  # Mostly internal processing
                keywords=["combine", "analyze", "compare", "contrast", "evaluate"],
                next_stages={"WRITING", "SEARCHING"},
            ),
            "WRITING": StageDefinition(
                name="WRITING",
                description="Writing the report or summary",
                tools={"write", "edit"},
                keywords=["write", "summarize", "report", "document", "create"],
                next_stages={"COMPLETION", "SYNTHESIZING"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Finalizing and presenting findings",
                tools={"read"},
                keywords=["done", "finish", "complete", "present"],
                next_stages=set(),
            ),
        }

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Get provider hints for research tasks.

        Returns:
            Provider preferences for research and summarization.
        """
        return {
            "preferred_providers": ["anthropic", "openai", "google"],
            "preferred_models": [
                "claude-sonnet-4-20250514",
                "gpt-4-turbo",
                "gemini-pro",
            ],
            "min_context_window": 100000,  # Need large context for research
            "requires_tool_calling": True,
            "prefers_extended_thinking": False,  # Research is more iterative
        }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Get evaluation criteria for research tasks.

        Returns:
            Criteria for evaluating research quality.
        """
        return [
            "Accuracy of information",
            "Source quality and reliability",
            "Comprehensiveness of research",
            "Clarity of summarization",
            "Proper attribution and citations",
            "Logical organization of findings",
            "Objectivity and balance",
        ]

    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
        """Add research-specific configuration.

        Args:
            config: Base configuration.

        Returns:
            Customized configuration.
        """
        config.metadata["requires_internet"] = True
        config.metadata["max_sources"] = 20
        config.metadata["preferred_source_types"] = [
            "official_documentation",
            "academic_papers",
            "reputable_news",
            "primary_sources",
        ]
        return config
