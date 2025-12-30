"""Research Prompt Contributor - Task hints and system prompt extensions for research."""

from typing import Dict, Optional

from victor.core.verticals.protocols import PromptContributorProtocol, TaskTypeHint


# Research-specific task type hints
RESEARCH_TASK_TYPE_HINTS: Dict[str, TaskTypeHint] = {
    "fact_check": TaskTypeHint(
        task_type="fact_check",
        hint="""[FACT-CHECK] Verify claims with multiple independent sources:
1. Search for original sources and official documentation
2. Cross-reference with authoritative databases
3. Check recency and relevance of sources
4. Note any conflicting information found""",
        tool_budget=12,
        priority_tools=["web_search", "web_fetch", "read"],
    ),
    "literature_review": TaskTypeHint(
        task_type="literature_review",
        hint="""[LITERATURE] Systematic review of existing knowledge:
1. Define scope and search criteria
2. Search academic and authoritative sources
3. Extract key findings and methodologies
4. Synthesize patterns and gaps
5. Provide structured bibliography""",
        tool_budget=20,
        priority_tools=["web_search", "web_fetch", "read", "write"],
    ),
    "competitive_analysis": TaskTypeHint(
        task_type="competitive_analysis",
        hint="""[ANALYSIS] Compare products, services, or approaches:
1. Identify key comparison criteria
2. Gather data from official sources
3. Create objective comparison matrix
4. Note strengths, weaknesses, limitations
5. Avoid promotional language""",
        tool_budget=15,
        priority_tools=["web_search", "web_fetch", "read", "write"],
    ),
    "trend_research": TaskTypeHint(
        task_type="trend_research",
        hint="""[TRENDS] Identify patterns and emerging developments:
1. Search recent news and publications
2. Look for quantitative data and statistics
3. Identify key players and innovations
4. Note methodology limitations
5. Distinguish facts from speculation""",
        tool_budget=15,
        priority_tools=["web_search", "web_fetch"],
    ),
    "technical_research": TaskTypeHint(
        task_type="technical_research",
        hint="""[TECHNICAL] Deep dive into technical topics:
1. Start with official documentation
2. Search code repositories and examples
3. Look for benchmarks and comparisons
4. Note version-specific information
5. Verify with multiple technical sources""",
        tool_budget=18,
        priority_tools=["web_search", "web_fetch", "code_search", "read"],
    ),
    "general_query": TaskTypeHint(
        task_type="general_query",
        hint="""[QUERY] Answer factual questions:
1. Search for authoritative sources
2. Provide direct answer with context
3. Cite sources with URLs
4. Note any uncertainty or limitations""",
        tool_budget=8,
        priority_tools=["web_search", "web_fetch"],
    ),
    # Default fallback for 'general' task type
    "general": TaskTypeHint(
        task_type="general",
        hint="""[GENERAL RESEARCH] For general research queries:
1. Use web_search to find relevant sources
2. Fetch key pages with web_fetch for details
3. Synthesize findings from multiple sources
4. Cite all sources with URLs
5. Note limitations or areas needing further research""",
        tool_budget=10,
        priority_tools=["web_search", "web_fetch", "read"],
    ),
}


class ResearchPromptContributor(PromptContributorProtocol):
    """Contributes research-specific prompts and task hints."""

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Return research-specific task type hints.

        Returns:
            Dict mapping task types to TaskTypeHint objects
        """
        return RESEARCH_TASK_TYPE_HINTS.copy()

    def get_system_prompt_section(self) -> str:
        """Return additional system prompt content for research context.

        Returns:
            System prompt section for research tasks
        """
        return """
## Research Quality Checklist

Before finalizing any research output:
- [ ] All claims have cited sources
- [ ] Sources are authoritative and recent
- [ ] Conflicting viewpoints acknowledged
- [ ] Limitations and uncertainties noted
- [ ] Statistical claims include methodology context
- [ ] URLs are provided for verification

## Source Hierarchy

1. **Primary sources**: Official documentation, academic papers, government data
2. **Secondary sources**: Reputable news outlets, industry reports, expert analyses
3. **Tertiary sources**: Encyclopedia entries, aggregated reviews (use sparingly)

Avoid: Social media posts, anonymous forums, outdated content (>2 years for fast-moving topics)
""".strip()

    def get_grounding_rules(self) -> str:
        """Get research-specific grounding rules.

        Returns:
            Grounding rules text for research tasks
        """
        return """GROUNDING: Base ALL responses on tool output only. Never fabricate sources or statistics.
Always cite URLs for claims. Acknowledge uncertainty when sources conflict.""".strip()

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Returns:
            Priority value (Research is specialized, so medium priority)
        """
        return 5

    def get_context_hints(self, task_type: Optional[str] = None) -> Optional[str]:
        """Return contextual hints based on detected task type."""
        if task_type and task_type in RESEARCH_TASK_TYPE_HINTS:
            return RESEARCH_TASK_TYPE_HINTS[task_type].hint
        return None
