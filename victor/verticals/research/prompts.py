"""Research Prompt Contributor - Task hints and system prompt extensions for research."""

from typing import Dict, Optional

from victor.verticals.protocols import PromptContributorProtocol


# Research-specific task type hints
RESEARCH_TASK_TYPE_HINTS: Dict[str, str] = {
    "fact_check": """[FACT-CHECK] Verify claims with multiple independent sources:
1. Search for original sources and official documentation
2. Cross-reference with authoritative databases
3. Check recency and relevance of sources
4. Note any conflicting information found""",

    "literature_review": """[LITERATURE] Systematic review of existing knowledge:
1. Define scope and search criteria
2. Search academic and authoritative sources
3. Extract key findings and methodologies
4. Synthesize patterns and gaps
5. Provide structured bibliography""",

    "competitive_analysis": """[ANALYSIS] Compare products, services, or approaches:
1. Identify key comparison criteria
2. Gather data from official sources
3. Create objective comparison matrix
4. Note strengths, weaknesses, limitations
5. Avoid promotional language""",

    "trend_research": """[TRENDS] Identify patterns and emerging developments:
1. Search recent news and publications
2. Look for quantitative data and statistics
3. Identify key players and innovations
4. Note methodology limitations
5. Distinguish facts from speculation""",

    "technical_research": """[TECHNICAL] Deep dive into technical topics:
1. Start with official documentation
2. Search code repositories and examples
3. Look for benchmarks and comparisons
4. Note version-specific information
5. Verify with multiple technical sources""",

    "general_query": """[QUERY] Answer factual questions:
1. Search for authoritative sources
2. Provide direct answer with context
3. Cite sources with URLs
4. Note any uncertainty or limitations""",

    # Default fallback for 'general' task type
    "general": """[GENERAL RESEARCH] For general research queries:
1. Use web_search to find relevant sources
2. Fetch key pages with web_fetch for details
3. Synthesize findings from multiple sources
4. Cite all sources with URLs
5. Note limitations or areas needing further research""",
}


class ResearchPromptContributor(PromptContributorProtocol):
    """Contributes research-specific prompts and task hints."""

    def get_task_type_hints(self) -> Dict[str, str]:
        """Return research-specific task type hints."""
        return RESEARCH_TASK_TYPE_HINTS

    def get_system_prompt_extension(self) -> Optional[str]:
        """Return additional system prompt content for research context."""
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
"""

    def get_context_hints(self, task_type: Optional[str] = None) -> Optional[str]:
        """Return contextual hints based on detected task type."""
        if task_type and task_type in RESEARCH_TASK_TYPE_HINTS:
            return RESEARCH_TASK_TYPE_HINTS[task_type]
        return None
