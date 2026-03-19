"""Serializable prompt metadata for the Research vertical."""

from __future__ import annotations

from typing import Any, Dict

RESEARCH_TASK_TYPE_HINTS: Dict[str, Dict[str, Any]] = {
    "fact_check": {
        "hint": """[FACT-CHECK] Verify claims with multiple independent sources:
1. Search for original sources and official documentation
2. Cross-reference with authoritative databases
3. Check recency and relevance of sources
4. Note any conflicting information found""",
        "tool_budget": 12,
        "priority_tools": ["web_search", "web_fetch", "read"],
    },
    "literature_review": {
        "hint": """[LITERATURE] Systematic review of existing knowledge:
1. Define scope and search criteria
2. Search academic and authoritative sources
3. Extract key findings and methodologies
4. Synthesize patterns and gaps
5. Provide structured bibliography""",
        "tool_budget": 20,
        "priority_tools": ["web_search", "web_fetch", "read", "write"],
    },
    "competitive_analysis": {
        "hint": """[ANALYSIS] Compare products, services, or approaches:
1. Identify key comparison criteria
2. Gather data from official sources
3. Create objective comparison matrix
4. Note strengths, weaknesses, limitations
5. Avoid promotional language""",
        "tool_budget": 15,
        "priority_tools": ["web_search", "web_fetch", "read", "write"],
    },
    "trend_research": {
        "hint": """[TRENDS] Identify patterns and emerging developments:
1. Search recent news and publications
2. Look for quantitative data and statistics
3. Identify key players and innovations
4. Note methodology limitations
5. Distinguish facts from speculation""",
        "tool_budget": 15,
        "priority_tools": ["web_search", "web_fetch"],
    },
    "technical_research": {
        "hint": """[TECHNICAL] Deep dive into technical topics:
1. Start with official documentation
2. Search code repositories and examples
3. Look for benchmarks and comparisons
4. Note version-specific information
5. Verify with multiple technical sources""",
        "tool_budget": 18,
        "priority_tools": ["web_search", "web_fetch", "code_search", "read"],
    },
    "general_query": {
        "hint": """[QUERY] Answer factual questions:
1. Search for authoritative sources
2. Provide direct answer with context
3. Cite sources with URLs
4. Note any uncertainty or limitations""",
        "tool_budget": 8,
        "priority_tools": ["web_search", "web_fetch"],
    },
    "general": {
        "hint": """[GENERAL RESEARCH] For general research queries:
1. Use web_search to find relevant sources
2. Fetch key pages with web_fetch for details
3. Synthesize findings from multiple sources
4. Cite all sources with URLs
5. Note limitations or areas needing further research""",
        "tool_budget": 10,
        "priority_tools": ["web_search", "web_fetch", "read"],
    },
}

RESEARCH_SYSTEM_PROMPT_SECTION = """
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

RESEARCH_GROUNDING_RULES = (
    """GROUNDING: Base ALL responses on tool output only. Never fabricate sources or statistics.
Always cite URLs for claims. Acknowledge uncertainty when sources conflict.""".strip()
)

RESEARCH_PROMPT_PRIORITY = 5

RESEARCH_PROMPT_TEMPLATES: Dict[str, str] = {
    "research_operations": RESEARCH_SYSTEM_PROMPT_SECTION,
}

__all__ = [
    "RESEARCH_GROUNDING_RULES",
    "RESEARCH_PROMPT_PRIORITY",
    "RESEARCH_PROMPT_TEMPLATES",
    "RESEARCH_SYSTEM_PROMPT_SECTION",
    "RESEARCH_TASK_TYPE_HINTS",
]
