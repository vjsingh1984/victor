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

"""Teams integration for Research vertical."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from victor.framework.teams import TeamFormation, TeamMemberSpec


@dataclass
class ResearchTeamSpec:
    """Specification for a research team."""

    name: str
    description: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50


RESEARCH_TEAM_SPECS: Dict[str, ResearchTeamSpec] = {
    "deep_research_team": ResearchTeamSpec(
        name="Deep Research Team",
        description="Comprehensive multi-source research with verification",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Search and gather information from multiple sources",
                name="Primary Researcher",
                tool_budget=30,
            ),
            TeamMemberSpec(
                role="analyst",
                goal="Analyze and extract key findings",
                name="Research Analyst",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Verify claims and cross-reference sources",
                name="Fact Verifier",
                tool_budget=20,
            ),
            TeamMemberSpec(
                role="writer",
                goal="Synthesize findings into comprehensive report",
                name="Report Writer",
                tool_budget=20,
            ),
        ],
        total_tool_budget=95,
    ),
    "fact_check_team": ResearchTeamSpec(
        name="Fact Check Team",
        description="Verify claims and statements",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="analyst",
                goal="Parse and identify claims to verify",
                name="Claim Analyst",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="researcher",
                goal="Search for evidence supporting or refuting claims",
                name="Evidence Researcher",
                tool_budget=35,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Evaluate evidence and determine verdicts",
                name="Verdict Reviewer",
                tool_budget=20,
            ),
        ],
        total_tool_budget=70,
    ),
    "literature_team": ResearchTeamSpec(
        name="Literature Review Team",
        description="Systematic academic literature review",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="planner",
                goal="Define review scope and search strategy",
                name="Review Planner",
                tool_budget=10,
            ),
            TeamMemberSpec(
                role="researcher",
                goal="Search academic sources and papers",
                name="Literature Searcher",
                tool_budget=35,
            ),
            TeamMemberSpec(
                role="analyst",
                goal="Screen and extract data from papers",
                name="Paper Analyst",
                tool_budget=30,
            ),
            TeamMemberSpec(
                role="writer",
                goal="Synthesize literature review",
                name="Review Writer",
                tool_budget=20,
            ),
        ],
        total_tool_budget=95,
    ),
    "competitive_team": ResearchTeamSpec(
        name="Competitive Analysis Team",
        description="Market and competitive research",
        formation=TeamFormation.PARALLEL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Research competitor products and features",
                name="Competitor Researcher",
                tool_budget=30,
            ),
            TeamMemberSpec(
                role="analyst",
                goal="Analyze market trends and opportunities",
                name="Market Analyst",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="writer",
                goal="Create competitive analysis report",
                name="Analysis Writer",
                tool_budget=15,
            ),
        ],
        total_tool_budget=70,
    ),
    "synthesis_team": ResearchTeamSpec(
        name="Synthesis Team",
        description="Combine multiple research sources into cohesive report",
        formation=TeamFormation.HIERARCHICAL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Gather and organize source materials",
                name="Source Gatherer",
                tool_budget=20,
            ),
            TeamMemberSpec(
                role="analyst",
                goal="Identify themes and connections",
                name="Theme Analyst",
                tool_budget=20,
            ),
            TeamMemberSpec(
                role="writer",
                goal="Write synthesized report",
                name="Synthesis Writer",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Review for coherence and completeness",
                name="Quality Reviewer",
                tool_budget=15,
            ),
        ],
        total_tool_budget=80,
    ),
}


def get_team_for_task(task_type: str) -> Optional[ResearchTeamSpec]:
    """Get appropriate team for task type."""
    mapping = {
        "research": "deep_research_team",
        "deep_research": "deep_research_team",
        "comprehensive": "deep_research_team",
        "fact_check": "fact_check_team",
        "verify": "fact_check_team",
        "verification": "fact_check_team",
        "literature": "literature_team",
        "academic": "literature_team",
        "papers": "literature_team",
        "competitive": "competitive_team",
        "market": "competitive_team",
        "competitor": "competitive_team",
        "synthesis": "synthesis_team",
        "combine": "synthesis_team",
        "report": "synthesis_team",
    }
    spec_name = mapping.get(task_type.lower())
    if spec_name:
        return RESEARCH_TEAM_SPECS.get(spec_name)
    return None


def list_team_types() -> List[str]:
    """List available team types."""
    return list(RESEARCH_TEAM_SPECS.keys())


__all__ = [
    "ResearchTeamSpec",
    "RESEARCH_TEAM_SPECS",
    "get_team_for_task",
    "list_team_types",
]
