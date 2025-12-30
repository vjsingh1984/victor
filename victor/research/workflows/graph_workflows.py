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

"""StateGraph-based workflows for Research vertical.

Provides LangGraph-compatible StateGraph workflows for complex research tasks
that benefit from:
- Typed state management
- Cyclic execution (search-verify-refine loops)
- Explicit retry limits
- Checkpoint/resume semantics
- Human-in-the-loop for fact-checking approvals

Example:
    from victor.research.workflows.graph_workflows import (
        create_deep_research_workflow,
        ResearchState,
    )

    graph = create_deep_research_workflow()
    result = await graph.compile().invoke(ResearchState(
        query="What are the latest developments in quantum computing?",
        sources=[],
    ))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, TypedDict

from victor.framework.graph import END, StateGraph, GraphConfig

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Typed State Definitions
# =============================================================================


class ResearchState(TypedDict, total=False):
    """Typed state for deep research workflows."""

    query: str
    search_terms: Optional[List[str]]
    sources: Optional[List[Dict[str, Any]]]
    analysis: Optional[Dict[str, Any]]
    verification_results: Optional[Dict[str, Any]]
    report: Optional[str]
    iteration_count: int
    max_iterations: int
    needs_more_sources: bool
    success: bool


class FactCheckState(TypedDict, total=False):
    """Typed state for fact-checking workflows."""

    claims_text: str
    parsed_claims: Optional[List[Dict[str, Any]]]
    evidence: Optional[List[Dict[str, Any]]]
    verdicts: Optional[List[Dict[str, Any]]]
    report: Optional[str]
    iteration: int
    max_iterations: int


class LiteratureState(TypedDict, total=False):
    """Typed state for literature review workflows."""

    research_question: str
    search_protocol: Optional[Dict[str, Any]]
    papers_found: Optional[List[Dict[str, Any]]]
    papers_screened: Optional[List[Dict[str, Any]]]
    extracted_data: Optional[Dict[str, Any]]
    synthesis: Optional[str]
    iteration: int


# =============================================================================
# Node Functions
# =============================================================================


async def clarify_query_node(state: ResearchState) -> ResearchState:
    """Clarify research query and generate search terms."""
    query = state.get("query", "")
    state["search_terms"] = [query]  # Would be expanded by LLM
    return state


async def search_sources_node(state: ResearchState) -> ResearchState:
    """Search for sources using generated terms."""
    state["sources"] = state.get("sources", [])
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    return state


async def analyze_sources_node(state: ResearchState) -> ResearchState:
    """Analyze gathered sources for key findings."""
    state["analysis"] = {
        "key_findings": [],
        "gaps": [],
        "quality_scores": {},
    }
    return state


async def verify_findings_node(state: ResearchState) -> ResearchState:
    """Verify key findings across sources."""
    state["verification_results"] = {
        "verified": [],
        "unverified": [],
        "contradictions": [],
    }
    return state


async def generate_report_node(state: ResearchState) -> ResearchState:
    """Generate comprehensive research report."""
    state["report"] = "Research Report"
    state["success"] = True
    return state


# Fact-check nodes
async def parse_claims_node(state: FactCheckState) -> FactCheckState:
    """Parse text to identify verifiable claims."""
    state["parsed_claims"] = []
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def search_evidence_node(state: FactCheckState) -> FactCheckState:
    """Search for evidence for/against claims."""
    state["evidence"] = []
    return state


async def evaluate_evidence_node(state: FactCheckState) -> FactCheckState:
    """Evaluate evidence and render verdicts."""
    state["verdicts"] = []
    return state


async def generate_verdict_report_node(state: FactCheckState) -> FactCheckState:
    """Generate fact-check report with verdicts."""
    state["report"] = "Fact Check Report"
    return state


# Literature review nodes
async def define_protocol_node(state: LiteratureState) -> LiteratureState:
    """Define review protocol and search strategy."""
    state["search_protocol"] = {
        "inclusion_criteria": [],
        "exclusion_criteria": [],
        "search_strings": [],
    }
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def search_literature_node(state: LiteratureState) -> LiteratureState:
    """Search academic databases."""
    state["papers_found"] = []
    return state


async def screen_papers_node(state: LiteratureState) -> LiteratureState:
    """Screen papers against inclusion criteria."""
    state["papers_screened"] = []
    return state


async def extract_data_node(state: LiteratureState) -> LiteratureState:
    """Extract data from selected papers."""
    state["extracted_data"] = {"themes": [], "findings": []}
    return state


async def synthesize_review_node(state: LiteratureState) -> LiteratureState:
    """Synthesize findings into literature review."""
    state["synthesis"] = "Literature Review"
    return state


# =============================================================================
# Condition Functions
# =============================================================================


def should_search_more(state: ResearchState) -> str:
    """Determine if more sources are needed."""
    sources = state.get("sources", [])
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if len(sources) >= 10 or iteration >= max_iter:
        return "analyze"
    if state.get("needs_more_sources", False):
        return "search"
    return "analyze"


def should_continue_verification(state: ResearchState) -> str:
    """Determine if verification is complete."""
    verification = state.get("verification_results", {})
    unverified = len(verification.get("unverified", []))
    contradictions = len(verification.get("contradictions", []))

    if unverified > 3 or contradictions > 0:
        iteration = state.get("iteration_count", 0)
        if iteration < state.get("max_iterations", 3):
            return "search"  # Find more sources
    return "report"


def should_continue_factcheck(state: FactCheckState) -> str:
    """Determine if fact-checking should continue."""
    verdicts = state.get("verdicts", [])
    claims = state.get("parsed_claims", [])

    if len(verdicts) < len(claims):
        return "continue"
    return "report"


# =============================================================================
# Workflow Factories
# =============================================================================


def create_deep_research_workflow() -> StateGraph[ResearchState]:
    """Create a deep research workflow with source verification.

    Implements:
    1. Clarify -> Search -> Analyze -> Verify
    2. If verification finds gaps, search for more sources
    3. Generate comprehensive report

    Returns:
        StateGraph for deep research
    """
    graph = StateGraph(ResearchState)

    graph.add_node("clarify", clarify_query_node)
    graph.add_node("search", search_sources_node)
    graph.add_node("analyze", analyze_sources_node)
    graph.add_node("verify", verify_findings_node)
    graph.add_node("report", generate_report_node)

    graph.add_edge("clarify", "search")
    graph.add_conditional_edge(
        "search",
        should_search_more,
        {"search": "search", "analyze": "analyze"},
    )
    graph.add_edge("analyze", "verify")
    graph.add_conditional_edge(
        "verify",
        should_continue_verification,
        {"search": "search", "report": "report"},
    )
    graph.add_edge("report", END)

    graph.set_entry_point("clarify")
    return graph


def create_fact_check_workflow() -> StateGraph[FactCheckState]:
    """Create a fact-checking workflow with verdict generation.

    Returns:
        StateGraph for fact-checking
    """
    graph = StateGraph(FactCheckState)

    graph.add_node("parse", parse_claims_node)
    graph.add_node("search", search_evidence_node)
    graph.add_node("evaluate", evaluate_evidence_node)
    graph.add_node("report", generate_verdict_report_node)

    graph.add_edge("parse", "search")
    graph.add_edge("search", "evaluate")
    graph.add_conditional_edge(
        "evaluate",
        should_continue_factcheck,
        {"continue": "search", "report": "report"},
    )
    graph.add_edge("report", END)

    graph.set_entry_point("parse")
    return graph


def create_literature_review_workflow() -> StateGraph[LiteratureState]:
    """Create a systematic literature review workflow.

    Returns:
        StateGraph for literature review
    """
    graph = StateGraph(LiteratureState)

    graph.add_node("protocol", define_protocol_node)
    graph.add_node("search", search_literature_node)
    graph.add_node("screen", screen_papers_node)
    graph.add_node("extract", extract_data_node)
    graph.add_node("synthesize", synthesize_review_node)

    graph.add_edge("protocol", "search")
    graph.add_edge("search", "screen")
    graph.add_edge("screen", "extract")
    graph.add_edge("extract", "synthesize")
    graph.add_edge("synthesize", END)

    graph.set_entry_point("protocol")
    return graph


class ResearchGraphExecutor:
    """Executor that integrates StateGraph with AgentOrchestrator for Research."""

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        checkpointer: Optional[Any] = None,
    ):
        self._orchestrator = orchestrator
        self._checkpointer = checkpointer

    async def run(
        self,
        graph: StateGraph,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        config: Optional[GraphConfig] = None,
    ):
        compiled = graph.compile(checkpointer=self._checkpointer)
        exec_config = config or GraphConfig()
        if self._checkpointer:
            exec_config.checkpointer = self._checkpointer
        return await compiled.invoke(initial_state, config=exec_config, thread_id=thread_id)


__all__ = [
    "ResearchState",
    "FactCheckState",
    "LiteratureState",
    "create_deep_research_workflow",
    "create_fact_check_workflow",
    "create_literature_review_workflow",
    "ResearchGraphExecutor",
]
