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

"""Research vertical workflows.

This package provides workflow definitions for common research tasks:
- Deep research with source verification
- Fact-checking
- Literature review
- Competitive analysis

Supports both standard and streaming execution via StreamingWorkflowExecutor.

Example:
    provider = ResearchWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("deep_research", orchestrator, context):
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
            print(f"Completed: {chunk.node_name}")
"""

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Type

from victor.core.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import (
    WorkflowBuilder,
    WorkflowDefinition,
    workflow,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.workflows.executor import WorkflowExecutor
    from victor.workflows.streaming import WorkflowStreamChunk
    from victor.workflows.streaming_executor import StreamingWorkflowExecutor


@workflow("deep_research", "Multi-source research with verification")
def deep_research_workflow() -> WorkflowDefinition:
    """Create deep research workflow.

    Performs comprehensive research with source verification.
    """
    return (
        WorkflowBuilder("deep_research")
        .set_metadata("category", "research")
        .set_metadata("complexity", "high")
        # Understand research question
        .add_agent(
            "understand",
            role="researcher",
            goal="Clarify research question and identify key search terms",
            tool_budget=10,
            allowed_tools=["read_file", "grep"],
            output_key="research_plan",
        )
        # Search multiple sources
        .add_agent(
            "search",
            role="researcher",
            goal="Search for information from multiple sources",
            tool_budget=25,
            allowed_tools=["web_search", "web_fetch"],
            input_mapping={"plan": "research_plan"},
            output_key="raw_sources",
        )
        # Extract and analyze
        .add_agent(
            "analyze",
            role="analyst",
            goal="Extract key facts and analyze source quality",
            tool_budget=20,
            allowed_tools=["web_fetch", "read_file"],
            input_mapping={"sources": "raw_sources"},
            output_key="analysis",
        )
        # Verify claims
        .add_agent(
            "verify",
            role="reviewer",
            goal="Cross-reference and verify key claims",
            tool_budget=15,
            allowed_tools=["web_search", "web_fetch"],
            input_mapping={"claims": "analysis"},
            output_key="verification",
        )
        # Synthesize report
        .add_agent(
            "synthesize",
            role="writer",
            goal="Create comprehensive research report with citations",
            tool_budget=15,
            allowed_tools=["write_file", "edit_files"],
            input_mapping={"findings": "verification"},
            next_nodes=[],
        )
        .build()
    )


@workflow("fact_check", "Fact verification workflow")
def fact_check_workflow() -> WorkflowDefinition:
    """Create fact-checking workflow.

    Verifies claims against authoritative sources.
    """
    return (
        WorkflowBuilder("fact_check")
        .set_metadata("category", "research")
        .set_metadata("complexity", "medium")
        # Parse claims
        .add_agent(
            "parse",
            role="analyst",
            goal="Identify specific claims to verify",
            tool_budget=10,
            allowed_tools=["read_file"],
            output_key="claims",
        )
        # Search for evidence
        .add_agent(
            "search_evidence",
            role="researcher",
            goal="Search for supporting or refuting evidence",
            tool_budget=20,
            allowed_tools=["web_search", "web_fetch"],
            input_mapping={"claims": "claims"},
            output_key="evidence",
        )
        # Evaluate evidence
        .add_agent(
            "evaluate",
            role="reviewer",
            goal="Evaluate evidence quality and reach conclusions",
            tool_budget=15,
            allowed_tools=["web_fetch"],
            input_mapping={"evidence": "evidence"},
            output_key="evaluation",
        )
        # Report findings
        .add_agent(
            "report",
            role="writer",
            goal="Create fact-check report with verdicts",
            tool_budget=10,
            allowed_tools=["write_file"],
            next_nodes=[],
        )
        .build()
    )


@workflow("literature_review", "Academic literature review")
def literature_review_workflow() -> WorkflowDefinition:
    """Create literature review workflow.

    Systematic review of academic and technical literature.
    """
    return (
        WorkflowBuilder("literature_review")
        .set_metadata("category", "research")
        .set_metadata("complexity", "high")
        # Define scope
        .add_agent(
            "scope",
            role="planner",
            goal="Define review scope and search strategy",
            tool_budget=10,
            allowed_tools=["read_file"],
            output_key="scope",
        )
        # Search literature
        .add_agent(
            "search_lit",
            role="researcher",
            goal="Search academic databases and repositories",
            tool_budget=30,
            allowed_tools=["web_search", "web_fetch"],
            input_mapping={"scope": "scope"},
            output_key="papers",
        )
        # Screen and select
        .add_agent(
            "screen",
            role="analyst",
            goal="Screen papers for relevance and quality",
            tool_budget=15,
            allowed_tools=["web_fetch", "read_file"],
            input_mapping={"papers": "papers"},
            output_key="selected",
        )
        # Extract data
        .add_agent(
            "extract",
            role="analyst",
            goal="Extract key findings and methodologies",
            tool_budget=20,
            allowed_tools=["web_fetch", "read_file"],
            input_mapping={"papers": "selected"},
            output_key="extracted",
        )
        # Synthesize
        .add_agent(
            "synthesize_review",
            role="writer",
            goal="Synthesize findings into literature review",
            tool_budget=15,
            allowed_tools=["write_file", "edit_files"],
            next_nodes=[],
        )
        .build()
    )


@workflow("competitive_analysis", "Market and competitive research")
def competitive_analysis_workflow() -> WorkflowDefinition:
    """Create competitive analysis workflow.

    Research competitors, market trends, and positioning.
    """
    return (
        WorkflowBuilder("competitive_analysis")
        .set_metadata("category", "research")
        .set_metadata("complexity", "medium")
        # Identify competitors
        .add_agent(
            "identify",
            role="researcher",
            goal="Identify key competitors and market segments",
            tool_budget=15,
            allowed_tools=["web_search", "web_fetch"],
            output_key="competitors",
        )
        # Research each competitor
        .add_agent(
            "research_competitors",
            role="researcher",
            goal="Research competitor products, features, and positioning",
            tool_budget=25,
            allowed_tools=["web_search", "web_fetch"],
            input_mapping={"competitors": "competitors"},
            output_key="competitor_data",
        )
        # Analyze market
        .add_agent(
            "analyze_market",
            role="analyst",
            goal="Analyze market trends and opportunities",
            tool_budget=15,
            allowed_tools=["web_search", "web_fetch"],
            output_key="market_analysis",
        )
        # Create report
        .add_agent(
            "report_analysis",
            role="writer",
            goal="Create competitive analysis report",
            tool_budget=10,
            allowed_tools=["write_file"],
            next_nodes=[],
        )
        .build()
    )


class ResearchWorkflowProvider(WorkflowProviderProtocol):
    """Provides research-specific workflows.

    Includes support for streaming execution via StreamingWorkflowExecutor
    for real-time progress updates during research workflows.

    Example:
        provider = ResearchWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Stream research execution
        async for chunk in provider.astream("deep_research", orchestrator, {}):
            print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
    """

    def __init__(self) -> None:
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        if self._workflows is None:
            self._workflows = {
                "deep_research": deep_research_workflow(),
                "fact_check": fact_check_workflow(),
                "literature_review": literature_review_workflow(),
                "competitive_analysis": competitive_analysis_workflow(),
            }
        return self._workflows

    def get_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Get workflow definitions for this vertical."""
        return self._load_workflows()

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        return self._load_workflows().get(name)

    def get_workflow_names(self) -> List[str]:
        return list(self._load_workflows().keys())

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns."""
        return [
            (r"deep\s+research", "deep_research"),
            (r"research\s+.*\s+thoroughly", "deep_research"),
            (r"comprehensive\s+research", "deep_research"),
            (r"fact\s*check", "fact_check"),
            (r"verify\s+(claim|statement)", "fact_check"),
            (r"is\s+it\s+true", "fact_check"),
            (r"literature\s+review", "literature_review"),
            (r"academic\s+review", "literature_review"),
            (r"papers?\s+on", "literature_review"),
            (r"competitive?\s+analysis", "competitive_analysis"),
            (r"market\s+research", "competitive_analysis"),
            (r"competitor", "competitive_analysis"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Get appropriate workflow for task type."""
        mapping = {
            "research": "deep_research",
            "fact_check": "fact_check",
            "verification": "fact_check",
            "literature": "literature_review",
            "academic": "literature_review",
            "competitive": "competitive_analysis",
            "market": "competitive_analysis",
        }
        return mapping.get(task_type.lower())

    def create_executor(
        self,
        orchestrator: "AgentOrchestrator",
    ) -> "WorkflowExecutor":
        """Create a standard workflow executor.

        Args:
            orchestrator: Agent orchestrator instance

        Returns:
            WorkflowExecutor for running workflows
        """
        from victor.workflows.executor import WorkflowExecutor

        return WorkflowExecutor(orchestrator)

    def create_streaming_executor(
        self,
        orchestrator: "AgentOrchestrator",
    ) -> "StreamingWorkflowExecutor":
        """Create a streaming workflow executor.

        Args:
            orchestrator: Agent orchestrator instance

        Returns:
            StreamingWorkflowExecutor for real-time progress streaming
        """
        from victor.workflows.streaming_executor import StreamingWorkflowExecutor

        return StreamingWorkflowExecutor(orchestrator)

    async def astream(
        self,
        workflow_name: str,
        orchestrator: "AgentOrchestrator",
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["WorkflowStreamChunk"]:
        """Stream workflow execution with real-time events.

        Convenience method that creates a streaming executor and
        streams the specified workflow.

        Args:
            workflow_name: Name of the workflow to execute
            orchestrator: Agent orchestrator instance
            context: Initial context data for the workflow

        Yields:
            WorkflowStreamChunk events during execution

        Raises:
            ValueError: If workflow_name is not found

        Example:
            provider = ResearchWorkflowProvider()
            async for chunk in provider.astream("fact_check", orchestrator, {}):
                if chunk.event_type == WorkflowEventType.NODE_START:
                    print(f"Starting: {chunk.node_name}")
        """
        workflow = self.get_workflow(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        executor = self.create_streaming_executor(orchestrator)
        async for chunk in executor.astream(workflow, context or {}):
            yield chunk

    def __repr__(self) -> str:
        return f"ResearchWorkflowProvider(workflows={len(self._load_workflows())})"


from victor.research.workflows.graph_workflows import (
    ResearchState,
    FactCheckState,
    LiteratureState,
    create_deep_research_workflow,
    create_fact_check_workflow,
    create_literature_review_workflow,
    ResearchGraphExecutor,
)

__all__ = [
    # WorkflowBuilder-based workflows
    "ResearchWorkflowProvider",
    "deep_research_workflow",
    "fact_check_workflow",
    "literature_review_workflow",
    "competitive_analysis_workflow",
    # StateGraph-based workflows
    "ResearchState",
    "FactCheckState",
    "LiteratureState",
    "create_deep_research_workflow",
    "create_fact_check_workflow",
    "create_literature_review_workflow",
    "ResearchGraphExecutor",
]
