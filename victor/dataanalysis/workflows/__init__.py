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

"""Data Analysis vertical workflows.

This package provides workflow definitions for common data analysis tasks:
- Exploratory Data Analysis (EDA)
- Data cleaning and preparation
- Statistical analysis
- Machine Learning pipeline

Supports both standard and streaming execution via StreamingWorkflowExecutor.

Example:
    provider = DataAnalysisWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("eda_workflow", orchestrator, context):
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
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.workflows.executor import WorkflowExecutor
    from victor.workflows.streaming import WorkflowStreamChunk
    from victor.workflows.streaming_executor import StreamingWorkflowExecutor


@workflow("eda_workflow", "Exploratory Data Analysis workflow")
def eda_workflow() -> WorkflowDefinition:
    """Create exploratory data analysis workflow.

    Comprehensive EDA with profiling, visualization, and insights.
    """
    return (
        WorkflowBuilder("eda_workflow")
        .set_metadata("category", "data_analysis")
        .set_metadata("complexity", "medium")
        # Load and understand data
        .add_agent(
            "load",
            role="researcher",
            goal="Load data and understand structure, types, and shape",
            tool_budget=15,
            allowed_tools=["read_file", "ls", "shell"],
            output_key="data_profile",
        )
        # Generate summary statistics
        .add_agent(
            "profile",
            role="analyst",
            goal="Generate summary statistics and data profiling",
            tool_budget=20,
            allowed_tools=["shell", "write_file"],
            input_mapping={"data": "data_profile"},
            output_key="statistics",
        )
        # Analyze distributions and correlations
        .add_agent(
            "analyze",
            role="analyst",
            goal="Analyze distributions, correlations, and patterns",
            tool_budget=20,
            allowed_tools=["shell", "write_file"],
            input_mapping={"stats": "statistics"},
            output_key="analysis",
        )
        # Create visualizations
        .add_agent(
            "visualize",
            role="executor",
            goal="Create visualizations for key findings",
            tool_budget=20,
            allowed_tools=["shell", "write_file"],
            input_mapping={"analysis": "analysis"},
            output_key="visualizations",
        )
        # Summarize insights
        .add_agent(
            "summarize",
            role="writer",
            goal="Summarize key insights and recommendations",
            tool_budget=10,
            allowed_tools=["write_file"],
            next_nodes=[],
        )
        .build()
    )


@workflow("data_cleaning", "Data cleaning and preparation workflow")
def data_cleaning_workflow() -> WorkflowDefinition:
    """Create data cleaning workflow.

    Systematic data cleaning with validation.
    """
    return (
        WorkflowBuilder("data_cleaning")
        .set_metadata("category", "data_analysis")
        .set_metadata("complexity", "medium")
        # Assess data quality
        .add_agent(
            "assess",
            role="analyst",
            goal="Assess data quality issues (missing, duplicates, outliers)",
            tool_budget=15,
            allowed_tools=["read_file", "shell"],
            output_key="quality_report",
        )
        # Plan cleaning strategy
        .add_agent(
            "plan",
            role="planner",
            goal="Plan data cleaning strategy based on issues",
            tool_budget=10,
            allowed_tools=["read_file"],
            input_mapping={"report": "quality_report"},
            output_key="cleaning_plan",
        )
        # Execute cleaning
        .add_agent(
            "clean",
            role="executor",
            goal="Execute data cleaning transformations",
            tool_budget=25,
            allowed_tools=["shell", "write_file", "read_file"],
            input_mapping={"plan": "cleaning_plan"},
            output_key="cleaned_data",
        )
        # Validate results
        .add_agent(
            "validate",
            role="reviewer",
            goal="Validate cleaned data meets quality standards",
            tool_budget=15,
            allowed_tools=["shell", "read_file"],
            input_mapping={"data": "cleaned_data"},
            output_key="validation",
        )
        # Document changes
        .add_agent(
            "document",
            role="writer",
            goal="Document cleaning steps and data lineage",
            tool_budget=10,
            allowed_tools=["write_file"],
            next_nodes=[],
        )
        .build()
    )


@workflow("statistical_analysis", "Statistical analysis workflow")
def statistical_analysis_workflow() -> WorkflowDefinition:
    """Create statistical analysis workflow.

    Hypothesis testing and statistical modeling.
    """
    return (
        WorkflowBuilder("statistical_analysis")
        .set_metadata("category", "data_analysis")
        .set_metadata("complexity", "high")
        # Understand research questions
        .add_agent(
            "formulate",
            role="researcher",
            goal="Formulate hypotheses and identify statistical tests",
            tool_budget=15,
            allowed_tools=["read_file", "web_search"],
            output_key="hypotheses",
        )
        # Prepare data for analysis
        .add_agent(
            "prepare",
            role="executor",
            goal="Prepare data for statistical analysis",
            tool_budget=20,
            allowed_tools=["shell", "read_file", "write_file"],
            input_mapping={"hypotheses": "hypotheses"},
            output_key="prepared_data",
        )
        # Run statistical tests
        .add_agent(
            "test",
            role="analyst",
            goal="Run statistical tests and calculate metrics",
            tool_budget=25,
            allowed_tools=["shell", "write_file"],
            input_mapping={"data": "prepared_data"},
            output_key="test_results",
        )
        # Interpret results
        .add_agent(
            "interpret",
            role="analyst",
            goal="Interpret results and draw conclusions",
            tool_budget=15,
            allowed_tools=["shell", "write_file"],
            input_mapping={"results": "test_results"},
            output_key="interpretation",
        )
        # Report findings
        .add_agent(
            "report",
            role="writer",
            goal="Create statistical report with visualizations",
            tool_budget=15,
            allowed_tools=["shell", "write_file"],
            next_nodes=[],
        )
        .build()
    )


@workflow("ml_pipeline", "Machine Learning pipeline workflow")
def ml_pipeline_workflow() -> WorkflowDefinition:
    """Create ML pipeline workflow.

    End-to-end machine learning pipeline.
    """
    return (
        WorkflowBuilder("ml_pipeline")
        .set_metadata("category", "data_analysis")
        .set_metadata("complexity", "high")
        # Problem understanding
        .add_agent(
            "understand",
            role="researcher",
            goal="Understand ML problem and success criteria",
            tool_budget=10,
            allowed_tools=["read_file", "web_search"],
            output_key="problem_definition",
        )
        # Feature engineering
        .add_agent(
            "engineer",
            role="executor",
            goal="Engineer features from raw data",
            tool_budget=25,
            allowed_tools=["shell", "read_file", "write_file"],
            input_mapping={"problem": "problem_definition"},
            output_key="features",
        )
        # Model training
        .add_agent(
            "train",
            role="executor",
            goal="Train and tune ML models",
            tool_budget=30,
            allowed_tools=["shell", "write_file"],
            input_mapping={"features": "features"},
            output_key="models",
        )
        # Model evaluation
        .add_agent(
            "evaluate",
            role="reviewer",
            goal="Evaluate models and select best",
            tool_budget=20,
            allowed_tools=["shell", "read_file", "write_file"],
            input_mapping={"models": "models"},
            output_key="evaluation",
        )
        # Document pipeline
        .add_agent(
            "document_ml",
            role="writer",
            goal="Document ML pipeline and model performance",
            tool_budget=10,
            allowed_tools=["write_file"],
            next_nodes=[],
        )
        .build()
    )


class DataAnalysisWorkflowProvider(WorkflowProviderProtocol):
    """Provides data analysis-specific workflows.

    Includes support for streaming execution via StreamingWorkflowExecutor
    for real-time progress updates during data analysis workflows.

    Example:
        provider = DataAnalysisWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Stream ML pipeline execution
        async for chunk in provider.astream("ml_pipeline", orchestrator, {}):
            print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
    """

    def __init__(self) -> None:
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        if self._workflows is None:
            self._workflows = {
                "eda_workflow": eda_workflow(),
                "data_cleaning": data_cleaning_workflow(),
                "statistical_analysis": statistical_analysis_workflow(),
                "ml_pipeline": ml_pipeline_workflow(),
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
            (r"explor(e|atory)\s+data", "eda_workflow"),
            (r"eda\b", "eda_workflow"),
            (r"data\s+profil", "eda_workflow"),
            (r"clean\s+(the\s+)?data", "data_cleaning"),
            (r"data\s+clean", "data_cleaning"),
            (r"handle\s+missing", "data_cleaning"),
            (r"statistic(al)?\s+analysis", "statistical_analysis"),
            (r"hypothesis\s+test", "statistical_analysis"),
            (r"correlation\s+analysis", "statistical_analysis"),
            (r"machine\s+learning", "ml_pipeline"),
            (r"ml\s+model", "ml_pipeline"),
            (r"train\s+(a\s+)?model", "ml_pipeline"),
            (r"predict", "ml_pipeline"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Get appropriate workflow for task type."""
        mapping = {
            "eda": "eda_workflow",
            "exploration": "eda_workflow",
            "profiling": "eda_workflow",
            "cleaning": "data_cleaning",
            "preparation": "data_cleaning",
            "statistics": "statistical_analysis",
            "hypothesis": "statistical_analysis",
            "ml": "ml_pipeline",
            "training": "ml_pipeline",
            "prediction": "ml_pipeline",
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
            provider = DataAnalysisWorkflowProvider()
            async for chunk in provider.astream("eda_workflow", orchestrator, {}):
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
        return f"DataAnalysisWorkflowProvider(workflows={len(self._load_workflows())})"


from victor.dataanalysis.workflows.graph_workflows import (
    EDAState,
    CleaningState,
    MLPipelineState,
    create_eda_workflow,
    create_cleaning_workflow,
    create_ml_pipeline_workflow,
    DataAnalysisGraphExecutor,
)

__all__ = [
    # WorkflowBuilder-based workflows
    "DataAnalysisWorkflowProvider",
    "eda_workflow",
    "data_cleaning_workflow",
    "statistical_analysis_workflow",
    "ml_pipeline_workflow",
    # StateGraph-based workflows
    "EDAState",
    "CleaningState",
    "MLPipelineState",
    "create_eda_workflow",
    "create_cleaning_workflow",
    "create_ml_pipeline_workflow",
    "DataAnalysisGraphExecutor",
]
