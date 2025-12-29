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

"""StateGraph-based workflows for Data Analysis vertical.

Provides LangGraph-compatible StateGraph workflows for complex data analysis
tasks that benefit from:
- Typed state management
- Cyclic execution (clean-validate-fix loops)
- Explicit retry limits
- Checkpoint/resume semantics

Example:
    from victor.verticals.data_analysis.workflows.graph_workflows import (
        create_eda_workflow,
        EDAState,
    )

    graph = create_eda_workflow()
    result = await graph.compile().invoke(EDAState(
        data_path="data.csv",
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


class EDAState(TypedDict, total=False):
    """Typed state for EDA workflows."""
    data_path: str
    data_shape: Optional[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]]
    visualizations: Optional[List[str]]
    insights: Optional[List[str]]
    report_path: Optional[str]
    iteration: int


class CleaningState(TypedDict, total=False):
    """Typed state for data cleaning workflows."""
    data_path: str
    quality_issues: Optional[List[Dict[str, Any]]]
    cleaning_plan: Optional[str]
    cleaned_data_path: Optional[str]
    validation_results: Optional[Dict[str, Any]]
    validation_passed: bool
    iteration: int
    max_iterations: int


class MLPipelineState(TypedDict, total=False):
    """Typed state for ML pipeline workflows."""
    data_path: str
    problem_type: str  # classification, regression, clustering
    features: Optional[List[str]]
    model_results: Optional[List[Dict[str, Any]]]
    best_model: Optional[str]
    metrics: Optional[Dict[str, float]]
    iteration: int
    max_iterations: int


# =============================================================================
# Node Functions - EDA
# =============================================================================


async def load_data_node(state: EDAState) -> EDAState:
    """Load data and understand structure."""
    state["data_shape"] = {"rows": 0, "cols": 0, "dtypes": {}}
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def profile_data_node(state: EDAState) -> EDAState:
    """Generate summary statistics."""
    state["statistics"] = {
        "numeric_summary": {},
        "categorical_summary": {},
        "missing_values": {},
    }
    return state


async def create_visualizations_node(state: EDAState) -> EDAState:
    """Create visualizations for key findings."""
    state["visualizations"] = []
    return state


async def synthesize_insights_node(state: EDAState) -> EDAState:
    """Synthesize key insights from analysis."""
    state["insights"] = []
    return state


async def generate_eda_report_node(state: EDAState) -> EDAState:
    """Generate EDA report."""
    state["report_path"] = "eda_report.html"
    return state


# =============================================================================
# Node Functions - Cleaning
# =============================================================================


async def assess_quality_node(state: CleaningState) -> CleaningState:
    """Assess data quality issues."""
    state["quality_issues"] = []
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def plan_cleaning_node(state: CleaningState) -> CleaningState:
    """Plan cleaning strategy."""
    state["cleaning_plan"] = "Cleaning plan"
    return state


async def execute_cleaning_node(state: CleaningState) -> CleaningState:
    """Execute cleaning transformations."""
    state["cleaned_data_path"] = "cleaned_data.csv"
    return state


async def validate_cleaning_node(state: CleaningState) -> CleaningState:
    """Validate cleaned data."""
    state["validation_results"] = {"checks_passed": 0, "checks_total": 0}
    state["validation_passed"] = True
    return state


# =============================================================================
# Node Functions - ML Pipeline
# =============================================================================


async def define_problem_node(state: MLPipelineState) -> MLPipelineState:
    """Define ML problem and metrics."""
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def engineer_features_node(state: MLPipelineState) -> MLPipelineState:
    """Engineer features from raw data."""
    state["features"] = []
    return state


async def train_models_node(state: MLPipelineState) -> MLPipelineState:
    """Train candidate models."""
    state["model_results"] = []
    return state


async def evaluate_models_node(state: MLPipelineState) -> MLPipelineState:
    """Evaluate and select best model."""
    state["best_model"] = "best_model"
    state["metrics"] = {}
    return state


# =============================================================================
# Condition Functions
# =============================================================================


def should_retry_cleaning(state: CleaningState) -> str:
    """Determine if cleaning should be retried."""
    if state.get("validation_passed", False):
        return "done"
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    if iteration >= max_iter:
        return "done"
    return "retry"


def should_tune_more(state: MLPipelineState) -> str:
    """Determine if more model tuning is needed."""
    metrics = state.get("metrics", {})
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # Check if performance is acceptable
    score = metrics.get("primary_metric", 0)
    if score >= 0.9 or iteration >= max_iter:
        return "done"
    return "tune"


# =============================================================================
# Workflow Factories
# =============================================================================


def create_eda_workflow() -> StateGraph[EDAState]:
    """Create an EDA workflow.

    Implements: Load -> Profile -> Visualize -> Synthesize -> Report

    Returns:
        StateGraph for EDA
    """
    graph = StateGraph(EDAState)

    graph.add_node("load", load_data_node)
    graph.add_node("profile", profile_data_node)
    graph.add_node("visualize", create_visualizations_node)
    graph.add_node("synthesize", synthesize_insights_node)
    graph.add_node("report", generate_eda_report_node)

    graph.add_edge("load", "profile")
    graph.add_edge("profile", "visualize")
    graph.add_edge("visualize", "synthesize")
    graph.add_edge("synthesize", "report")
    graph.add_edge("report", END)

    graph.set_entry_point("load")
    return graph


def create_cleaning_workflow() -> StateGraph[CleaningState]:
    """Create a data cleaning workflow with validation loop.

    Implements:
    1. Assess -> Plan -> Clean -> Validate
    2. If validation fails, retry (up to max_iterations)

    Returns:
        StateGraph for data cleaning
    """
    graph = StateGraph(CleaningState)

    graph.add_node("assess", assess_quality_node)
    graph.add_node("plan", plan_cleaning_node)
    graph.add_node("clean", execute_cleaning_node)
    graph.add_node("validate", validate_cleaning_node)

    graph.add_edge("assess", "plan")
    graph.add_edge("plan", "clean")
    graph.add_edge("clean", "validate")

    graph.add_conditional_edge(
        "validate",
        should_retry_cleaning,
        {"retry": "assess", "done": END},
    )

    graph.set_entry_point("assess")
    return graph


def create_ml_pipeline_workflow() -> StateGraph[MLPipelineState]:
    """Create an ML pipeline workflow with tuning loop.

    Implements:
    1. Define -> Engineer -> Train -> Evaluate
    2. If metrics insufficient, tune more (up to max_iterations)

    Returns:
        StateGraph for ML pipeline
    """
    graph = StateGraph(MLPipelineState)

    graph.add_node("define", define_problem_node)
    graph.add_node("engineer", engineer_features_node)
    graph.add_node("train", train_models_node)
    graph.add_node("evaluate", evaluate_models_node)

    graph.add_edge("define", "engineer")
    graph.add_edge("engineer", "train")
    graph.add_edge("train", "evaluate")

    graph.add_conditional_edge(
        "evaluate",
        should_tune_more,
        {"tune": "train", "done": END},
    )

    graph.set_entry_point("define")
    return graph


class DataAnalysisGraphExecutor:
    """Executor that integrates StateGraph with AgentOrchestrator for Data Analysis."""

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
    "EDAState",
    "CleaningState",
    "MLPipelineState",
    "create_eda_workflow",
    "create_cleaning_workflow",
    "create_ml_pipeline_workflow",
    "DataAnalysisGraphExecutor",
]
