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

"""Teams integration for Data Analysis vertical."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from victor.framework.teams import TeamFormation, TeamMemberSpec


@dataclass
class DataAnalysisTeamSpec:
    """Specification for a data analysis team."""

    name: str
    description: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50


DATA_ANALYSIS_TEAM_SPECS: Dict[str, DataAnalysisTeamSpec] = {
    "eda_team": DataAnalysisTeamSpec(
        name="EDA Team",
        description="Exploratory Data Analysis team",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Load and understand data structure",
                name="Data Loader",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="analyst",
                goal="Generate summary statistics and profiles",
                name="Data Profiler",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Create visualizations",
                name="Visualizer",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="writer",
                goal="Summarize insights",
                name="Insight Writer",
                tool_budget=15,
            ),
        ],
        total_tool_budget=80,
    ),
    "cleaning_team": DataAnalysisTeamSpec(
        name="Data Cleaning Team",
        description="Data quality and preparation team",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="analyst",
                goal="Assess data quality issues",
                name="Quality Assessor",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="planner",
                goal="Plan cleaning strategy",
                name="Cleaning Planner",
                tool_budget=10,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Execute cleaning transformations",
                name="Data Cleaner",
                tool_budget=30,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Validate cleaned data",
                name="Quality Validator",
                tool_budget=15,
            ),
        ],
        total_tool_budget=70,
    ),
    "statistics_team": DataAnalysisTeamSpec(
        name="Statistical Analysis Team",
        description="Hypothesis testing and statistical modeling",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Formulate hypotheses and research design",
                name="Research Designer",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Prepare data and run tests",
                name="Statistical Analyst",
                tool_budget=35,
            ),
            TeamMemberSpec(
                role="analyst",
                goal="Interpret results",
                name="Result Interpreter",
                tool_budget=20,
            ),
            TeamMemberSpec(
                role="writer",
                goal="Write statistical report",
                name="Report Writer",
                tool_budget=15,
            ),
        ],
        total_tool_budget=85,
    ),
    "ml_team": DataAnalysisTeamSpec(
        name="Machine Learning Team",
        description="End-to-end ML pipeline team",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Understand problem and define metrics",
                name="Problem Definer",
                tool_budget=10,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Engineer features",
                name="Feature Engineer",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Train and tune models",
                name="Model Trainer",
                tool_budget=35,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Evaluate and select models",
                name="Model Evaluator",
                tool_budget=20,
            ),
        ],
        total_tool_budget=90,
    ),
    "visualization_team": DataAnalysisTeamSpec(
        name="Visualization Team",
        description="Create charts and dashboards",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="analyst",
                goal="Identify key visualizations needed",
                name="Viz Analyst",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Create charts and plots",
                name="Chart Creator",
                tool_budget=35,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Review for clarity and accuracy",
                name="Viz Reviewer",
                tool_budget=15,
            ),
        ],
        total_tool_budget=65,
    ),
}


def get_team_for_task(task_type: str) -> Optional[DataAnalysisTeamSpec]:
    """Get appropriate team for task type."""
    mapping = {
        "eda": "eda_team",
        "exploration": "eda_team",
        "exploratory": "eda_team",
        "profiling": "eda_team",
        "clean": "cleaning_team",
        "cleaning": "cleaning_team",
        "preparation": "cleaning_team",
        "prepare": "cleaning_team",
        "statistics": "statistics_team",
        "statistical": "statistics_team",
        "hypothesis": "statistics_team",
        "test": "statistics_team",
        "ml": "ml_team",
        "machine_learning": "ml_team",
        "training": "ml_team",
        "model": "ml_team",
        "visualization": "visualization_team",
        "visualize": "visualization_team",
        "chart": "visualization_team",
        "plot": "visualization_team",
    }
    spec_name = mapping.get(task_type.lower())
    if spec_name:
        return DATA_ANALYSIS_TEAM_SPECS.get(spec_name)
    return None


def list_team_types() -> List[str]:
    """List available team types."""
    return list(DATA_ANALYSIS_TEAM_SPECS.keys())


__all__ = [
    "DataAnalysisTeamSpec",
    "DATA_ANALYSIS_TEAM_SPECS",
    "get_team_for_task",
    "list_team_types",
]
