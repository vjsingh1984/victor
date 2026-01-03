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

Uses YAML-first architecture with Python escape hatches for complex conditions
and transforms that cannot be expressed in YAML.

Example:
    provider = DataAnalysisWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("eda_workflow", orchestrator, context):
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
            print(f"Completed: {chunk.node_name}")

Available workflows (all YAML-defined):
- eda_pipeline: Full EDA with parallel statistics and visualizations
- eda_quick: Lightweight EDA for quick analysis
- data_cleaning: Systematic cleaning with validation loop
- data_cleaning_quick: Automated cleaning without human review
- statistical_analysis: Hypothesis testing and statistical modeling
- ml_pipeline: End-to-end ML pipeline with hyperparameter tuning
- ml_quick: Quick baseline model training
"""

from typing import List, Optional, Tuple

from victor.framework.workflows import BaseYAMLWorkflowProvider


class DataAnalysisWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides data analysis-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which handles:
    - YAML workflow loading and caching
    - Escape hatches registration
    - Standard and streaming executor creation
    - Workflow retrieval methods

    Example:
        provider = DataAnalysisWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Stream ML pipeline execution
        async for chunk in provider.astream("ml_pipeline", orchestrator, {}):
            print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
    """

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for DataAnalysis escape hatches.

        Returns:
            Fully qualified module path to escape_hatches.py
        """
        return "victor.dataanalysis.escape_hatches"

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns.

        Returns:
            List of (regex_pattern, workflow_name) tuples for auto-triggering
        """
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
        """Get appropriate workflow for task type.

        Args:
            task_type: Type of data analysis task

        Returns:
            Workflow name string or None if no mapping exists
        """
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


# Register DataAnalysis domain handlers when this module is loaded
from victor.dataanalysis.handlers import register_handlers as _register_handlers

_register_handlers()

__all__ = [
    # YAML-first workflow provider
    "DataAnalysisWorkflowProvider",
]
