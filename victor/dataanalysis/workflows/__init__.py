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

    # Compile and execute (recommended - uses UnifiedWorkflowCompiler with caching)
    result = await provider.run_compiled_workflow("eda_pipeline", {"data_path": "data.csv"})

    # Stream execution with real-time progress
    async for node_id, state in provider.stream_compiled_workflow("eda_pipeline", context):
        print(f"Completed: {node_id}")

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

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading with two-level caching
    - UnifiedWorkflowCompiler integration for consistent execution
    - Checkpointing support for resumable analysis pipelines

    Example:
        provider = DataAnalysisWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Execute with caching (recommended)
        result = await provider.run_compiled_workflow("ml_pipeline", {"data": "train.csv"})

        # Stream with real-time progress
        async for node_id, state in provider.stream_compiled_workflow("ml_pipeline", {}):
            print(f"Completed: {node_id}")
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
