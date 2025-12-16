"""Data Analysis Mode Configuration - Budget settings for analysis tasks."""

from dataclasses import dataclass
from typing import Dict, Optional

from victor.verticals.protocols import ModeConfigProviderProtocol


@dataclass
class DataAnalysisModeConfig:
    """Configuration for a data analysis mode."""
    tool_budget: int
    max_iterations: int
    description: str
    allowed_stages: list[str]


# Data analysis-specific mode configurations
DATA_ANALYSIS_MODE_CONFIGS: Dict[str, DataAnalysisModeConfig] = {
    "quick": DataAnalysisModeConfig(
        tool_budget=10,
        max_iterations=20,
        description="Quick data overview and basic stats",
        allowed_stages=["INITIAL", "DATA_LOADING", "EXPLORATION", "COMPLETION"],
    ),
    "standard": DataAnalysisModeConfig(
        tool_budget=25,
        max_iterations=50,
        description="Standard analysis with cleaning and visualization",
        allowed_stages=["INITIAL", "DATA_LOADING", "EXPLORATION", "CLEANING", "ANALYSIS", "VISUALIZATION", "COMPLETION"],
    ),
    "comprehensive": DataAnalysisModeConfig(
        tool_budget=50,
        max_iterations=100,
        description="Full analysis pipeline with ML and reporting",
        allowed_stages=["INITIAL", "DATA_LOADING", "EXPLORATION", "CLEANING", "ANALYSIS", "VISUALIZATION", "REPORTING", "COMPLETION"],
    ),
    "research": DataAnalysisModeConfig(
        tool_budget=80,
        max_iterations=150,
        description="Deep research analysis with multiple iterations",
        allowed_stages=["INITIAL", "DATA_LOADING", "EXPLORATION", "CLEANING", "ANALYSIS", "VISUALIZATION", "REPORTING", "COMPLETION"],
    ),
}

# Default tool budgets by analysis complexity
DATA_ANALYSIS_DEFAULT_TOOL_BUDGETS: Dict[str, int] = {
    "data_profiling": 8,
    "basic_stats": 10,
    "visualization": 12,
    "correlation": 15,
    "regression": 20,
    "clustering": 20,
    "time_series": 25,
    "ml_pipeline": 40,
    "full_report": 50,
}


class DataAnalysisModeConfigProvider(ModeConfigProviderProtocol):
    """Provides mode configurations for data analysis tasks."""

    def get_mode_configs(self) -> Dict[str, Dict]:
        """Return available analysis modes as dictionaries."""
        return {
            name: {
                "tool_budget": config.tool_budget,
                "max_iterations": config.max_iterations,
                "description": config.description,
                "allowed_stages": config.allowed_stages,
            }
            for name, config in DATA_ANALYSIS_MODE_CONFIGS.items()
        }

    def get_default_tool_budget(self, task_type: Optional[str] = None) -> int:
        """Return default tool budget for a task type."""
        if task_type and task_type in DATA_ANALYSIS_DEFAULT_TOOL_BUDGETS:
            return DATA_ANALYSIS_DEFAULT_TOOL_BUDGETS[task_type]
        return 25  # Standard analysis budget

    def get_default_max_iterations(self, task_type: Optional[str] = None) -> int:
        """Return default max iterations for a task type."""
        budget = self.get_default_tool_budget(task_type)
        # Analysis typically needs 2x iterations per tool call
        return budget * 2

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to analysis mode."""
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "standard",
            "complex": "comprehensive",
            "highly_complex": "research",
        }
        return mapping.get(complexity, "standard")
