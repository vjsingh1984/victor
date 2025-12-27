"""Research Mode Configuration - Budget and iteration settings for research tasks."""

from dataclasses import dataclass
from typing import Dict, Optional

from victor.verticals.protocols import ModeConfigProviderProtocol


@dataclass
class ResearchModeConfig:
    """Configuration for a research mode."""

    tool_budget: int
    max_iterations: int
    description: str
    allowed_stages: list[str]


# Research-specific mode configurations
RESEARCH_MODE_CONFIGS: Dict[str, ResearchModeConfig] = {
    "quick": ResearchModeConfig(
        tool_budget=5,
        max_iterations=10,
        description="Fast lookup for simple factual queries",
        allowed_stages=["INITIAL", "SEARCHING", "COMPLETION"],
    ),
    "standard": ResearchModeConfig(
        tool_budget=15,
        max_iterations=30,
        description="Balanced research with source verification",
        allowed_stages=["INITIAL", "SEARCHING", "READING", "SYNTHESIZING", "COMPLETION"],
    ),
    "deep": ResearchModeConfig(
        tool_budget=30,
        max_iterations=60,
        description="Comprehensive research with full verification cycle",
        allowed_stages=[
            "INITIAL",
            "SEARCHING",
            "READING",
            "SYNTHESIZING",
            "WRITING",
            "VERIFICATION",
            "COMPLETION",
        ],
    ),
    "academic": ResearchModeConfig(
        tool_budget=50,
        max_iterations=100,
        description="Thorough literature review with extensive citation",
        allowed_stages=[
            "INITIAL",
            "SEARCHING",
            "READING",
            "SYNTHESIZING",
            "WRITING",
            "VERIFICATION",
            "COMPLETION",
        ],
    ),
}

# Default tool budgets by task complexity
RESEARCH_DEFAULT_TOOL_BUDGETS: Dict[str, int] = {
    "simple_lookup": 3,
    "fact_check": 8,
    "comparison": 12,
    "trend_analysis": 20,
    "literature_review": 40,
    "comprehensive_report": 50,
}


class ResearchModeConfigProvider(ModeConfigProviderProtocol):
    """Provides mode configurations for research tasks."""

    def get_mode_configs(self) -> Dict[str, Dict]:
        """Return available research modes as dictionaries."""
        return {
            name: {
                "tool_budget": config.tool_budget,
                "max_iterations": config.max_iterations,
                "description": config.description,
                "allowed_stages": config.allowed_stages,
            }
            for name, config in RESEARCH_MODE_CONFIGS.items()
        }

    def get_default_tool_budget(self, task_type: Optional[str] = None) -> int:
        """Return default tool budget for a task type."""
        if task_type and task_type in RESEARCH_DEFAULT_TOOL_BUDGETS:
            return RESEARCH_DEFAULT_TOOL_BUDGETS[task_type]
        return 15  # Standard research budget

    def get_default_max_iterations(self, task_type: Optional[str] = None) -> int:
        """Return default max iterations for a task type."""
        budget = self.get_default_tool_budget(task_type)
        # Research typically needs 2x iterations per tool call
        return budget * 2

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to research mode."""
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "standard",
            "complex": "deep",
            "highly_complex": "academic",
        }
        return mapping.get(complexity, "standard")
