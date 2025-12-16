"""DevOps Mode Configuration - Budget and iteration settings for infrastructure tasks."""

from dataclasses import dataclass
from typing import Dict, Optional

from victor.verticals.protocols import ModeConfigProviderProtocol


@dataclass
class DevOpsModeConfig:
    """Configuration for a DevOps mode."""
    tool_budget: int
    max_iterations: int
    description: str
    allowed_stages: list[str]


# DevOps-specific mode configurations
DEVOPS_MODE_CONFIGS: Dict[str, DevOpsModeConfig] = {
    "quick": DevOpsModeConfig(
        tool_budget=8,
        max_iterations=15,
        description="Quick configuration tweaks and small changes",
        allowed_stages=["INITIAL", "IMPLEMENTATION", "COMPLETION"],
    ),
    "standard": DevOpsModeConfig(
        tool_budget=20,
        max_iterations=40,
        description="Standard infrastructure changes with validation",
        allowed_stages=["INITIAL", "ASSESSMENT", "IMPLEMENTATION", "VALIDATION", "COMPLETION"],
    ),
    "comprehensive": DevOpsModeConfig(
        tool_budget=40,
        max_iterations=80,
        description="Full infrastructure setup with all stages",
        allowed_stages=["INITIAL", "ASSESSMENT", "PLANNING", "IMPLEMENTATION", "VALIDATION", "DEPLOYMENT", "MONITORING", "COMPLETION"],
    ),
    "migration": DevOpsModeConfig(
        tool_budget=60,
        max_iterations=120,
        description="Large-scale infrastructure migrations",
        allowed_stages=["INITIAL", "ASSESSMENT", "PLANNING", "IMPLEMENTATION", "VALIDATION", "DEPLOYMENT", "MONITORING", "COMPLETION"],
    ),
}

# Default tool budgets by task complexity
DEVOPS_DEFAULT_TOOL_BUDGETS: Dict[str, int] = {
    "dockerfile_simple": 5,
    "dockerfile_complex": 10,
    "docker_compose": 12,
    "ci_cd_basic": 15,
    "ci_cd_advanced": 25,
    "kubernetes_manifest": 15,
    "kubernetes_helm": 25,
    "terraform_module": 20,
    "terraform_full": 40,
    "monitoring_setup": 20,
}


class DevOpsModeConfigProvider(ModeConfigProviderProtocol):
    """Provides mode configurations for DevOps tasks."""

    def get_mode_configs(self) -> Dict[str, Dict]:
        """Return available DevOps modes as dictionaries."""
        return {
            name: {
                "tool_budget": config.tool_budget,
                "max_iterations": config.max_iterations,
                "description": config.description,
                "allowed_stages": config.allowed_stages,
            }
            for name, config in DEVOPS_MODE_CONFIGS.items()
        }

    def get_default_tool_budget(self, task_type: Optional[str] = None) -> int:
        """Return default tool budget for a task type."""
        if task_type and task_type in DEVOPS_DEFAULT_TOOL_BUDGETS:
            return DEVOPS_DEFAULT_TOOL_BUDGETS[task_type]
        return 20  # Standard DevOps budget

    def get_default_max_iterations(self, task_type: Optional[str] = None) -> int:
        """Return default max iterations for a task type."""
        budget = self.get_default_tool_budget(task_type)
        # DevOps typically needs 2x iterations per tool call
        return budget * 2

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to DevOps mode."""
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "standard",
            "complex": "comprehensive",
            "highly_complex": "migration",
        }
        return mapping.get(complexity, "standard")
