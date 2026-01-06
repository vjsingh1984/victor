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

"""Benchmark mode configurations using central registry.

This module registers benchmark-specific operational modes with the central
ModeConfigRegistry and exports a registry-based provider for protocol
compatibility.

Benchmark modes are optimized for evaluation scenarios:
- fast: Quick evaluation with lower tool budget for rapid iteration
- default: Balanced settings for standard benchmark tasks
- thorough: Comprehensive analysis with higher budgets for complex tasks
"""

from __future__ import annotations

from typing import Dict

from victor.core.mode_config import (
    ModeConfig,
    ModeConfigRegistry,
    ModeDefinition,
    RegistryBasedModeConfigProvider,
)


# =============================================================================
# Benchmark-Specific Modes (Registered with Central Registry)
# =============================================================================

_BENCHMARK_MODES: Dict[str, ModeDefinition] = {
    "fast": ModeDefinition(
        name="fast",
        tool_budget=15,
        max_iterations=8,
        temperature=0.2,
        description="Quick evaluation with minimal tool calls",
        exploration_multiplier=0.5,
        priority_tools=[
            "read",
            "grep",
            "edit",
            "shell",
        ],
        metadata={
            "timeout_seconds": 120,
            "quality_threshold": 0.6,
            "max_file_reads": 10,
        },
    ),
    "default": ModeDefinition(
        name="default",
        tool_budget=30,
        max_iterations=15,
        temperature=0.2,
        description="Balanced settings for standard benchmark tasks",
        exploration_multiplier=1.0,
        priority_tools=[
            "read",
            "grep",
            "code_search",
            "edit",
            "shell",
            "test",
        ],
        metadata={
            "timeout_seconds": 300,
            "quality_threshold": 0.75,
            "max_file_reads": 25,
        },
    ),
    "thorough": ModeDefinition(
        name="thorough",
        tool_budget=50,
        max_iterations=20,
        temperature=0.3,
        description="Comprehensive analysis for complex benchmark tasks",
        exploration_multiplier=1.5,
        priority_tools=[
            "read",
            "grep",
            "glob",
            "code_search",
            "ast_search",
            "find_definition",
            "find_references",
            "edit",
            "write",
            "shell",
            "test",
            "git_diff",
        ],
        metadata={
            "timeout_seconds": 600,
            "quality_threshold": 0.9,
            "max_file_reads": 50,
            "enable_verification": True,
        },
    ),
}

# Benchmark-specific task type budgets
_BENCHMARK_TASK_BUDGETS: Dict[str, int] = {
    # SWE-bench style tasks
    "bug_fix": 25,
    "feature_addition": 35,
    "code_refactor": 30,
    # HumanEval/MBPP style tasks
    "function_generation": 10,
    "algorithm_implementation": 15,
    # Analysis tasks
    "codebase_understanding": 20,
    "test_generation": 20,
    # Verification tasks
    "test_execution": 8,
    "patch_verification": 12,
}


# =============================================================================
# Register with Central Registry
# =============================================================================


def _register_benchmark_modes() -> None:
    """Register benchmark modes with the central registry.

    This function is idempotent - safe to call multiple times.
    Called by BenchmarkModeConfigProvider.__init__ when provider is instantiated
    during vertical integration. Module-level auto-registration removed to avoid
    load-order coupling.
    """
    registry = ModeConfigRegistry.get_instance()
    registry.register_vertical(
        name="benchmark",
        modes=_BENCHMARK_MODES,
        task_budgets=_BENCHMARK_TASK_BUDGETS,
        default_mode="default",
        default_budget=30,
    )


# NOTE: Import-time auto-registration removed (SOLID compliance)
# Registration happens when BenchmarkModeConfigProvider is instantiated during
# vertical integration. The provider's __init__ calls _register_benchmark_modes()
# for idempotent registration.


# =============================================================================
# Provider (Protocol Compatibility)
# =============================================================================


class BenchmarkModeConfigProvider(RegistryBasedModeConfigProvider):
    """Mode configuration provider for benchmark vertical.

    Uses the central ModeConfigRegistry but provides benchmark-specific
    complexity mapping and evaluation-optimized configurations.

    Example:
        provider = BenchmarkModeConfigProvider()
        modes = provider.get_mode_configs()
        default_mode = provider.get_default_mode()
    """

    def __init__(self) -> None:
        """Initialize benchmark mode provider."""
        # Ensure registration (idempotent - handles singleton reset)
        _register_benchmark_modes()
        super().__init__(
            vertical="benchmark",
            default_mode="default",
            default_budget=30,
        )

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to benchmark mode.

        Args:
            complexity: Complexity level (trivial, simple, moderate, complex, highly_complex)

        Returns:
            Recommended mode name for benchmark evaluation
        """
        mapping = {
            "trivial": "fast",
            "simple": "fast",
            "moderate": "default",
            "complex": "thorough",
            "highly_complex": "thorough",
        }
        return mapping.get(complexity, "default")

    def get_mode_for_benchmark(self, benchmark_name: str) -> str:
        """Get recommended mode for a specific benchmark.

        Args:
            benchmark_name: Name of the benchmark (e.g., "swe_bench", "humaneval")

        Returns:
            Recommended mode name
        """
        benchmark_modes = {
            "humaneval": "fast",
            "mbpp": "fast",
            "swe_bench": "thorough",
            "swe_bench_lite": "default",
            "apps": "default",
        }
        return benchmark_modes.get(benchmark_name.lower(), "default")

    def get_timeout_for_mode(self, mode_name: str) -> int:
        """Get timeout in seconds for a mode.

        Args:
            mode_name: Name of the mode

        Returns:
            Timeout in seconds
        """
        registry = ModeConfigRegistry.get_instance()
        mode = registry.get_mode("benchmark", mode_name)
        if mode and mode.metadata:
            return mode.metadata.get("timeout_seconds", 300)
        return 300

    def get_quality_threshold(self, mode_name: str) -> float:
        """Get quality threshold for a mode.

        Args:
            mode_name: Name of the mode

        Returns:
            Quality threshold (0.0 to 1.0)
        """
        registry = ModeConfigRegistry.get_instance()
        mode = registry.get_mode("benchmark", mode_name)
        if mode and mode.metadata:
            return mode.metadata.get("quality_threshold", 0.75)
        return 0.75


def get_mode_config(mode_name: str) -> ModeConfig | None:
    """Get a specific mode configuration.

    Args:
        mode_name: Name of the mode

    Returns:
        ModeConfig or None if not found
    """
    registry = ModeConfigRegistry.get_instance()
    modes = registry.get_modes("benchmark")
    return modes.get(mode_name.lower())


def get_tool_budget(mode_name: str | None = None, task_type: str | None = None) -> int:
    """Get tool budget based on mode or task type.

    Args:
        mode_name: Optional mode name
        task_type: Optional task type

    Returns:
        Recommended tool budget
    """
    registry = ModeConfigRegistry.get_instance()
    return registry.get_tool_budget(
        vertical="benchmark",
        mode_name=mode_name,
        task_type=task_type,
    )


__all__ = [
    "BenchmarkModeConfigProvider",
    "get_mode_config",
    "get_tool_budget",
]
