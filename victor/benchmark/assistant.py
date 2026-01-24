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

"""BenchmarkVertical - Domain-specific vertical for AI coding benchmarks.

This vertical demonstrates ISP-compliant protocol registration, where only the
protocols actually implemented by the vertical are registered, rather than
inheriting from all possible protocol interfaces.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.core.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.tools.tool_names import ToolNames

# Import ISP-compliant provider protocols
from victor.core.verticals.protocols.providers import (
    HandlerProvider,
    ModeConfigProvider,
    TieredToolConfigProvider,
    ToolProvider,
    WorkflowProvider,
)

# Phase 3: Import framework capabilities
from victor.framework.capabilities import FileOperationsCapability

# Phase 2.1: Protocol auto-registration decorator
from victor.core.verticals.protocol_decorators import register_protocols

if TYPE_CHECKING:
    from victor.core.verticals.protocols import ModeConfigProviderProtocol
    from victor.core.vertical_types import TieredToolConfig


@register_protocols
class BenchmarkVertical(VerticalBase):
    """Vertical for AI coding benchmark evaluation.

    This vertical configures agents for evaluating coding tasks from benchmarks
    like SWE-bench, HumanEval, and MBPP. It provides:

    - Optimized tool selection for code analysis and modification
    - Benchmark-specific stages (UNDERSTANDING → ANALYSIS → IMPLEMENTATION → VERIFICATION)
    - Workflows for common benchmark patterns
    - Metrics collection through framework observability

    ISP Compliance:
        This vertical explicitly declares which protocols it implements through
        protocol registration, rather than inheriting from all possible protocol
        interfaces. This follows the Interface Segregation Principle (ISP) by
        implementing only needed protocols.

        Implemented Protocols:
        - ToolProvider: Provides tools optimized for benchmark task execution
        - WorkflowProvider: Provides YAML-based workflows (swe_bench, passk, etc.)
        - ModeConfigProvider: Provides mode configurations (fast, default, thorough)
        - HandlerProvider: Provides benchmark compute handlers
        - TieredToolConfigProvider: Provides tiered tool configuration
    """

    name = "benchmark"
    description = "AI coding benchmark evaluation and performance testing"
    version = "1.0.0"

    # Phase 3: Framework file operations capability (read, write, edit, grep)
    _file_ops = FileOperationsCapability()

    # Benchmark-specific stages
    STAGE_UNDERSTANDING = "UNDERSTANDING"
    STAGE_ANALYSIS = "ANALYSIS"
    STAGE_IMPLEMENTATION = "IMPLEMENTATION"
    STAGE_VERIFICATION = "VERIFICATION"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Tools optimized for benchmark task execution.

        Phase 3: Uses framework FileOperationsCapability for common file operations
        to reduce code duplication and maintain consistency across verticals.

        Returns a curated set of tools for:
        - Code reading and understanding
        - Code search and navigation
        - Code modification
        - Test execution and verification

        This method is part of the ToolProvider protocol.
        """
        # Start with framework file operations (read, write, edit, grep)
        tools = cls._file_ops.get_tool_list()

        # Add benchmark-specific tools
        tools.extend(
            [
                # Core reading/navigation (always needed)
                ToolNames.LS,
                # Code-specific tools
                ToolNames.CODE_SEARCH,  # Semantic code search
                ToolNames.SYMBOL,  # Find symbol definitions
                ToolNames.REFS,  # Find references
                # Modification tools
                ToolNames.EDIT,
                # Execution/verification
                ToolNames.SHELL,
                ToolNames.TEST,
                # Git for patch generation (unified git tool)
                ToolNames.GIT,
                ToolNames.DIFF,  # Create patches
            ]
        )

        return tools

    @classmethod
    def get_system_prompt(cls) -> str:
        """System prompt optimized for benchmark task execution."""
        return """You are an expert software engineer solving coding tasks from benchmarks.

Your approach:
1. UNDERSTAND the problem completely before making changes
2. ANALYZE the codebase to find relevant files and understand context
3. IMPLEMENT the solution with minimal, focused changes
4. VERIFY your changes work correctly

Guidelines:
- Read and understand the issue/task description carefully
- Search the codebase to find all relevant files
- Make the smallest change that solves the problem
- Follow the existing code style and patterns
- Generate clean, applicable patches when requested
- Verify your solution by running tests if available

You are being evaluated on:
- Correctness: Does your solution pass the tests?
- Efficiency: Did you find the right files quickly?
- Quality: Is your code clean and maintainable?
"""

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Benchmark-specific conversation stages.

        Stages map to benchmark task phases and influence tool selection.
        """
        return {
            cls.STAGE_UNDERSTANDING: StageDefinition(
                name=cls.STAGE_UNDERSTANDING,
                description="Understanding the task/issue",
                tools={ToolNames.READ, ToolNames.GREP, ToolNames.LS},
                keywords=[
                    "understand",
                    "issue",
                    "problem",
                    "task",
                    "bug",
                    "error",
                    "failing",
                ],
                next_stages={cls.STAGE_ANALYSIS},
            ),
            cls.STAGE_ANALYSIS: StageDefinition(
                name=cls.STAGE_ANALYSIS,
                description="Analyzing codebase for relevant files",
                tools={
                    ToolNames.CODE_SEARCH,
                    ToolNames.SYMBOL,
                    ToolNames.REFS,
                    ToolNames.GREP,
                },
                keywords=[
                    "find",
                    "search",
                    "locate",
                    "where",
                    "definition",
                    "references",
                    "usage",
                ],
                next_stages={cls.STAGE_IMPLEMENTATION},
            ),
            cls.STAGE_IMPLEMENTATION: StageDefinition(
                name=cls.STAGE_IMPLEMENTATION,
                description="Implementing the solution",
                tools={ToolNames.EDIT, ToolNames.WRITE, ToolNames.READ},
                keywords=[
                    "fix",
                    "change",
                    "modify",
                    "update",
                    "add",
                    "remove",
                    "implement",
                    "patch",
                ],
                next_stages={cls.STAGE_VERIFICATION},
            ),
            cls.STAGE_VERIFICATION: StageDefinition(
                name=cls.STAGE_VERIFICATION,
                description="Verifying the solution",
                tools={ToolNames.TEST, ToolNames.SHELL, ToolNames.GIT},
                keywords=[
                    "test",
                    "verify",
                    "check",
                    "run",
                    "validate",
                    "confirm",
                ],
                next_stages=set(),  # Terminal stage
            ),
        }

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Hints for LLM provider configuration."""
        return {
            "temperature": 0.2,  # Lower temperature for deterministic coding
            "max_tokens": 4096,  # Sufficient for code generation
            "stop_sequences": [],  # Let model complete naturally
        }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Evaluation criteria for benchmark tasks."""
        return [
            "Correctness: Solution passes all provided tests",
            "Efficiency: Minimal tool calls to find solution",
            "Quality: Clean, maintainable code following project style",
            "Completeness: All aspects of the issue are addressed",
        ]

    @classmethod
    def get_tiered_tool_config(cls) -> Optional["TieredToolConfig"]:
        """Tiered tool configuration for benchmark evaluation.

        Returns tool tiers optimized for benchmark task execution:
        - Mandatory: Always available (read, ls, grep)
        - Core: Vertical-specific essentials (code_search, edit)
        - Semantic: Selected based on task similarity

        Returns:
            TieredToolConfig with proper tool tiers for benchmark evaluation
        """
        from victor.core.vertical_types import TieredToolConfig

        return TieredToolConfig(
            mandatory={
                ToolNames.READ,
                ToolNames.LS,
                ToolNames.GREP,
            },
            vertical_core={
                ToolNames.CODE_SEARCH,
                ToolNames.EDIT,
                ToolNames.WRITE,
                ToolNames.SHELL,
            },
            semantic_pool={
                ToolNames.SYMBOL,
                ToolNames.REFS,
                ToolNames.TEST,
                ToolNames.GIT,
                ToolNames.DIFF,
            },
            readonly_only_for_analysis=False,
        )

    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
        """Apply benchmark-specific configuration tweaks."""
        # Add benchmark-specific metadata
        config.metadata["tool_budget"] = 30  # Recommended for complex benchmark tasks
        config.metadata["thinking"] = True  # Enable extended thinking for complex reasoning

        return config

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get benchmark-specific workflow provider.

        Returns the BenchmarkWorkflowProvider which provides YAML-based workflows
        for AI coding benchmarks:
        - swe_bench: SWE-bench style issue resolution
        - code_generation: HumanEval/MBPP function generation
        - passk_generation: Multi-attempt pass@k evaluation
        - live_code_bench: Live code execution benchmark
        - big_code_bench: Large-scale code understanding
        - aider_polyglot: Multi-language code modification

        This method is part of the WorkflowProvider protocol.

        Returns:
            BenchmarkWorkflowProvider instance
        """
        from victor.benchmark.workflows import BenchmarkWorkflowProvider

        return BenchmarkWorkflowProvider()

    @classmethod
    def get_workflows(cls) -> Dict[str, Any]:
        """Get benchmark workflow definitions.

        Returns available workflows from BenchmarkWorkflowProvider.

        This method is part of the WorkflowProvider protocol.

        Returns:
            Dict mapping workflow names to workflow definitions
        """
        provider = cls.get_workflow_provider()
        return provider.get_workflows() if provider else {}

    @classmethod
    def get_mode_config_provider(cls) -> Optional["ModeConfigProviderProtocol"]:
        """Get benchmark-specific mode configuration provider.

        Returns a provider with modes optimized for benchmark evaluation:
        - fast: Quick evaluation with lower tool budget (15) and fewer turns (8)
        - default: Balanced settings (30 tool_budget, 15 max_turns)
        - thorough: Comprehensive analysis (50 tool_budget, 20 max_turns)

        Returns:
            BenchmarkModeConfigProvider instance
        """
        from victor.benchmark.mode_config import BenchmarkModeConfigProvider

        return BenchmarkModeConfigProvider()

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get benchmark compute handlers for workflow execution.

        Provides handlers for benchmark-specific workflow nodes like:
        - passk_generation: Multi-attempt code generation with pass@k metrics
        - git_diff_generator: Create git diffs from code changes

        This method is part of the HandlerProvider protocol.

        Returns:
            Dict mapping handler name to handler instance
        """
        try:
            from victor.benchmark.handlers import HANDLERS

            return HANDLERS
        except ImportError:
            # Handlers not yet implemented
            return {}

    # NOTE: The following getters are auto-generated by VerticalExtensionLoaderMeta:
    # - get_safety_extension()
    # - get_prompt_contributor()
    # - get_team_spec_provider()
    # - get_capability_provider()
    # - get_enrichment_strategy()
    # - get_middleware()
    # - get_rl_config_provider()
    # - get_rl_hooks()
    #
    # get_extensions() is inherited from VerticalBase with full caching support.
    # To clear all caches, use cls.clear_config_cache().


# Protocol registration is now handled by @register_protocols decorator
# which auto-detects implemented protocols:
# - ToolProvider (get_tools)
# - WorkflowProvider (get_workflows)
# - ModeConfigProvider (get_mode_config_provider)
# - HandlerProvider (get_handlers)
# - TieredToolConfigProvider (get_tiered_tool_config)
#
# ISP Compliance Note:
# This vertical implements only the protocols it needs. The @register_protocols
# decorator auto-detects and registers these protocols at class decoration time.
