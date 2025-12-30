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

"""Evaluation harness for benchmark testing.

This module provides infrastructure for evaluating coding agents against
standardized benchmarks like SWE-bench, HumanEval, MBPP, etc.

Example usage:
    from victor.evaluation import (
        EvaluationHarness,
        EvaluationConfig,
        BenchmarkType,
        SWEBenchRunner,
        get_harness,
    )
    import asyncio

    async def run_agent(task):
        '''Your agent implementation.'''
        # ... agent logic ...
        return agent_output

    async def evaluate():
        # Get harness and register runner
        harness = get_harness()
        harness.register_runner(SWEBenchRunner())

        # Configure evaluation
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH,
            model="claude-3-opus",
            max_tasks=10,
            timeout_per_task=300,
        )

        # Run evaluation
        result = await harness.run_evaluation(config, run_agent)

        # Generate report
        print(harness.generate_report(result, format="text"))
        print(f"Pass rate: {result.pass_rate:.1%}")

    asyncio.run(evaluate())
"""

from victor.evaluation.protocol import (
    BenchmarkMetadata,
    BenchmarkTask,
    BenchmarkType,
    CodeQualityMetrics,
    EvaluationConfig,
    EvaluationMetric,
    EvaluationResult,
    LeaderboardEntry,
    TaskResult,
    TaskStatus,
)
from victor.evaluation.harness import (
    BaseBenchmarkRunner,
    BenchmarkRunner,
    EvaluationHarness,
    TaskEnvironment,
    get_harness,
)
from victor.evaluation.benchmarks import (
    HumanEvalRunner,
    MBPPRunner,
    SWEBenchRunner,
)
from victor.evaluation.code_quality import (
    CodeQualityAnalyzer,
    BatchCodeAnalyzer,
    LintResult,
)
from victor.evaluation.pass_at_k import (
    PassAtKEvaluator,
    PassAtKResult,
    AggregatePassAtKResult,
    GreedyVsSamplingComparison,
    pass_at_k,
    generate_report as generate_pass_at_k_report,
)
from victor.evaluation.analyzers import (
    AnalyzerRegistry,
    get_code_quality_analyzer,
    get_pass_at_k_evaluator,
)

# Code generation harness (HumanEval, MBPP - provider-only, no tools)
from victor.evaluation.code_generation_harness import (
    CodeGenMetrics,
    CodeGenResult,
    CodeGenerationRunner,
    CodeGenerationBenchmark,
    create_code_gen_runner,
)

# Agentic harness (SWE-bench, Aider Polyglot - with tools and file editing)
from victor.evaluation.agentic_harness import (
    AgenticValidationType,
    ToolCall,
    FileEdit,
    AgenticExecutionTrace,
    AgenticTaskResult,
    AgenticMetrics,
    AgenticValidator,
    PatchApplicationValidator,
    TestPassingValidator,
    FileEditValidator,
    ToolUsageValidator,
    AgenticBenchmarkRunner,
    generate_agentic_report,
)

# Agent adapter (connects Victor orchestrator to agentic benchmarks)
from victor.evaluation.agent_adapter import (
    AdapterConfig,
    VictorAgentAdapter,
    create_victor_agent_callback,
    run_agentic_task,
)

# SWE-bench dataset loader and workspace management
from victor.evaluation.swe_bench_loader import (
    SWEBenchConfig,
    SWEBenchInstance,
    SWEBenchLoader,
    SWEBenchWorkspaceManager,
    load_swe_bench_tasks,
    setup_swe_bench_workspace,
    get_swe_bench_repos,
)

# Multi-language test runners for agentic benchmarks
from victor.evaluation.test_runners import (
    Language,
    TestRunnerConfig,
    TestResult,
    TestRunResults,
    BaseTestRunner,
    PythonTestRunner,
    JavaScriptTestRunner,
    GoTestRunner,
    RustTestRunner,
    JavaTestRunner,
    TestRunnerRegistry,
    detect_language,
    is_test_tool_available,
)

# Environment setup utilities
from victor.evaluation.env_setup import (
    SetupStrategy,
    SetupResult,
    EnvironmentConfig,
    EnvironmentSetup,
    validate_environment,
    quick_setup,
)

# Baseline test validation for SWE-bench
from victor.evaluation.baseline_validator import (
    BaselineStatus,
    TestBaseline,
    BaselineValidationResult,
    BaselineCache,
    BaselineValidator,
    quick_validate_baseline,
    check_environment_health,
    TestStatus,
    get_test_status,
)

# Test result correlation and scoring
from victor.evaluation.result_correlation import (
    FailureCategory,
    TestCorrelation,
    SWEBenchScore,
    CorrelationReport,
    ResultCorrelator,
    correlate_validation_results,
    analyze_failure_patterns,
    save_correlation_report,
)

# Evaluation orchestrator for end-to-end SWE-bench evaluation
from victor.evaluation.evaluation_orchestrator import (
    EvaluationStage,
    TaskProgress,
    OrchestratorConfig,
    EvaluationSummary,
    EvaluationOrchestrator,
    run_swe_bench_evaluation,
)

__all__ = [
    # Protocol types
    "BenchmarkMetadata",
    "BenchmarkTask",
    "BenchmarkType",
    "CodeQualityMetrics",
    "EvaluationConfig",
    "EvaluationMetric",
    "EvaluationResult",
    "LeaderboardEntry",
    "TaskResult",
    "TaskStatus",
    # Harness
    "BaseBenchmarkRunner",
    "BenchmarkRunner",
    "EvaluationHarness",
    "TaskEnvironment",
    "get_harness",
    # Benchmark runners
    "HumanEvalRunner",
    "MBPPRunner",
    "SWEBenchRunner",
    # Code quality
    "CodeQualityAnalyzer",
    "BatchCodeAnalyzer",
    "LintResult",
    # Pass@k
    "PassAtKEvaluator",
    "PassAtKResult",
    "AggregatePassAtKResult",
    "GreedyVsSamplingComparison",
    "pass_at_k",
    "generate_pass_at_k_report",
    # Registry
    "AnalyzerRegistry",
    "get_code_quality_analyzer",
    "get_pass_at_k_evaluator",
    # Code generation harness (HumanEval, MBPP - provider-only)
    "CodeGenMetrics",
    "CodeGenResult",
    "CodeGenerationRunner",
    "CodeGenerationBenchmark",
    "create_code_gen_runner",
    # Agentic harness (SWE-bench, Aider Polyglot - with tools)
    "AgenticValidationType",
    "ToolCall",
    "FileEdit",
    "AgenticExecutionTrace",
    "AgenticTaskResult",
    "AgenticMetrics",
    "AgenticValidator",
    "PatchApplicationValidator",
    "TestPassingValidator",
    "FileEditValidator",
    "ToolUsageValidator",
    "AgenticBenchmarkRunner",
    "generate_agentic_report",
    # SWE-bench loader and workspace management
    "SWEBenchConfig",
    "SWEBenchInstance",
    "SWEBenchLoader",
    "SWEBenchWorkspaceManager",
    "load_swe_bench_tasks",
    "setup_swe_bench_workspace",
    "get_swe_bench_repos",
    # Multi-language test runners
    "Language",
    "TestRunnerConfig",
    "TestResult",
    "TestRunResults",
    "BaseTestRunner",
    "PythonTestRunner",
    "JavaScriptTestRunner",
    "GoTestRunner",
    "RustTestRunner",
    "JavaTestRunner",
    "TestRunnerRegistry",
    "detect_language",
    "is_test_tool_available",
    # Environment setup utilities
    "SetupStrategy",
    "SetupResult",
    "EnvironmentConfig",
    "EnvironmentSetup",
    "validate_environment",
    "quick_setup",
    # Baseline test validation
    "BaselineStatus",
    "TestBaseline",
    "BaselineValidationResult",
    "BaselineCache",
    "BaselineValidator",
    "quick_validate_baseline",
    "check_environment_health",
    "TestStatus",
    "get_test_status",
    # Test result correlation and scoring
    "FailureCategory",
    "TestCorrelation",
    "SWEBenchScore",
    "CorrelationReport",
    "ResultCorrelator",
    "correlate_validation_results",
    "analyze_failure_patterns",
    "save_correlation_report",
    # Evaluation orchestrator
    "EvaluationStage",
    "TaskProgress",
    "OrchestratorConfig",
    "EvaluationSummary",
    "EvaluationOrchestrator",
    "run_swe_bench_evaluation",
]
