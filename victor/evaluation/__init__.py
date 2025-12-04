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
    SWEBenchRunner,
)

__all__ = [
    # Protocol types
    "BenchmarkMetadata",
    "BenchmarkTask",
    "BenchmarkType",
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
    "SWEBenchRunner",
]
