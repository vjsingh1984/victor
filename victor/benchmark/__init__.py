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

"""Benchmark Vertical - High-level API for AI coding evaluations.

This vertical provides a framework-aligned approach to running benchmarks,
using the same patterns as other verticals (Coding, Research, DevOps).

Usage:
    from victor import Agent
    from victor.benchmark import BenchmarkVertical

    # Create agent with benchmark vertical
    agent = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        vertical=BenchmarkVertical,
    )

    # Run a benchmark task
    result = await agent.run(task_prompt)

    # Or use the high-level benchmark API
    from victor.benchmark import BenchmarkAgent

    bench_agent = await BenchmarkAgent.create(provider="anthropic")
    result = await bench_agent.execute_task(benchmark_task)
"""

from victor.benchmark.assistant import BenchmarkVertical
from victor.benchmark.agent import (
    BenchmarkAgent,
    BenchmarkAgentConfig,
    ExecutionTrace,
    create_benchmark_agent,
)
from victor.benchmark.task_bridge import (
    benchmark_task_to_framework_task,
    framework_result_to_benchmark_result,
    build_benchmark_prompt,
)
from victor.benchmark.harness_integration import (
    create_agent_callback,
    HighLevelEvaluationRunner,
)
from victor.benchmark.workflows import BenchmarkWorkflowProvider

__all__ = [
    # Vertical
    "BenchmarkVertical",
    # Agent API
    "BenchmarkAgent",
    "BenchmarkAgentConfig",
    "ExecutionTrace",
    "create_benchmark_agent",
    # Task bridge
    "benchmark_task_to_framework_task",
    "framework_result_to_benchmark_result",
    "build_benchmark_prompt",
    # Harness integration
    "create_agent_callback",
    "HighLevelEvaluationRunner",
    # Workflow provider
    "BenchmarkWorkflowProvider",
]
