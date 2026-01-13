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

"""Parallel execution framework for workflow nodes and tasks.

This module provides a unified framework for parallel execution across
all verticals, replacing duplicated patterns with a single, configurable
implementation.

Quick Start:
    from victor.framework.parallel import (
        execute_parallel,
        JoinStrategy,
        ErrorStrategy,
    )

    # Simple parallel execution
    result = await execute_parallel(
        [task1, task2, task3],
        join_strategy=JoinStrategy.ALL,
    )

    # With custom configuration
    result = await execute_parallel(
        [fetch_data, fetch_metadata],
        join_strategy=JoinStrategy.MAJORITY,
        error_strategy=ErrorStrategy.COLLECT_ERRORS,
        max_concurrent=5,
    )

Advanced Usage:
    from victor.framework.parallel import (
        ParallelExecutor,
        ParallelConfig,
        ResourceLimit,
    )

    config = ParallelConfig(
        join_strategy=JoinStrategy.N_OF_M,
        n_of_m=2,
        error_strategy=ErrorStrategy.CONTINUE_ALL,
        resource_limit=ResourceLimit(max_concurrent=10, timeout=30.0),
    )

    executor = ParallelExecutor(config)
    result = await executor.execute(tasks, context)

YAML Integration:
    # In workflow YAML:
    - id: parallel_analysis
      type: compute
      handler: parallel_executor
      config:
        join_strategy: majority
        error_strategy: collect_errors
        resource_limit:
          max_concurrent: 5
          timeout: 30.0
      tasks: [analyze1, analyze2, analyze3]
      output: combined_results
"""

from victor.framework.parallel.executor import (
    ProgressCallback,
    ProgressEvent,
    ParallelExecutor,
    ParallelExecutorHandler,
    create_parallel_executor,
    execute_parallel,
    execute_parallel_with_config,
    register_parallel_handler,
)
from victor.framework.parallel.protocols import (
    ErrorStrategyProtocol,
    JoinStrategyProtocol,
    ParallelExecutionResult,
    ParallelExecutorProtocol,
    ResultAggregatorProtocol,
)
from victor.framework.parallel.strategies import (
    AllJoinStrategy,
    AnyJoinStrategy,
    CollectErrorsErrorStrategy,
    ContinueAllErrorStrategy,
    create_error_strategy,
    create_join_strategy,
    ErrorStrategy,
    FailFastErrorStrategy,
    FirstJoinStrategy,
    JoinStrategy,
    MajorityJoinStrategy,
    NOfMJoinStrategy,
    ParallelConfig,
    ResourceLimit,
    validate_error_strategy,
    validate_join_strategy,
)

__all__ = [
    # Main executor
    "ParallelExecutor",
    "ParallelExecutorHandler",
    # Configuration
    "ParallelConfig",
    "ResourceLimit",
    # Enums
    "JoinStrategy",
    "ErrorStrategy",
    # Result classes
    "ParallelExecutionResult",
    "ProgressEvent",
    # Progress callback
    "ProgressCallback",
    # Strategy classes
    "AllJoinStrategy",
    "AnyJoinStrategy",
    "FirstJoinStrategy",
    "NOfMJoinStrategy",
    "MajorityJoinStrategy",
    "FailFastErrorStrategy",
    "ContinueAllErrorStrategy",
    "CollectErrorsErrorStrategy",
    # Factory functions
    "create_parallel_executor",
    "create_join_strategy",
    "create_error_strategy",
    "execute_parallel",
    "execute_parallel_with_config",
    "register_parallel_handler",
    # Validation
    "validate_join_strategy",
    "validate_error_strategy",
    # Protocols
    "JoinStrategyProtocol",
    "ErrorStrategyProtocol",
    "ResultAggregatorProtocol",
    "ParallelExecutorProtocol",
]

# Version of the parallel framework
__version__ = "1.0.0"
