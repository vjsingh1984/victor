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

"""Unified Optimization Module for Victor.

This module provides comprehensive optimization capabilities organized into
three submodules:

- **victor.optimization.workflow**: Workflow profiling and optimization
  Analyzes workflow execution, detects bottlenecks, generates optimized variants.

- **victor.optimization.runtime**: Lazy loading and parallel execution
  Provides lazy component loading, adaptive parallel execution, work stealing.

- **victor.optimization.core**: Hot path utilities
  Fast JSON serialization, lazy imports, memoization, performance monitoring.

Usage:
    # Workflow optimization
    from victor.optimization import WorkflowOptimizer, WorkflowProfiler
    optimizer = WorkflowOptimizer()
    profile = await optimizer.analyze_workflow(workflow_id)

    # Runtime optimization
    from victor.optimization import LazyComponentLoader, AdaptiveParallelExecutor
    loader = LazyComponentLoader()
    executor = AdaptiveParallelExecutor()

    # Hot path utilities
    from victor.optimization import json_dumps, json_loads, LazyImport
    data = json_dumps({"key": "value"})

    # Or import from specific submodules
    from victor.optimization.workflow import WorkflowOptimizer
    from victor.optimization.runtime import LazyComponentLoader
    from victor.optimization.core import json_dumps
"""

# =============================================================================
# Workflow Optimization (from victor.optimization.workflow)
# =============================================================================
from victor.optimization.workflow.models import (
    Bottleneck,
    BottleneckSeverity,
    BottleneckType,
    NodeStatistics,
    OptimizationOpportunity,
    OptimizationStrategy,
    OptimizationStrategyType,
    WorkflowProfile,
)
from victor.optimization.workflow.profiler import WorkflowProfiler
from victor.optimization.workflow.strategies import (
    ParallelizationStrategy,
    PruningStrategy,
    ToolSelectionStrategy,
)
from victor.optimization.workflow.generator import WorkflowVariantGenerator
from victor.optimization.workflow.search import HillClimbingOptimizer
from victor.optimization.workflow.evaluator import VariantEvaluator
from victor.optimization.workflow.optimizer import WorkflowOptimizer
from victor.optimization.workflow.validator import (
    ConstraintViolation,
    OptimizationValidationResult,
    OptimizationValidator,
    ValidationRecommendation,
)

# =============================================================================
# Runtime Optimization (from victor.optimization.runtime)
# =============================================================================
from victor.optimization.runtime import (
    # Lazy loading
    LazyComponentLoader,
    LoadingStats,
    ComponentDescriptor,
    lazy_load,
    set_global_loader,
    get_global_loader,
    # Parallel execution
    AdaptiveParallelExecutor,
    OptimizationStrategy as RuntimeOptimizationStrategy,  # Avoid conflict
    PerformanceMetrics,
    TaskWithPriority,
    create_adaptive_executor,
    execute_parallel_optimized,
)

# =============================================================================
# Hot Path Optimization (from victor.optimization.core)
# =============================================================================
from victor.optimization.core import (
    LazyImport,
    lazy_import,
    json_dumps,
    json_loads,
    json_dump,
    json_load,
    ThreadSafeMemoized,
    cached_property,
    timed,
    retry,
    async_retry,
    PerformanceMonitor,
)

__all__ = [
    # ==========================================================================
    # Workflow Optimization
    # ==========================================================================
    # Models
    "WorkflowProfile",
    "NodeStatistics",
    "Bottleneck",
    "BottleneckType",
    "BottleneckSeverity",
    "OptimizationOpportunity",
    "OptimizationStrategy",
    "OptimizationStrategyType",
    # Core components
    "WorkflowProfiler",
    "WorkflowOptimizer",
    "WorkflowVariantGenerator",
    "VariantEvaluator",
    "HillClimbingOptimizer",
    # Strategies
    "PruningStrategy",
    "ParallelizationStrategy",
    "ToolSelectionStrategy",
    # Validation
    "OptimizationValidator",
    "OptimizationValidationResult",
    "ValidationRecommendation",
    "ConstraintViolation",
    # ==========================================================================
    # Runtime Optimization
    # ==========================================================================
    # Lazy loading
    "LazyComponentLoader",
    "LoadingStats",
    "ComponentDescriptor",
    "lazy_load",
    "set_global_loader",
    "get_global_loader",
    # Parallel execution
    "AdaptiveParallelExecutor",
    "RuntimeOptimizationStrategy",
    "PerformanceMetrics",
    "TaskWithPriority",
    "create_adaptive_executor",
    "execute_parallel_optimized",
    # ==========================================================================
    # Hot Path Optimization
    # ==========================================================================
    "LazyImport",
    "lazy_import",
    "json_dumps",
    "json_loads",
    "json_dump",
    "json_load",
    "ThreadSafeMemoized",
    "cached_property",
    "timed",
    "retry",
    "async_retry",
    "PerformanceMonitor",
]
