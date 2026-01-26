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

"""Workflow optimization algorithms for Victor.

This is the canonical location for workflow optimization capabilities including:
- Performance profiling and bottleneck detection
- Optimization strategies (pruning, parallelization, tool selection)
- Search algorithms (hill climbing, simulated annealing)
- Variant generation and evaluation
- Safety validation and rollback

Usage:
    from victor.optimization.workflow import WorkflowOptimizer, WorkflowProfiler

    optimizer = WorkflowOptimizer()
    profile = await optimizer.analyze_workflow(workflow_id)
    suggestions = await optimizer.suggest_optimizations(workflow_id)
"""

# Import from local workflow module files
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

__all__ = [
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
]
