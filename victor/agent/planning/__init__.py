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

"""Autonomous planning module for goal-oriented execution.

This package provides infrastructure for Victor to accept high-level goals
and autonomously generate and execute structured plans.

Components:
- ExecutionPlan: A plan consisting of steps with dependencies
- PlanStep: Individual step in a plan
- AutonomousPlanner: Generates and executes plans
- HierarchicalPlanner: Hierarchical task decomposition engine
- TaskDecomposition: Task graph management and algorithms

Example:
    from victor.agent.planning import AutonomousPlanner, ExecutionPlan

    planner = AutonomousPlanner(orchestrator)

    # Generate plan
    plan = await planner.plan_for_goal("Implement user authentication")

    # Review plan
    print(plan.to_markdown())

    # Execute plan
    result = await planner.execute_plan(plan, auto_approve=True)

Example - Hierarchical Planning:
    from victor.agent.planning import HierarchicalPlanner

    planner = HierarchicalPlanner(orchestrator=orchestrator)

    # Decompose task
    graph = await planner.decompose_task("Implement user authentication")

    # Get next tasks
    tasks = await planner.suggest_next_tasks(graph)
"""

from victor.agent.planning.autonomous import (
    AutonomousPlanner,
    PLANNING_SYSTEM_PROMPT,
)
from victor.agent.planning.base import (
    ExecutionPlan,
    PlanResult,
    PlanStep,
    StepResult,
    StepStatus,
    StepType,
)
from victor.agent.planning.hierarchical_planner import (
    DECOMPOSITION_SYSTEM_PROMPT,
    HierarchicalPlanner,
)
from victor.agent.planning.task_decomposition import (
    ComplexityScore,
    DependencyEdge,
    DependencyType,
    SimpleTask,
    Task,
    TaskDecomposition,
    TaskGraph,
    TaskNode,
    TaskStatus,
    UpdatedPlan,
    ValidationResult,
)

__all__ = [
    # Base data structures (autonomous planner)
    "ExecutionPlan",
    "PlanResult",
    "PlanStep",
    "StepResult",
    "StepStatus",
    "StepType",
    # Autonomous planner
    "AutonomousPlanner",
    "PLANNING_SYSTEM_PROMPT",
    # Hierarchical planner
    "HierarchicalPlanner",
    "DECOMPOSITION_SYSTEM_PROMPT",
    # Task decomposition (new NetworkX-based + legacy classes)
    "SimpleTask",
    "TaskDecomposition",
    "TaskNode",
    "TaskStatus",
    "DependencyType",
    "DependencyEdge",
    # Legacy classes (for backward compatibility with hierarchical_planner)
    "Task",
    "TaskGraph",
    "ComplexityScore",
    "ValidationResult",
    "UpdatedPlan",
]
