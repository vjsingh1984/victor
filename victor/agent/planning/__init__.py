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

Example:
    from victor.agent.planning import AutonomousPlanner, ExecutionPlan

    planner = AutonomousPlanner(orchestrator)

    # Generate plan
    plan = await planner.plan_for_goal("Implement user authentication")

    # Review plan
    print(plan.to_markdown())

    # Execute plan
    result = await planner.execute_plan(plan, auto_approve=True)
"""

from victor.agent.planning.base import (
    ExecutionPlan,
    PlanResult,
    PlanStep,
    StepResult,
    StepStatus,
    StepType,
)
from victor.agent.planning.autonomous import (
    AutonomousPlanner,
    PLANNING_SYSTEM_PROMPT,
)

__all__ = [
    # Base data structures
    "ExecutionPlan",
    "PlanResult",
    "PlanStep",
    "StepResult",
    "StepStatus",
    "StepType",
    # Planner
    "AutonomousPlanner",
    "PLANNING_SYSTEM_PROMPT",
]
