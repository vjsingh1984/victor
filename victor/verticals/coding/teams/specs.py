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

"""Team specifications for coding tasks.

Provides pre-defined team configurations for common software development
workflows including feature implementation, bug fixing, refactoring,
and code review.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from victor.framework.teams import TeamFormation, TeamMemberSpec


@dataclass
class CodingRoleConfig:
    """Configuration for a coding-specific role.

    Attributes:
        base_role: Base agent role (researcher, planner, executor, reviewer)
        tools: Tools available to this role
        tool_budget: Default tool budget
        description: Role description
    """

    base_role: str
    tools: List[str]
    tool_budget: int
    description: str = ""


# Coding-specific roles with tool allocations
CODING_ROLES: Dict[str, CodingRoleConfig] = {
    "code_researcher": CodingRoleConfig(
        base_role="researcher",
        tools=[
            "read_file",
            "grep",
            "code_search",
            "semantic_code_search",
            "project_overview",
            "symbols",
            "references",
            "list_directory",
        ],
        tool_budget=25,
        description="Researches and analyzes codebase patterns",
    ),
    "code_planner": CodingRoleConfig(
        base_role="planner",
        tools=[
            "read_file",
            "project_overview",
            "plan_files",
            "grep",
        ],
        tool_budget=15,
        description="Plans implementation approach",
    ),
    "code_executor": CodingRoleConfig(
        base_role="executor",
        tools=[
            "read_file",
            "write_file",
            "edit_files",
            "bash",
            "git_status",
            "git_diff",
        ],
        tool_budget=40,
        description="Implements code changes",
    ),
    "code_reviewer": CodingRoleConfig(
        base_role="reviewer",
        tools=[
            "read_file",
            "git_diff",
            "run_tests",
            "bash",
            "grep",
        ],
        tool_budget=20,
        description="Reviews code and runs tests",
    ),
    "test_writer": CodingRoleConfig(
        base_role="executor",
        tools=[
            "read_file",
            "write_file",
            "run_tests",
            "bash",
            "test_file",
        ],
        tool_budget=30,
        description="Writes and runs tests",
    ),
    "doc_writer": CodingRoleConfig(
        base_role="executor",
        tools=[
            "read_file",
            "write_file",
            "edit_files",
            "grep",
        ],
        tool_budget=20,
        description="Writes documentation",
    ),
    "security_reviewer": CodingRoleConfig(
        base_role="researcher",
        tools=[
            "read_file",
            "grep",
            "code_search",
            "bash",
        ],
        tool_budget=20,
        description="Reviews code for security issues",
    ),
}


@dataclass
class CodingTeamSpec:
    """Specification for a coding team.

    Attributes:
        name: Team name
        description: Team description
        formation: How agents are organized
        members: Team member specifications
        total_tool_budget: Total tool budget for the team
        max_iterations: Maximum iterations
    """

    name: str
    description: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50


# Pre-defined team specifications
CODING_TEAM_SPECS: Dict[str, CodingTeamSpec] = {
    "feature_team": CodingTeamSpec(
        name="Feature Implementation Team",
        description="End-to-end feature implementation with research, planning, implementation, and review",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze codebase for relevant patterns and dependencies",
                name="Code Researcher",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="planner",
                goal="Design implementation approach based on research",
                name="Implementation Planner",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Implement the feature according to plan",
                name="Feature Implementer",
                tool_budget=40,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Review code and run tests",
                name="Code Reviewer",
                tool_budget=20,
            ),
        ],
        total_tool_budget=100,
    ),
    "bug_fix_team": CodingTeamSpec(
        name="Bug Fix Team",
        description="Systematic bug investigation and fix with verification",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Investigate bug root cause through code analysis",
                name="Bug Investigator",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Apply the fix based on investigation",
                name="Bug Fixer",
                tool_budget=25,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Verify fix with tests",
                name="Fix Verifier",
                tool_budget=20,
            ),
        ],
        total_tool_budget=70,
    ),
    "refactoring_team": CodingTeamSpec(
        name="Refactoring Team",
        description="Safe refactoring with analysis and testing",
        formation=TeamFormation.HIERARCHICAL,
        members=[
            TeamMemberSpec(
                role="planner",
                goal="Plan refactoring approach and identify affected areas",
                name="Refactoring Planner",
                tool_budget=20,
                is_manager=True,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Execute refactoring changes",
                name="Refactoring Executor",
                tool_budget=35,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Ensure tests pass and code quality maintained",
                name="Quality Verifier",
                tool_budget=20,
            ),
        ],
        total_tool_budget=75,
    ),
    "review_team": CodingTeamSpec(
        name="Code Review Team",
        description="Comprehensive code review with parallel analysis",
        formation=TeamFormation.PARALLEL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Check for security vulnerabilities",
                name="Security Reviewer",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="researcher",
                goal="Check code style and conventions",
                name="Style Reviewer",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="researcher",
                goal="Check logic correctness and edge cases",
                name="Logic Reviewer",
                tool_budget=15,
            ),
            TeamMemberSpec(
                role="planner",
                goal="Synthesize findings into actionable feedback",
                name="Review Synthesizer",
                tool_budget=10,
            ),
        ],
        total_tool_budget=55,
    ),
    "testing_team": CodingTeamSpec(
        name="Testing Team",
        description="Comprehensive test coverage improvement",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze existing test coverage and identify gaps",
                name="Coverage Analyzer",
                tool_budget=20,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Write tests for uncovered areas",
                name="Test Writer",
                tool_budget=35,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Run tests and verify coverage improvement",
                name="Test Verifier",
                tool_budget=15,
            ),
        ],
        total_tool_budget=70,
    ),
    "documentation_team": CodingTeamSpec(
        name="Documentation Team",
        description="Generate and update documentation",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze code to understand functionality",
                name="Code Analyzer",
                tool_budget=20,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Write documentation",
                name="Doc Writer",
                tool_budget=25,
            ),
        ],
        total_tool_budget=45,
    ),
}


def get_team_for_task(task_type: str) -> Optional[CodingTeamSpec]:
    """Get appropriate team specification for task type.

    Args:
        task_type: Type of task (feature, bug, review, etc.)

    Returns:
        CodingTeamSpec or None if no matching team
    """
    mapping = {
        # Feature tasks
        "feature": "feature_team",
        "implement": "feature_team",
        "new_feature": "feature_team",
        "add": "feature_team",
        # Bug fix tasks
        "bug": "bug_fix_team",
        "fix": "bug_fix_team",
        "bugfix": "bug_fix_team",
        "debug": "bug_fix_team",
        # Refactoring tasks
        "refactor": "refactoring_team",
        "refactoring": "refactoring_team",
        "restructure": "refactoring_team",
        # Review tasks
        "review": "review_team",
        "code_review": "review_team",
        "audit": "review_team",
        # Testing tasks
        "test": "testing_team",
        "testing": "testing_team",
        "coverage": "testing_team",
        # Documentation tasks
        "doc": "documentation_team",
        "documentation": "documentation_team",
        "docs": "documentation_team",
    }
    spec_name = mapping.get(task_type.lower())
    if spec_name:
        return CODING_TEAM_SPECS.get(spec_name)
    return None


def get_role_config(role_name: str) -> Optional[CodingRoleConfig]:
    """Get configuration for a coding role.

    Args:
        role_name: Role name

    Returns:
        CodingRoleConfig or None
    """
    return CODING_ROLES.get(role_name.lower())


def list_team_types() -> List[str]:
    """List all available team types.

    Returns:
        List of team type names
    """
    return list(CODING_TEAM_SPECS.keys())


def list_roles() -> List[str]:
    """List all available coding roles.

    Returns:
        List of role names
    """
    return list(CODING_ROLES.keys())


__all__ = [
    # Types
    "CodingRoleConfig",
    "CodingTeamSpec",
    # Role configurations
    "CODING_ROLES",
    # Team specifications
    "CODING_TEAM_SPECS",
    # Helper functions
    "get_team_for_task",
    "get_role_config",
    "list_team_types",
    "list_roles",
]
