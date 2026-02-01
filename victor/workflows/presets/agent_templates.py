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

"""Preset agent configurations for common roles.

These presets provide optimized configurations for different agent types,
including role-specific tool sets, budgets, and prompts.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentPreset:
    """Preset configuration for an agent role.

    Attributes:
        name: Unique preset name
        role: Agent role (researcher, executor, etc.)
        description: What this agent specializes in
        goal_template: Default goal template
        default_tool_budget: Default tool call limit
        allowed_tools: Tools this agent can use (None = role defaults)
        system_prompt_template: Optional custom system prompt
        llm_config: Optional LLM configuration
        example_goals: Example goals for this preset
    """

    name: str
    role: str
    description: str
    goal_template: str
    default_tool_budget: int = 15
    allowed_tools: Optional[list[str]] = None
    system_prompt_template: Optional[str] = None
    llm_config: Optional[dict[str, Any]] = None
    example_goals: list[str] = field(default_factory=list)

    def to_workflow_builder_params(self, goal: str) -> dict[str, Any]:
        """Convert to WorkflowBuilder.add_agent() parameters.

        Args:
            goal: Specific goal for this instance

        Returns:
            Dictionary of parameters for WorkflowBuilder.add_agent()
        """
        params = {
            "role": self.role,
            "goal": goal or self.goal_template,
            "tool_budget": self.default_tool_budget,
        }

        if self.allowed_tools:
            params["allowed_tools"] = self.allowed_tools

        if self.llm_config:
            params["llm_config"] = self.llm_config

        return params


# =============================================================================
# RESEARCHER PRESETS
# =============================================================================

CODE_RESEARCHER = AgentPreset(
    name="code_researcher",
    role="researcher",
    description="Explores codebase to find patterns, implementations, and dependencies",
    goal_template="Research {topic} in the codebase and provide detailed findings",
    default_tool_budget=20,
    allowed_tools=[
        "read",
        "code_search",
        "search",
        "list_directory",
        "git_diff",
        "web_search",
    ],
    example_goals=[
        "Research authentication patterns in the codebase",
        "Find all uses of the caching layer",
        "Explore error handling patterns across modules",
    ],
)

DOCUMENTATION_RESEARCHER = AgentPreset(
    name="documentation_researcher",
    role="researcher",
    description="Researches documentation, README files, and inline comments",
    goal_template="Research documentation about {topic}",
    default_tool_budget=15,
    allowed_tools=[
        "read",
        "search",
        "web_search",
        "list_directory",
    ],
    example_goals=[
        "Research API documentation for endpoints",
        "Find installation and setup instructions",
        "Research configuration options",
    ],
)

SECURITY_RESEARCHER = AgentPreset(
    name="security_researcher",
    role="researcher",
    description="Analyzes code for security vulnerabilities and best practices",
    goal_template="Analyze security aspects of {component}",
    default_tool_budget=25,
    allowed_tools=[
        "read",
        "code_search",
        "search",
        "grep",
        "web_search",
    ],
    llm_config={"temperature": 0.1},  # Lower temperature for consistent analysis
    example_goals=[
        "Analyze authentication flow for security issues",
        "Review input validation across the codebase",
        "Research dependency vulnerabilities",
    ],
)

# =============================================================================
# PLANNER PRESETS
# =============================================================================()

TASK_PLANNER = AgentPreset(
    name="task_planner",
    role="planner",
    description="Breaks down complex tasks into actionable steps",
    goal_template="Create a detailed implementation plan for {task}",
    default_tool_budget=10,
    allowed_tools=[
        "read",
        "search",
        "code_search",
        "list_directory",
    ],
    example_goals=[
        "Create plan for implementing user authentication",
        "Break down refactoring of the payment module",
        "Plan database migration strategy",
    ],
)

ARCHITECTURE_PLANNER = AgentPreset(
    name="architecture_planner",
    role="planner",
    description="Designs system architecture and component integration",
    goal_template="Design architecture for {system}",
    default_tool_budget=15,
    allowed_tools=[
        "read",
        "code_search",
        "search",
        "list_directory",
        "web_search",
    ],
    example_goals=[
        "Design microservices architecture for the API",
        "Plan integration with third-party payment provider",
        "Design caching strategy for high-traffic endpoints",
    ],
)

# =============================================================================
# EXECUTOR PRESETS
# =============================================================================

CODE_WRITER = AgentPreset(
    name="code_writer",
    role="executor",
    description="Writes new code and implements features",
    goal_template="Implement {feature} following best practices",
    default_tool_budget=25,
    allowed_tools=[
        "read",
        "write",
        "edit",
        "create_file",
        "search",
        "code_search",
        "shell",
        "test",
    ],
    example_goals=[
        "Implement user registration endpoint",
        "Add data validation layer",
        "Create unit tests for payment module",
    ],
)

BUG_FIXER = AgentPreset(
    name="bug_fixer",
    role="executor",
    description="Diagnoses and fixes bugs in existing code",
    goal_template="Fix the bug: {bug_description}",
    default_tool_budget=30,
    allowed_tools=[
        "read",
        "write",
        "edit",
        "search",
        "code_search",
        "shell",
        "test",
        "git_diff",
        "git_log",
    ],
    example_goals=[
        "Fix the authentication token expiration bug",
        "Resolve race condition in order processing",
        "Fix memory leak in background job handler",
    ],
)

REFACTORING_AGENT = AgentPreset(
    name="refactoring_agent",
    role="executor",
    description="Refactors code for improved maintainability and performance",
    goal_template="Refactor {component} to improve {aspect}",
    default_tool_budget=35,
    allowed_tools=[
        "read",
        "write",
        "edit",
        "code_search",
        "search",
        "test",
        "shell",
        "lint",
    ],
    llm_config={"temperature": 0.2},  # Lower temp for consistent refactoring
    example_goals=[
        "Refactor user service to follow SOLID principles",
        "Optimize database queries in report generation",
        "Extract common validation logic into shared utilities",
    ],
)

# =============================================================================
# REVIEWER PRESETS
# =============================================================================()

CODE_REVIEWER = AgentPreset(
    name="code_reviewer",
    role="reviewer",
    description="Reviews code changes for quality, security, and best practices",
    goal_template="Review {code_change} and provide feedback",
    default_tool_budget=15,
    allowed_tools=[
        "read",
        "code_search",
        "search",
        "git_diff",
        "test",
        "lint",
    ],
    llm_config={"temperature": 0.1},  # Very low temp for consistent reviews
    example_goals=[
        "Review pull request for user authentication feature",
        "Review database migration script for safety",
        "Review error handling implementation",
    ],
)

SECURITY_AUDITOR = AgentPreset(
    name="security_auditor",
    role="reviewer",
    description="Performs security audits and identifies vulnerabilities",
    goal_template="Perform security audit of {component}",
    default_tool_budget=20,
    allowed_tools=[
        "read",
        "code_search",
        "search",
        "grep",
        "git_diff",
        "web_search",
    ],
    llm_config={"temperature": 0.0},  # Zero temp for rigorous security analysis
    example_goals=[
        "Audit authentication flow for OWASP Top 10",
        "Review input validation for SQL injection risks",
        "Audit API rate limiting implementation",
    ],
)

PERFORMANCE_REVIEWER = AgentPreset(
    name="performance_reviewer",
    role="reviewer",
    description="Analyzes code for performance bottlenecks and optimization opportunities",
    goal_template="Analyze performance of {component}",
    default_tool_budget=18,
    allowed_tools=[
        "read",
        "code_search",
        "search",
        "profiler",
        "shell",
    ],
    example_goals=[
        "Analyze database query performance in reporting module",
        "Review memory usage in data processing pipeline",
        "Identify optimization opportunities in API endpoints",
    ],
)

# =============================================================================
# TESTER PRESETS
# =============================================================================()

TEST_WRITER = AgentPreset(
    name="test_writer",
    role="tester",
    description="Writes comprehensive unit and integration tests",
    goal_template="Write tests for {component}",
    default_tool_budget=20,
    allowed_tools=[
        "read",
        "write",
        "edit",
        "create_file",
        "code_search",
        "search",
        "test",
        "shell",
    ],
    example_goals=[
        "Write unit tests for payment processing module",
        "Create integration tests for API endpoints",
        "Add edge case tests for validation logic",
    ],
)

TEST_TRIAGER = AgentPreset(
    name="test_triage_agent",
    role="tester",
    description="Analyzes test failures and creates actionable bug reports",
    goal_template="Analyze test failures in {test_suite}",
    default_tool_budget=15,
    allowed_tools=[
        "read",
        "test",
        "shell",
        "git_log",
        "git_diff",
        "search",
    ],
    example_goals=[
        "Analyze test failures in authentication suite",
        "Investigate flaky integration tests",
        "Review test coverage gaps",
    ],
)

# =============================================================================
# PRESET REGISTRY
# =============================================================================()

_AGENT_PRESETS: dict[str, AgentPreset] = {
    # Researchers
    "code_researcher": CODE_RESEARCHER,
    "documentation_researcher": DOCUMENTATION_RESEARCHER,
    "security_researcher": SECURITY_RESEARCHER,
    # Planners
    "task_planner": TASK_PLANNER,
    "architecture_planner": ARCHITECTURE_PLANNER,
    # Executors
    "code_writer": CODE_WRITER,
    "bug_fixer": BUG_FIXER,
    "refactoring_agent": REFACTORING_AGENT,
    # Reviewers
    "code_reviewer": CODE_REVIEWER,
    "security_auditor": SECURITY_AUDITOR,
    "performance_reviewer": PERFORMANCE_REVIEWER,
    # Testers
    "test_writer": TEST_WRITER,
    "test_triage_agent": TEST_TRIAGER,
}


def get_agent_preset(name: str) -> Optional[AgentPreset]:
    """Get an agent preset by name.

    Args:
        name: Preset name (e.g., "code_researcher", "bug_fixer")

    Returns:
        AgentPreset if found, None otherwise

    Example:
        preset = get_agent_preset("code_reviewer")
        params = preset.to_workflow_builder_params("Review PR #123")
    """
    return _AGENT_PRESETS.get(name)


def list_agent_presets() -> list[str]:
    """List all available agent preset names.

    Returns:
        List of preset names

    Example:
        presets = list_agent_presets()
        for name in presets:
            preset = get_agent_preset(name)
            print(f"{name}: {preset.description}")
    """
    return list(_AGENT_PRESETS.keys())


def list_agent_presets_by_role() -> dict[str, list[str]]:
    """List agent presets grouped by role.

    Returns:
        Dictionary mapping roles to lists of preset names

    Example:
        by_role = list_agent_presets_by_role()
        for role, presets in by_role.items():
            print(f"{role}: {', '.join(presets)}")
    """
    by_role: dict[str, list[str]] = {}
    for name, preset in _AGENT_PRESETS.items():
        if preset.role not in by_role:
            by_role[preset.role] = []
        by_role[preset.role].append(name)
    return by_role


__all__ = [
    "AgentPreset",
    "get_agent_preset",
    "list_agent_presets",
    "list_agent_presets_by_role",
    # Individual presets
    "CODE_RESEARCHER",
    "DOCUMENTATION_RESEARCHER",
    "SECURITY_RESEARCHER",
    "TASK_PLANNER",
    "ARCHITECTURE_PLANNER",
    "CODE_WRITER",
    "BUG_FIXER",
    "REFACTORING_AGENT",
    "CODE_REVIEWER",
    "SECURITY_AUDITOR",
    "PERFORMANCE_REVIEWER",
    "TEST_WRITER",
    "TEST_TRIAGER",
]
