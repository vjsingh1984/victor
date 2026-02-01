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

"""Validation rules for YAML workflows.

This module defines validation rules and best practices for workflow definitions.
Each rule implements a specific check and can be enabled/disabled independently.

Rule Categories:
    - SYNTAX: YAML syntax and structure
    - SCHEMA: Node schema validation
    - CONNECTIONS: Node reference validation
    - BEST_PRACTICES: Workflow design best practices
    - SECURITY: Security and safety checks
    - COMPLEXITY: Workflow complexity analysis
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


class RuleCategory(Enum):
    """Category of validation rule."""

    SYNTAX = "syntax"
    SCHEMA = "schema"
    CONNECTIONS = "connections"
    BEST_PRACTICES = "best_practices"
    SECURITY = "security"
    COMPLEXITY = "complexity"
    TEAM = "team"


@dataclass
class ValidationIssue:
    """A validation issue found by a rule.

    Attributes:
        rule_id: Unique identifier for the rule
        severity: Severity level of the issue
        category: Category of the rule
        message: Human-readable issue description
        location: Location in the workflow (node_id, field, etc.)
        suggestion: Suggested fix (optional)
        context: Additional context for debugging
    """

    rule_id: str
    severity: Severity
    category: RuleCategory
    message: str
    location: str
    suggestion: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
            "context": self.context,
        }


class ValidationRule(ABC):
    """Base class for validation rules.

    Each rule implements specific validation logic for workflows.
    Rules can be enabled/disabled independently and can have custom severity levels.
    """

    def __init__(
        self,
        rule_id: str,
        category: RuleCategory,
        severity: Severity = Severity.ERROR,
        enabled: bool = True,
    ):
        """Initialize the validation rule.

        Args:
            rule_id: Unique identifier for this rule
            category: Category of the rule
            severity: Default severity level
            enabled: Whether the rule is enabled
        """
        self.rule_id = rule_id
        self.category = category
        self.severity = severity
        self.enabled = enabled

    @abstractmethod
    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check the workflow for issues.

        Args:
            workflow: Workflow definition dictionary

        Returns:
            List of validation issues found
        """
        ...

    def create_issue(
        self,
        message: str,
        location: str,
        severity: Optional[Severity] = None,
        suggestion: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationIssue:
        """Create a validation issue.

        Args:
            message: Issue description
            location: Location in workflow
            severity: Override default severity
            suggestion: Suggested fix
            context: Additional context

        Returns:
            ValidationIssue instance
        """
        return ValidationIssue(
            rule_id=self.rule_id,
            severity=severity or self.severity,
            category=self.category,
            message=message,
            location=location,
            suggestion=suggestion,
            context=context or {},
        )


class NodeIDFormatRule(ValidationRule):
    """Check that node IDs follow naming conventions.

    Node IDs should be:
    - Lowercase with underscores
    - Start with a letter
    - Contain only alphanumeric characters and underscores
    - Not exceed 100 characters
    """

    # Valid pattern: lowercase letter followed by alphanumeric/underscore
    PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="node_id_format",
            category=RuleCategory.SYNTAX,
            severity=Severity.WARNING,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check node ID format."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            for node in wf_def.get("nodes", []):
                node_id = node.get("id", "")
                if not node_id:
                    issues.append(
                        self.create_issue(
                            message="Node missing 'id' field",
                            location=f"{wf_name}:<unknown>",
                            severity=Severity.ERROR,
                        )
                    )
                    continue

                if len(node_id) > 100:
                    issues.append(
                        self.create_issue(
                            message=f"Node ID too long ({len(node_id)} > 100 chars)",
                            location=f"{wf_name}:{node_id}",
                            suggestion="Use shorter, descriptive IDs",
                        )
                    )

                if not self.PATTERN.match(node_id):
                    issues.append(
                        self.create_issue(
                            message=f"Node ID should be lowercase with underscores (got: '{node_id}')",
                            location=f"{wf_name}:{node_id}",
                            suggestion=f"Rename to '{node_id.lower().replace('-', '_')}'",
                        )
                    )

        return issues


class RequiredFieldsRule(ValidationRule):
    """Check that nodes have required fields based on their type."""

    REQUIRED_FIELDS = {
        "agent": ["role", "goal"],
        "compute": ["handler"],
        "condition": ["condition", "branches"],
        "parallel": ["parallel_nodes", "join_strategy"],
        "transform": ["transform"],
        "hitl": ["hitl_type", "prompt"],
        "team": ["team_formation", "members", "goal"],
    }

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="required_fields",
            category=RuleCategory.SCHEMA,
            severity=Severity.ERROR,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check required fields."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            for node in wf_def.get("nodes", []):
                node_id = node.get("id", "<unknown>")
                node_type = node.get("type", "")

                if node_type in self.REQUIRED_FIELDS:
                    required = self.REQUIRED_FIELDS[node_type]
                    for field in required:
                        if field not in node or not node[field]:
                            issues.append(
                                self.create_issue(
                                    message=f"{node_type.upper()} node missing required field: '{field}'",
                                    location=f"{wf_name}:{node_id}",
                                    suggestion=f"Add '{field}' field to node",
                                    context={"node_type": node_type, "missing_field": field},
                                )
                            )

        return issues


class ConnectionReferencesRule(ValidationRule):
    """Check that all node references (next, branches, parallel_nodes) are valid."""

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="connection_references",
            category=RuleCategory.CONNECTIONS,
            severity=Severity.ERROR,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check connection references."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            # Build node ID set
            node_ids = {node.get("id") for node in wf_def.get("nodes", [])}
            node_ids.discard(None)  # Remove None if present

            for node in wf_def.get("nodes", []):
                node_id = node.get("id", "<unknown>")

                # Check 'next' references
                for next_node in node.get("next", []):
                    if next_node not in node_ids:
                        issues.append(
                            self.create_issue(
                                message=f"Invalid 'next' reference: '{next_node}' not found",
                                location=f"{wf_name}:{node_id}",
                                context={"reference": next_node, "reference_type": "next"},
                            )
                        )

                # Check branch references
                if "branches" in node:
                    for branch_value, branch_target in node["branches"].items():
                        if branch_target not in node_ids and branch_target != "__end__":
                            issues.append(
                                self.create_issue(
                                    message=f"Invalid branch target: '{branch_target}' not found",
                                    location=f"{wf_name}:{node_id}",
                                    context={
                                        "branch": branch_value,
                                        "target": branch_target,
                                    },
                                )
                            )

                # Check parallel_nodes references
                if "parallel_nodes" in node:
                    for parallel_node in node["parallel_nodes"]:
                        if parallel_node not in node_ids:
                            issues.append(
                                self.create_issue(
                                    message=f"Invalid parallel_nodes reference: '{parallel_node}' not found",
                                    location=f"{wf_name}:{node_id}",
                                    context={"reference": parallel_node},
                                )
                            )

        return issues


class CircularDependencyRule(ValidationRule):
    """Check for circular dependencies in workflow graph."""

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="circular_dependency",
            category=RuleCategory.CONNECTIONS,
            severity=Severity.ERROR,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check for circular dependencies using DFS."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            # Build adjacency list
            adj = {}
            for node in wf_def.get("nodes", []):
                node_id = node.get("id")
                if node_id:
                    adj[node_id] = set(node.get("next", []))
                    # Add branch targets
                    if "branches" in node:
                        for target in node["branches"].values():
                            if target != "__end__":
                                adj[node_id].add(target)

            # Detect cycles using DFS
            visited = set()
            rec_stack = set()

            def dfs(node_id: str, path: list[str]) -> Optional[list[str]]:
                """DFS to detect cycles."""
                visited.add(node_id)
                rec_stack.add(node_id)

                for neighbor in adj.get(node_id, []):
                    if neighbor not in visited:
                        result = dfs(neighbor, path + [neighbor])
                        if result:
                            return result
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        return path[cycle_start:] + [neighbor]

                rec_stack.remove(node_id)
                return None

            for node_id in adj:
                if node_id not in visited:
                    cycle = dfs(node_id, [node_id])
                    if cycle:
                        issues.append(
                            self.create_issue(
                                message=f"Circular dependency detected: {' -> '.join(cycle)}",
                                location=f"{wf_name}:{cycle[0]}",
                                suggestion="Break the cycle by removing one of the connections",
                                context={"cycle": cycle},
                            )
                        )
                        break

        return issues


class TeamFormationRule(ValidationRule):
    """Validate team node configuration.

    Checks:
    - Valid formation type (one of 8 formations)
    - Required member fields (id, role, goal)
    - Recursion depth (1-10)
    - Tool budget (positive integer)
    """

    VALID_FORMATIONS = {
        "sequential",
        "parallel",
        "hierarchical",
        "pipeline",
        "consensus",
        "round_robin",
        "dynamic",
        "expert",
    }

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="team_formation",
            category=RuleCategory.TEAM,
            severity=Severity.ERROR,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check team node configuration."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            for node in wf_def.get("nodes", []):
                if node.get("type") != "team":
                    continue

                node_id = node.get("id", "<unknown>")

                # Check formation type
                formation = node.get("team_formation")
                if not formation:
                    issues.append(
                        self.create_issue(
                            message="Team node missing 'team_formation' field",
                            location=f"{wf_name}:{node_id}",
                        )
                    )
                elif formation not in self.VALID_FORMATIONS:
                    issues.append(
                        self.create_issue(
                            message=f"Invalid team formation: '{formation}'. Must be one of: {', '.join(sorted(self.VALID_FORMATIONS))}",
                            location=f"{wf_name}:{node_id}",
                            suggestion=f"Use one of: {', '.join(sorted(self.VALID_FORMATIONS))}",
                        )
                    )

                # Check members
                members = node.get("members", [])
                if not members:
                    issues.append(
                        self.create_issue(
                            message="Team node has no members defined",
                            location=f"{wf_name}:{node_id}",
                            suggestion="Add at least one team member",
                        )
                    )
                else:
                    for i, member in enumerate(members):
                        member_id = member.get("id", f"<member_{i}>")

                        # Check required member fields
                        if not member.get("role"):
                            issues.append(
                                self.create_issue(
                                    message="Team member missing 'role' field",
                                    location=f"{wf_name}:{node_id}.members[{i}]",
                                    context={"member": member_id},
                                )
                            )

                        if not member.get("goal"):
                            issues.append(
                                self.create_issue(
                                    message="Team member missing 'goal' field",
                                    location=f"{wf_name}:{node_id}.members[{i}]",
                                    context={"member": member_id},
                                )
                            )

                # Check recursion depth
                max_iterations = node.get("max_iterations")
                if max_iterations is not None:
                    if (
                        not isinstance(max_iterations, int)
                        or max_iterations < 1
                        or max_iterations > 10
                    ):
                        issues.append(
                            self.create_issue(
                                message=f"Invalid max_iterations: {max_iterations}. Must be between 1 and 10",
                                location=f"{wf_name}:{node_id}",
                                severity=Severity.WARNING,
                                suggestion="Set max_iterations between 1 and 10 to prevent excessive loops",
                            )
                        )

                # Check tool budget
                total_tool_budget = node.get("total_tool_budget")
                if total_tool_budget is not None:
                    if not isinstance(total_tool_budget, int) or total_tool_budget < 1:
                        issues.append(
                            self.create_issue(
                                message=f"Invalid total_tool_budget: {total_tool_budget}. Must be positive integer",
                                location=f"{wf_name}:{node_id}",
                            )
                        )

        return issues


class GoalQualityRule(ValidationRule):
    """Check goal description quality.

    Goals should:
    - Be at least 20 characters
    - Describe what to accomplish (not just how)
    - Not be too generic (e.g., "do task")
    """

    GENERIC_GOALS = {"do task", "execute", "run", "perform task", "complete task"}

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="goal_quality",
            category=RuleCategory.BEST_PRACTICES,
            severity=Severity.WARNING,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check goal quality."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            for node in wf_def.get("nodes", []):
                node_id = node.get("id", "<unknown>")
                node_type = node.get("type", "")

                # Check agent node goals
                if node_type == "agent":
                    goal = node.get("goal", "")
                    if not goal:
                        issues.append(
                            self.create_issue(
                                message="Agent node missing 'goal' field",
                                location=f"{wf_name}:{node_id}",
                                severity=Severity.ERROR,
                            )
                        )
                    else:
                        goal_lower = goal.lower().strip()

                        if len(goal) < 20:
                            issues.append(
                                self.create_issue(
                                    message=f"Goal description too short ({len(goal)} < 20 chars)",
                                    location=f"{wf_name}:{node_id}",
                                    suggestion="Provide more detailed goal description",
                                )
                            )

                        if goal_lower in self.GENERIC_GOALS:
                            issues.append(
                                self.create_issue(
                                    message=f"Goal too generic: '{goal}'",
                                    location=f"{wf_name}:{node_id}",
                                    suggestion="Describe what the agent should accomplish",
                                )
                            )

                # Check team node goals
                elif node_type == "team":
                    goal = node.get("goal", "")
                    if not goal:
                        issues.append(
                            self.create_issue(
                                message="Team node missing 'goal' field",
                                location=f"{wf_name}:{node_id}",
                                severity=Severity.ERROR,
                            )
                        )
                    elif len(goal) < 30:
                        issues.append(
                            self.create_issue(
                                message=f"Team goal description too short ({len(goal)} < 30 chars)",
                                location=f"{wf_name}:{node_id}",
                                suggestion="Provide detailed team goal",
                            )
                        )

        return issues


class ToolBudgetRule(ValidationRule):
    """Check tool budget configuration.

    Validates:
    - Tool budget is positive integer
    - Tool budget is reasonable (warn if > 100)
    - Tool budget matches task complexity
    """

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="tool_budget",
            category=RuleCategory.BEST_PRACTICES,
            severity=Severity.WARNING,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check tool budget."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            for node in wf_def.get("nodes", []):
                node_id = node.get("id", "<unknown>")
                node_type = node.get("type", "")

                # Check agent node tool budgets
                if node_type == "agent":
                    tool_budget = node.get("tool_budget")
                    if tool_budget is not None:
                        if not isinstance(tool_budget, int) or tool_budget < 1:
                            issues.append(
                                self.create_issue(
                                    message=f"Invalid tool_budget: {tool_budget}. Must be positive integer",
                                    location=f"{wf_name}:{node_id}",
                                    severity=Severity.ERROR,
                                )
                            )
                        elif tool_budget > 100:
                            issues.append(
                                self.create_issue(
                                    message=f"Tool budget very high ({tool_budget} > 100)",
                                    location=f"{wf_name}:{node_id}",
                                    suggestion="Consider reducing tool budget to prevent excessive API calls",
                                )
                            )

        return issues


class DisconnectedNodesRule(ValidationRule):
    """Check for disconnected nodes (unreachable from start).

    Nodes that are not reachable from the workflow's entry point may indicate
    missing connections or orphaned code.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="disconnected_nodes",
            category=RuleCategory.CONNECTIONS,
            severity=Severity.WARNING,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check for disconnected nodes."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            # Build adjacency list
            adj = {}
            node_ids = set()
            for node in wf_def.get("nodes", []):
                node_id = node.get("id")
                if node_id:
                    node_ids.add(node_id)
                    adj[node_id] = set(node.get("next", []))

            # Find start node (first node or specified entry_point)
            start_node = wf_def.get("entry_point")
            if not start_node and node_ids:
                # Use first node as start
                start_node = next(iter(node_ids), None)

            if not start_node:
                continue

            # BFS from start node
            visited = set()
            queue = [start_node]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Find disconnected nodes
            disconnected = node_ids - visited
            if disconnected:
                issues.append(
                    self.create_issue(
                        message=f"Found {len(disconnected)} disconnected node(s): {', '.join(sorted(disconnected))}",
                        location=f"{wf_name}:{start_node}",
                        suggestion="Add connections to include these nodes in the workflow",
                        context={"disconnected_nodes": list(disconnected)},
                    )
                )

        return issues


class DuplicateNodeIDsRule(ValidationRule):
    """Check for duplicate node IDs within a workflow."""

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="duplicate_node_ids",
            category=RuleCategory.SYNTAX,
            severity=Severity.ERROR,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Check for duplicate node IDs."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            node_ids = []
            for node in wf_def.get("nodes", []):
                node_id = node.get("id")
                if node_id:
                    node_ids.append(node_id)

            # Find duplicates
            seen = set()
            duplicates = set()
            for node_id in node_ids:
                if node_id in seen:
                    duplicates.add(node_id)
                seen.add(node_id)

            if duplicates:
                issues.append(
                    self.create_issue(
                        message=f"Duplicate node IDs found: {', '.join(sorted(duplicates))}",
                        location=f"{wf_name}",
                        suggestion="Ensure each node has a unique ID",
                        context={"duplicates": list(duplicates)},
                    )
                )

        return issues


class ComplexityAnalysisRule(ValidationRule):
    """Analyze workflow complexity.

    Metrics:
    - Total node count
    - Maximum nesting depth
    - Cyclomatic complexity (branches + 1)
    """

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="complexity_analysis",
            category=RuleCategory.COMPLEXITY,
            severity=Severity.INFO,
            enabled=enabled,
        )

    def check(self, workflow: dict[str, Any]) -> list[ValidationIssue]:
        """Analyze workflow complexity."""
        issues = []

        for wf_name, wf_def in workflow.get("workflows", {}).items():
            nodes = wf_def.get("nodes", [])

            # Count nodes
            node_count = len(nodes)

            # Calculate cyclomatic complexity
            complexity = 1  # Base complexity
            for node in nodes:
                if node.get("type") == "condition":
                    branches = node.get("branches", {})
                    if branches:
                        complexity += len(branches) - 1

            # Estimate maximum depth
            max_depth = self._calculate_max_depth(wf_def)

            issues.append(
                self.create_issue(
                    message=f"Workflow complexity: {node_count} nodes, depth ~{max_depth}, cyclomatic complexity {complexity}",
                    location=f"{wf_name}",
                    context={
                        "node_count": node_count,
                        "max_depth": max_depth,
                        "cyclomatic_complexity": complexity,
                    },
                )
            )

            # Warn if too complex
            if node_count > 50:
                issues.append(
                    self.create_issue(
                        severity=Severity.WARNING,
                        message=f"Workflow has many nodes ({node_count} > 50). Consider splitting into sub-workflows",
                        location=f"{wf_name}",
                        suggestion="Break into smaller, reusable workflows",
                    )
                )

            if max_depth > 10:
                issues.append(
                    self.create_issue(
                        severity=Severity.WARNING,
                        message=f"Workflow deeply nested (depth {max_depth} > 10)",
                        location=f"{wf_name}",
                        suggestion="Flatten workflow structure",
                    )
                )

        return issues

    def _calculate_max_depth(self, wf_def: dict[str, Any]) -> int:
        """Calculate maximum nesting depth."""
        # Build adjacency list
        adj = {}
        for node in wf_def.get("nodes", []):
            node_id = node.get("id")
            if node_id:
                next_nodes = node.get("next", [])
                adj[node_id] = next_nodes

        # Find start node
        start_node = wf_def.get("entry_point")
        if not start_node and adj:
            start_node = next(iter(adj.keys()), None)

        if not start_node:
            return 0

        # DFS to find longest path
        visited = set()

        def dfs(node_id: str, depth: int) -> int:
            if node_id in visited:
                return depth  # Prevent infinite recursion
            visited.add(node_id)

            max_depth = depth
            for neighbor in adj.get(node_id, []):
                max_depth = max(max_depth, dfs(neighbor, depth + 1))

            visited.remove(node_id)
            return max_depth

        return dfs(start_node, 1)


# Default rule set
DEFAULT_RULES = [
    NodeIDFormatRule(),
    RequiredFieldsRule(),
    ConnectionReferencesRule(),
    CircularDependencyRule(),
    TeamFormationRule(),
    GoalQualityRule(),
    ToolBudgetRule(),
    DisconnectedNodesRule(),
    DuplicateNodeIDsRule(),
    ComplexityAnalysisRule(),
]


__all__ = [
    "ValidationRule",
    "ValidationIssue",
    "Severity",
    "RuleCategory",
    "DEFAULT_RULES",
    # Specific rules
    "NodeIDFormatRule",
    "RequiredFieldsRule",
    "ConnectionReferencesRule",
    "CircularDependencyRule",
    "TeamFormationRule",
    "GoalQualityRule",
    "ToolBudgetRule",
    "DisconnectedNodesRule",
    "DuplicateNodeIDsRule",
    "ComplexityAnalysisRule",
]
