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

"""Tests for workflow linter.

Tests the comprehensive workflow validation and linting system including:
- Validation rules
- Linter functionality
- Report generation
- CLI integration
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from victor.workflows.linter import WorkflowLinter, LinterResult, lint_dict
from victor.workflows.validation_rules import (
    NodeIDFormatRule,
    RequiredFieldsRule,
    ConnectionReferencesRule,
    CircularDependencyRule,
    TeamFormationRule,
    GoalQualityRule,
    ToolBudgetRule,
    DisconnectedNodesRule,
    DuplicateNodeIDsRule,
    ComplexityAnalysisRule,
    Severity,
    RuleCategory,
    ValidationIssue,
    ValidationRule,
)


class TestValidationRule:
    """Tests for ValidationRule base class."""

    def test_rule_initialization(self):
        """Test rule initialization."""
        rule = NodeIDFormatRule()
        assert rule.rule_id == "node_id_format"
        assert rule.category == RuleCategory.SYNTAX
        assert rule.severity == Severity.WARNING
        assert rule.enabled is True

    def test_create_issue(self):
        """Test issue creation."""
        rule = NodeIDFormatRule()
        issue = rule.create_issue(
            message="Test issue",
            location="test_workflow:test_node",
            suggestion="Fix it",
        )

        assert issue.rule_id == "node_id_format"
        assert issue.message == "Test issue"
        assert issue.location == "test_workflow:test_node"
        assert issue.suggestion == "Fix it"
        assert issue.severity == Severity.WARNING


class TestNodeIDFormatRule:
    """Tests for NodeIDFormatRule."""

    @pytest.fixture
    def valid_workflow(self) -> Dict[str, Any]:
        """Valid workflow for testing."""
        return {
            "workflows": {
                "test_workflow": {
                    "description": "Test workflow",
                    "nodes": [
                        {
                            "id": "start_node",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Test goal",
                        },
                        {"id": "end_node", "type": "agent", "role": "writer", "goal": "End goal"},
                    ],
                }
            }
        }

    def test_valid_node_ids(self, valid_workflow):
        """Test valid node IDs pass validation."""
        rule = NodeIDFormatRule()
        issues = rule.check(valid_workflow)
        assert len(issues) == 0

    def test_uppercase_node_id(self, valid_workflow):
        """Test uppercase node IDs are flagged."""
        valid_workflow["workflows"]["test_workflow"]["nodes"][0]["id"] = "StartNode"
        rule = NodeIDFormatRule()
        issues = rule.check(valid_workflow)
        assert len(issues) == 1
        assert "uppercase" in issues[0].message.lower() or "lowercase" in issues[0].message.lower()

    def test_hyphen_in_node_id(self, valid_workflow):
        """Test hyphens in node IDs are flagged."""
        valid_workflow["workflows"]["test_workflow"]["nodes"][0]["id"] = "start-node"
        rule = NodeIDFormatRule()
        issues = rule.check(valid_workflow)
        assert len(issues) == 1
        assert "start_node" in issues[0].suggestion.lower()

    def test_missing_node_id(self, valid_workflow):
        """Test missing node IDs are flagged."""
        valid_workflow["workflows"]["test_workflow"]["nodes"][0]["id"] = ""
        rule = NodeIDFormatRule()
        issues = rule.check(valid_workflow)
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR

    def test_too_long_node_id(self, valid_workflow):
        """Test overly long node IDs are flagged."""
        valid_workflow["workflows"]["test_workflow"]["nodes"][0]["id"] = "a" * 101
        rule = NodeIDFormatRule()
        issues = rule.check(valid_workflow)
        assert len(issues) == 1
        assert "too long" in issues[0].message.lower()


class TestRequiredFieldsRule:
    """Tests for RequiredFieldsRule."""

    @pytest.fixture
    def workflow_template(self) -> Dict[str, Any]:
        """Workflow template for testing."""
        return {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [],
                }
            }
        }

    def test_agent_node_missing_role(self, workflow_template):
        """Test agent node without role is flagged."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {"id": "test", "type": "agent", "goal": "Test"}
        )
        rule = RequiredFieldsRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1
        assert "role" in issues[0].message.lower()

    def test_agent_node_missing_goal(self, workflow_template):
        """Test agent node without goal is flagged."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {"id": "test", "type": "agent", "role": "researcher"}
        )
        rule = RequiredFieldsRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1
        assert "goal" in issues[0].message.lower()

    def test_compute_node_missing_handler(self, workflow_template):
        """Test compute node without handler is flagged."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {"id": "test", "type": "compute"}
        )
        rule = RequiredFieldsRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1
        assert "handler" in issues[0].message.lower()

    def test_team_node_missing_fields(self, workflow_template):
        """Test team node without required fields."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {"id": "test", "type": "team"}
        )
        rule = RequiredFieldsRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 3  # team_formation, members, goal


class TestConnectionReferencesRule:
    """Tests for ConnectionReferencesRule."""

    @pytest.fixture
    def workflow_template(self) -> Dict[str, Any]:
        """Workflow template for testing."""
        return {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [
                        {"id": "node1", "type": "agent", "role": "researcher", "goal": "Test"},
                        {"id": "node2", "type": "agent", "role": "writer", "goal": "Test"},
                    ],
                }
            }
        }

    def test_valid_next_references(self, workflow_template):
        """Test valid next references."""
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["next"] = ["node2"]
        rule = ConnectionReferencesRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 0

    def test_invalid_next_reference(self, workflow_template):
        """Test invalid next reference."""
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["next"] = ["nonexistent"]
        rule = ConnectionReferencesRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1
        assert "nonexistent" in issues[0].message

    def test_valid_branch_references(self, workflow_template):
        """Test valid branch references."""
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["type"] = "condition"
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["condition"] = "test"
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["branches"] = {
            "true": "node2",
            "false": "__end__",
        }
        rule = ConnectionReferencesRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 0

    def test_invalid_branch_reference(self, workflow_template):
        """Test invalid branch reference."""
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["type"] = "condition"
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["branches"] = {
            "true": "nonexistent"
        }
        rule = ConnectionReferencesRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1

    def test_parallel_nodes_reference(self, workflow_template):
        """Test parallel_nodes references."""
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["type"] = "parallel"
        workflow_template["workflows"]["test_workflow"]["nodes"][0]["parallel_nodes"] = [
            "node2",
            "nonexistent",
        ]
        rule = ConnectionReferencesRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1


class TestCircularDependencyRule:
    """Tests for CircularDependencyRule."""

    def test_detect_cycle(self):
        """Test circular dependency detection."""
        workflow = {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [
                        {
                            "id": "node1",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Test",
                            "next": ["node2"],
                        },
                        {
                            "id": "node2",
                            "type": "agent",
                            "role": "writer",
                            "goal": "Test",
                            "next": ["node3"],
                        },
                        {
                            "id": "node3",
                            "type": "agent",
                            "role": "analyst",
                            "goal": "Test",
                            "next": ["node1"],  # Creates cycle
                        },
                    ],
                }
            }
        }
        rule = CircularDependencyRule()
        issues = rule.check(workflow)
        assert len(issues) == 1
        assert "circular" in issues[0].message.lower()

    def test_no_cycle(self):
        """Test workflow without cycle."""
        workflow = {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [
                        {
                            "id": "node1",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Test",
                            "next": ["node2"],
                        },
                        {
                            "id": "node2",
                            "type": "agent",
                            "role": "writer",
                            "goal": "Test",
                        },
                    ],
                }
            }
        }
        rule = CircularDependencyRule()
        issues = rule.check(workflow)
        assert len(issues) == 0


class TestTeamFormationRule:
    """Tests for TeamFormationRule."""

    @pytest.fixture
    def workflow_template(self) -> Dict[str, Any]:
        """Workflow template for testing."""
        return {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [],
                }
            }
        }

    def test_valid_team_node(self, workflow_template):
        """Test valid team node."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {
                "id": "team1",
                "type": "team",
                "team_formation": "parallel",
                "goal": "Test team goal",
                "members": [
                    {"id": "member1", "role": "researcher", "goal": "Research"},
                    {"id": "member2", "role": "writer", "goal": "Write"},
                ],
            }
        )
        rule = TeamFormationRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 0

    def test_invalid_formation_type(self, workflow_template):
        """Test invalid formation type."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {
                "id": "team1",
                "type": "team",
                "team_formation": "invalid_formation",
                "goal": "Test",
                "members": [],
            }
        )
        rule = TeamFormationRule()
        issues = rule.check(workflow_template)
        # Should have at least 2 issues: invalid formation + no members
        assert len(issues) >= 1
        formation_issues = [i for i in issues if "invalid" in i.message.lower()]
        assert len(formation_issues) >= 1

    def test_team_member_missing_role(self, workflow_template):
        """Test team member without role."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {
                "id": "team1",
                "type": "team",
                "team_formation": "parallel",
                "goal": "Test",
                "members": [{"id": "member1", "goal": "Test"}],
            }
        )
        rule = TeamFormationRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1
        assert "role" in issues[0].message.lower()

    def test_invalid_max_iterations(self, workflow_template):
        """Test invalid max_iterations."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {
                "id": "team1",
                "type": "team",
                "team_formation": "parallel",
                "goal": "Test",
                "members": [{"id": "member1", "role": "researcher", "goal": "Test"}],
                "max_iterations": 15,  # Too high
            }
        )
        rule = TeamFormationRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1
        assert "max_iterations" in issues[0].message.lower()


class TestGoalQualityRule:
    """Tests for GoalQualityRule."""

    @pytest.fixture
    def workflow_template(self) -> Dict[str, Any]:
        """Workflow template for testing."""
        return {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [],
                }
            }
        }

    def test_good_goal(self, workflow_template):
        """Test good quality goal."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {
                "id": "agent1",
                "type": "agent",
                "role": "researcher",
                "goal": "Research the latest AI trends and summarize findings",
            }
        )
        rule = GoalQualityRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 0

    def test_short_goal(self, workflow_template):
        """Test goal that's too short."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {"id": "agent1", "type": "agent", "role": "researcher", "goal": "Do task"}
        )
        rule = GoalQualityRule()
        issues = rule.check(workflow_template)
        # Should have at least 2 issues: too short + generic
        assert len(issues) >= 1
        short_issues = [i for i in issues if "too short" in i.message.lower()]
        assert len(short_issues) >= 1

    def test_generic_goal(self, workflow_template):
        """Test generic goal."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {"id": "agent1", "type": "agent", "role": "researcher", "goal": "execute"}
        )
        rule = GoalQualityRule()
        issues = rule.check(workflow_template)
        # Should have at least 2 issues: too short + generic
        assert len(issues) >= 1
        generic_issues = [i for i in issues if "generic" in i.message.lower()]
        assert len(generic_issues) >= 1


class TestToolBudgetRule:
    """Tests for ToolBudgetRule."""

    @pytest.fixture
    def workflow_template(self) -> Dict[str, Any]:
        """Workflow template for testing."""
        return {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [],
                }
            }
        }

    def test_reasonable_budget(self, workflow_template):
        """Test reasonable tool budget."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {
                "id": "agent1",
                "type": "agent",
                "role": "researcher",
                "goal": "Test",
                "tool_budget": 20,
            }
        )
        rule = ToolBudgetRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 0

    def test_very_high_budget(self, workflow_template):
        """Test very high tool budget."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {
                "id": "agent1",
                "type": "agent",
                "role": "researcher",
                "goal": "Test",
                "tool_budget": 150,
            }
        )
        rule = ToolBudgetRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1
        assert "very high" in issues[0].message.lower()

    def test_negative_budget(self, workflow_template):
        """Test negative tool budget."""
        workflow_template["workflows"]["test_workflow"]["nodes"].append(
            {
                "id": "agent1",
                "type": "agent",
                "role": "researcher",
                "goal": "Test",
                "tool_budget": -5,
            }
        )
        rule = ToolBudgetRule()
        issues = rule.check(workflow_template)
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR


class TestDisconnectedNodesRule:
    """Tests for DisconnectedNodesRule."""

    def test_disconnected_nodes(self):
        """Test detection of disconnected nodes."""
        workflow = {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "entry_point": "node1",
                    "nodes": [
                        {
                            "id": "node1",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Test",
                            "next": ["node2"],
                        },
                        {"id": "node2", "type": "agent", "role": "writer", "goal": "Test"},
                        {
                            "id": "node3",
                            "type": "agent",
                            "role": "analyst",
                            "goal": "Test",
                        },  # Disconnected
                    ],
                }
            }
        }
        rule = DisconnectedNodesRule()
        issues = rule.check(workflow)
        assert len(issues) == 1
        assert "disconnected" in issues[0].message.lower()

    def test_all_connected(self):
        """Test workflow with all nodes connected."""
        workflow = {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "entry_point": "node1",
                    "nodes": [
                        {
                            "id": "node1",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Test",
                            "next": ["node2"],
                        },
                        {"id": "node2", "type": "agent", "role": "writer", "goal": "Test"},
                    ],
                }
            }
        }
        rule = DisconnectedNodesRule()
        issues = rule.check(workflow)
        assert len(issues) == 0


class TestDuplicateNodeIDsRule:
    """Tests for DuplicateNodeIDsRule."""

    def test_duplicate_ids(self):
        """Test detection of duplicate node IDs."""
        workflow = {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [
                        {"id": "node1", "type": "agent", "role": "researcher", "goal": "Test"},
                        {"id": "node1", "type": "agent", "role": "writer", "goal": "Test"},
                        {"id": "node2", "type": "agent", "role": "analyst", "goal": "Test"},
                    ],
                }
            }
        }
        rule = DuplicateNodeIDsRule()
        issues = rule.check(workflow)
        assert len(issues) == 1
        assert "duplicate" in issues[0].message.lower()
        assert "node1" in issues[0].message


class TestComplexityAnalysisRule:
    """Tests for ComplexityAnalysisRule."""

    def test_simple_workflow(self):
        """Test simple workflow complexity."""
        workflow = {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "entry_point": "node1",
                    "nodes": [
                        {
                            "id": "node1",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Test",
                            "next": ["node2"],
                        },
                        {"id": "node2", "type": "agent", "role": "writer", "goal": "Test"},
                    ],
                }
            }
        }
        rule = ComplexityAnalysisRule()
        issues = rule.check(workflow)
        assert len(issues) == 1
        assert issues[0].severity == Severity.INFO
        assert "complexity" in issues[0].message.lower()

    def test_complex_workflow(self):
        """Test complex workflow warning."""
        # Create workflow with 51 nodes
        nodes = []
        for i in range(51):
            nodes.append(
                {
                    "id": f"node{i}",
                    "type": "agent",
                    "role": "researcher",
                    "goal": f"Task {i}",
                }
            )

        workflow = {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "entry_point": "node0",
                    "nodes": nodes,
                }
            }
        }
        rule = ComplexityAnalysisRule()
        issues = rule.check(workflow)
        # Should have info issue + warning about too many nodes
        assert len(issues) == 2
        warning_issues = [i for i in issues if i.severity == Severity.WARNING]
        assert len(warning_issues) == 1
        assert "many nodes" in warning_issues[0].message.lower()


class TestWorkflowLinter:
    """Tests for WorkflowLinter."""

    @pytest.fixture
    def valid_workflow(self) -> Dict[str, Any]:
        """Valid workflow for testing."""
        return {
            "workflows": {
                "test_workflow": {
                    "description": "Test workflow",
                    "nodes": [
                        {
                            "id": "start_node",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Conduct comprehensive research on AI trends",
                            "tool_budget": 25,
                            "next": ["end_node"],
                        },
                        {
                            "id": "end_node",
                            "type": "agent",
                            "role": "writer",
                            "goal": "Write detailed report on findings",
                            "tool_budget": 15,
                        },
                    ],
                }
            }
        }

    def test_lint_valid_workflow(self, valid_workflow):
        """Test linting a valid workflow."""
        linter = WorkflowLinter()
        result = linter.lint_dict(valid_workflow)
        assert result.is_valid
        assert result.error_count == 0

    def test_lint_invalid_workflow(self):
        """Test linting an invalid workflow."""
        workflow = {
            "workflows": {
                "test_workflow": {
                    "description": "Test",
                    "nodes": [
                        {
                            "id": "Start-Node",  # Invalid ID format
                            "type": "agent",
                            "role": "researcher",
                            "goal": "x",  # Too short
                            "tool_budget": -5,  # Invalid
                        },
                        {
                            "id": "Start-Node",  # Duplicate ID
                            "type": "agent",
                            "role": "writer",
                            "goal": "test",
                            "next": ["nonexistent"],  # Invalid reference
                        },
                    ],
                }
            }
        }
        linter = WorkflowLinter()
        result = linter.lint_dict(workflow)
        assert not result.is_valid
        assert result.error_count > 0

    def test_lint_file(self, valid_workflow, tmp_path):
        """Test linting a file."""
        import yaml

        workflow_file = tmp_path / "test_workflow.yaml"
        with open(workflow_file, "w") as f:
            yaml.dump(valid_workflow, f)

        linter = WorkflowLinter()
        result = linter.lint_file(workflow_file)
        assert result.files_checked == 1
        assert result.workflow_count == 1

    def test_lint_directory(self, valid_workflow, tmp_path):
        """Test linting a directory."""
        import yaml

        # Create multiple workflow files
        for i in range(3):
            workflow_file = tmp_path / f"workflow{i}.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(valid_workflow, f)

        linter = WorkflowLinter()
        result = linter.lint_directory(tmp_path)
        assert result.files_checked == 3
        assert result.workflow_count == 3

    def test_enable_disable_rules(self, valid_workflow):
        """Test enabling/disabling rules."""
        linter = WorkflowLinter()

        # Disable a rule
        linter.disable_rule("node_id_format")
        assert linter.get_rule("node_id_format").enabled is False

        # Enable a rule
        linter.enable_rule("node_id_format")
        assert linter.get_rule("node_id_format").enabled is True

    def test_set_rule_severity(self, valid_workflow):
        """Test changing rule severity."""
        linter = WorkflowLinter()
        linter.set_rule_severity("tool_budget", Severity.ERROR)
        assert linter.get_rule("tool_budget").severity == Severity.ERROR

    def test_add_custom_rule(self, valid_workflow):
        """Test adding custom rule."""

        class CustomRule(ValidationRule):
            def __init__(self):
                super().__init__(
                    rule_id="custom_rule",
                    category=RuleCategory.BEST_PRACTICES,
                    severity=Severity.INFO,
                )

            def check(self, workflow):
                return [
                    self.create_issue(
                        message="Custom check",
                        location="test",
                    )
                ]

        linter = WorkflowLinter()
        linter.add_rule(CustomRule())
        result = linter.lint_dict(valid_workflow)
        assert "custom_rule" in [i.rule_id for i in result.issues]


class TestLinterResult:
    """Tests for LinterResult."""

    @pytest.fixture
    def sample_issues(self):
        """Create sample issues for testing."""
        return [
            ValidationIssue(
                rule_id="rule1",
                severity=Severity.ERROR,
                category=RuleCategory.SYNTAX,
                message="Error message",
                location="workflow:node1",
            ),
            ValidationIssue(
                rule_id="rule2",
                severity=Severity.WARNING,
                category=RuleCategory.BEST_PRACTICES,
                message="Warning message",
                location="workflow:node2",
            ),
            ValidationIssue(
                rule_id="rule3",
                severity=Severity.INFO,
                category=RuleCategory.COMPLEXITY,
                message="Info message",
                location="workflow:node3",
            ),
        ]

    def test_issue_counts(self, sample_issues):
        """Test issue count properties."""
        result = LinterResult(issues=sample_issues)
        assert result.error_count == 1
        assert result.warning_count == 1
        assert result.info_count == 1
        assert result.suggestion_count == 0

    def test_has_errors(self, sample_issues):
        """Test has_errors property."""
        result = LinterResult(issues=sample_issues)
        assert result.has_errors is True
        assert result.is_valid is False

    def test_filter_by_severity(self, sample_issues):
        """Test filtering by severity."""
        result = LinterResult(issues=sample_issues)
        errors = result.get_issues_by_severity(Severity.ERROR)
        assert len(errors) == 1
        assert errors[0].severity == Severity.ERROR

    def test_filter_by_category(self, sample_issues):
        """Test filtering by category."""
        result = LinterResult(issues=sample_issues)
        syntax_issues = result.get_issues_by_category(RuleCategory.SYNTAX)
        assert len(syntax_issues) == 1
        assert syntax_issues[0].category == RuleCategory.SYNTAX

    def test_filter_by_location(self, sample_issues):
        """Test filtering by location."""
        result = LinterResult(issues=sample_issues)
        node1_issues = result.get_issues_by_location("workflow:node1")
        assert len(node1_issues) == 1

    def test_generate_text_report(self, sample_issues):
        """Test text report generation."""
        result = LinterResult(issues=sample_issues, duration_seconds=0.5)
        report = result.generate_report(format="text")
        assert "ERROR" in report
        assert "WARNING" in report
        assert "Error message" in report
        assert "0.50s" in report

    def test_generate_json_report(self, sample_issues):
        """Test JSON report generation."""
        import json

        result = LinterResult(issues=sample_issues)
        report = result.generate_report(format="json")
        data = json.loads(report)
        assert "summary" in data
        assert "issues" in data
        assert data["summary"]["error_count"] == 1
        assert len(data["issues"]) == 3

    def test_generate_markdown_report(self, sample_issues):
        """Test Markdown report generation."""
        result = LinterResult(issues=sample_issues)
        report = result.generate_report(format="markdown")
        assert "# Workflow Linting Report" in report
        assert "## Summary" in report
        assert "## Issues" in report
        assert "❌" in report
        assert "⚠️" in report


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_lint_dict(self):
        """Test lint_dict convenience function."""
        workflow = {
            "workflows": {
                "test": {
                    "description": "Test",
                    "nodes": [
                        {
                            "id": "node1",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Test goal",
                        }
                    ],
                }
            }
        }
        result = lint_dict(workflow)
        assert isinstance(result, LinterResult)
        assert result.workflow_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
