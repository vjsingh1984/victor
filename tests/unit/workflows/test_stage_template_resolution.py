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

"""Tests for stage template resolution in YAML workflow loader.

Tests for the _resolve_stage_templates() function that enables workflows
to reference predefined stage templates from the WorkflowTemplateRegistry.
"""

import pytest
from typing import Dict, Any

from victor.workflows.yaml_loader import (
    _resolve_stage_templates,
    _deep_merge_dicts,
    load_workflow_from_yaml,
    YAMLWorkflowError,
)
from victor.workflows.template_registry import (
    WorkflowTemplateRegistry,
    get_workflow_template_registry,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the global registry before each test."""
    registry = get_workflow_template_registry()
    registry.clear()

    # Register test stage templates
    registry.register_stage_template(
        "test_stage",
        {
            "description": "Test stage template",
            "type": "agent",
            "role": "executor",
            "name": "Test Stage",
            "goal": "Execute test task",
            "tools": ["read", "write"],
            "tool_budget": 15,
            "llm_config": {"temperature": 0.3},
            "timeout": 120,
        },
    )

    registry.register_stage_template(
        "lint_check_stage",
        {
            "description": "Run linters for code quality",
            "type": "compute",
            "name": "Run Linters",
            "tools": ["shell"],
            "inputs": {"commands": ["$ctx.lint_command", "$ctx.format_check_command"]},
            "output": "lint_results",
            "constraints": ["llm", "write"],
            "timeout": 180,
        },
    )

    registry.register_stage_template(
        "nested_template",
        {
            "description": "Template with nested structure",
            "type": "agent",
            "role": "analyst",
            "name": "Nested Template",
            "goal": "Analyze data",
            "tools": ["read", "grep"],
            "llm_config": {"temperature": 0.2, "model_hint": "claude-3-haiku"},
            "tool_budget": 20,
            "timeout": 180,
        },
    )

    yield

    # Clean up after test
    registry.clear()


# =============================================================================
# Deep Merge Tests
# =============================================================================


class TestDeepMergeDicts:
    """Tests for _deep_merge_dicts() function."""

    def test_simple_merge(self):
        """Test simple dict merging."""
        base = {"a": 1, "b": 2}
        overrides = {"b": 99, "c": 3}
        result = _deep_merge_dicts(base, overrides)

        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_dict_merge(self):
        """Test nested dict merging."""
        base = {"a": 1, "b": {"x": 10, "y": 20}}
        overrides = {"b": {"y": 99}, "c": 3}
        result = _deep_merge_dicts(base, overrides)

        assert result == {"a": 1, "b": {"x": 10, "y": 99}, "c": 3}

    def test_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base = {"a": [1, 2, 3]}
        overrides = {"a": [4, 5]}
        result = _deep_merge_dicts(base, overrides)

        assert result == {"a": [4, 5]}

    def test_deep_copy_immutability(self):
        """Test that original dicts are not mutated."""
        base = {"a": 1, "b": {"x": 10}}
        overrides = {"b": {"y": 20}}

        original_base = base.copy()
        original_base_inner = base["b"].copy()

        result = _deep_merge_dicts(base, overrides)

        # Verify original dict unchanged
        assert base == original_base
        assert base["b"] == original_base_inner

    def test_empty_overrides(self):
        """Test merging with empty overrides."""
        base = {"a": 1, "b": 2}
        overrides = {}
        result = _deep_merge_dicts(base, overrides)

        assert result == {"a": 1, "b": 2}

    def test_empty_base(self):
        """Test merging with empty base."""
        base = {}
        overrides = {"a": 1, "b": 2}
        result = _deep_merge_dicts(base, overrides)

        assert result == {"a": 1, "b": 2}


# =============================================================================
# Stage Template Resolution Tests
# =============================================================================


class TestResolveStageTemplates:
    """Tests for _resolve_stage_templates() function."""

    def test_resolve_single_stage_reference(self):
        """Test resolving a single stage template reference."""
        node_list = [
            {
                "id": "my_task",
                "stage": "test_stage",
                "next": ["next_node"],
            }
        ]

        result = _resolve_stage_templates(node_list)

        assert len(result) == 1
        node = result[0]

        # Should have all template fields
        assert node["id"] == "my_task"
        assert node["type"] == "agent"
        assert node["role"] == "executor"
        assert node["name"] == "Test Stage"
        assert node["goal"] == "Execute test task"
        assert node["tools"] == ["read", "write"]
        assert node["tool_budget"] == 15
        assert node["llm_config"] == {"temperature": 0.3}
        assert node["timeout"] == 120
        assert node["next"] == ["next_node"]

        # Should not have stage reference after resolution
        assert "stage" not in node

    def test_resolve_with_overrides(self):
        """Test resolving stage template with overrides."""
        node_list = [
            {
                "id": "custom_lint",
                "stage": "lint_check_stage",
                "overrides": {"timeout": 240, "name": "Custom Lint Check"},
                "next": ["type_check"],
            }
        ]

        result = _resolve_stage_templates(node_list)

        assert len(result) == 1
        node = result[0]

        # Should have overridden values
        assert node["timeout"] == 240
        assert node["name"] == "Custom Lint Check"

        # Should have other template values
        assert node["type"] == "compute"
        assert node["tools"] == ["shell"]

        # Should not have overrides key
        assert "overrides" not in node
        assert "stage" not in node

    def test_resolve_nested_overrides(self):
        """Test resolving with nested override fields."""
        node_list = [
            {
                "id": "nested_task",
                "stage": "nested_template",
                "overrides": {
                    "llm_config": {"model_hint": "claude-3-sonnet"},
                    "timeout": 300,
                },
            }
        ]

        result = _resolve_stage_templates(node_list)

        assert len(result) == 1
        node = result[0]

        # Should have merged nested llm_config
        assert node["llm_config"]["temperature"] == 0.2  # From template
        assert node["llm_config"]["model_hint"] == "claude-3-sonnet"  # Override

        # Should have overridden timeout
        assert node["timeout"] == 300

    def test_node_level_field_overrides(self):
        """Test that node-level fields override template fields."""
        node_list = [
            {
                "id": "my_task",
                "stage": "test_stage",
                "timeout": 999,  # Direct field override
                "next": ["next_node"],
            }
        ]

        result = _resolve_stage_templates(node_list)

        assert len(result) == 1
        node = result[0]

        # Direct field override should take precedence
        assert node["timeout"] == 999

    def test_mixed_stage_and_regular_nodes(self):
        """Test resolving mix of stage references and regular nodes."""
        node_list = [
            {"id": "stage_ref", "stage": "test_stage"},
            {
                "id": "regular_node",
                "type": "agent",
                "role": "executor",
                "goal": "Regular task",
            },
        ]

        result = _resolve_stage_templates(node_list)

        assert len(result) == 2

        # First node should be resolved from template
        assert result[0]["id"] == "stage_ref"
        assert result[0]["type"] == "agent"
        assert result[0]["role"] == "executor"

        # Second node should be unchanged
        assert result[1]["id"] == "regular_node"
        assert result[1]["type"] == "agent"
        assert result[1]["goal"] == "Regular task"

    def test_missing_stage_template_error(self):
        """Test that missing stage template raises error."""
        node_list = [
            {
                "id": "my_task",
                "stage": "nonexistent_stage",
            }
        ]

        with pytest.raises(YAMLWorkflowError) as exc_info:
            _resolve_stage_templates(node_list)

        assert "nonexistent_stage" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_missing_id_in_resolved_node(self):
        """Test that resolved node without ID raises error."""
        # Create a template without an ID field
        registry = get_workflow_template_registry()
        registry.register_stage_template(
            "no_id_template",
            {
                "type": "agent",
                "role": "executor",
                # Missing 'id' field
            },
        )

        node_list = [
            {
                "stage": "no_id_template",
                # No ID provided in node either
            }
        ]

        with pytest.raises(YAMLWorkflowError) as exc_info:
            _resolve_stage_templates(node_list)

        assert "missing 'id' field" in str(exc_info.value)

    def test_preserves_node_id_over_template_id(self):
        """Test that node ID is always used, even if template has ID."""
        # Create a template with an ID field (unusual but possible)
        registry = get_workflow_template_registry()
        registry.register_stage_template(
            "id_template",
            {
                "id": "template_id",
                "type": "agent",
                "role": "executor",
                "name": "Template ID",
            },
        )

        node_list = [
            {
                "id": "node_id",  # Node's ID should win
                "stage": "id_template",
            }
        ]

        result = _resolve_stage_templates(node_list)

        assert result[0]["id"] == "node_id"

    def test_overrides_with_both_section_and_direct_fields(self):
        """Test when both overrides section and direct fields are present."""
        node_list = [
            {
                "id": "my_task",
                "stage": "test_stage",
                "timeout": 150,  # Direct field
                "overrides": {"tool_budget": 25},  # Overrides section
                "next": ["next_node"],
            }
        ]

        result = _resolve_stage_templates(node_list)

        assert len(result) == 1
        node = result[0]

        # Direct field should override template
        assert node["timeout"] == 150

        # Overrides section should also apply
        assert node["tool_budget"] == 25

        # Other template values should be preserved
        assert node["tools"] == ["read", "write"]

    def test_empty_node_list(self):
        """Test resolving empty node list."""
        result = _resolve_stage_templates([])

        assert result == []

    def test_no_stage_references(self):
        """Test node list with no stage references."""
        node_list = [
            {"id": "node1", "type": "agent", "role": "executor"},
            {"id": "node2", "type": "compute", "name": "Compute task"},
        ]

        result = _resolve_stage_templates(node_list)

        # Should return nodes unchanged
        assert len(result) == 2
        assert result[0]["id"] == "node1"
        assert result[1]["id"] == "node2"


# =============================================================================
# Integration Tests
# =============================================================================


class TestStageTemplateIntegration:
    """Integration tests for stage template resolution in workflow loading."""

    def test_load_workflow_with_stage_templates(self):
        """Test loading a complete workflow with stage template references."""
        yaml_content = """
workflows:
  test_workflow:
    description: "Test workflow with stage templates"
    nodes:
      - id: step1
        stage: test_stage
        next: [step2]

      - id: step2
        stage: lint_check_stage
        overrides:
          timeout: 240
        next: [step3]

      - id: step3
        type: agent
        role: reviewer
        goal: "Review results"
"""

        workflow = load_workflow_from_yaml(yaml_content, "test_workflow")

        assert workflow.name == "test_workflow"
        assert len(workflow.nodes) == 3

        # Verify step1 was resolved from template
        step1 = workflow.nodes["step1"]
        assert hasattr(step1, "role")
        assert step1.role == "executor"

        # Verify step2 was resolved with override
        step2 = workflow.nodes["step2"]
        # Timeout should be overridden
        # (assuming ComputeNode has timeout attribute)

        # Verify step3 is regular node
        step3 = workflow.nodes["step3"]
        assert hasattr(step3, "role")
        assert step3.role == "reviewer"

    def test_stage_template_preserves_flow_structure(self):
        """Test that stage templates don't break workflow flow."""
        yaml_content = """
workflows:
  linear_flow:
    description: "Linear workflow with templates"
    nodes:
      - id: step1
        stage: test_stage

      - id: step2
        stage: lint_check_stage

      - id: step3
        type: transform
        transform: "status = 'done'"
"""

        workflow = load_workflow_from_yaml(yaml_content, "linear_flow")

        # Verify auto-chaining works with resolved templates
        step1 = workflow.nodes["step1"]
        step2 = workflow.nodes["step2"]
        step3 = workflow.nodes["step3"]

        # Nodes should be auto-chained
        assert "step2" in step1.next_nodes
        assert "step3" in step2.next_nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
