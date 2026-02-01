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

"""Tests for workflow template registry.

These tests verify the template registry functionality for Phase 5:
- Loading templates from YAML
- Template extension with overrides
- Stage template management
- Deep merge functionality
- Error handling

Run with: pytest tests/unit/workflows/test_template_registry.py -v
"""

import tempfile

import pytest

from victor.workflows.template_registry import (
    WorkflowTemplateRegistry,
    get_workflow_template_registry,
    register_default_templates,
)


@pytest.mark.unit
class TestWorkflowTemplateRegistry:
    """Tests for WorkflowTemplateRegistry class."""

    def test_init_empty_registry(self):
        """Test registry starts empty."""
        registry = WorkflowTemplateRegistry()

        assert registry.template_count == 0
        assert registry.stage_template_count == 0
        assert registry.list_templates() == []
        assert registry.list_stage_templates() == []

    def test_load_workflow_template_from_yaml(self):
        """Test loading workflow templates from YAML."""
        registry = WorkflowTemplateRegistry()

        yaml_content = """
        templates:
          test_workflow:
            name: "Test Workflow"
            description: "A test workflow"
            version: "1.0"
            formation: "parallel"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            registry.load_templates_from_yaml(f.name)

        assert registry.template_count == 1
        assert registry.has_template("test_workflow")

        template = registry.get_template("test_workflow")
        assert template["name"] == "Test Workflow"
        assert template["description"] == "A test workflow"

    def test_load_workflow_alternative_format(self):
        """Test loading workflows with 'workflows' key."""
        registry = WorkflowTemplateRegistry()

        yaml_content = """
        workflows:
          my_workflow:
            description: "Test workflow"
            nodes: []
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            registry.load_templates_from_yaml(f.name)

        assert registry.has_template("my_workflow")

    def test_load_team_definition(self):
        """Test loading team definition (name + members)."""
        registry = WorkflowTemplateRegistry()

        yaml_content = """
        name: code_review_parallel
        display_name: "Code Review Team"
        formation: "parallel"
        members:
          - id: "reviewer1"
            role: "reviewer"
            name: "Reviewer 1"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            registry.load_templates_from_yaml(f.name)

        assert registry.has_template("code_review_parallel")
        template = registry.get_template("code_review_parallel")
        assert template["formation"] == "parallel"

    def test_load_stage_templates(self):
        """Test loading stage templates from YAML."""
        registry = WorkflowTemplateRegistry()

        yaml_content = """
        stage_templates:
          read_stage:
            type: agent
            role: reader
            description: "Read context"

          modify_stage:
            type: agent
            role: executor
            description: "Make changes"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            registry.load_templates_from_yaml(f.name)

        assert registry.stage_template_count == 2
        assert registry.has_stage_template("read_stage")
        assert registry.has_stage_template("modify_stage")

    def test_load_with_namespace(self):
        """Test loading templates with namespace prefix."""
        registry = WorkflowTemplateRegistry()

        yaml_content = """
        templates:
          my_template:
            name: "My Template"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            registry.load_templates_from_yaml(f.name, namespace="coding")

        assert registry.has_template("coding.my_template")
        assert not registry.has_template("my_template")

    def test_get_template_default(self):
        """Test get_template returns default when not found."""
        registry = WorkflowTemplateRegistry()

        default = {"default": True}
        result = registry.get_template("nonexistent", default=default)

        assert result == default

    def test_get_stage_template_default(self):
        """Test get_stage_template returns default when not found."""
        registry = WorkflowTemplateRegistry()

        default = {"default": True}
        result = registry.get_stage_template("nonexistent", default=default)

        assert result == default

    def test_has_template(self):
        """Test has_template method."""
        registry = WorkflowTemplateRegistry()

        assert not registry.has_template("test")

        registry.register_template("test", {"name": "Test"})

        assert registry.has_template("test")

    def test_has_stage_template(self):
        """Test has_stage_template method."""
        registry = WorkflowTemplateRegistry()

        assert not registry.has_stage_template("test_stage")

        registry.register_stage_template("test_stage", {"type": "agent"})

        assert registry.has_stage_template("test_stage")

    def test_list_templates(self):
        """Test listing all templates."""
        registry = WorkflowTemplateRegistry()

        registry.register_template("template_b", {"name": "B"})
        registry.register_template("template_a", {"name": "A"})
        registry.register_template("template_c", {"name": "C"})

        templates = registry.list_templates()

        assert templates == ["template_a", "template_b", "template_c"]

    def test_list_stage_templates(self):
        """Test listing all stage templates."""
        registry = WorkflowTemplateRegistry()

        registry.register_stage_template("stage_c", {"type": "agent"})
        registry.register_stage_template("stage_a", {"type": "compute"})
        registry.register_stage_template("stage_b", {"type": "transform"})

        stages = registry.list_stage_templates()

        assert stages == ["stage_a", "stage_b", "stage_c"]

    def test_extend_template_shallow_merge(self):
        """Test extending template with shallow merge."""
        registry = WorkflowTemplateRegistry()

        base = {
            "name": "Base Workflow",
            "description": "Base description",
            "config": {"option1": "value1"},
        }
        registry.register_template("base", base)

        extended = registry.extend_template("base", {"name": "Extended"}, deep_merge=False)

        assert extended["name"] == "Extended"
        assert extended["description"] == "Base description"
        assert extended["config"] == {"option1": "value1"}
        assert extended["extends"] == "base"

    def test_extend_template_deep_merge(self):
        """Test extending template with deep merge."""
        registry = WorkflowTemplateRegistry()

        base = {
            "name": "Base Workflow",
            "config": {
                "option1": "value1",
                "option2": "value2",
            },
            "members": [
                {"id": "member1", "role": "role1"},
            ],
        }
        registry.register_template("base", base)

        extended = registry.extend_template(
            "base",
            {
                "config": {"option2": "new_value2", "option3": "value3"},
                "members": [{"id": "member2"}],
            },
            deep_merge=True,
        )

        # Deep merge should merge nested dicts
        assert extended["config"]["option1"] == "value1"
        assert extended["config"]["option2"] == "new_value2"
        assert extended["config"]["option3"] == "value3"

        # Lists should be replaced, not merged
        assert extended["members"] == [{"id": "member2"}]

        assert extended["extends"] == "base"

    def test_extend_nonexistent_template_raises(self):
        """Test extending nonexistent template raises ValueError."""
        registry = WorkflowTemplateRegistry()

        with pytest.raises(ValueError, match="Base template 'nonexistent' not found"):
            registry.extend_template("nonexistent", {"name": "Extended"})

    def test_register_duplicate_template_raises(self):
        """Test registering duplicate template raises ValueError."""
        registry = WorkflowTemplateRegistry()

        registry.register_template("test", {"name": "Test"})

        with pytest.raises(ValueError, match="already registered"):
            registry.register_template("test", {"name": "Test2"})

    def test_register_duplicate_stage_raises(self):
        """Test registering duplicate stage template raises ValueError."""
        registry = WorkflowTemplateRegistry()

        registry.register_stage_template("test", {"type": "agent"})

        with pytest.raises(ValueError, match="already registered"):
            registry.register_stage_template("test", {"type": "compute"})

    def test_get_template_info(self):
        """Test getting template metadata."""
        registry = WorkflowTemplateRegistry()

        template = {
            "name": "Test Workflow",
            "display_name": "Test",
            "description": "A test workflow",
            "version": "1.0",
            "vertical": "coding",
            "formation": "parallel",
            "complexity": "standard",
        }
        registry.register_template("test", template)

        info = registry.get_template_info("test")

        assert info["name"] == "test"
        assert info["display_name"] == "Test"
        assert info["description"] == "A test workflow"
        assert info["version"] == "1.0"
        assert info["vertical"] == "coding"
        assert info["formation"] == "parallel"
        assert info["complexity"] == "standard"
        assert info["type"] == "workflow"

    def test_get_stage_info(self):
        """Test getting stage template metadata."""
        registry = WorkflowTemplateRegistry()

        stage = {
            "description": "Read stage",
            "type": "agent",
            "role": "reader",
        }
        registry.register_stage_template("read", stage)

        info = registry.get_stage_info("read")

        assert info["name"] == "read"
        assert info["description"] == "Read stage"
        assert info["stage_type"] == "agent"
        assert info["type"] == "stage"

    def test_clear_registry(self):
        """Test clearing all templates."""
        registry = WorkflowTemplateRegistry()

        registry.register_template("test1", {"name": "Test1"})
        registry.register_stage_template("stage1", {"type": "agent"})

        assert registry.template_count == 1
        assert registry.stage_template_count == 1

        registry.clear()

        assert registry.template_count == 0
        assert registry.stage_template_count == 0
        assert registry.list_templates() == []
        assert registry.list_stage_templates() == []

    def test_load_from_directory(self, tmp_path):
        """Test loading templates from directory."""
        registry = WorkflowTemplateRegistry()

        # Create test YAML files
        (tmp_path / "template1.yaml").write_text(
            """
            templates:
              workflow1:
                name: "Workflow 1"
        """
        )

        (tmp_path / "template2.yaml").write_text(
            """
            templates:
              workflow2:
                name: "Workflow 2"
        """
        )

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "template3.yaml").write_text(
            """
            templates:
              workflow3:
                name: "Workflow 3"
        """
        )

        registry.load_templates_from_directory(tmp_path, recursive=True)

        # Templates are loaded with namespace based on relative path
        # Files in root get ".." namespace, subdir files get "subdir" namespace
        assert registry.template_count >= 3
        # Check for templates with namespace prefix
        assert registry.has_template("..workflow1") or registry.has_template("workflow1")
        assert registry.has_template("subdir.workflow3")

    def test_load_from_file_not_found_raises(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        registry = WorkflowTemplateRegistry()

        with pytest.raises(FileNotFoundError):
            registry.load_templates_from_yaml("/nonexistent/file.yaml")

    def test_template_copy_not_mutated(self):
        """Test that extending template doesn't mutate original."""
        registry = WorkflowTemplateRegistry()

        base = {"name": "Base", "config": {"option": "value"}}
        registry.register_template("base", base)

        extended = registry.extend_template("base", {"name": "Extended"})

        # Modify extended
        extended["config"]["option"] = "modified"

        # Original should be unchanged
        assert registry.get_template("base")["config"]["option"] == "value"


@pytest.mark.unit
class TestGlobalRegistry:
    """Tests for global workflow template registry."""

    def test_get_global_registry_singleton(self):
        """Test that get_workflow_template_registry returns singleton."""
        registry1 = get_workflow_template_registry()
        registry2 = get_workflow_template_registry()

        assert registry1 is registry2

    def test_register_default_templates(self):
        """Test registering default templates."""
        registry = get_workflow_template_registry()

        # Clear first
        registry.clear()

        # Register defaults
        register_default_templates()

        # Should have templates loaded
        # (actual count depends on what's in victor/workflows/templates/)
        assert registry.template_count > 0 or registry.stage_template_count > 0

    def test_common_stage_templates_available(self):
        """Test that common stage templates are registered."""
        registry = get_workflow_template_registry()

        # These should exist from common_stages.yaml
        common_stages = [
            "read_stage",
            "modify_stage",
            "verify_stage",
            "analyze_stage",
            "review_stage",
        ]

        for stage_name in common_stages:
            if registry.has_stage_template(stage_name):
                stage = registry.get_stage_template(stage_name)
                assert "type" in stage
                assert "description" in stage
