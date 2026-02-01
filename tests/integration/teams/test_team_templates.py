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

"""Tests for team template system."""

import pytest
from pathlib import Path

from victor.workflows.team_templates import (
    TeamTemplate,
    TeamMemberSpec,
    TeamTemplateRegistry,
    get_registry,
    get_template,
    register_template,
    list_templates,
    search_templates,
)


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    members = [
        TeamMemberSpec(
            id="member1",
            role="researcher",
            name="Member One",
            goal="Research the topic",
            tool_budget=20,
        ),
        TeamMemberSpec(
            id="member2",
            role="executor",
            name="Member Two",
            goal="Execute the plan",
            tool_budget=30,
        ),
    ]

    return TeamTemplate(
        name="test_template",
        display_name="Test Template",
        description="A test template",
        formation="sequential",
        members=members,
        vertical="general",
        complexity="quick",
        max_iterations=25,
        total_tool_budget=50,
        timeout_seconds=300,
    )


@pytest.fixture
def registry(tmp_path):
    """Create a test registry."""
    # Use a temporary directory to avoid loading YAML templates
    return TeamTemplateRegistry(template_dir=tmp_path / "templates")


class TestTeamMemberSpec:
    """Tests for TeamMemberSpec."""

    def test_create_member_spec(self):
        """Test creating a member specification."""
        spec = TeamMemberSpec(
            id="test_member",
            role="researcher",
            name="Test Member",
            goal="Test goal",
            backstory="Test backstory",
            expertise=["test1", "test2"],
            tool_budget=25,
        )

        assert spec.id == "test_member"
        assert spec.role == "researcher"
        assert spec.name == "Test Member"
        assert spec.backstory == "Test backstory"
        assert spec.expertise == ["test1", "test2"]
        assert spec.tool_budget == 25

    def test_to_member(self):
        """Test converting to TeamMember."""
        spec = TeamMemberSpec(
            id="test_member",
            role="researcher",
            name="Test Member",
            goal="Test goal",
            tool_budget=25,
        )

        member = spec.to_member()

        assert member.id == "test_member"
        assert member.role.value == "researcher"
        assert member.name == "Test Member"
        assert member.goal == "Test goal"
        assert member.tool_budget == 25

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "test_member",
            "role": "researcher",
            "name": "Test Member",
            "goal": "Test goal",
            "tool_budget": 25,
            "backstory": "Test backstory",
            "expertise": ["test1", "test2"],
        }

        spec = TeamMemberSpec.from_dict(data)

        assert spec.id == "test_member"
        assert spec.backstory == "Test backstory"
        assert spec.expertise == ["test1", "test2"]


class TestTeamTemplate:
    """Tests for TeamTemplate."""

    def test_create_template(self, sample_template):
        """Test creating a template."""
        assert sample_template.name == "test_template"
        assert sample_template.formation == "sequential"
        assert len(sample_template.members) == 2
        assert sample_template.max_iterations == 25
        assert sample_template.total_tool_budget == 50

    def test_template_validation_empty_members(self):
        """Test validation fails with no members."""
        with pytest.raises(ValueError, match="must have at least one member"):
            TeamTemplate(
                name="invalid",
                display_name="Invalid",
                description="Invalid template",
                formation="sequential",
                members=[],
            )

    def test_template_validation_invalid_formation(self):
        """Test validation fails with invalid formation."""
        members = [
            TeamMemberSpec(
                id="m1",
                role="researcher",
                name="M1",
                goal="Test",
            )
        ]

        with pytest.raises(ValueError, match="Invalid formation"):
            TeamTemplate(
                name="invalid",
                display_name="Invalid",
                description="Invalid template",
                formation="invalid_formation",
                members=members,
            )

    def test_template_validation_hierarchical_no_manager(self):
        """Test hierarchical validation fails without manager."""
        members = [
            TeamMemberSpec(
                id="m1",
                role="researcher",
                name="M1",
                goal="Test",
                can_delegate=False,
            )
        ]

        with pytest.raises(ValueError, match="exactly one member"):
            TeamTemplate(
                name="invalid",
                display_name="Invalid",
                description="Invalid template",
                formation="hierarchical",
                members=members,
            )

    def test_to_team_config(self, sample_template):
        """Test converting to TeamConfig."""
        config = sample_template.to_team_config(
            goal="Custom goal",
            context={"key": "value"},
        )

        assert config.name == "Test Template"
        assert config.goal == "Custom goal"
        assert config.formation.value == "sequential"
        assert len(config.members) == 2
        assert config.shared_context == {"key": "value"}

    def test_to_dict(self, sample_template):
        """Test converting to dictionary."""
        data = sample_template.to_dict()

        assert data["name"] == "test_template"
        assert data["display_name"] == "Test Template"
        assert data["formation"] == "sequential"
        assert len(data["members"]) == 2
        assert "member1" in [m["id"] for m in data["members"]]

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "test_template",
            "display_name": "Test Template",
            "description": "Test description",
            "formation": "sequential",
            "members": [
                {
                    "id": "m1",
                    "role": "researcher",
                    "name": "M1",
                    "goal": "Test",
                    "tool_budget": 20,
                }
            ],
            "vertical": "general",
            "complexity": "quick",
        }

        template = TeamTemplate.from_dict(data)

        assert template.name == "test_template"
        assert len(template.members) == 1
        assert template.members[0].id == "m1"


class TestTeamTemplateRegistry:
    """Tests for TeamTemplateRegistry."""

    def test_register_template(self, registry, sample_template):
        """Test registering a template."""
        registry.register(sample_template)

        retrieved = registry.get_template("test_template")
        assert retrieved is not None
        assert retrieved.name == "test_template"

    def test_get_template_not_found(self, registry):
        """Test getting non-existent template."""
        retrieved = registry.get_template("nonexistent")
        assert retrieved is None

    def test_list_templates_no_filters(self, registry, sample_template):
        """Test listing all templates."""
        registry.register(sample_template)

        templates = registry.list_templates()
        assert "test_template" in templates

    def test_list_templates_with_filters(self, registry, sample_template):
        """Test listing templates with filters."""
        registry.register(sample_template)

        # Filter by vertical
        templates = registry.list_templates(vertical="general")
        assert "test_template" in templates

        # Filter by formation
        templates = registry.list_templates(formation="sequential")
        assert "test_template" in templates

        # Filter by complexity
        templates = registry.list_templates(complexity="quick")
        assert "test_template" in templates

        # Filter with no matches
        templates = registry.list_templates(vertical="coding")
        assert "test_template" not in templates

    def test_search_templates(self, registry, sample_template):
        """Test searching templates."""
        registry.register(sample_template)

        results = registry.search("test")
        assert len(results) > 0
        assert sample_template in results

    def test_get_by_vertical(self, registry):
        """Test getting templates by vertical."""
        template1 = TeamTemplate(
            name="coding_template",
            display_name="Coding",
            description="Coding template",
            formation="sequential",
            members=[
                TeamMemberSpec(
                    id="m1",
                    role="researcher",
                    name="M1",
                    goal="Test",
                )
            ],
            vertical="coding",
        )

        template2 = TeamTemplate(
            name="general_template",
            display_name="General",
            description="General template",
            formation="sequential",
            members=[
                TeamMemberSpec(
                    id="m1",
                    role="researcher",
                    name="M1",
                    goal="Test",
                )
            ],
            vertical="general",
        )

        registry.register(template1)
        registry.register(template2)

        coding_templates = registry.get_by_vertical("coding")
        assert len(coding_templates) == 1
        assert coding_templates[0].name == "coding_template"

    def test_get_by_formation(self, registry):
        """Test getting templates by formation."""
        template1 = TeamTemplate(
            name="parallel_template",
            display_name="Parallel",
            description="Parallel template",
            formation="parallel",
            members=[
                TeamMemberSpec(
                    id="m1",
                    role="researcher",
                    name="M1",
                    goal="Test",
                )
            ],
        )

        template2 = TeamTemplate(
            name="sequential_template",
            display_name="Sequential",
            description="Sequential template",
            formation="sequential",
            members=[
                TeamMemberSpec(
                    id="m1",
                    role="researcher",
                    name="M1",
                    goal="Test",
                )
            ],
        )

        registry.register(template1)
        registry.register(template2)

        parallel_templates = registry.get_by_formation("parallel")
        assert len(parallel_templates) == 1
        assert parallel_templates[0].name == "parallel_template"

    def test_validate_template_valid(self, registry, sample_template):
        """Test validating a valid template."""
        errors = registry.validate_template(sample_template)
        assert len(errors) == 0

    def test_validate_template_invalid(self, registry):
        """Test validating an invalid template."""
        # Template with duplicate member IDs
        template = TeamTemplate(
            name="invalid",
            display_name="Invalid",
            description="Invalid template",
            formation="sequential",
            members=[
                TeamMemberSpec(
                    id="duplicate",
                    role="researcher",
                    name="M1",
                    goal="Test",
                ),
                TeamMemberSpec(
                    id="duplicate",  # Duplicate ID
                    role="executor",
                    name="M2",
                    goal="Test",
                ),
            ],
        )

        errors = registry.validate_template(template)
        assert len(errors) > 0
        assert any("duplicate" in error.lower() for error in errors)

    def test_invalidate_cache(self, registry, sample_template):
        """Test cache invalidation."""
        registry.register(sample_template)
        assert registry.get_template("test_template") is not None

        registry.invalidate_cache()
        # After invalidation, templates are cleared
        assert len(registry._templates) == 0


class TestTemplateFiles:
    """Tests for loading templates from YAML files."""

    @pytest.mark.integration
    def test_load_template_from_yaml(self):
        """Test loading a template from YAML file."""
        template_path = (
            Path(__file__).parent.parent.parent.parent
            / "victor/workflows/templates/coding/code_review_parallel.yaml"
        )

        if not template_path.exists():
            pytest.skip("Template file not found")

        template = TeamTemplate.from_yaml(template_path)

        assert template.name == "code_review_parallel"
        assert template.formation == "parallel"
        assert len(template.members) == 4
        assert template.vertical == "coding"

    @pytest.mark.integration
    def test_load_all_templates(self):
        """Test loading all templates from directory."""
        registry = get_registry()
        registry.load_templates(force_reload=True)

        templates = registry.list_templates()
        # Should have at least 20 templates
        assert len(templates) >= 20

    @pytest.mark.integration
    def test_template_validation(self):
        """Test that all built-in templates are valid."""
        registry = get_registry()
        registry.load_templates(force_reload=True)

        template_names = registry.list_templates()
        for name in template_names:
            template = registry.get_template(name)
            assert template is not None

            errors = registry.validate_template(template)
            assert len(errors) == 0, f"Template {name} has validation errors: {errors}"


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_registry_singleton(self):
        """Test that get_registry returns singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_register_template_global(self, sample_template):
        """Test global register_template function."""
        register_template(sample_template)
        registry = get_registry()
        retrieved = registry.get_template("test_template")
        assert retrieved is not None

    def test_get_template_global(self, sample_template):
        """Test global get_template function."""
        register_template(sample_template)
        retrieved = get_template("test_template")
        assert retrieved is not None

    def test_list_templates_global(self, sample_template):
        """Test global list_templates function."""
        register_template(sample_template)
        templates = list_templates()
        assert "test_template" in templates

    def test_search_templates_global(self, sample_template):
        """Test global search_templates function."""
        register_template(sample_template)
        results = search_templates("test")
        assert len(results) > 0


@pytest.mark.integration
class TestTemplateApplication:
    """Integration tests for template application."""

    def test_apply_to_workflow(self):
        """Test applying template to create workflow node."""
        template = get_template("code_review_parallel")
        if not template:
            pytest.skip("Template not available")

        node = template.to_team_node(
            node_id="review_node",
            goal="Review authentication changes",
            output_key="review_results",
        )

        assert node.id == "review_node"
        assert node.team_formation == "parallel"
        assert len(node.members) == 4
        assert node.output_key == "review_results"

    def test_apply_to_team_config(self):
        """Test applying template to create team config."""
        template = get_template("code_review_parallel")
        if not template:
            pytest.skip("Template not available")

        config = template.to_team_config(
            goal="Review PR #123",
            context={"pr_number": 123},
        )

        assert config.goal == "Review PR #123"
        assert config.formation.value == "parallel"
        assert len(config.members) == 4
        assert config.shared_context == {"pr_number": 123}
