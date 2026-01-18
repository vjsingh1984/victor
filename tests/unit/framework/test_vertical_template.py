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

"""Unit tests for VerticalTemplate module.

Tests the VerticalTemplate dataclass and related components including:
- VerticalMetadata
- ExtensionSpecs
- MiddlewareSpec
- SafetyPatternSpec
- PromptHintSpec
- WorkflowSpec
- TeamSpec
- CapabilitySpec
"""

import pytest

from victor.framework.vertical_template import (
    VerticalTemplate,
    VerticalMetadata,
    ExtensionSpecs,
    MiddlewareSpec,
    SafetyPatternSpec,
    PromptHintSpec,
    WorkflowSpec,
    TeamSpec,
    TeamRoleSpec,
    CapabilitySpec,
)
from victor.core.vertical_types import StageDefinition


class TestVerticalMetadata:
    """Tests for VerticalMetadata dataclass."""

    def test_create_minimal_metadata(self):
        """Test creating minimal metadata."""
        metadata = VerticalMetadata(
            name="test",
            description="Test vertical",
        )

        assert metadata.name == "test"
        assert metadata.description == "Test vertical"
        assert metadata.version == "1.0.0"
        assert metadata.license == "Apache-2.0"
        assert metadata.category == "general"
        assert metadata.tags == []
        assert metadata.provider_hints == {}
        assert metadata.evaluation_criteria == []

    def test_create_full_metadata(self):
        """Test creating metadata with all fields."""
        metadata = VerticalMetadata(
            name="test",
            description="Test vertical",
            version="2.0.0",
            author="Test Author",
            license="MIT",
            category="testing",
            tags=["test", "example"],
            provider_hints={"preferred": ["anthropic"]},
            evaluation_criteria=["accuracy", "speed"],
        )

        assert metadata.version == "2.0.0"
        assert metadata.author == "Test Author"
        assert metadata.license == "MIT"
        assert metadata.category == "testing"
        assert metadata.tags == ["test", "example"]
        assert metadata.provider_hints == {"preferred": ["anthropic"]}
        assert metadata.evaluation_criteria == ["accuracy", "speed"]

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = VerticalMetadata(
            name="test",
            description="Test vertical",
            version="1.5.0",
            tags=["test"],
        )

        data = metadata.to_dict()

        assert data["name"] == "test"
        assert data["description"] == "Test vertical"
        assert data["version"] == "1.5.0"
        assert data["tags"] == ["test"]

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "name": "test",
            "description": "Test vertical",
            "version": "1.5.0",
            "author": "Test Author",
            "tags": ["test"],
        }

        metadata = VerticalMetadata.from_dict(data)

        assert metadata.name == "test"
        assert metadata.description == "Test vertical"
        assert metadata.version == "1.5.0"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test"]


class TestMiddlewareSpec:
    """Tests for MiddlewareSpec dataclass."""

    def test_create_middleware_spec(self):
        """Test creating middleware specification."""
        spec = MiddlewareSpec(
            name="test_middleware",
            class_name="TestMiddleware",
            module="victor.test.middleware",
            enabled=True,
            config={"option": "value"},
        )

        assert spec.name == "test_middleware"
        assert spec.class_name == "TestMiddleware"
        assert spec.module == "victor.test.middleware"
        assert spec.enabled is True
        assert spec.config == {"option": "value"}

    def test_middleware_spec_defaults(self):
        """Test middleware specification with defaults."""
        spec = MiddlewareSpec(
            name="test",
            class_name="Test",
            module="victor.test",
        )

        assert spec.enabled is True
        assert spec.config == {}

    def test_middleware_spec_to_dict(self):
        """Test converting middleware spec to dictionary."""
        spec = MiddlewareSpec(
            name="test",
            class_name="Test",
            module="victor.test",
            enabled=False,
        )

        data = spec.to_dict()

        assert data["name"] == "test"
        assert data["class_name"] == "Test"
        assert data["module"] == "victor.test"
        assert data["enabled"] is False

    def test_middleware_spec_from_dict(self):
        """Test creating middleware spec from dictionary."""
        data = {
            "name": "test",
            "class_name": "Test",
            "module": "victor.test",
            "enabled": True,
            "config": {"key": "value"},
        }

        spec = MiddlewareSpec.from_dict(data)

        assert spec.name == "test"
        assert spec.config == {"key": "value"}


class TestSafetyPatternSpec:
    """Tests for SafetyPatternSpec dataclass."""

    def test_create_safety_pattern_spec(self):
        """Test creating safety pattern specification."""
        spec = SafetyPatternSpec(
            name="dangerous_command",
            pattern=r"rm -rf",
            description="Dangerous file deletion",
            severity="critical",
            category="commands",
        )

        assert spec.name == "dangerous_command"
        assert spec.pattern == r"rm -rf"
        assert spec.description == "Dangerous file deletion"
        assert spec.severity == "critical"
        assert spec.category == "commands"

    def test_safety_pattern_spec_defaults(self):
        """Test safety pattern specification with defaults."""
        spec = SafetyPatternSpec(
            name="test",
            pattern="test.*",
            description="Test pattern",
        )

        assert spec.severity == "medium"
        assert spec.category == "general"

    def test_safety_pattern_spec_serialization(self):
        """Test safety pattern serialization."""
        spec = SafetyPatternSpec(
            name="test",
            pattern="test.*",
            description="Test",
            severity="high",
        )

        # Test to_dict
        data = spec.to_dict()
        assert data["severity"] == "high"

        # Test from_dict
        spec2 = SafetyPatternSpec.from_dict(data)
        assert spec2.severity == "high"


class TestPromptHintSpec:
    """Tests for PromptHintSpec dataclass."""

    def test_create_prompt_hint_spec(self):
        """Test creating prompt hint specification."""
        spec = PromptHintSpec(
            task_type="create",
            hint="[CREATE] Create file immediately",
            tool_budget=5,
            priority_tools=["write"],
        )

        assert spec.task_type == "create"
        assert spec.hint == "[CREATE] Create file immediately"
        assert spec.tool_budget == 5
        assert spec.priority_tools == ["write"]

    def test_prompt_hint_spec_defaults(self):
        """Test prompt hint specification with defaults."""
        spec = PromptHintSpec(
            task_type="test",
            hint="Test hint",
        )

        assert spec.tool_budget == 10
        assert spec.priority_tools == []

    def test_prompt_hint_spec_serialization(self):
        """Test prompt hint serialization."""
        spec = PromptHintSpec(
            task_type="edit",
            hint="[EDIT] Edit file",
            tool_budget=8,
            priority_tools=["read", "edit"],
        )

        data = spec.to_dict()
        assert data["tool_budget"] == 8
        assert data["priority_tools"] == ["read", "edit"]

        spec2 = PromptHintSpec.from_dict(data)
        assert spec2.tool_budget == 8


class TestExtensionSpecs:
    """Tests for ExtensionSpecs dataclass."""

    def test_create_empty_extensions(self):
        """Test creating empty extension specifications."""
        extensions = ExtensionSpecs()

        assert extensions.middleware == []
        assert extensions.safety_patterns == []
        assert extensions.prompt_hints == []
        assert extensions.handlers == {}
        assert extensions.personas == {}
        assert extensions.composed_chains == {}

    def test_create_extensions_with_middleware(self):
        """Test creating extensions with middleware."""
        middleware = [
            MiddlewareSpec(
                name="test1",
                class_name="Test1",
                module="victor.test",
            ),
            MiddlewareSpec(
                name="test2",
                class_name="Test2",
                module="victor.test",
            ),
        ]

        extensions = ExtensionSpecs(middleware=middleware)

        assert len(extensions.middleware) == 2
        assert extensions.middleware[0].name == "test1"
        assert extensions.middleware[1].name == "test2"

    def test_create_extensions_with_prompt_hints(self):
        """Test creating extensions with prompt hints."""
        hints = [
            PromptHintSpec(
                task_type="create",
                hint="[CREATE]",
            ),
            PromptHintSpec(
                task_type="edit",
                hint="[EDIT]",
            ),
        ]

        extensions = ExtensionSpecs(prompt_hints=hints)

        assert len(extensions.prompt_hints) == 2

    def test_extensions_serialization(self):
        """Test extensions serialization."""
        middleware = [
            MiddlewareSpec(
                name="test",
                class_name="Test",
                module="victor.test",
            )
        ]

        extensions = ExtensionSpecs(middleware=middleware)
        data = extensions.to_dict()

        assert "middleware" in data
        assert len(data["middleware"]) == 1

        extensions2 = ExtensionSpecs.from_dict(data)
        assert len(extensions2.middleware) == 1


class TestWorkflowSpec:
    """Tests for WorkflowSpec dataclass."""

    def test_create_workflow_spec(self):
        """Test creating workflow specification."""
        spec = WorkflowSpec(
            name="test_workflow",
            description="Test workflow",
            yaml_path="workflows/test.yaml",
            handler_module="victor.test.handlers",
        )

        assert spec.name == "test_workflow"
        assert spec.description == "Test workflow"
        assert spec.yaml_path == "workflows/test.yaml"
        assert spec.handler_module == "victor.test.handlers"

    def test_workflow_spec_optional_fields(self):
        """Test workflow specification with optional fields."""
        spec = WorkflowSpec(
            name="test",
            description="Test workflow",
        )

        assert spec.yaml_path is None
        assert spec.handler_module is None


class TestTeamSpec:
    """Tests for TeamSpec dataclass."""

    def test_create_team_spec(self):
        """Test creating team specification."""
        roles = [
            TeamRoleSpec(
                name="reviewer",
                display_name="Reviewer",
                description="Code reviewer",
                persona="You are a reviewer",
                tool_categories=["analysis"],
                capabilities=["review"],
            )
        ]

        spec = TeamSpec(
            name="review_team",
            display_name="Review Team",
            description="Code review team",
            formation="parallel",
            communication_style="structured",
            max_iterations=5,
            roles=roles,
        )

        assert spec.name == "review_team"
        assert spec.formation == "parallel"
        assert spec.max_iterations == 5
        assert len(spec.roles) == 1
        assert spec.roles[0].name == "reviewer"

    def test_team_spec_defaults(self):
        """Test team specification with defaults."""
        spec = TeamSpec(
            name="test",
            display_name="Test",
            description="Test team",
        )

        assert spec.formation == "parallel"
        assert spec.communication_style == "structured"
        assert spec.max_iterations == 5
        assert spec.roles == []

    def test_team_role_spec(self):
        """Test team role specification."""
        role = TeamRoleSpec(
            name="specialist",
            display_name="Specialist",
            description="Domain specialist",
            persona="You are a specialist",
            tool_categories=["analysis", "tools"],
            capabilities=["analyze"],
        )

        assert role.name == "specialist"
        assert role.tool_categories == ["analysis", "tools"]
        assert role.capabilities == ["analyze"]


class TestCapabilitySpec:
    """Tests for CapabilitySpec dataclass."""

    def test_create_capability_spec(self):
        """Test creating capability specification."""
        spec = CapabilitySpec(
            name="test_capability",
            type="tool",
            description="Test capability",
            enabled=True,
            handler="victor.test.handler",
            config={"option": "value"},
        )

        assert spec.name == "test_capability"
        assert spec.type == "tool"
        assert spec.enabled is True
        assert spec.handler == "victor.test.handler"
        assert spec.config == {"option": "value"}

    def test_capability_spec_defaults(self):
        """Test capability specification with defaults."""
        spec = CapabilitySpec(
            name="test",
            type="workflow",
            description="Test",
        )

        assert spec.enabled is True
        assert spec.handler is None
        assert spec.config == {}


class TestVerticalTemplate:
    """Tests for VerticalTemplate dataclass."""

    @pytest.fixture
    def minimal_template(self):
        """Create a minimal valid template."""
        return VerticalTemplate(
            metadata=VerticalMetadata(
                name="test",
                description="Test vertical",
            ),
            tools=["read", "write"],
            system_prompt="You are a test assistant",
            stages={
                "INITIAL": StageDefinition(
                    name="INITIAL",
                    description="Initial stage",
                    tools={"read"},
                    keywords=["what"],
                    next_stages={"COMPLETION"},
                ),
                "COMPLETION": StageDefinition(
                    name="COMPLETION",
                    description="Completion stage",
                    tools=set(),
                    keywords=["done"],
                    next_stages=set(),
                ),
            },
        )

    def test_create_minimal_template(self, minimal_template):
        """Test creating minimal template."""
        assert minimal_template.metadata.name == "test"
        assert minimal_template.tools == ["read", "write"]
        assert len(minimal_template.stages) == 2
        assert "INITIAL" in minimal_template.stages
        assert "COMPLETION" in minimal_template.stages

    def test_create_template_with_extensions(self, minimal_template):
        """Test creating template with extensions."""
        extensions = ExtensionSpecs(
            middleware=[
                MiddlewareSpec(
                    name="test",
                    class_name="Test",
                    module="victor.test",
                )
            ],
            prompt_hints=[
                PromptHintSpec(
                    task_type="create",
                    hint="[CREATE]",
                )
            ],
        )

        template = VerticalTemplate(
            metadata=minimal_template.metadata,
            tools=minimal_template.tools,
            system_prompt=minimal_template.system_prompt,
            stages=minimal_template.stages,
            extensions=extensions,
        )

        assert len(template.extensions.middleware) == 1
        assert len(template.extensions.prompt_hints) == 1

    def test_template_validation_valid(self, minimal_template):
        """Test validation of valid template."""
        errors = minimal_template.validate()
        assert len(errors) == 0

    def test_template_validation_missing_name(self, minimal_template):
        """Test validation fails with missing name."""
        minimal_template.metadata.name = ""
        errors = minimal_template.validate()
        assert len(errors) > 0
        assert any("name" in e.lower() for e in errors)

    def test_template_validation_missing_description(self, minimal_template):
        """Test validation fails with missing description."""
        minimal_template.metadata.description = ""
        errors = minimal_template.validate()
        assert len(errors) > 0

    def test_template_validation_empty_tools(self, minimal_template):
        """Test validation fails with empty tools."""
        minimal_template.tools = []
        errors = minimal_template.validate()
        assert len(errors) > 0
        assert any("tool" in e.lower() for e in errors)

    def test_template_validation_missing_required_stages(self):
        """Test validation fails with missing required stages."""
        template = VerticalTemplate(
            metadata=VerticalMetadata(
                name="test",
                description="Test",
            ),
            tools=["read"],
            system_prompt="Test prompt",
            stages={
                "INITIAL": StageDefinition(
                    name="INITIAL",
                    description="Initial",
                    tools=set(),
                    keywords=[],
                    next_stages=set(),
                ),
                # Missing COMPLETION stage
            },
        )

        errors = template.validate()
        assert len(errors) > 0
        assert any("COMPLETION" in e for e in errors)

    def test_template_is_valid(self, minimal_template):
        """Test is_valid method."""
        assert minimal_template.is_valid() is True

        minimal_template.metadata.name = ""
        assert minimal_template.is_valid() is False

    def test_template_to_dict(self, minimal_template):
        """Test converting template to dictionary."""
        data = minimal_template.to_dict()

        assert "metadata" in data
        assert "tools" in data
        assert "system_prompt" in data
        assert "stages" in data

        assert data["metadata"]["name"] == "test"
        assert data["tools"] == ["read", "write"]
        assert "INITIAL" in data["stages"]
        assert "COMPLETION" in data["stages"]

    def test_template_from_dict(self, minimal_template):
        """Test creating template from dictionary."""
        data = minimal_template.to_dict()
        template = VerticalTemplate.from_dict(data)

        assert template.metadata.name == "test"
        assert template.tools == ["read", "write"]
        assert len(template.stages) == 2

    def test_template_roundtrip_serialization(self, minimal_template):
        """Test template survives roundtrip through dict."""
        # Convert to dict
        data = minimal_template.to_dict()

        # Convert back
        template2 = VerticalTemplate.from_dict(data)

        # Verify
        assert template2.metadata.name == minimal_template.metadata.name
        assert template2.tools == minimal_template.tools
        assert len(template2.stages) == len(minimal_template.stages)

    def test_template_with_all_components(self):
        """Test template with all components populated."""
        template = VerticalTemplate(
            metadata=VerticalMetadata(
                name="complete",
                description="Complete test vertical",
                version="1.0.0",
                tags=["test", "complete"],
            ),
            tools=["read", "write", "grep"],
            system_prompt="Complete prompt",
            stages={
                "INITIAL": StageDefinition(
                    name="INITIAL",
                    description="Initial",
                    tools={"read"},
                    keywords=[],
                    next_stages={"COMPLETION"},
                ),
                "COMPLETION": StageDefinition(
                    name="COMPLETION",
                    description="Complete",
                    tools=set(),
                    keywords=[],
                    next_stages=set(),
                ),
            },
            extensions=ExtensionSpecs(
                middleware=[
                    MiddlewareSpec(
                        name="test",
                        class_name="Test",
                        module="victor.test",
                    )
                ]
            ),
            workflows=[
                WorkflowSpec(
                    name="test_workflow",
                    description="Test workflow",
                )
            ],
            teams=[
                TeamSpec(
                    name="test_team",
                    display_name="Test Team",
                    description="Test",
                )
            ],
            capabilities=[
                CapabilitySpec(
                    name="test_cap",
                    type="tool",
                    description="Test capability",
                )
            ],
            custom_config={"key": "value"},
        )

        # Should be valid
        assert template.is_valid()

        # Should serialize correctly
        data = template.to_dict()
        assert "workflows" in data
        assert "teams" in data
        assert "capabilities" in data
        assert "custom_config" in data
