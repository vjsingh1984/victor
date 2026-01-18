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

"""Unit tests for VerticalTemplateRegistry module.

Tests the VerticalTemplateRegistry class including:
- Template registration and retrieval
- YAML loading and saving
- Template validation
- Query and listing operations
- Thread safety
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from victor.framework.vertical_template import (
    VerticalTemplate,
    VerticalMetadata,
    ExtensionSpecs,
)
from victor.framework.vertical_template_registry import VerticalTemplateRegistry
from victor.core.vertical_types import StageDefinition


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_template():
    """Create a valid template for testing."""
    return VerticalTemplate(
        metadata=VerticalMetadata(
            name="test_vertical",
            description="Test vertical for registry",
            version="1.0.0",
            category="testing",
            tags=["test"],
        ),
        tools=["read", "write", "grep"],
        system_prompt="You are a test assistant",
        stages={
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Initial stage",
                tools={"read"},
                keywords=["what", "how"],
                next_stages={"EXECUTION"},
            ),
            "EXECUTION": StageDefinition(
                name="EXECUTION",
                description="Execution stage",
                tools={"read", "write"},
                keywords=["do", "make"],
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
        extensions=ExtensionSpecs(),
    )


@pytest.fixture
def sample_template_yaml(temp_dir):
    """Create a sample template YAML file."""
    yaml_path = temp_dir / "test_template.yaml"

    data = {
        "metadata": {
            "name": "yaml_test",
            "description": "Test from YAML",
            "version": "1.0.0",
            "category": "test",
        },
        "tools": ["read", "write"],
        "system_prompt": "Test prompt",
        "stages": {
            "INITIAL": {
                "name": "INITIAL",
                "description": "Initial",
                "tools": ["read"],
                "keywords": ["what"],
                "next_stages": ["COMPLETION"],
            },
            "COMPLETION": {
                "name": "COMPLETION",
                "description": "Complete",
                "tools": [],
                "keywords": ["done"],
                "next_stages": [],
            },
        },
        "extensions": {
            "middleware": [],
            "safety_patterns": [],
            "prompt_hints": [],
            "handlers": {},
            "personas": {},
            "composed_chains": {},
        },
        "workflows": [],
        "teams": [],
        "capabilities": [],
        "custom_config": {},
        "file_templates": {},
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)

    return yaml_path


class TestVerticalTemplateRegistry:
    """Tests for VerticalTemplateRegistry class."""

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        registry1 = VerticalTemplateRegistry.get_instance()
        registry2 = VerticalTemplateRegistry.get_instance()

        assert registry1 is registry2

    def test_register_template(self, valid_template):
        """Test registering a template."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        retrieved = registry.get("test_vertical")
        assert retrieved is not None
        assert retrieved.metadata.name == "test_vertical"

    def test_register_invalid_template(self):
        """Test registering invalid template raises error."""
        from victor.framework.vertical_template import VerticalTemplate, VerticalMetadata

        # Create invalid template (missing required stages)
        invalid_template = VerticalTemplate(
            metadata=VerticalMetadata(
                name="invalid",
                description="Invalid template",
            ),
            tools=["read"],
            system_prompt="Test",
            stages={
                "INITIAL": StageDefinition(
                    name="INITIAL",
                    description="Initial",
                    tools=set(),
                    keywords=[],
                    next_stages=set(),
                )
                # Missing COMPLETION stage
            },
        )

        registry = VerticalTemplateRegistry()

        with pytest.raises(ValueError, match="Invalid template"):
            registry.register(invalid_template)

    def test_register_overwrite(self, valid_template):
        """Test registering with overwrite flag."""
        registry = VerticalTemplateRegistry()

        # Register first time
        registry.register(valid_template, overwrite=False)

        # Try registering again without overwrite
        with pytest.raises(ValueError, match="already exists"):
            registry.register(valid_template, overwrite=False)

        # Should succeed with overwrite=True
        registry.register(valid_template, overwrite=True)

    def test_unregister_template(self, valid_template):
        """Test unregistering a template."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        assert registry.get("test_vertical") is not None

        registry.unregister("test_vertical")

        assert registry.get("test_vertical") is None

    def test_get_nonexistent_template(self):
        """Test getting non-existent template returns None."""
        registry = VerticalTemplateRegistry()
        result = registry.get("nonexistent")

        assert result is None

    def test_list_all_templates(self, valid_template):
        """Test listing all templates."""
        registry = VerticalTemplateRegistry()

        # Empty initially
        assert len(registry.list_all()) == 0

        # Register template
        registry.register(valid_template)

        # List should contain one template
        templates = registry.list_all()
        assert len(templates) == 1
        assert templates[0].metadata.name == "test_vertical"

    def test_list_names(self, valid_template):
        """Test listing template names."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        names = registry.list_names()
        assert "test_vertical" in names

    def test_find_by_category(self, valid_template):
        """Test finding templates by category."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        # Find by category
        results = registry.find_by_category("testing")
        assert len(results) == 1
        assert results[0].metadata.name == "test_vertical"

        # Different category should return empty
        results = registry.find_by_category("other")
        assert len(results) == 0

    def test_find_by_tag(self, valid_template):
        """Test finding templates by tag."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        # Find by tag
        results = registry.find_by_tag("test")
        assert len(results) == 1
        assert results[0].metadata.name == "test_vertical"

        # Different tag should return empty
        results = registry.find_by_tag("other")
        assert len(results) == 0

    def test_clear_registry(self, valid_template):
        """Test clearing all templates."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        assert len(registry.list_all()) == 1

        registry.clear()

        assert len(registry.list_all()) == 0

    def test_load_from_yaml(self, sample_template_yaml):
        """Test loading template from YAML file."""
        registry = VerticalTemplateRegistry()
        template = registry.load_from_yaml(sample_template_yaml)

        assert template is not None
        assert template.metadata.name == "yaml_test"
        assert template.tools == ["read", "write"]

    def test_load_from_nonexistent_file(self, temp_dir):
        """Test loading from non-existent file returns None."""
        registry = VerticalTemplateRegistry()
        template = registry.load_from_yaml(temp_dir / "nonexistent.yaml")

        assert template is None

    def test_load_and_register(self, sample_template_yaml):
        """Test loading and registering template from YAML."""
        registry = VerticalTemplateRegistry()
        template = registry.load_and_register(sample_template_yaml)

        assert template is not None
        assert registry.get("yaml_test") is not None

    def test_load_directory(self, temp_dir):
        """Test loading all templates from directory."""
        registry = VerticalTemplateRegistry()

        # Create multiple YAML files
        for i in range(3):
            yaml_path = temp_dir / f"template_{i}.yaml"
            data = {
                "metadata": {
                    "name": f"test_{i}",
                    "description": f"Test {i}",
                    "version": "1.0.0",
                },
                "tools": ["read"],
                "system_prompt": "Test",
                "stages": {
                    "INITIAL": {
                        "name": "INITIAL",
                        "description": "Initial",
                        "tools": [],
                        "keywords": [],
                        "next_stages": ["COMPLETION"],
                    },
                    "COMPLETION": {
                        "name": "COMPLETION",
                        "description": "Complete",
                        "tools": [],
                        "keywords": [],
                        "next_stages": [],
                    },
                },
                "extensions": {
                    "middleware": [],
                    "safety_patterns": [],
                    "prompt_hints": [],
                    "handlers": {},
                    "personas": {},
                    "composed_chains": {},
                },
                "workflows": [],
                "teams": [],
                "capabilities": [],
                "custom_config": {},
                "file_templates": {},
            }

            with open(yaml_path, "w") as f:
                yaml.safe_dump(data, f)

        # Load all templates
        count = registry.load_directory(temp_dir)

        assert count == 3
        assert len(registry.list_all()) == 3

    def test_save_to_yaml(self, valid_template, temp_dir):
        """Test saving template to YAML file."""
        registry = VerticalTemplateRegistry()
        output_path = temp_dir / "output.yaml"

        success = registry.save_to_yaml(valid_template, output_path)

        assert success is True
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = yaml.safe_load(f)

        assert data["metadata"]["name"] == "test_vertical"
        assert data["tools"] == ["read", "write", "grep"]

    def test_save_invalid_template(self, temp_dir):
        """Test saving invalid template fails."""
        from victor.framework.vertical_template import VerticalTemplate, VerticalMetadata

        invalid_template = VerticalTemplate(
            metadata=VerticalMetadata(name="x", description="x"),
            tools=[],  # Empty tools - invalid
            system_prompt="x",
            stages={},
        )

        registry = VerticalTemplateRegistry()
        output_path = temp_dir / "output.yaml"

        success = registry.save_to_yaml(invalid_template, output_path, validate=True)

        assert success is False
        assert not output_path.exists()

    def test_export_to_dict(self, valid_template):
        """Test exporting template to dictionary."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        data = registry.export_to_dict("test_vertical")

        assert data is not None
        assert data["metadata"]["name"] == "test_vertical"

    def test_export_nonexistent_template(self):
        """Test exporting non-existent template returns None."""
        registry = VerticalTemplateRegistry()
        data = registry.export_to_dict("nonexistent")

        assert data is None

    def test_import_from_dict(self):
        """Test importing template from dictionary."""
        data = {
            "metadata": {
                "name": "imported",
                "description": "Imported template",
                "version": "1.0.0",
            },
            "tools": ["read"],
            "system_prompt": "Test",
            "stages": {
                "INITIAL": {
                    "name": "INITIAL",
                    "description": "Initial",
                    "tools": [],
                    "keywords": [],
                    "next_stages": ["COMPLETION"],
                },
                "COMPLETION": {
                    "name": "COMPLETION",
                    "description": "Complete",
                    "tools": [],
                    "keywords": [],
                    "next_stages": [],
                },
            },
            "extensions": {
                "middleware": [],
                "safety_patterns": [],
                "prompt_hints": [],
                "handlers": {},
                "personas": {},
                "composed_chains": {},
            },
            "workflows": [],
            "teams": [],
            "capabilities": [],
            "custom_config": {},
            "file_templates": {},
        }

        registry = VerticalTemplateRegistry()
        template = registry.import_from_dict(data)

        assert template is not None
        assert template.metadata.name == "imported"

    def test_get_statistics(self, valid_template):
        """Test getting registry statistics."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        stats = registry.get_statistics()

        assert stats["total_templates"] == 1
        assert "test_vertical" in stats["template_names"]
        assert stats["categories"]["testing"] == 1

    def test_validate_registry(self):
        """Test validating all templates in registry."""
        from victor.framework.vertical_template import VerticalTemplate, VerticalMetadata

        registry = VerticalTemplateRegistry()

        # Add valid template
        registry.register(valid_template())

        # Add invalid template
        invalid = VerticalTemplate(
            metadata=VerticalMetadata(name="inv", description="Invalid"),
            tools=[],  # Invalid
            system_prompt="x",
            stages={},
        )

        # Manually add to registry (bypassing validation)
        registry._templates["inv"] = invalid

        errors = registry.validate_registry()

        # Should have errors for the invalid template
        assert len(errors) > 0
        assert any("inv" in e for e in errors)

    def test_is_registry_valid(self, valid_template):
        """Test checking if all templates are valid."""
        registry = VerticalTemplateRegistry()
        registry.register(valid_template)

        assert registry.is_registry_valid() is True


class TestRegistryThreadSafety:
    """Tests for thread safety of registry operations."""

    def test_concurrent_registration(self, valid_template):
        """Test concurrent template registration."""
        import threading

        registry = VerticalTemplateRegistry()
        errors = []

        def register_template(i):
            try:
                template = VerticalTemplate(
                    metadata=VerticalMetadata(
                        name=f"test_{i}",
                        description=f"Test {i}",
                    ),
                    tools=["read"],
                    system_prompt="Test",
                    stages={
                        "INITIAL": StageDefinition(
                            name="INITIAL",
                            description="Initial",
                            tools=set(),
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
                )
                registry.register(template, overwrite=True)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=register_template, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0

        # Should have all templates registered
        assert len(registry.list_all()) == 10


class TestGlobalConvenienceFunctions:
    """Tests for global convenience functions."""

    def test_get_template_registry(self):
        """Test getting global registry instance."""
        from victor.framework.vertical_template_registry import get_template_registry

        registry = get_template_registry()
        assert isinstance(registry, VerticalTemplateRegistry)

    def test_register_template(self, valid_template):
        """Test register_template convenience function."""
        from victor.framework.vertical_template_registry import (
            register_template,
            get_template,
        )

        register_template(valid_template)

        retrieved = get_template("test_vertical")
        assert retrieved is not None
        assert retrieved.metadata.name == "test_vertical"

    def test_list_templates(self, valid_template):
        """Test list_templates convenience function."""
        from victor.framework.vertical_template_registry import (
            register_template,
            list_templates,
        )

        registry = VerticalTemplateRegistry.get_instance()
        registry.clear()  # Clear any existing templates

        register_template(valid_template)

        templates = list_templates()
        assert len(templates) == 1
        assert templates[0].metadata.name == "test_vertical"
