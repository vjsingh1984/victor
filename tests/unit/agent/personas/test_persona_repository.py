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

"""Unit tests for PersonaRepository.

Tests persona storage, versioning, import/export functionality.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from victor.agent.personas.persona_repository import (
    PersonaConflictError,
    PersonaRepository,
    PersonaVersion,
)
from victor.agent.personas.types import (
    CommunicationStyle,
    Feedback,
    PersonalityType,
    Persona,
    PersonaConstraints,
)


@pytest.fixture
def temp_repository(tmp_path):
    """Create a temporary repository with file storage."""
    storage_file = tmp_path / "personas.yaml"
    return PersonaRepository(storage_path=storage_file)


@pytest.fixture
def in_memory_repository():
    """Create an in-memory repository."""
    return PersonaRepository()


@pytest.fixture
def sample_persona():
    """Create a sample persona for testing."""
    return Persona(
        id="test_persona",
        name="Test Persona",
        description="A test persona",
        personality=PersonalityType.PRAGMATIC,
        communication_style=CommunicationStyle.TECHNICAL,
        expertise=["coding", "testing"],
        backstory="Test backstory",
    )


class TestPersonaRepository:
    """Test PersonaRepository class."""

    def test_save_new_persona(self, in_memory_repository, sample_persona):
        """Test saving a new persona."""
        saved = in_memory_repository.save(sample_persona)

        assert saved.id == "test_persona"
        assert saved.version == 1

    def test_save_existing_persona_versioning(self, in_memory_repository, sample_persona):
        """Test that saving existing persona creates new version."""
        in_memory_repository.save(sample_persona)

        # Modify and save again
        sample_persona.version = 2
        sample_persona.expertise.append("performance")
        saved = in_memory_repository.save(sample_persona, "Added performance expertise")

        assert saved.version == 2

        # Check version history
        history = in_memory_repository.get_version_history("test_persona")
        assert len(history) == 2
        assert history[0].version == 1
        assert history[1].version == 2

    def test_get_persona_success(self, in_memory_repository, sample_persona):
        """Test getting an existing persona."""
        in_memory_repository.save(sample_persona)

        retrieved = in_memory_repository.get("test_persona")

        assert retrieved is not None
        assert retrieved.id == "test_persona"
        assert retrieved.name == "Test Persona"

    def test_get_persona_not_found(self, in_memory_repository):
        """Test getting non-existent persona."""
        retrieved = in_memory_repository.get("nonexistent")
        assert retrieved is None

    def test_list_all_personas(self, in_memory_repository):
        """Test listing all personas."""
        persona1 = Persona(
            id="persona1",
            name="Persona 1",
            description="First",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        persona2 = Persona(
            id="persona2",
            name="Persona 2",
            description="Second",
            personality=PersonalityType.CREATIVE,
            communication_style=CommunicationStyle.CASUAL,
            expertise=["design"],
        )

        in_memory_repository.save(persona1)
        in_memory_repository.save(persona2)

        all_personas = in_memory_repository.list_all()

        assert len(all_personas) == 2
        persona_ids = {p.id for p in all_personas}
        assert persona_ids == {"persona1", "persona2"}

    def test_delete_persona_success(self, in_memory_repository, sample_persona):
        """Test deleting an existing persona."""
        in_memory_repository.save(sample_persona)

        result = in_memory_repository.delete("test_persona")

        assert result is True
        assert in_memory_repository.get("test_persona") is None

    def test_delete_persona_not_found(self, in_memory_repository):
        """Test deleting non-existent persona."""
        result = in_memory_repository.delete("nonexistent")
        assert result is False

    def test_exists_persona(self, in_memory_repository, sample_persona):
        """Test checking if persona exists."""
        in_memory_repository.save(sample_persona)

        assert in_memory_repository.exists("test_persona") is True
        assert in_memory_repository.exists("nonexistent") is False

    def test_get_version_history(self, in_memory_repository, sample_persona):
        """Test getting version history for persona."""
        in_memory_repository.save(sample_persona, "Initial version")

        sample_persona.version = 2
        sample_persona.name = "Updated Persona"
        in_memory_repository.save(sample_persona, "Updated name")

        history = in_memory_repository.get_version_history("test_persona")

        assert len(history) == 2
        assert history[0].version == 1
        assert history[1].version == 2
        assert history[1].change_description == "Updated name"

    def test_get_version_history_empty(self, in_memory_repository):
        """Test getting version history for non-existent persona."""
        history = in_memory_repository.get_version_history("nonexistent")
        assert history == []

    def test_get_specific_version(self, in_memory_repository, sample_persona):
        """Test getting a specific version of a persona."""
        original_name = sample_persona.name
        in_memory_repository.save(sample_persona)

        # Create a NEW persona object with version 2
        from copy import deepcopy

        v2_persona = deepcopy(sample_persona)
        v2_persona.version = 2
        v2_persona.name = "Updated"
        in_memory_repository.save(v2_persona)

        # Get version 1
        v1 = in_memory_repository.get_version("test_persona", 1)

        assert v1 is not None
        assert v1.version == 1
        assert v1.name == original_name

    def test_get_specific_version_not_found(self, in_memory_repository, sample_persona):
        """Test getting non-existent version."""
        in_memory_repository.save(sample_persona)

        v5 = in_memory_repository.get_version("test_persona", 5)
        assert v5 is None


class TestPersonaValidation:
    """Test persona validation."""

    def test_validate_valid_persona(self, in_memory_repository):
        """Test validating a valid persona."""
        persona = Persona(
            id="valid",
            name="Valid Persona",
            description="A valid persona",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        errors = in_memory_repository.validate(persona)
        assert errors == []

    def test_validate_missing_id(self, in_memory_repository):
        """Test validating persona with missing ID."""
        persona = Persona(
            id="",
            name="No ID",
            description="Missing ID",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
        )

        errors = in_memory_repository.validate(persona)
        assert "Persona ID is required" in errors

    def test_validate_missing_name(self, in_memory_repository):
        """Test validating persona with missing name."""
        persona = Persona(
            id="no_name",
            name="",
            description="No name",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
        )

        errors = in_memory_repository.validate(persona)
        assert "Persona name is required" in errors

    def test_validate_missing_description(self, in_memory_repository):
        """Test validating persona with missing description."""
        persona = Persona(
            id="no_desc",
            name="No Description",
            description="",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
        )

        errors = in_memory_repository.validate(persona)
        assert "Persona description is required" in errors

    def test_validate_conflicting_tools(self, in_memory_repository):
        """Test validating persona with conflicting tool constraints."""
        persona = Persona(
            id="conflict",
            name="Conflict",
            description="Conflicting constraints",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            constraints=PersonaConstraints(
                preferred_tools={"read_file", "write_file"},
                forbidden_tools={"write_file", "delete_file"},  # Overlap!
            ),
        )

        errors = in_memory_repository.validate(persona)
        assert any("preferred and forbidden" in err for err in errors)

    def test_validate_invalid_max_tool_calls(self, in_memory_repository):
        """Test validating persona with invalid max_tool_calls."""
        persona = Persona(
            id="invalid",
            name="Invalid",
            description="Invalid constraint",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            constraints=PersonaConstraints(max_tool_calls=0),
        )

        errors = in_memory_repository.validate(persona)
        assert any("max_tool_calls must be at least 1" in err for err in errors)


class TestPersonaConflictDetection:
    """Test persona conflict detection."""

    def test_no_conflict_same_version(self, in_memory_repository, sample_persona):
        """Test that same version doesn't conflict."""
        in_memory_repository.save(sample_persona)

        # Saving same version again should not conflict
        try:
            in_memory_repository.save(sample_persona)
        except PersonaConflictError:
            pytest.fail("Should not raise conflict for same version")

    def test_conflict_personality_change_without_version_bump(
        self, in_memory_repository, sample_persona
    ):
        """Test conflict when personality changes without version bump."""
        # Save original
        in_memory_repository.save(sample_persona)

        # Try to save another persona with same ID/version but different personality
        # This simulates a concurrent modification conflict
        conflicting = Persona(
            id=sample_persona.id,
            name=sample_persona.name,
            description=sample_persona.description,
            personality=PersonalityType.CREATIVE,  # Changed!
            communication_style=sample_persona.communication_style,
            expertise=list(sample_persona.expertise),  # Copy list
            version=1,  # Same version as existing!
        )

        # Should detect conflict when trying to save conflicting version
        with pytest.raises(PersonaConflictError):
            in_memory_repository.save(conflicting)

    def test_no_conflict_with_version_bump(self, in_memory_repository, sample_persona):
        """Test that version bump prevents conflict."""
        in_memory_repository.save(sample_persona)

        # Change personality but increment version
        modified = Persona(
            id=sample_persona.id,
            name=sample_persona.name,
            description=sample_persona.description,
            personality=PersonalityType.CREATIVE,
            communication_style=sample_persona.communication_style,
            version=2,  # Incremented!
        )

        # Should not raise
        in_memory_repository.save(modified)


class TestPersonaExport:
    """Test persona export functionality."""

    def test_export_to_yaml(self, in_memory_repository, sample_persona, tmp_path):
        """Test exporting persona to YAML file."""
        in_memory_repository.save(sample_persona)

        output_file = tmp_path / "exported.yaml"
        in_memory_repository.export_to_yaml("test_persona", output_file)

        assert output_file.exists()

        # Verify content
        with output_file.open("r") as f:
            data = yaml.safe_load(f)

        assert data["id"] == "test_persona"
        assert data["name"] == "Test Persona"

    def test_export_to_yaml_not_found(self, in_memory_repository, tmp_path):
        """Test exporting non-existent persona to YAML."""
        output_file = tmp_path / "exported.yaml"

        with pytest.raises(ValueError, match="Persona not found"):
            in_memory_repository.export_to_yaml("nonexistent", output_file)

    def test_export_to_json(self, in_memory_repository, sample_persona, tmp_path):
        """Test exporting persona to JSON file."""
        in_memory_repository.save(sample_persona)

        output_file = tmp_path / "exported.json"
        in_memory_repository.export_to_json("test_persona", output_file)

        assert output_file.exists()

        # Verify content
        with output_file.open("r") as f:
            data = json.load(f)

        assert data["id"] == "test_persona"
        assert data["name"] == "Test Persona"

    def test_export_to_json_not_found(self, in_memory_repository, tmp_path):
        """Test exporting non-existent persona to JSON."""
        output_file = tmp_path / "exported.json"

        with pytest.raises(ValueError, match="Persona not found"):
            in_memory_repository.export_to_json("nonexistent", output_file)

    def test_export_all_to_yaml(self, in_memory_repository, tmp_path):
        """Test exporting all personas to YAML."""
        persona1 = Persona(
            id="p1",
            name="Persona 1",
            description="First",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
        )

        persona2 = Persona(
            id="p2",
            name="Persona 2",
            description="Second",
            personality=PersonalityType.CREATIVE,
            communication_style=CommunicationStyle.CASUAL,
        )

        in_memory_repository.save(persona1)
        in_memory_repository.save(persona2)

        output_file = tmp_path / "all.yaml"
        in_memory_repository.export_all_to_yaml(output_file)

        assert output_file.exists()

        # Verify content
        with output_file.open("r") as f:
            data = yaml.safe_load(f)

        assert "personas" in data
        assert "p1" in data["personas"]
        assert "p2" in data["personas"]


class TestPersonaImport:
    """Test persona import functionality."""

    def test_import_from_yaml(self, in_memory_repository, tmp_path):
        """Test importing persona from YAML file."""
        # Create test file
        data = {
            "id": "imported",
            "name": "Imported Persona",
            "description": "Imported from YAML",
            "personality": "creative",
            "communication_style": "casual",
            "expertise": ["innovation", "design"],
            "version": 1,
        }

        input_file = tmp_path / "import.yaml"
        with input_file.open("w") as f:
            yaml.safe_dump(data, f)

        # Import
        persona = in_memory_repository.import_from_yaml(input_file)

        assert persona.id == "imported"
        assert persona.name == "Imported Persona"
        assert persona.personality == PersonalityType.CREATIVE
        assert "innovation" in persona.expertise

    def test_import_from_yaml_invalid(self, in_memory_repository, tmp_path):
        """Test importing invalid YAML file."""
        # Create file with missing required fields
        data = {"id": "incomplete"}  # Missing name, description, etc.

        input_file = tmp_path / "invalid.yaml"
        with input_file.open("w") as f:
            yaml.safe_dump(data, f)

        with pytest.raises(ValueError, match="Missing required field"):
            in_memory_repository.import_from_yaml(input_file)

    def test_import_from_json(self, in_memory_repository, tmp_path):
        """Test importing persona from JSON file."""
        # Create test file
        data = {
            "id": "imported_json",
            "name": "Imported from JSON",
            "description": "JSON import test",
            "personality": "systematic",
            "communication_style": "formal",
            "expertise": ["architecture"],
            "version": 1,
        }

        input_file = tmp_path / "import.json"
        with input_file.open("w") as f:
            json.dump(data, f)

        # Import
        persona = in_memory_repository.import_from_json(input_file)

        assert persona.id == "imported_json"
        assert persona.name == "Imported from JSON"
        assert persona.personality == PersonalityType.SYSTEMATIC

    def test_import_from_json_invalid(self, in_memory_repository, tmp_path):
        """Test importing invalid JSON file."""
        # Create file with missing fields
        data = {"id": "incomplete"}

        input_file = tmp_path / "invalid.json"
        with input_file.open("w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Missing required field"):
            in_memory_repository.import_from_json(input_file)

    def test_import_all_from_yaml(self, in_memory_repository, tmp_path):
        """Test importing multiple personas from YAML."""
        # Create test file with multiple personas
        data = {
            "personas": {
                "persona1": {
                    "id": "p1",
                    "name": "Persona 1",
                    "description": "First",
                    "personality": "pragmatic",
                    "communication_style": "technical",
                },
                "persona2": {
                    "id": "p2",
                    "name": "Persona 2",
                    "description": "Second",
                    "personality": "creative",
                    "communication_style": "casual",
                },
            }
        }

        input_file = tmp_path / "all.yaml"
        with input_file.open("w") as f:
            yaml.safe_dump(data, f)

        # Import
        count = in_memory_repository.import_all_from_yaml(input_file)

        assert count == 2
        assert in_memory_repository.exists("p1")
        assert in_memory_repository.exists("p2")

    def test_import_with_constraints(self, in_memory_repository, tmp_path):
        """Test importing persona with constraints."""
        data = {
            "id": "constrained",
            "name": "Constrained",
            "description": "Has constraints",
            "personality": "cautious",
            "communication_style": "formal",
            "constraints": {
                "max_tool_calls": 30,
                "preferred_tools": ["read_file", "analyze"],
                "forbidden_tools": ["delete", "execute_code"],
                "response_length": "long",
                "explanation_depth": "detailed",
            },
        }

        input_file = tmp_path / "constrained.yaml"
        with input_file.open("w") as f:
            yaml.safe_dump(data, f)

        persona = in_memory_repository.import_from_yaml(input_file)

        assert persona.constraints.max_tool_calls == 30
        assert "read_file" in persona.constraints.preferred_tools
        assert "delete" in persona.constraints.forbidden_tools
        assert persona.constraints.response_length == "long"
        assert persona.constraints.explanation_depth == "detailed"

    def test_import_with_prompt_templates(self, in_memory_repository, tmp_path):
        """Test importing persona with prompt templates."""
        data = {
            "id": "templated",
            "name": "Templated",
            "description": "Has templates",
            "personality": "supportive",
            "communication_style": "educational",
            "prompt_templates": {
                "system_prompt": "You are a helpful assistant.",
                "greeting": "Hello! How can I help?",
                "farewell": "Goodbye!",
            },
        }

        input_file = tmp_path / "templated.yaml"
        with input_file.open("w") as f:
            yaml.safe_dump(data, f)

        persona = in_memory_repository.import_from_yaml(input_file)

        assert persona.prompt_templates is not None
        assert "helpful assistant" in persona.prompt_templates.system_prompt
        assert persona.prompt_templates.greeting == "Hello! How can I help?"
        assert persona.prompt_templates.farewell == "Goodbye!"


class TestPersonaFileStorage:
    """Test file-based persona storage."""

    def test_save_to_storage(self, temp_repository, sample_persona):
        """Test that personas are persisted to file."""
        temp_repository.save(sample_persona)

        # Verify file was created
        assert temp_repository._storage_path.exists()

    def test_load_from_storage(self, temp_repository, sample_persona):
        """Test that personas are loaded from file on initialization."""
        # Save to storage
        temp_repository.save(sample_persona)

        # Create new repository pointing to same file
        new_repository = PersonaRepository(storage_path=temp_repository._storage_path)

        # Should load saved persona
        loaded = new_repository.get("test_persona")
        assert loaded is not None
        assert loaded.name == "Test Persona"


class TestPersonaVersion:
    """Test PersonaVersion class."""

    def test_version_entry_creation(self):
        """Test creating a version history entry."""
        persona = Persona(
            id="test",
            name="Test",
            description="Test",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
        )

        version = PersonaVersion(
            version=1, persona=persona, timestamp=datetime.utcnow(), change_description="Initial"
        )

        assert version.version == 1
        assert version.persona == persona
        assert version.change_description == "Initial"
        assert isinstance(version.timestamp, datetime)
