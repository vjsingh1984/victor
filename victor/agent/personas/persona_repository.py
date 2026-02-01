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

"""Persona repository for storage and management.

This module provides the PersonaRepository class which handles:
- CRUD operations for personas
- Persona validation and conflict detection
- Persona versioning and history tracking
- Import/export from YAML/JSON formats
- Thread-safe persona storage

The repository provides a persistence abstraction layer that can work with
different storage backends (in-memory, file system, database).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from victor.agent.personas.types import (
    CommunicationStyle,
    PersonalityType,
    Persona,
    PersonaConstraints,
    PromptTemplates,
)

logger = logging.getLogger(__name__)


class PersonaConflictError(Exception):
    """Raised when a persona conflict is detected."""

    pass


class PersonaVersion:
    """Version history entry for a persona.

    Attributes:
        version: Version number
        persona: Persona at this version
        timestamp: When this version was created
        change_description: Description of changes
    """

    def __init__(
        self,
        version: int,
        persona: Persona,
        timestamp: datetime,
        change_description: str = "",
    ) -> None:
        self.version = version
        self.persona = persona
        self.timestamp = timestamp
        self.change_description = change_description


class PersonaRepository:
    """Repository for persona storage and management.

    Provides thread-safe CRUD operations, validation, versioning,
    and import/export functionality for personas.

    By default uses in-memory storage. Can be extended to support
    file system or database backends.

    Attributes:
        _personas: Dictionary of persona_id -> Persona
        _versions: Dictionary of persona_id -> List[PersonaVersion]
        _lock: Thread lock for concurrent access
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """Initialize the repository.

        Args:
            storage_path: Optional path for file-based storage
        """
        self._personas: dict[str, Persona] = {}
        self._versions: dict[str, list[PersonaVersion]] = {}
        self._lock = threading.RLock()
        self._storage_path = storage_path

        # Load from storage if path provided
        if storage_path and storage_path.exists():
            self._load_from_storage()

    def save(self, persona: Persona, change_description: str = "") -> Persona:
        """Save a persona to the repository.

        Creates a new version if persona already exists.

        Args:
            persona: Persona to save
            change_description: Optional description of changes

        Returns:
            Saved persona

        Raises:
            PersonaConflictError: If persona conflicts with existing
        """
        with self._lock:
            # Check for conflicts
            if persona.id in self._personas:
                existing = self._personas[persona.id]

                # Check for conflicts if:
                # 1. Same version but different content (concurrent modification)
                # 2. Version going backward (shouldn't happen)
                if existing.version == persona.version:
                    # Same version - check if content actually changed
                    # If significant attributes changed, this is a conflict
                    if self._detect_conflict(existing, persona):
                        raise PersonaConflictError(
                            f"Persona {persona.id} has conflicting changes at same version"
                        )
                elif persona.version < existing.version:
                    # Version going backward - conflict
                    raise PersonaConflictError(
                        f"Persona {persona.id} version {persona.version} is older than existing version {existing.version}"
                    )
                # If new.version > existing.version, it's a valid update

            # Create version history entry
            if persona.id not in self._versions:
                self._versions[persona.id] = []

            version_entry = PersonaVersion(
                version=persona.version,
                persona=persona,
                timestamp=datetime.utcnow(),
                change_description=change_description,
            )
            self._versions[persona.id].append(version_entry)

            # Save persona
            self._personas[persona.id] = persona

            # Persist to storage if configured
            if self._storage_path:
                self._save_to_storage()

            logger.debug(f"Saved persona {persona.id} version {persona.version}")
            return persona

    def get(self, persona_id: str) -> Optional[Persona]:
        """Get a persona by ID.

        Args:
            persona_id: Persona identifier

        Returns:
            Persona or None if not found
        """
        with self._lock:
            return self._personas.get(persona_id)

    def list_all(self) -> list[Persona]:
        """List all personas in the repository.

        Returns:
            List of all personas
        """
        with self._lock:
            return list(self._personas.values())

    def delete(self, persona_id: str) -> bool:
        """Delete a persona from the repository.

        Args:
            persona_id: Persona to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if persona_id in self._personas:
                del self._personas[persona_id]
                if persona_id in self._versions:
                    del self._versions[persona_id]

                # Persist to storage if configured
                if self._storage_path:
                    self._save_to_storage()

                logger.debug(f"Deleted persona {persona_id}")
                return True
            return False

    def exists(self, persona_id: str) -> bool:
        """Check if a persona exists.

        Args:
            persona_id: Persona identifier

        Returns:
            True if persona exists
        """
        with self._lock:
            return persona_id in self._personas

    def get_version_history(self, persona_id: str) -> list[PersonaVersion]:
        """Get version history for a persona.

        Args:
            persona_id: Persona identifier

        Returns:
            List of version entries, oldest first
        """
        with self._lock:
            return self._versions.get(persona_id, [])

    def get_version(self, persona_id: str, version: int) -> Optional[Persona]:
        """Get a specific version of a persona.

        Args:
            persona_id: Persona identifier
            version: Version number

        Returns:
            Persona at specified version or None if not found
        """
        with self._lock:
            history = self._versions.get(persona_id, [])
            for entry in history:
                if entry.version == version:
                    return entry.persona
            return None

    def export_to_yaml(self, persona_id: str, output_path: Path) -> None:
        """Export a persona to YAML file.

        Args:
            persona_id: Persona to export
            output_path: Output file path

        Raises:
            ValueError: If persona not found
        """
        persona = self.get(persona_id)
        if persona is None:
            raise ValueError(f"Persona not found: {persona_id}")

        definition = persona.to_dict()

        with output_path.open("w") as f:
            yaml.safe_dump(definition, f, default_flow_style=False, sort_keys=False)

        logger.debug(f"Exported persona {persona_id} to {output_path}")

    def export_to_json(self, persona_id: str, output_path: Path) -> None:
        """Export a persona to JSON file.

        Args:
            persona_id: Persona to export
            output_path: Output file path

        Raises:
            ValueError: If persona not found
        """
        persona = self.get(persona_id)
        if persona is None:
            raise ValueError(f"Persona not found: {persona_id}")

        definition = persona.to_dict()

        with output_path.open("w") as f:
            json.dump(definition, f, indent=2)

        logger.debug(f"Exported persona {persona_id} to {output_path}")

    def import_from_yaml(self, input_path: Path) -> Persona:
        """Import a persona from YAML file.

        Args:
            input_path: Input file path

        Returns:
            Imported persona

        Raises:
            ValueError: If file is invalid
        """
        with input_path.open("r") as f:
            data = yaml.safe_load(f)

        return self._import_from_dict(data)

    def import_from_json(self, input_path: Path) -> Persona:
        """Import a persona from JSON file.

        Args:
            input_path: Input file path

        Returns:
            Imported persona

        Raises:
            ValueError: If file is invalid
        """
        with input_path.open("r") as f:
            data = json.load(f)

        return self._import_from_dict(data)

    def export_all_to_yaml(self, output_path: Path) -> None:
        """Export all personas to a YAML file.

        Args:
            output_path: Output file path
        """
        definitions = {
            persona_id: persona.to_dict() for persona_id, persona in self._personas.items()
        }

        with output_path.open("w") as f:
            yaml.safe_dump(
                {"personas": definitions},
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        logger.debug(f"Exported {len(self._personas)} personas to {output_path}")

    def import_all_from_yaml(self, input_path: Path) -> int:
        """Import multiple personas from a YAML file.

        Args:
            input_path: Input file path

        Returns:
            Number of personas imported

        Raises:
            ValueError: If file is invalid
        """
        with input_path.open("r") as f:
            data = yaml.safe_load(f)

        if "personas" not in data:
            raise ValueError("Invalid persona file: missing 'personas' key")

        count = 0
        for persona_data in data["personas"].values():
            try:
                self._import_from_dict(persona_data)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import persona: {e}")

        logger.debug(f"Imported {count} personas from {input_path}")
        return count

    def validate(self, persona: Persona) -> list[str]:
        """Validate a persona and return any issues.

        Args:
            persona: Persona to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        if not persona.id:
            errors.append("Persona ID is required")

        if not persona.name:
            errors.append("Persona name is required")

        if not persona.description:
            errors.append("Persona description is required")

        # Check constraints
        if persona.constraints:
            # Check for tool conflicts
            if persona.constraints.preferred_tools and persona.constraints.forbidden_tools:
                overlap = persona.constraints.preferred_tools & persona.constraints.forbidden_tools
                if overlap:
                    errors.append(f"Tools cannot be both preferred and forbidden: {overlap}")

            # Check max_tool_calls
            if (
                persona.constraints.max_tool_calls is not None
                and persona.constraints.max_tool_calls < 1
            ):
                errors.append("max_tool_calls must be at least 1")

        return errors

    def _detect_conflict(self, existing: Persona, new: Persona) -> bool:
        """Detect if two persona versions conflict.

        Args:
            existing: Existing persona
            new: New persona version

        Returns:
            True if conflict detected
        """
        # Check for incompatible changes
        # e.g., changing personality and communication style simultaneously
        # without proper version increment

        if existing.personality != new.personality:
            # Personality change should increment version
            if new.version <= existing.version:
                return True

        if existing.communication_style != new.communication_style:
            # Communication style change should increment version
            if new.version <= existing.version:
                return True

        return False

    def _import_from_dict(self, data: dict[str, Any]) -> Persona:
        """Import a persona from dictionary.

        Args:
            data: Persona definition dictionary

        Returns:
            Imported persona

        Raises:
            ValueError: If data is invalid
        """
        # Validate required fields
        required = ["id", "name", "description", "personality", "communication_style"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Parse enums
        personality = PersonalityType(data["personality"])
        communication = CommunicationStyle(data["communication_style"])

        # Parse constraints
        constraints = None
        if "constraints" in data and data["constraints"]:
            c_data = data["constraints"]
            constraints = PersonaConstraints(
                max_tool_calls=c_data.get("max_tool_calls"),
                preferred_tools=set(c_data.get("preferred_tools", [])),
                forbidden_tools=set(c_data.get("forbidden_tools", [])),
                response_length=c_data.get("response_length", "medium"),
                explanation_depth=c_data.get("explanation_depth", "standard"),
            )

        # Parse prompt templates
        templates = None
        if "prompt_templates" in data and data["prompt_templates"]:
            t_data = data["prompt_templates"]
            templates = PromptTemplates(
                system_prompt=t_data.get("system_prompt", ""),
                task_prompt=t_data.get("task_prompt"),
                greeting=t_data.get("greeting"),
                farewell=t_data.get("farewell"),
            )

        # Create persona
        persona = Persona(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            personality=personality,
            communication_style=communication,
            expertise=data.get("expertise", []),
            backstory=data.get("backstory"),
            constraints=constraints,
            prompt_templates=templates,
            version=data.get("version", 1),
        )

        # Validate
        errors = self.validate(persona)
        if errors:
            raise ValueError(f"Invalid persona: {', '.join(errors)}")

        # Save
        return self.save(persona, change_description="Imported from file")

    def _load_from_storage(self) -> None:
        """Load personas from storage path.

        Attempts to load from YAML or JSON file.
        """
        if not self._storage_path:
            return

        try:
            if self._storage_path.suffix == ".yaml":
                self.import_all_from_yaml(self._storage_path)
            elif self._storage_path.suffix == ".json":
                with self._storage_path.open("r") as f:
                    data = json.load(f)
                    for persona_data in data.get("personas", {}).values():
                        self._import_from_dict(persona_data)
            else:
                logger.warning(f"Unsupported storage file format: {self._storage_path.suffix}")
        except Exception as e:
            logger.warning(f"Failed to load personas from storage: {e}")

    def _save_to_storage(self) -> None:
        """Save personas to storage path."""
        if not self._storage_path:
            return

        try:
            if self._storage_path.suffix == ".yaml":
                self.export_all_to_yaml(self._storage_path)
            elif self._storage_path.suffix == ".json":
                data = {
                    "personas": {
                        persona_id: persona.to_dict()
                        for persona_id, persona in self._personas.items()
                    }
                }
                with self._storage_path.open("w") as f:
                    json.dump(data, f, indent=2)
            else:
                logger.warning(f"Unsupported storage file format: {self._storage_path.suffix}")
        except Exception as e:
            logger.warning(f"Failed to save personas to storage: {e}")
