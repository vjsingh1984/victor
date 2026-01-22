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

"""Fixtures for persona integration tests.

Provides isolated persona repositories to prevent test pollution
and version conflicts from persisting personas to the actual YAML file.
"""

from pathlib import Path
from typing import Dict

import pytest

from victor.agent.personas.persona_manager import PersonaManager
from victor.agent.personas.persona_repository import PersonaRepository
from victor.agent.personas.types import (
    CommunicationStyle,
    Persona,
    PersonalityType,
)


@pytest.fixture
def temp_persona_repository(tmp_path: Path) -> PersonaRepository:
    """Create a persona repository with temporary storage.

    This fixture ensures tests don't pollute the actual persona YAML file
    and prevents version conflicts between test runs.

    Args:
        tmp_path: Pytest's temporary directory fixture

    Returns:
        PersonaRepository with temporary storage path
    """
    repo = PersonaRepository(storage_path=tmp_path / "test_personas.yaml")
    return repo


@pytest.fixture
def persona_manager(temp_persona_repository: PersonaRepository) -> PersonaManager:
    """Create a persona manager with temporary repository.

    This replaces the original fixture that used the default PersonaManager,
    which was persisting to the actual YAML file and causing conflicts.

    Args:
        temp_persona_repository: Temporary repository fixture

    Returns:
        PersonaManager with isolated repository
    """
    manager = PersonaManager(repository=temp_persona_repository, auto_load=False)
    # Load predefined personas into the temporary repository
    _load_predefined_personas(manager)
    return manager


def _load_predefined_personas(manager: PersonaManager) -> None:
    """Load predefined personas into manager.

    These personas are used across multiple tests and provide
    consistent test data for persona integration tests.

    Args:
        manager: PersonaManager instance to populate
    """
    predefined = [
        Persona(
            id="senior_developer",
            name="Senior Developer",
            description="Experienced, pragmatic developer",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding", "debugging", "testing", "code_review"],
        ),
        Persona(
            id="security_expert",
            name="Security Expert",
            description="Security-focused specialist",
            personality=PersonalityType.CAUTIOUS,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["security", "vulnerabilities", "auditing"],
        ),
        Persona(
            id="mentor",
            name="Programming Mentor",
            description="Patient educator",
            personality=PersonalityType.SUPPORTIVE,
            communication_style=CommunicationStyle.EDUCATIONAL,
            expertise=["teaching", "explanation", "skill_development"],
        ),
    ]

    for persona in predefined:
        manager.repository.save(persona)
