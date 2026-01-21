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

"""Dynamic agent persona system for adaptive behavior.

This module provides a comprehensive persona management system that enables:
- Dynamic persona loading and adaptation based on context
- Predefined persona library for common use cases
- Custom persona creation and validation
- Context-aware persona adaptation
- Feedback-driven persona improvement

Key Components:
- Persona: Base persona definition with traits, expertise, and constraints
- AdaptedPersona: Context-aware persona with dynamic adjustments
- PersonaManager: Central persona management and adaptation
- PersonaRepository: Storage and persistence layer

Example Usage:
    from victor.agent.personas import PersonaManager

    # Load predefined persona
    manager = PersonaManager()
    persona = manager.load_persona("senior_developer")

    # Adapt persona to context
    adapted = manager.adapt_persona(
        persona,
        context={"task_type": "security_review", "user_preference": "thorough"}
    )

    # Create custom persona
    custom = manager.create_custom_persona(
        name="Code Review Expert",
        traits={
            "personality": "critical",
            "communication_style": "constructive",
            "expertise": ["testing", "quality", "standards"]
        }
    )
"""

from __future__ import annotations

from victor.agent.personas.persona_manager import PersonaManager
from victor.agent.personas.persona_repository import PersonaRepository
from victor.agent.personas.types import (
    AdaptedPersona,
    CommunicationStyle,
    Feedback,
    Persona,
    PersonaConstraints,
    PersonalityType,
    PromptTemplates,
)

__all__ = [
    "Persona",
    "AdaptedPersona",
    "Feedback",
    "PersonaManager",
    "PersonaRepository",
    "PersonalityType",
    "CommunicationStyle",
    "PersonaConstraints",
    "PromptTemplates",
]
