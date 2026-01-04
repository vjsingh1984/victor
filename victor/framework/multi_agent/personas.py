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

"""Generic persona traits for multi-agent systems."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class CommunicationStyle(Enum):
    """Communication styles for agent personas.

    Defines how an agent communicates with users and other agents.
    """

    FORMAL = "formal"
    """Formal, professional communication with proper grammar and structure."""

    CASUAL = "casual"
    """Relaxed, conversational tone with informal language."""

    TECHNICAL = "technical"
    """Precise, technical language with domain-specific terminology."""

    CONCISE = "concise"
    """Brief, to-the-point responses with minimal elaboration."""


class ExpertiseLevel(Enum):
    """Expertise levels for agent personas.

    Indicates the depth of knowledge and experience an agent represents.
    """

    NOVICE = "novice"
    """Entry-level with basic understanding of fundamentals."""

    INTERMEDIATE = "intermediate"
    """Solid working knowledge with practical experience."""

    EXPERT = "expert"
    """Deep expertise with comprehensive understanding."""

    SPECIALIST = "specialist"
    """Highly specialized knowledge in a narrow domain."""


@dataclass
class PersonaTraits:
    """Generic traits that define an agent persona.

    PersonaTraits provides a flexible way to define agent characteristics
    without coupling to specific implementations. These traits influence
    how agents behave, communicate, and approach problems.

    Attributes:
        name: Display name for this persona.
        role: The role this persona fulfills (e.g., "researcher", "reviewer").
        description: Detailed description of the persona's purpose and behavior.
        communication_style: How the persona communicates (default: TECHNICAL).
        expertise_level: Level of expertise represented (default: EXPERT).
        verbosity: How verbose responses should be, 0.0-1.0 (default: 0.5).
        strengths: List of areas where this persona excels.
        weaknesses: List of areas where this persona may struggle.
        preferred_tools: Tools this persona is most effective with.
        risk_tolerance: Willingness to take risks, 0.0-1.0 (default: 0.5).
        creativity: Level of creative/novel approaches, 0.0-1.0 (default: 0.5).
        custom_traits: Additional domain-specific traits.

    Example:
        persona = PersonaTraits(
            name="Security Auditor",
            role="security_reviewer",
            description="Identifies vulnerabilities and security issues",
            communication_style=CommunicationStyle.FORMAL,
            expertise_level=ExpertiseLevel.SPECIALIST,
            strengths=["vulnerability detection", "threat modeling"],
            preferred_tools=["static_analysis", "dependency_check"],
            risk_tolerance=0.2,  # Very risk-averse
        )
    """

    name: str
    role: str
    description: str
    communication_style: CommunicationStyle = CommunicationStyle.TECHNICAL
    expertise_level: ExpertiseLevel = ExpertiseLevel.EXPERT
    verbosity: float = 0.5
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    preferred_tools: List[str] = field(default_factory=list)
    risk_tolerance: float = 0.5
    creativity: float = 0.5
    custom_traits: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate trait values after initialization."""
        if not 0.0 <= self.verbosity <= 1.0:
            raise ValueError(f"verbosity must be between 0.0 and 1.0, got {self.verbosity}")
        if not 0.0 <= self.risk_tolerance <= 1.0:
            raise ValueError(
                f"risk_tolerance must be between 0.0 and 1.0, got {self.risk_tolerance}"
            )
        if not 0.0 <= self.creativity <= 1.0:
            raise ValueError(f"creativity must be between 0.0 and 1.0, got {self.creativity}")

    def to_system_prompt_fragment(self) -> str:
        """Generate a system prompt fragment for this persona.

        Creates a text block that can be included in a system prompt
        to establish the persona's character and behavior.

        Returns:
            Formatted system prompt fragment.

        Example:
            fragment = persona.to_system_prompt_fragment()
            system_prompt = f"You are an AI assistant.\\n\\n{fragment}"
        """
        lines = [
            f"You are {self.name}, a {self.role}.",
            f"Description: {self.description}",
            f"Communication style: {self.communication_style.value}",
        ]
        if self.strengths:
            lines.append(f"Strengths: {', '.join(self.strengths)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona traits to dictionary for serialization.

        Returns:
            Dictionary representation with enum values converted to strings.
        """
        data = asdict(self)
        data["communication_style"] = self.communication_style.value
        data["expertise_level"] = self.expertise_level.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaTraits":
        """Create PersonaTraits from dictionary.

        Args:
            data: Dictionary with persona trait values.

        Returns:
            PersonaTraits instance.
        """
        data = data.copy()
        if isinstance(data.get("communication_style"), str):
            data["communication_style"] = CommunicationStyle(data["communication_style"])
        if isinstance(data.get("expertise_level"), str):
            data["expertise_level"] = ExpertiseLevel(data["expertise_level"])
        return cls(**data)


@dataclass
class PersonaTemplate:
    """Template for creating personas with defaults.

    PersonaTemplate allows defining a base persona configuration that
    can be customized when creating concrete PersonaTraits instances.

    Attributes:
        base_traits: The base PersonaTraits to use as template.
        overrides: Default overrides to apply when creating instances.

    Example:
        # Create a template for code reviewers
        base = PersonaTraits(
            name="Code Reviewer",
            role="reviewer",
            description="Reviews code for quality and correctness",
            communication_style=CommunicationStyle.TECHNICAL,
        )
        template = PersonaTemplate(base_traits=base)

        # Create specialized reviewers from template
        security_reviewer = template.create(
            name="Security Reviewer",
            description="Reviews code for security vulnerabilities",
            strengths=["vulnerability detection"],
        )
    """

    base_traits: PersonaTraits
    overrides: Dict[str, Any] = field(default_factory=dict)

    def create(self, **kwargs: Any) -> PersonaTraits:
        """Create a PersonaTraits instance from this template.

        Merges base traits with overrides and any additional kwargs.

        Args:
            **kwargs: Additional trait values to override.

        Returns:
            New PersonaTraits instance.

        Example:
            persona = template.create(name="Custom Name", verbosity=0.8)
        """
        # Convert base traits to dict, handling enums
        merged = asdict(self.base_traits)
        merged["communication_style"] = self.base_traits.communication_style
        merged["expertise_level"] = self.base_traits.expertise_level

        # Apply overrides and kwargs
        merged.update(self.overrides)
        merged.update(kwargs)

        return PersonaTraits(**merged)


__all__ = [
    "CommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    "PersonaTraits",
]
