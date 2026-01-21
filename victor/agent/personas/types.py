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

"""Persona type definitions to avoid circular imports.

This module defines the core data types for the persona system.
All other persona modules import from here to avoid circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PersonalityType(Enum):
    """Core personality types for agents."""

    CURIOUS = "curious"
    CAUTIOUS = "cautious"
    CREATIVE = "creative"
    PRAGMATIC = "pragmatic"
    SYSTEMATIC = "systematic"
    CRITICAL = "critical"
    SUPPORTIVE = "supportive"
    METHODICAL = "methodical"


class CommunicationStyle(Enum):
    """Communication style preferences."""

    CONCISE = "concise"
    VERBOSE = "verbose"
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    ACCESSIBLE = "accessible"
    EDUCATIONAL = "educational"
    DIRECT = "direct"
    CONSTRUCTIVE = "constructive"


@dataclass
class PersonaConstraints:
    """Behavioral and operational constraints for a persona.

    Attributes:
        max_tool_calls: Maximum number of tool calls per interaction
        preferred_tools: Tools this persona prefers to use
        forbidden_tools: Tools this persona should not use
        response_length: Target response length (short/medium/long)
        explanation_depth: Depth of explanations (brief/standard/detailed)
    """

    max_tool_calls: Optional[int] = None
    preferred_tools: Optional[Set[str]] = None
    forbidden_tools: Optional[Set[str]] = None
    response_length: str = "medium"  # short, medium, long
    explanation_depth: str = "standard"  # brief, standard, detailed

    def __post_init__(self) -> None:
        """Validate constraints."""
        if self.preferred_tools is None:
            self.preferred_tools = set()
        if self.forbidden_tools is None:
            self.forbidden_tools = set()

        # Validate response_length
        valid_lengths = {"short", "medium", "long"}
        if self.response_length not in valid_lengths:
            raise ValueError(
                f"response_length must be one of {valid_lengths}, got {self.response_length}"
            )

        # Validate explanation_depth
        valid_depths = {"brief", "standard", "detailed"}
        if self.explanation_depth not in valid_depths:
            raise ValueError(
                f"explanation_depth must be one of {valid_depths}, got {self.explanation_depth}"
            )


@dataclass
class PromptTemplates:
    """Prompt templates for different interaction types.

    Attributes:
        system_prompt: Template for system prompt generation
        task_prompt: Template for task-specific prompts
        greeting: Optional greeting message
        farewell: Optional farewell message
    """

    system_prompt: str
    task_prompt: Optional[str] = None
    greeting: Optional[str] = None
    farewell: Optional[str] = None


@dataclass
class Persona:
    """Base persona definition.

    A persona defines an agent's personality, communication style, expertise areas,
    and behavioral constraints. Personas enable adaptive agent behavior by modifying
    system prompts, tool selection, and interaction patterns.

    Attributes:
        id: Unique identifier for the persona
        name: Display name
        description: Brief description of the persona
        personality: Core personality type
        communication_style: Preferred communication style
        expertise: List of expertise areas
        backstory: Optional background story/context
        constraints: Behavioral and operational constraints
        prompt_templates: Custom prompt templates
        created_at: Timestamp of creation
        version: Persona version for tracking changes
    """

    id: str
    name: str
    description: str
    personality: PersonalityType
    communication_style: CommunicationStyle
    expertise: List[str] = field(default_factory=list)
    backstory: Optional[str] = None
    constraints: Optional[PersonaConstraints] = None
    prompt_templates: Optional[PromptTemplates] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.constraints is None:
            self.constraints = PersonaConstraints()

    def matches_expertise(self, required_expertise: Set[str]) -> float:
        """Calculate expertise match score (0.0 to 1.0).

        Args:
            required_expertise: Set of required expertise areas

        Returns:
            Match score from 0.0 (no match) to 1.0 (perfect match)
        """
        if not required_expertise:
            return 0.5  # Neutral score when no requirements

        persona_expertise = {e.lower() for e in self.expertise}
        required = {e.lower() for e in required_expertise}

        matches = len(persona_expertise & required)
        return matches / len(required)

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary representation.

        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "personality": self.personality.value,
            "communication_style": self.communication_style.value,
            "expertise": self.expertise,
            "backstory": self.backstory,
            "constraints": (
                {
                    "max_tool_calls": self.constraints.max_tool_calls,
                    "preferred_tools": list(self.constraints.preferred_tools or []),
                    "forbidden_tools": list(self.constraints.forbidden_tools or []),
                    "response_length": self.constraints.response_length,
                    "explanation_depth": self.constraints.explanation_depth,
                }
                if self.constraints
                else None
            ),
            "prompt_templates": (
                {
                    "system_prompt": self.prompt_templates.system_prompt,
                    "task_prompt": self.prompt_templates.task_prompt,
                    "greeting": self.prompt_templates.greeting,
                    "farewell": self.prompt_templates.farewell,
                }
                if self.prompt_templates
                else None
            ),
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }


@dataclass
class ContextAdjustment:
    """Context-specific adjustments to a persona.

    Attributes:
        task_type: Type of task triggering the adjustment
        personality_override: Optional personality override
        communication_override: Optional communication style override
        additional_expertise: Expertise areas to add for this context
        constraint_modifications: Temporary constraint changes
        prompt_modifications: Additional prompt content
    """

    task_type: str
    personality_override: Optional[PersonalityType] = None
    communication_override: Optional[CommunicationStyle] = None
    additional_expertise: List[str] = field(default_factory=list)
    constraint_modifications: Optional[Dict[str, Any]] = None
    prompt_modifications: Optional[List[str]] = None


@dataclass
class DynamicTrait:
    """Runtime-calculated trait based on context.

    Attributes:
        name: Trait name
        value: Calculated trait value
        confidence: Confidence score (0.0 to 1.0)
        reason: Explanation for why this trait was selected
    """

    name: str
    value: Any
    confidence: float
    reason: str


@dataclass
class AdaptedPersona:
    """Context-adapted persona with dynamic traits.

    An AdaptedPersona extends a base persona with context-specific adjustments
    and dynamically calculated traits based on the current task, user preferences,
    and conversation history.

    Attributes:
        base_persona: Original persona being adapted
        context_adjustments: Applied context adjustments
        dynamic_traits: Calculated traits for this context
        adaptation_reason: Explanation of why adaptations were made
        adapted_at: When this adaptation was created
    """

    base_persona: Persona
    context_adjustments: List[ContextAdjustment] = field(default_factory=list)
    dynamic_traits: List[DynamicTrait] = field(default_factory=list)
    adaptation_reason: str = ""
    adapted_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def personality(self) -> PersonalityType:
        """Get effective personality after adaptations."""
        for adjustment in reversed(self.context_adjustments):
            if adjustment.personality_override:
                return adjustment.personality_override
        return self.base_persona.personality

    @property
    def communication_style(self) -> CommunicationStyle:
        """Get effective communication style after adaptations."""
        for adjustment in reversed(self.context_adjustments):
            if adjustment.communication_override:
                return adjustment.communication_override
        return self.base_persona.communication_style

    @property
    def expertise(self) -> List[str]:
        """Get effective expertise list after additions."""
        base_expertise = set(self.base_persona.expertise)
        for adjustment in self.context_adjustments:
            base_expertise.update(adjustment.additional_expertise)
        return list(base_expertise)

    @property
    def constraints(self) -> PersonaConstraints:
        """Get effective constraints after modifications."""
        base = self.base_persona.constraints or PersonaConstraints()

        # Apply modifications
        modifications: Dict[str, Any] = {}
        for adjustment in self.context_adjustments:
            if adjustment.constraint_modifications:
                modifications.update(adjustment.constraint_modifications)

        if not modifications:
            return base

        # Create modified constraints
        return PersonaConstraints(
            max_tool_calls=modifications.get("max_tool_calls", base.max_tool_calls),
            preferred_tools=set(
                modifications.get("preferred_tools", list(base.preferred_tools or []))
            ),
            forbidden_tools=set(
                modifications.get("forbidden_tools", list(base.forbidden_tools or []))
            ),
            response_length=modifications.get("response_length", base.response_length),
            explanation_depth=modifications.get(
                "explanation_depth", base.explanation_depth
            ),
        )

    def generate_system_prompt(self) -> str:
        """Generate system prompt for this adapted persona.

        Returns:
            System prompt string incorporating all adaptations
        """
        # Start with base template
        if self.base_persona.prompt_templates:
            template = self.base_persona.prompt_templates.system_prompt
        else:
            template = self._default_system_prompt()

        # Apply persona traits
        prompt = template.format(
            name=self.base_persona.name,
            personality=self.personality.value,
            communication_style=self.communication_style.value,
            expertise=", ".join(self.expertise),
            backstory=self.base_persona.backstory or "",
        )

        # Add dynamic traits
        if self.dynamic_traits:
            prompt += "\n\nAdditional Context:\n"
            for trait in self.dynamic_traits:
                prompt += f"- {trait.name}: {trait.value} (confidence: {trait.confidence:.0%})\n"

        # Add adaptation reason
        if self.adaptation_reason:
            prompt += f"\nAdaptation: {self.adaptation_reason}"

        return prompt

    def _default_system_prompt(self) -> str:
        """Generate default system prompt template."""
        return """You are {name}, an AI assistant with the following characteristics:

Personality: {personality}
Communication Style: {communication_style}
Expertise: {expertise}
{backstory}

Please respond in a manner consistent with this persona, maintaining the specified
communication style and leveraging your expertise areas to provide helpful,
accurate responses."""


@dataclass
class Feedback:
    """User feedback on persona performance.

    Attributes:
        persona_id: Persona being evaluated
        success_rating: Rating from 1.0 (poor) to 5.0 (excellent)
        user_comments: Optional textual feedback
        suggested_improvements: Suggested changes to persona
        context: Context in which feedback was given
        timestamp: When feedback was provided
    """

    persona_id: str
    success_rating: float
    user_comments: Optional[str] = None
    suggested_improvements: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate feedback."""
        if not (1.0 <= self.success_rating <= 5.0):
            raise ValueError(
                f"success_rating must be between 1.0 and 5.0, got {self.success_rating}"
            )
