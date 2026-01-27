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
        temperature: LLM temperature for this persona (0.0-1.0)
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
    temperature: float = 0.7
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.constraints is None:
            self.constraints = PersonaConstraints()

        # Validate temperature
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {self.temperature}")

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

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for this persona.

        Returns the system prompt from prompt_templates if available,
        otherwise generates a default system prompt.
        """
        if self.prompt_templates:
            return self.prompt_templates.system_prompt

        # Generate default system prompt
        return f"""You are {self.name}, an AI assistant with the following characteristics:

Personality: {self.personality.value}
Communication Style: {self.communication_style.value}
Expertise: {", ".join(self.expertise) if self.expertise else "General knowledge"}
{f"Backstory: {self.backstory}" if self.backstory else ""}

Please respond in a manner consistent with this persona, maintaining the specified
communication style and leveraging your expertise areas to provide helpful,
accurate responses."""

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
            "temperature": self.temperature,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }


@dataclass
class ContextAdjustment:
    """Context-specific adjustments to a persona.

    This dataclass supports both legacy and experimental API patterns
    for backward compatibility during API evolution.

    Legacy Attributes:
        task_type: Type of task triggering the adjustment
        personality_override: Optional personality override
        communication_override: Optional communication style override
        additional_expertise: Expertise areas to add for this context
        constraint_modifications: Temporary constraint changes
        prompt_modifications: Additional prompt content

    Experimental Attributes (Phase 3):
        temperature: LLM temperature adjustment (0.0-1.0)
        verbosity: Verbosity level ("concise", "standard", "verbose")
        tool_preference: Preferred tools for this context
        expertise_boost: Expertise areas to boost for this context
        constraint_relaxations: Temporary constraint relaxations

    Note: The experimental attributes provide a more expressive API for
    persona adaptation based on episodic and semantic memory.
    """

    # Legacy attributes (for backward compatibility)
    task_type: Optional[str] = None
    personality_override: Optional[PersonalityType] = None
    communication_override: Optional[CommunicationStyle] = None
    additional_expertise: List[str] = field(default_factory=list)
    constraint_modifications: Optional[Dict[str, Any]] = None
    prompt_modifications: Optional[List[str]] = None

    # Experimental attributes (Phase 3 memory-driven adaptation)
    temperature: Optional[float] = None
    verbosity: Optional[str] = None
    tool_preference: Optional[List[str]] = None
    expertise_boost: Optional[List[str]] = None
    constraint_relaxations: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize defaults and validate the adjustment."""
        # additional_expertise has default_factory=list
        if self.tool_preference is None:
            self.tool_preference = []
        if self.expertise_boost is None:
            self.expertise_boost = []
        if self.constraint_relaxations is None:
            self.constraint_relaxations = {}

        # Map experimental to legacy for compatibility
        if self.expertise_boost and not self.additional_expertise:
            self.additional_expertise = self.expertise_boost
        if self.constraint_relaxations and not self.constraint_modifications:
            self.constraint_modifications = self.constraint_relaxations


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

    This dataclass supports both legacy and experimental API patterns.

    Legacy Attributes:
        base_persona: Original persona being adapted
        context_adjustments: List of applied context adjustments
        dynamic_traits: Calculated traits for this context
        adaptation_reason: Explanation of why adaptations were made
        adapted_at: When this adaptation was created

    Experimental Attributes (Phase 3):
        context_type: Type identifier for the adaptation context
        adaptations: Single ContextAdjustment object
        confidence: Confidence score for the adaptation (0.0-1.0)
        reasoning: Explanation for the adaptation

    Note: The experimental API provides a simpler, more focused interface
    for memory-driven persona adaptation.
    """

    base_persona: Persona

    # Legacy attributes
    context_adjustments: List[ContextAdjustment] = field(default_factory=list)
    dynamic_traits: List[DynamicTrait] = field(default_factory=list)
    adaptation_reason: str = ""
    adapted_at: datetime = field(default_factory=datetime.utcnow)

    # Experimental attributes (Phase 3 memory-driven adaptation)
    context_type: Optional[str] = None
    adaptations: Optional[ContextAdjustment] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize and normalize the adapted persona."""
        # Handle experimental API: convert to legacy format
        if self.adaptations and not self.context_adjustments:
            self.context_adjustments = [self.adaptations]

        # Use reasoning as adaptation_reason if set
        if self.reasoning and not self.adaptation_reason:
            self.adaptation_reason = self.reasoning

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
            # Also add experimental expertise_boost
            if adjustment.expertise_boost:
                base_expertise.update(adjustment.expertise_boost)
        return list(base_expertise)

    @property
    def temperature(self) -> float:
        """Get effective temperature after adjustments.

        Returns the temperature from adaptations if set, otherwise
        returns the base persona's default temperature.
        """
        for adjustment in reversed(self.context_adjustments):
            if adjustment.temperature is not None:
                return adjustment.temperature
        # Default to 0.7 if not set
        return getattr(self.base_persona, "temperature", 0.7)

    @property
    def constraints(self) -> PersonaConstraints:
        """Get effective constraints after modifications."""
        base = self.base_persona.constraints or PersonaConstraints()

        # Apply modifications
        modifications: Dict[str, Any] = {}
        for adjustment in self.context_adjustments:
            # Legacy constraint_modifications
            if adjustment.constraint_modifications:
                modifications.update(adjustment.constraint_modifications)
            # Experimental constraint_relaxations
            if adjustment.constraint_relaxations:
                modifications.update(adjustment.constraint_relaxations)

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
            explanation_depth=modifications.get("explanation_depth", base.explanation_depth),
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

    This dataclass supports both legacy and experimental API patterns
    for memory-driven persona evolution.

    Legacy Attributes:
        persona_id: Persona being evaluated
        success_rating: Rating from 1.0 (poor) to 5.0 (excellent)
        user_comments: Optional textual feedback
        suggested_improvements: Suggested changes to persona
        context: Context in which feedback was given
        timestamp: When feedback was provided

    Experimental Attributes (Phase 3):
        task_type: Type of task the feedback applies to
        success_rate: Success rate (0.0-1.0)
        average_reward: Average reward from episodes
        feedback_data: Structured feedback data

    Note: The experimental API supports aggregated feedback from
    episodic memory for persona evolution.
    """

    persona_id: str

    # Legacy attributes
    success_rating: Optional[float] = None
    user_comments: Optional[str] = None
    suggested_improvements: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Experimental attributes (Phase 3 memory-driven evolution)
    task_type: Optional[str] = None
    success_rate: Optional[float] = None
    average_reward: Optional[float] = None
    feedback_data: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate and normalize feedback."""
        # Validate success_rating if set
        if self.success_rating is not None and not (1.0 <= self.success_rating <= 5.0):
            raise ValueError(
                f"success_rating must be between 1.0 and 5.0, got {self.success_rating}"
            )

        # Map experimental to legacy for compatibility
        if self.feedback_data:
            # Merge feedback_data into context
            if self.context is None:
                self.context = {}
            self.context.update(self.feedback_data)

            # Map success_rate to success_rating if not set
            if self.success_rating is None and self.success_rate is not None:
                # Convert 0-1 scale to 1-5 scale
                self.success_rating = 1.0 + (self.success_rate * 4.0)
