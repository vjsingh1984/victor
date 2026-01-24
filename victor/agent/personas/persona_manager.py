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

"""Persona manager for dynamic agent persona management.

This module provides the PersonaManager class which handles:
- Loading predefined personas from YAML configuration
- Adapting personas based on context (task type, user preferences)
- Creating and validating custom personas
- Matching personas to tasks based on suitability
- Incorporating feedback to improve personas
- Exporting/importing personas for sharing

The persona system enables adaptive agent behavior by modifying:
- System prompts and personality
- Communication style and verbosity
- Tool selection preferences
- Expertise focus areas
- Behavioral constraints
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from victor.agent.personas.persona_repository import PersonaRepository
from victor.agent.personas.types import (
    AdaptedPersona,
    CommunicationStyle,
    ContextAdjustment,
    DynamicTrait,
    Feedback,
    Persona,
    PersonaConstraints,
    PersonalityType,
    PromptTemplates,
)
from victor.core.events import IEventBackend, MessagingEvent
from victor.core.mode_config import ModeConfig

logger = logging.getLogger(__name__)


class PersonaManager:
    """Central manager for dynamic agent personas.

    The PersonaManager handles loading, adapting, creating, and managing personas.
    It integrates with the event bus for persona lifecycle events and maintains
    a repository for persona persistence.

    Attributes:
        repository: Persona storage backend
        event_bus: Event bus for publishing persona events
    """

    def __init__(
        self,
        repository: Optional[PersonaRepository] = None,
        event_bus: Optional[IEventBackend] = None,
        auto_load: bool = True,
    ) -> None:
        """Initialize the persona manager.

        Args:
            repository: Optional persona repository (defaults to in-memory with YAML loading)
            event_bus: Optional event bus for persona events
            auto_load: Whether to auto-load personas from default YAML location
        """
        from pathlib import Path

        if repository is None:
            # Create repository with default YAML path
            default_yaml = (
                Path(__file__).parent.parent.parent / "config" / "personas" / "agent_personas.yaml"
            )
            if default_yaml.exists() and auto_load:
                repository = PersonaRepository(storage_path=default_yaml)
                logger.info(f"Loaded personas from: {default_yaml}")
            else:
                repository = PersonaRepository()

        self.repository = repository
        self._event_bus = event_bus
        self._adaptation_cache: Dict[Tuple[str, str], AdaptedPersona] = {}

    def load_persona(self, persona_id: str) -> Persona:
        """Load a persona by ID.

        Args:
            persona_id: Unique persona identifier

        Returns:
            Loaded persona

        Raises:
            ValueError: If persona not found
        """
        logger.debug(f"Loading persona: {persona_id}")

        persona = self.repository.get(persona_id)
        if persona is None:
            raise ValueError(f"Persona not found: {persona_id}")

        # Publish event
        self._publish_event("persona.loaded", {"persona_id": persona_id})

        return persona

    def adapt_persona(self, persona: Persona, context: Dict[str, Any]) -> AdaptedPersona:
        """Adapt a persona based on context.

        Context may include:
        - task_type: Type of task being performed
        - user_preference: User-specified preferences
        - conversation_history: Previous conversation context
        - urgency: How quickly the task needs completion
        - complexity: Task complexity level

        Args:
            persona: Base persona to adapt
            context: Adaptation context

        Returns:
            Adapted persona with context-specific adjustments
        """
        logger.debug(f"Adapting persona {persona.id} to context: {context}")

        # Check cache
        cache_key = (persona.id, str(sorted(context.items())))
        if cache_key in self._adaptation_cache:
            return self._adaptation_cache[cache_key]

        # Determine adjustments
        adjustments = self._determine_adjustments(persona, context)
        dynamic_traits = self._calculate_dynamic_traits(persona, context)
        reason = self._explain_adaptations(persona, context, adjustments)

        # Create adapted persona
        adapted = AdaptedPersona(
            base_persona=persona,
            context_adjustments=adjustments,
            dynamic_traits=dynamic_traits,
            adaptation_reason=reason,
        )

        # Cache result
        self._adaptation_cache[cache_key] = adapted

        # Publish event
        self._publish_event(
            "persona.adapted",
            {
                "persona_id": persona.id,
                "context": context,
                "adjustments": len(adjustments),
            },
        )

        return adapted

    def create_custom_persona(
        self, name: str, traits: Dict[str, Any], persona_id: Optional[str] = None
    ) -> Persona:
        """Create a custom persona from traits.

        Args:
            name: Persona display name
            traits: Dictionary of persona traits
            persona_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Created persona

        Raises:
            ValueError: If required traits are missing or invalid
        """
        logger.debug(f"Creating custom persona: {name}")

        # Validate required fields
        if "personality" not in traits:
            raise ValueError("Missing required trait: personality")
        if "communication_style" not in traits:
            raise ValueError("Missing required trait: communication_style")
        if "description" not in traits:
            raise ValueError("Missing required trait: description")

        # Parse personality
        personality_value = traits["personality"]
        if isinstance(personality_value, str):
            personality = PersonalityType(personality_value)
        else:
            personality = personality_value

        # Parse communication style
        communication_value = traits["communication_style"]
        if isinstance(communication_value, str):
            communication = CommunicationStyle(communication_value)
        else:
            communication = communication_value

        # Create persona
        persona = Persona(
            id=persona_id or f"custom_{name.lower().replace(' ', '_')}",
            name=name,
            description=traits["description"],
            personality=personality,
            communication_style=communication,
            expertise=traits.get("expertise", []),
            backstory=traits.get("backstory"),
            constraints=self._parse_constraints(traits.get("constraints")),
            prompt_templates=self._parse_prompt_templates(traits.get("prompt_templates")),
        )

        # Validate persona
        self._validate_persona(persona)

        # Save to repository
        self.repository.save(persona)

        # Publish event
        self._publish_event("persona.created", {"persona_id": persona.id, "name": name})

        return persona

    def get_suitable_personas(
        self, task: str, min_score: float = 0.3
    ) -> List[Tuple[Persona, float]]:
        """Get personas suitable for a given task, ranked by match score.

        Args:
            task: Task description or type
            min_score: Minimum match score (0.0 to 1.0)

        Returns:
            List of (persona, score) tuples, sorted by score descending
        """
        logger.debug(f"Finding suitable personas for task: {task}")

        # Extract expertise requirements from task
        required_expertise = self._extract_expertise_from_task(task)

        # Get all personas
        all_personas = self.repository.list_all()

        # Score each persona
        scored = []
        for persona in all_personas:
            score = persona.matches_expertise(required_expertise)

            # Boost score for personality/task alignment
            if self._personality_matches_task(persona, task):
                score *= 1.2

            if score >= min_score:
                scored.append((persona, min(score, 1.0)))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Found {len(scored)} suitable personas")
        return scored

    def get_suggested_persona(self, task: str) -> Optional[Persona]:
        """Get the single best persona for a given task.

        This is a convenience method that returns the highest-scoring persona
        from get_suitable_personas(), or None if no persona meets the minimum threshold.

        Args:
            task: Task description or type

        Returns:
            Best matching Persona, or None if no suitable persona found
        """
        logger.debug(f"Getting suggested persona for task: {task}")

        suitable = self.get_suitable_personas(task, min_score=0.4)

        if suitable:
            best_persona, score = suitable[0]
            logger.info(f"Suggested persona: {best_persona.name} (score: {score:.2%})")
            return best_persona

        logger.debug("No suitable persona found")
        return None

    def merge_personas(
        self, personas: List[Persona], merged_name: str, merged_id: Optional[str] = None
    ) -> Persona:
        """Merge multiple personas into a hybrid persona.

        This method combines expertise, constraints, and traits from multiple personas
        to create a hybrid persona. Useful for creating specialized personas that
        combine characteristics from multiple predefined personas.

        Merging Strategy:
        - Personality: Uses the first persona's personality as base
        - Communication: Uses the most common communication style
        - Expertise: Union of all expertise areas (deduplicated)
        - Constraints: Merges preferred_tools (union), forbidden_tools (union), and uses max of max_tool_calls
        - Description: Combines all descriptions
        - Backstory: Concatenates all backstories

        Args:
            personas: List of personas to merge (must have at least 2)
            merged_name: Name for the merged persona
            merged_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Merged persona with combined characteristics

        Raises:
            ValueError: If fewer than 2 personas provided or personas conflict
        """
        if len(personas) < 2:
            raise ValueError(f"Must provide at least 2 personas to merge, got {len(personas)}")

        logger.debug(f"Merging {len(personas)} personas into: {merged_name}")

        # Validate no critical conflicts
        self._validate_merge_compatibility(personas)

        # Determine personality (use first persona's as base)
        personality = personas[0].personality

        # Determine most common communication style
        comm_styles = [p.communication_style for p in personas]
        communication = max(set(comm_styles), key=comm_styles.count)

        # Combine expertise (union, deduplicated)
        all_expertise = set()
        for persona in personas:
            all_expertise.update(persona.expertise)
        expertise = sorted(list(all_expertise))

        # Merge constraints
        constraints = self._merge_constraints([p.constraints for p in personas])

        # Combine descriptions
        descriptions = [p.description for p in personas]
        combined_description = f"Hybrid persona combining: {'; '.join(descriptions)}"

        # Combine backstories
        backstories = [p.backstory for p in personas if p.backstory]
        combined_backstory = "\n\n".join(backstories) if backstories else None

        # Create merged persona
        merged_persona = Persona(
            id=merged_id or f"merged_{'_'.join(p.id for p in personas)}",
            name=merged_name,
            description=combined_description,
            personality=personality,
            communication_style=communication,
            expertise=expertise,
            backstory=combined_backstory,
            constraints=constraints,
            prompt_templates=self._merge_prompt_templates([p.prompt_templates for p in personas]),
        )

        # Validate merged persona
        self.validate_persona(merged_persona)

        # Save to repository
        self.repository.save(merged_persona)

        # Publish event
        self._publish_event(
            "persona.merged",
            {
                "merged_id": merged_persona.id,
                "source_personas": [p.id for p in personas],
                "expertise_count": len(expertise),
            },
        )

        logger.info(
            f"Created merged persona: {merged_persona.id} with {len(expertise)} expertise areas"
        )
        return merged_persona

    def validate_persona(self, persona: Persona) -> None:
        """Validate a persona definition.

        This public method performs comprehensive validation of a persona,
        checking for required fields, valid enum values, and logical consistency.

        Args:
            persona: Persona to validate

        Raises:
            ValueError: If persona is invalid with specific error message
        """
        logger.debug(f"Validating persona: {persona.id}")

        self._validate_persona(persona)

        logger.debug(f"Persona {persona.id} is valid")

    def update_persona_from_feedback(self, persona_id: str, feedback: Feedback) -> None:
        """Update a persona based on user feedback.

        Args:
            persona_id: Persona to update
            feedback: User feedback to incorporate

        Raises:
            ValueError: If persona not found
        """
        logger.debug(f"Updating persona {persona_id} from feedback")

        persona = self.repository.get(persona_id)
        if persona is None:
            raise ValueError(f"Persona not found: {persona_id}")

        # Apply suggested improvements
        if feedback.suggested_improvements:
            self._apply_improvements(persona, feedback.suggested_improvements)

            # Increment version
            persona.version += 1

            # Save updated persona
            self.repository.save(persona)

            # Clear adaptation cache for this persona
            self._clear_cache_for_persona(persona_id)

        # Publish event
        self._publish_event(
            "persona.updated",
            {
                "persona_id": persona_id,
                "rating": feedback.success_rating,
                "version": persona.version,
            },
        )

    def evolve_persona(self, persona: Persona, feedback_list: List[Feedback]) -> Persona:
        """Evolve a persona based on aggregated feedback from episodic memory.

        This method creates an evolved version of a persona by applying
        multiple feedback items to improve persona performance based on
        historical outcomes.

        Args:
            persona: Base persona to evolve
            feedback_list: List of feedback items from episodic memory

        Returns:
            Evolved persona with improvements applied

        Example:
            feedback = [
                Feedback(
                    persona_id=persona.id,
                    task_type="code_review",
                    success_rate=0.75,
                    average_reward=7.5,
                    feedback_data={"best_style": "formal"}
                )
            ]
            evolved = manager.evolve_persona(persona, feedback)
        """
        if not feedback_list:
            return persona

        logger.info(f"Evolving persona {persona.id} based on {len(feedback_list)} feedback items")

        # Create a copy of the persona to evolve
        from dataclasses import replace

        evolved = replace(persona, version=persona.version + 1)

        # Aggregate feedback data
        best_styles = []
        worst_styles = []
        suggestions = []

        for feedback in feedback_list:
            # Extract feedback data
            if feedback.feedback_data:
                if "best_style" in feedback.feedback_data:
                    best_styles.append(feedback.feedback_data["best_style"])
                if "worst_style" in feedback.feedback_data:
                    worst_styles.append(feedback.feedback_data["worst_style"])
                if "improvement_suggestion" in feedback.feedback_data:
                    suggestions.append(feedback.feedback_data["improvement_suggestion"])

        # Apply communication style evolution if we have clear preferences
        if best_styles:
            # Count best styles
            from collections import Counter

            style_counts = Counter(best_styles)
            most_common_style = style_counts.most_common(1)[0][0]

            # Map string to CommunicationStyle enum
            style_map = {
                "formal": CommunicationStyle.FORMAL,
                "casual": CommunicationStyle.CASUAL,
                "concise": CommunicationStyle.CONCISE,
                "verbose": CommunicationStyle.VERBOSE,
                "technical": CommunicationStyle.TECHNICAL,
                "accessible": CommunicationStyle.ACCESSIBLE,
            }

            if most_common_style in style_map:
                evolved.communication_style = style_map[most_common_style]
                logger.debug(f"Evolved communication style to: {most_common_style}")

        # Apply suggestions to improve the persona
        if suggestions:
            improvements = {"suggestions": suggestions}
            self._apply_improvements(evolved, improvements)

        # Update system prompt based on feedback
        if evolved.prompt_templates:
            # Add feedback-based improvements to prompt
            current_prompt = evolved.prompt_templates.system_prompt
            feedback_section = "\n\nPerformance Feedback:\n"
            for suggestion in suggestions:
                feedback_section += f"- {suggestion}\n"
            evolved.prompt_templates.system_prompt = current_prompt + feedback_section

        logger.info(f"Successfully evolved persona {persona.id} to version {evolved.version}")

        # Publish evolution event
        self._publish_event(
            "persona.evolved",
            {
                "persona_id": persona.id,
                "original_version": persona.version,
                "evolved_version": evolved.version,
                "feedback_count": len(feedback_list),
            },
        )

        return evolved

    def export_persona(self, persona_id: str) -> Dict[str, Any]:
        """Export a persona for sharing.

        Args:
            persona_id: Persona to export

        Returns:
            Persona definition dictionary

        Raises:
            ValueError: If persona not found
        """
        persona = self.repository.get(persona_id)
        if persona is None:
            raise ValueError(f"Persona not found: {persona_id}")

        return persona.to_dict()

    def import_persona(self, definition: Dict[str, Any]) -> Persona:
        """Import a persona from definition.

        Args:
            definition: Persona definition dictionary

        Returns:
            Imported persona

        Raises:
            ValueError: If definition is invalid
        """
        # Validate required fields
        required = ["id", "name", "description", "personality", "communication_style"]
        for field in required:
            if field not in definition:
                raise ValueError(f"Missing required field: {field}")

        # Parse enums
        personality = PersonalityType(definition["personality"])
        communication = CommunicationStyle(definition["communication_style"])

        # Create persona
        persona = Persona(
            id=definition["id"],
            name=definition["name"],
            description=definition["description"],
            personality=personality,
            communication_style=communication,
            expertise=definition.get("expertise", []),
            backstory=definition.get("backstory"),
            constraints=self._parse_constraints(definition.get("constraints")),
            prompt_templates=self._parse_prompt_templates(definition.get("prompt_templates")),
        )

        # Validate and save
        self._validate_persona(persona)
        self.repository.save(persona)

        # Publish event
        self._publish_event("persona.imported", {"persona_id": persona.id})

        return persona

    def _determine_adjustments(
        self, persona: Persona, context: Dict[str, Any]
    ) -> List[ContextAdjustment]:
        """Determine context-specific adjustments.

        Args:
            persona: Base persona
            context: Adaptation context

        Returns:
            List of adjustments to apply
        """
        adjustments = []

        # Task type adjustments
        task_type = context.get("task_type", "")
        if task_type == "security_review":
            adjustments.append(
                ContextAdjustment(
                    task_type=task_type,
                    personality_override=PersonalityType.CAUTIOUS,
                    communication_override=CommunicationStyle.FORMAL,
                    additional_expertise=["security", "vulnerabilities"],
                    constraint_modifications={
                        "explanation_depth": "detailed",
                        "response_length": "long",
                    },
                )
            )
        elif task_type == "debugging":
            adjustments.append(
                ContextAdjustment(
                    task_type=task_type,
                    personality_override=PersonalityType.METHODICAL,
                    additional_expertise=["debugging", "troubleshooting"],
                    constraint_modifications={"explanation_depth": "detailed"},
                )
            )

        # Urgency adjustments
        urgency = context.get("urgency", "normal")
        if urgency == "high":
            adjustments.append(
                ContextAdjustment(
                    task_type="urgency",
                    communication_override=CommunicationStyle.CONCISE,
                    constraint_modifications={"response_length": "short"},
                )
            )

        # User preference adjustments
        user_pref = context.get("user_preference", "")
        if user_pref == "thorough":
            adjustments.append(
                ContextAdjustment(
                    task_type="user_preference",
                    constraint_modifications={"explanation_depth": "detailed"},
                )
            )

        return adjustments

    def _calculate_dynamic_traits(
        self, persona: Persona, context: Dict[str, Any]
    ) -> List[DynamicTrait]:
        """Calculate dynamic traits based on context.

        Args:
            persona: Base persona
            context: Adaptation context

        Returns:
            List of calculated dynamic traits
        """
        traits = []

        # Task complexity trait
        complexity = context.get("complexity", "medium")
        traits.append(
            DynamicTrait(
                name="task_complexity_handling",
                value=complexity,
                confidence=0.9,
                reason=f"Based on provided complexity level: {complexity}",
            )
        )

        # Urgency adaptation
        urgency = context.get("urgency", "normal")
        if urgency == "high":
            traits.append(
                DynamicTrait(
                    name="efficiency_focus",
                    value=True,
                    confidence=0.95,
                    reason="High urgency requires efficient responses",
                )
            )

        # Expertise activation
        task_type = context.get("task_type", "")
        relevant_expertise = [exp for exp in persona.expertise if exp.lower() in task_type.lower()]
        if relevant_expertise:
            traits.append(
                DynamicTrait(
                    name="activated_expertise",
                    value=relevant_expertise,
                    confidence=0.85,
                    reason=f"Expertise areas relevant to {task_type}",
                )
            )

        return traits

    def _explain_adaptations(
        self, persona: Persona, context: Dict[str, Any], adjustments: List[ContextAdjustment]
    ) -> str:
        """Generate explanation for adaptations.

        Args:
            persona: Base persona
            context: Adaptation context
            adjustments: Applied adjustments

        Returns:
            Human-readable explanation
        """
        reasons = []

        if adjustments:
            reasons.append(f"Applied {len(adjustments)} context-specific adjustments")

        task_type = context.get("task_type", "")
        if task_type:
            reasons.append(f"Optimized for task type: {task_type}")

        urgency = context.get("urgency", "")
        if urgency == "high":
            reasons.append("Increased conciseness for high urgency")

        user_pref = context.get("user_preference", "")
        if user_pref:
            reasons.append(f"Incorporated user preference: {user_pref}")

        return "; ".join(reasons) if reasons else "Standard persona application"

    def _extract_expertise_from_task(self, task: str) -> Set[str]:
        """Extract expertise requirements from task description.

        Args:
            task: Task description

        Returns:
            Set of required expertise areas
        """
        task_lower = task.lower()

        # Map task keywords to expertise
        expertise_map = {
            "security": {"security", "vulnerabilities", "auditing"},
            "performance": {"performance", "optimization", "profiling"},
            "test": {"testing", "quality", "verification"},
            "debug": {"debugging", "troubleshooting", "diagnostics"},
            "architecture": {"architecture", "design", "structure"},
            "review": {"review", "quality", "standards"},
            "code": {"coding", "implementation", "development"},
        }

        required = set()
        for keyword, areas in expertise_map.items():
            if keyword in task_lower:
                required.update(areas)

        return required

    def _personality_matches_task(self, persona: Persona, task: str) -> bool:
        """Check if personality matches task requirements.

        Args:
            persona: Persona to check
            task: Task description

        Returns:
            True if personality is well-suited to task
        """
        task_lower = task.lower()

        # Certain personalities are better for certain tasks
        if "security" in task_lower:
            return persona.personality in [
                PersonalityType.CAUTIOUS,
                PersonalityType.METHODICAL,
            ]

        if "creative" in task_lower or "design" in task_lower:
            return persona.personality in [
                PersonalityType.CREATIVE,
                PersonalityType.CURIOUS,
            ]

        if "debug" in task_lower:
            return persona.personality in [
                PersonalityType.METHODICAL,
                PersonalityType.SYSTEMATIC,
            ]

        return False

    def _parse_constraints(
        self, constraints_data: Optional[Dict[str, Any]]
    ) -> Optional[PersonaConstraints]:
        """Parse constraints from dictionary.

        Args:
            constraints_data: Optional constraints dictionary

        Returns:
            PersonaConstraints or None
        """
        if not constraints_data:
            return None

        return PersonaConstraints(
            max_tool_calls=constraints_data.get("max_tool_calls"),
            preferred_tools=set(constraints_data.get("preferred_tools", [])),
            forbidden_tools=set(constraints_data.get("forbidden_tools", [])),
            response_length=constraints_data.get("response_length", "medium"),
            explanation_depth=constraints_data.get("explanation_depth", "standard"),
        )

    def _parse_prompt_templates(
        self, templates_data: Optional[Dict[str, Any]]
    ) -> Optional[PromptTemplates]:
        """Parse prompt templates from dictionary.

        Args:
            templates_data: Optional templates dictionary

        Returns:
            PromptTemplates or None
        """
        if not templates_data:
            return None

        return PromptTemplates(
            system_prompt=templates_data.get("system_prompt", ""),
            task_prompt=templates_data.get("task_prompt"),
            greeting=templates_data.get("greeting"),
            farewell=templates_data.get("farewell"),
        )

    def _validate_persona(self, persona: Persona) -> None:
        """Validate persona definition.

        Args:
            persona: Persona to validate

        Raises:
            ValueError: If persona is invalid
        """
        if not persona.id:
            raise ValueError("Persona ID is required")

        if not persona.name:
            raise ValueError("Persona name is required")

        if not persona.description:
            raise ValueError("Persona description is required")

        # Validate constraints don't conflict
        if persona.constraints:
            if persona.constraints.preferred_tools and persona.constraints.forbidden_tools:
                overlap = persona.constraints.preferred_tools & persona.constraints.forbidden_tools
                if overlap:
                    raise ValueError(f"Tools cannot be both preferred and forbidden: {overlap}")

    def _apply_improvements(self, persona: Persona, improvements: Dict[str, Any]) -> None:
        """Apply suggested improvements to persona.

        Args:
            persona: Persona to improve
            improvements: Suggested improvements
        """
        # Update expertise
        if "add_expertise" in improvements:
            new_expertise = improvements["add_expertise"]
            for exp in new_expertise:
                if exp not in persona.expertise:
                    persona.expertise.append(exp)

        # Update communication style
        if "communication_style" in improvements:
            persona.communication_style = CommunicationStyle(improvements["communication_style"])

        # Update constraints
        if "constraints" in improvements:
            new_constraints = self._parse_constraints(improvements["constraints"])
            if new_constraints:
                if persona.constraints:
                    # Merge constraints
                    if new_constraints.preferred_tools:
                        persona.constraints.preferred_tools.update(new_constraints.preferred_tools)
                    if new_constraints.forbidden_tools:
                        persona.constraints.forbidden_tools.update(new_constraints.forbidden_tools)
                else:
                    persona.constraints = new_constraints

    def _clear_cache_for_persona(self, persona_id: str) -> None:
        """Clear adaptation cache for a persona.

        Args:
            persona_id: Persona ID to clear cache for
        """
        keys_to_remove = [key for key in self._adaptation_cache.keys() if key[0] == persona_id]
        for key in keys_to_remove:
            del self._adaptation_cache[key]

    def _validate_merge_compatibility(self, personas: List[Persona]) -> None:
        """Validate that personas can be merged without conflicts.

        Args:
            personas: List of personas to validate

        Raises:
            ValueError: If personas have critical conflicts
        """
        # Check for conflicting personalities (cautious + creative = bad idea)
        personalities = {p.personality for p in personas}
        conflicting_pairs = [
            (PersonalityType.CAUTIOUS, PersonalityType.CREATIVE),
            (PersonalityType.CRITICAL, PersonalityType.SUPPORTIVE),
        ]

        for pair in conflicting_pairs:
            if pair[0] in personalities and pair[1] in personalities:
                raise ValueError(
                    f"Cannot merge personas with conflicting personalities: "
                    f"{pair[0].value} and {pair[1].value}"
                )

        # Check for conflicting communication styles
        comm_styles = {p.communication_style for p in personas}
        if CommunicationStyle.CONCISE in comm_styles and CommunicationStyle.VERBOSE in comm_styles:
            logger.warning(
                "Merging concise and verbose communication styles may result in inconsistent behavior"
            )

    def _merge_constraints(
        self, constraints_list: List[Optional[PersonaConstraints]]
    ) -> PersonaConstraints:
        """Merge multiple constraint sets.

        Args:
            constraints_list: List of constraints to merge

        Returns:
            Merged PersonaConstraints
        """
        # Filter out None constraints
        valid_constraints = [c for c in constraints_list if c is not None]

        if not valid_constraints:
            return PersonaConstraints()

        # Merge max_tool_calls (use maximum)
        max_calls = max(c.max_tool_calls for c in valid_constraints if c.max_tool_calls is not None)

        # Merge preferred_tools (union)
        preferred = set()
        for c in valid_constraints:
            if c.preferred_tools:
                preferred.update(c.preferred_tools)

        # Merge forbidden_tools (union)
        forbidden = set()
        for c in valid_constraints:
            if c.forbidden_tools:
                forbidden.update(c.forbidden_tools)

        # For response_length, use the most verbose (short < medium < long)
        length_priority = {"short": 1, "medium": 2, "long": 3}
        max_priority = 0
        response_length = "medium"
        for c in valid_constraints:
            priority = length_priority.get(c.response_length, 2)
            if priority > max_priority:
                max_priority = priority
                response_length = c.response_length

        # For explanation_depth, use the most detailed (brief < standard < detailed)
        depth_priority = {"brief": 1, "standard": 2, "detailed": 3}
        max_depth_priority = 0
        explanation_depth = "standard"
        for c in valid_constraints:
            priority = depth_priority.get(c.explanation_depth, 2)
            if priority > max_depth_priority:
                max_depth_priority = priority
                explanation_depth = c.explanation_depth

        return PersonaConstraints(
            max_tool_calls=max_calls or None,
            preferred_tools=preferred or None,
            forbidden_tools=forbidden or None,
            response_length=response_length,
            explanation_depth=explanation_depth,
        )

    def _merge_prompt_templates(
        self, templates_list: List[Optional[PromptTemplates]]
    ) -> Optional[PromptTemplates]:
        """Merge multiple prompt template sets.

        Args:
            templates_list: List of prompt templates to merge

        Returns:
            Merged PromptTemplates or None if all are None
        """
        # Filter out None templates
        valid_templates = [t for t in templates_list if t is not None]

        if not valid_templates:
            return None

        # Use the first template as base
        base = valid_templates[0]

        # Combine system prompts with separators
        if len(valid_templates) > 1:
            combined_system = "\n\n---\n\n".join(t.system_prompt for t in valid_templates)
        else:
            combined_system = base.system_prompt

        return PromptTemplates(
            system_prompt=combined_system,
            task_prompt=base.task_prompt,
            greeting=base.greeting,
            farewell=base.farewell,
        )

    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish persona event to event bus.

        Args:
            event_type: Type of event
            data: Event data
        """
        if self._event_bus:
            try:
                event = MessagingEvent(
                    topic=f"persona.{event_type}",
                    data=data,
                )
                # Publish fire-and-forget
                import asyncio

                asyncio.create_task(self._event_bus.publish(event))
            except Exception as e:
                logger.warning(f"Failed to publish persona event: {e}")
