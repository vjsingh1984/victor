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

"""Usage examples for the Persona Manager system.

This module demonstrates how to use the dynamic persona management system
for adaptive agent behavior in Victor AI.
"""

from __future__ import annotations

import logging
from victor.agent.personas import PersonaManager, Persona, Feedback
from victor.agent.personas.types import PersonalityType, CommunicationStyle

logger = logging.getLogger(__name__)


def example_load_persona() -> None:
    """Example: Load a predefined persona."""
    print("\n=== Example: Load Predefined Persona ===\n")

    manager = PersonaManager()

    # Load a predefined persona
    senior_developer = manager.load_persona("senior_developer")

    print(f"Loaded: {senior_developer.name}")
    print(f"Personality: {senior_developer.personality.value}")
    print(f"Communication: {senior_developer.communication_style.value}")
    print(f"Expertise: {', '.join(senior_developer.expertise)}")


def example_adapt_persona() -> None:
    """Example: Adapt a persona based on context."""
    print("\n=== Example: Adapt Persona to Context ===\n")

    manager = PersonaManager()

    # Load base persona
    developer = manager.load_persona("senior_developer")

    # Adapt for security review context
    adapted = manager.adapt_persona(
        developer,
        context={
            "task_type": "security_review",
            "urgency": "normal",
            "user_preference": "thorough",
            "complexity": "high",
        },
    )

    print(f"Original personality: {developer.personality.value}")
    print(f"Adapted personality: {adapted.personality.value}")
    print(f"Adapted communication: {adapted.communication_style.value}")
    print(f"Adaptation reason: {adapted.adaptation_reason}")
    print(f"\nDynamic traits:")
    for trait in adapted.dynamic_traits:
        print(f"  - {trait.name}: {trait.value} (confidence: {trait.confidence:.0%})")


def example_create_custom_persona() -> None:
    """Example: Create a custom persona."""
    print("\n=== Example: Create Custom Persona ===\n")

    manager = PersonaManager()

    # Create a custom persona
    custom = manager.create_custom_persona(
        name="API Design Specialist",
        traits={
            "personality": PersonalityType.SYSTEMATIC,
            "communication_style": CommunicationStyle.TECHNICAL,
            "description": "Expert in API design, REST, GraphQL, and microservices",
            "expertise": ["api_design", "rest", "graphql", "microservices", "documentation"],
            "backstory": (
                "You have designed APIs for systems serving millions of requests. "
                "You understand the importance of clear contracts, versioning, "
                "and backward compatibility."
            ),
            "constraints": {
                "max_tool_calls": 45,
                "response_length": "long",
                "explanation_depth": "detailed",
            },
        },
    )

    print(f"Created: {custom.name}")
    print(f"ID: {custom.id}")
    print(f"Expertise areas: {len(custom.expertise)}")


def example_get_suggested_persona() -> None:
    """Example: Get suggested persona for a task."""
    print("\n=== Example: Get Suggested Persona ===\n")

    manager = PersonaManager()

    # Get suggested persona for a task
    task = "Review this code for security vulnerabilities"
    suggested = manager.get_suggested_persona(task)

    if suggested:
        print(f"Task: {task}")
        print(f"Suggested persona: {suggested.name}")
        print(f"Reasoning: Security-focused tasks benefit from {suggested.personality.value} personality")
    else:
        print("No suitable persona found")


def example_merge_personas() -> None:
    """Example: Merge multiple personas."""
    print("\n=== Example: Merge Personas ===\n")

    manager = PersonaManager()

    # Load multiple personas
    security = manager.load_persona("security_expert")
    performance = manager.load_persona("performance_specialist")

    # Merge them to create a hybrid
    secure_performant = manager.merge_personas(
        personas=[security, performance],
        merged_name="Security & Performance Expert",
        merged_id="secure_perf_expert",
    )

    print(f"Merged persona: {secure_performant.name}")
    print(f"Combined expertise ({len(secure_performant.expertise)} areas):")
    for exp in secure_performant.expertise:
        print(f"  - {exp}")

    if secure_performant.constraints:
        print(f"\nConstraints:")
        print(f"  Max tool calls: {secure_performant.constraints.max_tool_calls}")
        print(f"  Response length: {secure_performant.constraints.response_length}")


def example_validate_persona() -> None:
    """Example: Validate a persona."""
    print("\n=== Example: Validate Persona ===\n")

    manager = PersonaManager()

    # Load a valid persona
    persona = manager.load_persona("senior_developer")

    try:
        manager.validate_persona(persona)
        print(f"✓ {persona.name} is valid")
    except ValueError as e:
        print(f"✗ Validation error: {e}")


def example_feedback_loop() -> None:
    """Example: Update persona from user feedback."""
    print("\n=== Example: Feedback-Driven Improvement ===\n")

    manager = PersonaManager()

    # Create initial persona
    persona = manager.create_custom_persona(
        name="Junior Developer Assistant",
        traits={
            "personality": PersonalityType.SUPPORTIVE,
            "communication_style": CommunicationStyle.EDUCATIONAL,
            "description": "Helpful assistant for junior developers",
            "expertise": ["coding", "learning", "explanation"],
        },
    )

    print(f"Initial version: {persona.version}")

    # User provides feedback
    feedback = Feedback(
        persona_id=persona.id,
        success_rating=4.0,
        user_comments="Great, but needs more testing expertise",
        suggested_improvements={
            "add_expertise": ["testing", "unit_tests", "integration_tests"],
            "communication_style": "constructive",
        },
    )

    # Apply feedback
    manager.update_persona_from_feedback(persona.id, feedback)

    # Reload to see changes
    updated = manager.load_persona(persona.id)
    print(f"Updated version: {updated.version}")
    print(f"New expertise: {', '.join(updated.expertise)}")


def example_export_import() -> None:
    """Example: Export and import personas."""
    print("\n=== Example: Export/Import Personas ===\n")

    manager = PersonaManager()

    # Export a persona
    persona = manager.load_persona("senior_developer")
    exported = manager.export_persona(persona.id)

    print(f"Exported persona with {len(exported)} fields")
    print(f"Fields: {', '.join(exported.keys())}")

    # Import into a different manager
    new_manager = PersonaManager()
    imported = new_manager.import_persona(exported)

    print(f"Imported: {imported.name} (ID: {imported.id})")


def example_complete_workflow() -> None:
    """Example: Complete persona management workflow."""
    print("\n=== Example: Complete Workflow ===\n")

    manager = PersonaManager()

    # Step 1: Get suggested persona for task
    task = "I need to design a scalable microservices architecture"
    suggested = manager.get_suggested_persona(task)

    if not suggested:
        print("No exact match found, finding best alternatives...")
        alternatives = manager.get_suitable_personas(task, min_score=0.3)
        if alternatives:
            suggested = alternatives[0][0]
            print(f"Using: {suggested.name}")

    # Step 2: Adapt to specific context
    adapted = manager.adapt_persona(
        suggested,
        context={
            "task_type": "architecture",
            "urgency": "normal",
            "complexity": "high",
            "user_preference": "detailed",
        },
    )

    print(f"\nSelected: {adapted.base_persona.name}")
    print(f"Adaptations: {len(adapted.context_adjustments)}")
    print(f"System prompt preview: {adapted.generate_system_prompt()[:200]}...")


def main() -> None:
    """Run all examples."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Victor AI - Dynamic Persona Manager Examples")
    print("=" * 70)

    try:
        example_load_persona()
        example_adapt_persona()
        example_create_custom_persona()
        example_get_suggested_persona()
        example_merge_personas()
        example_validate_persona()
        example_feedback_loop()
        example_export_import()
        example_complete_workflow()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
