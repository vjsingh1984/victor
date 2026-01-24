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

"""Integration tests for persona system.

Tests integration with orchestrator, conversations, and workflows.
"""

from __future__ import annotations

import pytest

from victor.agent.personas.persona_manager import PersonaManager
from victor.agent.personas.persona_repository import PersonaRepository
from victor.agent.personas.types import (
    AdaptedPersona,
    CommunicationStyle,
    Feedback,
    PersonalityType,
    Persona,
)


@pytest.mark.integration
class TestPersonaOrchestratorIntegration:
    """Test integration with orchestrator."""

    def test_persona_with_orchestrator_initialization(self, persona_manager):
        """Test that personas can be integrated with orchestrator initialization."""
        # In a real integration, this would test loading personas during
        # orchestrator bootstrap
        persona = persona_manager.load_persona("senior_developer")

        assert persona.id == "senior_developer"
        assert "coding" in persona.expertise

    def test_persona_modifies_system_prompt(self, persona_manager):
        """Test that persona generates appropriate system prompt."""
        persona = persona_manager.load_persona("mentor")

        adapted = AdaptedPersona(base_persona=persona)
        system_prompt = adapted.generate_system_prompt()

        assert "Mentor" in system_prompt
        assert "supportive" in system_prompt.lower()
        assert "educational" in system_prompt.lower()

    def test_persona_constraints_affect_tool_selection(self, persona_manager):
        """Test that persona constraints influence tool selection."""
        persona = persona_manager.load_persona("security_expert")

        # Security expert should have constraints
        if persona.constraints:
            # Check that forbidden tools are respected
            if persona.constraints.forbidden_tools:
                assert len(persona.constraints.forbidden_tools) > 0

    @pytest.mark.skip(reason="Requires actual orchestrator setup")
    def test_persona_in_orchestrator_chat(self, persona_manager):
        """Test persona behavior in actual orchestrator chat."""
        # This would test full orchestrator integration
        # Requires actual orchestrator setup
        pass


@pytest.mark.integration
class TestPersonaConversationIntegration:
    """Test integration with conversation flow."""

    def test_persona_in_conversation_context(self, persona_manager):
        """Test using persona within conversation context."""
        # Simulate conversation start
        task_context = {
            "task_type": "code_review",
            "urgency": "normal",
            "user_preference": "thorough",
        }

        # Load and adapt persona
        base_persona = persona_manager.load_persona("senior_developer")
        adapted = persona_manager.adapt_persona(base_persona, task_context)

        # Should adapt to context
        assert isinstance(adapted, AdaptedPersona)
        assert len(adapted.context_adjustments) > 0

    def test_persona_switching_mid_conversation(self, persona_manager):
        """Test switching personas during conversation."""
        # Start with one persona
        persona1 = persona_manager.load_persona("senior_developer")

        # Simulate some conversation
        context = {"task_type": "coding", "stage": "implementation"}
        adapted1 = persona_manager.adapt_persona(persona1, context)

        # Switch to different persona
        persona2 = persona_manager.load_persona("security_expert")
        context["task_type"] = "security_review"
        adapted2 = persona_manager.adapt_persona(persona2, context)

        # Should have different personalities
        assert adapted1.personality != adapted2.personality

    def test_persona_persistence_across_sessions(self, persona_manager):
        """Test that persona choices persist across sessions."""
        # Simulate session 1
        persona_id = "senior_developer"
        persona1 = persona_manager.load_persona(persona_id)

        # Simulate session end and new session start
        persona2 = persona_manager.load_persona(persona_id)

        # Should get same persona
        assert persona1.id == persona2.id
        assert persona1.name == persona2.name


@pytest.mark.integration
class TestPersonaWorkflowIntegration:
    """Test integration with workflow execution."""

    def test_persona_in_workflow_definition(self, persona_manager):
        """Test using personas within workflow definitions."""
        # Simulate workflow that requires specific persona
        workflow_task = "Perform security audit"

        # Get suitable persona for task
        suitable = persona_manager.get_suitable_personas(workflow_task)

        assert len(suitable) > 0
        # Security expert should rank highly
        persona_ids = [p[0].id for p in suitable]
        assert "security_expert" in persona_ids

    def test_persona_adaptation_for_workflow_stages(self, persona_manager):
        """Test persona adaptation for different workflow stages."""
        persona = persona_manager.load_persona("senior_developer")

        # Use recognized task types that trigger adjustments
        # Security review stage
        security_context = {"task_type": "security_review", "urgency": "normal"}
        security_adapted = persona_manager.adapt_persona(persona, security_context)

        # Debugging stage with high urgency
        debugging_context = {"task_type": "debugging", "urgency": "high"}
        debugging_adapted = persona_manager.adapt_persona(persona, debugging_context)

        # Should have different adaptations
        # Security review should add caution personality
        assert security_adapted.personality == PersonalityType.CAUTIOUS
        # High urgency debugging should be concise
        assert debugging_adapted.communication_style == CommunicationStyle.CONCISE

        # Should have dynamic traits for context
        assert len(security_adapted.dynamic_traits) > 0
        assert len(debugging_adapted.dynamic_traits) > 0

    @pytest.mark.skip(reason="Requires actual workflow engine")
    def test_persona_workflow_execution(self, persona_manager):
        """Test persona behavior in actual workflow execution."""
        # This would test full workflow integration
        # Requires actual workflow engine
        pass


@pytest.mark.integration
class TestPersonaFeedbackIntegration:
    """Test feedback integration in persona system."""

    def test_feedback_updates_persona_in_repository(self, persona_manager):
        """Test that feedback updates persona in repository."""
        # Create custom persona
        traits = {
            "personality": "pragmatic",
            "communication_style": "technical",
            "description": "Test persona for feedback",
            "expertise": ["coding"],
        }

        persona = persona_manager.create_custom_persona("Feedback Test", traits)
        original_version = persona.version

        # Submit feedback
        feedback = Feedback(
            persona_id=persona.id,
            success_rating=4.5,
            user_comments="Great work, but add more performance focus",
            suggested_improvements={
                "add_expertise": ["performance", "optimization"],
                "communication_style": "accessible",
            },
        )

        persona_manager.update_persona_from_feedback(persona.id, feedback)

        # Reload and verify updates
        updated = persona_manager.load_persona(persona.id)
        assert updated.version == original_version + 1
        assert "performance" in updated.expertise
        assert updated.communication_style == CommunicationStyle.ACCESSIBLE

    def test_feedback_from_multiple_users(self, persona_manager):
        """Test aggregating feedback from multiple users."""
        # Create persona
        traits = {
            "personality": "creative",
            "communication_style": "casual",
            "description": "Multi-user feedback test",
            "expertise": ["design"],
        }

        persona = persona_manager.create_custom_persona("Multi-User Test", traits)

        # User 1 feedback
        feedback1 = Feedback(
            persona_id=persona.id,
            success_rating=4.0,
            suggested_improvements={"add_expertise": ["ux_design"]},
        )

        # User 2 feedback
        feedback2 = Feedback(
            persona_id=persona.id,
            success_rating=3.5,
            suggested_improvements={"add_expertise": ["ui_patterns"]},
        )

        persona_manager.update_persona_from_feedback(persona.id, feedback1)
        persona_manager.update_persona_from_feedback(persona.id, feedback2)

        # Should have accumulated improvements
        updated = persona_manager.load_persona(persona.id)
        assert "ux_design" in updated.expertise
        assert "ui_patterns" in updated.expertise

    def test_low_rating_triggers_review(self, persona_manager):
        """Test that low ratings trigger persona review."""
        # Create persona
        traits = {
            "personality": "critical",
            "communication_style": "direct",
            "description": "Low-rated persona",
            "expertise": ["review"],
        }

        persona = persona_manager.create_custom_persona("Low Rated", traits)

        # Submit low rating
        feedback = Feedback(
            persona_id=persona.id,
            success_rating=2.0,  # Low rating
            user_comments="Too harsh, not constructive enough",
            suggested_improvements={
                "communication_style": "constructive",
                "personality": "supportive",
            },
        )

        persona_manager.update_persona_from_feedback(persona.id, feedback)

        # Should update based on feedback
        updated = persona_manager.load_persona(persona.id)
        assert updated.communication_style == CommunicationStyle.CONSTRUCTIVE


@pytest.mark.integration
class TestPersonaCustomization:
    """Test persona customization scenarios."""

    def test_create_domain_specific_persona(self, persona_manager):
        """Test creating persona for specific domain."""
        traits = {
            "personality": "methodical",
            "communication_style": "formal",
            "description": "Specialized for data science tasks",
            "expertise": [
                "data_analysis",
                "statistics",
                "machine_learning",
                "visualization",
            ],
            "backstory": "Expert data scientist with 10 years of experience",
            "constraints": {
                "preferred_tools": ["analyze_data", "visualize", "train_model"],
                "response_length": "long",
                "explanation_depth": "detailed",
            },
        }

        persona = persona_manager.create_custom_persona("Data Scientist", traits)

        assert "data_analysis" in persona.expertise
        assert persona.constraints is not None
        assert "analyze_data" in persona.constraints.preferred_tools

    def test_create_role_based_persona(self, persona_manager):
        """Test creating persona for specific role."""
        traits = {
            "personality": "systematic",
            "communication_style": "technical",
            "description": "DevOps engineer persona",
            "expertise": ["deployment", "ci_cd", "monitoring", "automation"],
            "constraints": {
                "max_tool_calls": 40,
                "preferred_tools": ["deploy", "configure_ci", "setup_monitoring"],
            },
        }

        persona = persona_manager.create_custom_persona("DevOps Engineer", traits)

        assert "deployment" in persona.expertise
        assert persona.constraints.max_tool_calls == 40

    def test_clone_and_modify_persona(self, persona_manager):
        """Test cloning existing persona with modifications."""
        # Load existing persona
        original = persona_manager.load_persona("senior_developer")

        # Create modified version
        modified_traits = {
            "personality": original.personality.value,
            "communication_style": original.communication_style.value,
            "description": original.description + " - Specialized for testing",
            "expertise": original.expertise + ["performance_testing", "load_testing"],
        }

        modified = persona_manager.create_custom_persona("Testing Specialist", modified_traits)

        # Should have additional expertise
        assert "performance_testing" in modified.expertise
        assert "coding" in modified.expertise  # From original


@pytest.mark.integration
class TestPersonaScalability:
    """Test persona system scalability."""

    def test_multiple_concurrent_adaptations(self, persona_manager):
        """Test adapting persona for multiple contexts concurrently."""
        persona = persona_manager.load_persona("senior_developer")

        contexts = [
            {"task_type": "coding", "urgency": "high"},
            {"task_type": "review", "urgency": "normal"},
            {"task_type": "debugging", "urgency": "low"},
            {"task_type": "planning", "urgency": "normal"},
            {"task_type": "testing", "urgency": "high"},
        ]

        # Adapt to all contexts
        adapted_personas = [persona_manager.adapt_persona(persona, ctx) for ctx in contexts]

        # Should create distinct adaptations
        assert len(adapted_personas) == len(contexts)
        assert len({id(a) for a in adapted_personas}) == len(contexts)  # All unique

    def test_large_persona_library(self, persona_manager):
        """Test handling large persona library."""
        # Create many personas
        for i in range(50):
            traits = {
                "personality": "pragmatic",
                "communication_style": "technical",
                "description": f"Persona {i}",
                "expertise": ["coding"],
            }
            persona_manager.create_custom_persona(f"Persona_{i}", traits)

        # Should be able to list all
        all_personas = persona_manager.repository.list_all()
        assert len(all_personas) >= 50

    def test_persona_search_performance(self, persona_manager):
        """Test persona search performance."""
        # Create many personas with different expertise
        expertise_areas = [
            "security",
            "performance",
            "testing",
            "debugging",
            "architecture",
            "design",
            "deployment",
            "monitoring",
        ]

        for i, area in enumerate(expertise_areas * 10):  # 80 personas
            traits = {
                "personality": "pragmatic",
                "communication_style": "technical",
                "description": f"Expert in {area}",
                "expertise": [area, "coding"],
            }
            persona_manager.create_custom_persona(f"{area}_expert_{i}", traits)

        # Search for security-related task
        import time

        start = time.time()
        suitable = persona_manager.get_suitable_personas("Security audit and review")
        elapsed = time.time() - start

        # Should complete quickly (< 1 second)
        assert elapsed < 1.0
        assert len(suitable) > 0


@pytest.mark.integration
class TestPersonaErrorHandling:
    """Test error handling in persona system."""

    def test_handle_invalid_persona_id(self, persona_manager):
        """Test handling of invalid persona ID."""
        with pytest.raises(ValueError):
            persona_manager.load_persona("nonexistent_persona")

    def test_handle_corrupt_feedback(self, persona_manager):
        """Test handling of corrupted feedback data."""
        persona = persona_manager.create_custom_persona(
            "Test",
            {
                "personality": "pragmatic",
                "communication_style": "technical",
                "description": "Test",
            },
        )

        # Invalid feedback (negative rating)
        with pytest.raises(ValueError):
            Feedback(persona_id=persona.id, success_rating=-1.0)

    def test_handle_import_export_errors(self, persona_manager, tmp_path):
        """Test handling of import/export errors."""
        # Try to export non-existent persona
        output_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(ValueError):
            persona_manager.repository.export_to_yaml("nonexistent", output_file)

        # Try to import invalid file
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # YAML parse error
            persona_manager.repository.import_from_yaml(invalid_file)
