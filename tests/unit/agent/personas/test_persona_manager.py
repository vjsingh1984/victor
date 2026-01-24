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

"""Unit tests for PersonaManager.

Tests persona loading, adaptation, creation, and feedback integration.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, Mock

import pytest

from victor.agent.personas.persona_manager import PersonaManager
from victor.agent.personas.persona_repository import PersonaRepository
from victor.agent.personas.types import (
    AdaptedPersona,
    CommunicationStyle,
    ContextAdjustment,
    DynamicTrait,
    Feedback,
    PersonalityType,
    Persona,
    PersonaConstraints,
    PromptTemplates,
)


@pytest.fixture
def repository():
    """Create a test repository."""
    return PersonaRepository()


@pytest.fixture
def persona_manager(repository):
    """Create a test persona manager."""
    return PersonaManager(repository=repository)


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
        constraints=PersonaConstraints(
            max_tool_calls=50,
            preferred_tools={"read_file", "write_file"},
            response_length="medium",
            explanation_depth="standard",
        ),
    )


class TestPersonaManager:
    """Test PersonaManager class."""

    def test_load_persona_success(self, persona_manager, sample_persona):
        """Test successful persona loading."""
        # Save persona first
        persona_manager.repository.save(sample_persona)

        # Load persona
        loaded = persona_manager.load_persona("test_persona")

        assert loaded.id == "test_persona"
        assert loaded.name == "Test Persona"
        assert loaded.personality == PersonalityType.PRAGMATIC

    def test_load_persona_not_found(self, persona_manager):
        """Test loading non-existent persona."""
        with pytest.raises(ValueError, match="Persona not found"):
            persona_manager.load_persona("nonexistent")

    def test_adapt_persona_basic(self, persona_manager, sample_persona):
        """Test basic persona adaptation."""
        context = {"task_type": "security_review", "urgency": "high"}

        adapted = persona_manager.adapt_persona(sample_persona, context)

        assert isinstance(adapted, AdaptedPersona)
        assert adapted.base_persona == sample_persona
        assert len(adapted.context_adjustments) > 0

    def test_context_aware_adaptation_security(self, persona_manager):
        """Test context-aware adaptation for security tasks."""
        persona = Persona(
            id="dev",
            name="Developer",
            description="Dev",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        context = {"task_type": "security_review"}
        adapted = persona_manager.adapt_persona(persona, context)

        # Should adjust personality to cautious for security
        assert adapted.personality == PersonalityType.CAUTIOUS

    def test_context_aware_adaptation_urgency(self, persona_manager):
        """Test context-aware adaptation for high urgency."""
        persona = Persona(
            id="dev",
            name="Developer",
            description="Dev",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.VERBOSE,
            expertise=["coding"],
        )

        context = {"urgency": "high"}
        adapted = persona_manager.adapt_persona(persona, context)

        # Should adjust communication to concise for urgency
        assert adapted.communication_style == CommunicationStyle.CONCISE

    def test_context_aware_adaptation_user_preference(self, persona_manager):
        """Test context-aware adaptation for user preferences."""
        persona = Persona(
            id="dev",
            name="Developer",
            description="Dev",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            constraints=PersonaConstraints(explanation_depth="standard"),
        )

        context = {"user_preference": "thorough"}
        adapted = persona_manager.adapt_persona(persona, context)

        # Should adjust explanation depth
        assert adapted.constraints.explanation_depth == "detailed"

    def test_adapted_persona_expertise_addition(self, persona_manager):
        """Test that adapted persona adds expertise from context."""
        persona = Persona(
            id="dev",
            name="Developer",
            description="Dev",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        context = {"task_type": "security_review"}
        adapted = persona_manager.adapt_persona(persona, context)

        # Should add security expertise
        assert "security" in adapted.expertise
        assert "vulnerabilities" in adapted.expertise

    def test_create_custom_persona_success(self, persona_manager):
        """Test creating a custom persona."""
        traits = {
            "personality": "creative",
            "communication_style": "casual",
            "description": "Creative developer",
            "expertise": ["design", "innovation"],
            "backstory": "Loves thinking outside the box",
        }

        persona = persona_manager.create_custom_persona("Creative Dev", traits)

        assert persona.id == "custom_creative_dev"  # Auto-generated with 'custom_' prefix
        assert persona.name == "Creative Dev"
        assert persona.personality == PersonalityType.CREATIVE
        assert persona.communication_style == CommunicationStyle.CASUAL
        assert "design" in persona.expertise

    def test_create_custom_persona_with_id(self, persona_manager):
        """Test creating custom persona with explicit ID."""
        traits = {
            "personality": "systematic",
            "communication_style": "formal",
            "description": "Formal auditor",
        }

        persona = persona_manager.create_custom_persona(
            "Auditor", traits, persona_id="custom_auditor"
        )

        assert persona.id == "custom_auditor"

    def test_create_custom_persona_missing_traits(self, persona_manager):
        """Test creating custom persona with missing required traits."""
        traits = {
            "personality": "pragmatic",
            # Missing communication_style
            "description": "Incomplete persona",
        }

        with pytest.raises(ValueError, match="Missing required trait"):
            persona_manager.create_custom_persona("Incomplete", traits)

    def test_create_custom_persona_with_constraints(self, persona_manager):
        """Test creating custom persona with constraints."""
        traits = {
            "personality": "cautious",
            "communication_style": "formal",
            "description": "Secure persona",
            "constraints": {
                "max_tool_calls": 30,
                "forbidden_tools": ["execute_code"],
                "response_length": "long",
            },
        }

        persona = persona_manager.create_custom_persona("Secure Dev", traits)

        assert persona.constraints.max_tool_calls == 30
        assert "execute_code" in persona.constraints.forbidden_tools
        assert persona.constraints.response_length == "long"

    def test_get_suitable_personas_by_task(self, persona_manager):
        """Test getting suitable personas for a task."""
        # Create multiple personas
        security_persona = Persona(
            id="security",
            name="Security Expert",
            description="Security specialist",
            personality=PersonalityType.CAUTIOUS,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["security", "auditing", "vulnerabilities"],
        )

        dev_persona = Persona(
            id="developer",
            name="Developer",
            description="General developer",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding", "testing"],
        )

        persona_manager.repository.save(security_persona)
        persona_manager.repository.save(dev_persona)

        # Find suitable for security task
        suitable = persona_manager.get_suitable_personas("Security audit needed")

        assert len(suitable) > 0
        # Security persona should rank high
        persona_ids = [p[0].id for p in suitable]
        assert "security" in persona_ids

    def test_get_suitable_personas_with_min_score(self, persona_manager):
        """Test getting suitable personas with minimum score threshold."""
        persona = Persona(
            id="generalist",
            name="Generalist",
            description="General knowledge",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],  # Limited expertise
        )

        persona_manager.repository.save(persona)

        # High threshold should filter out non-matching personas
        suitable = persona_manager.get_suitable_personas("Deep security audit", min_score=0.8)

        # Generalist should not match highly for security
        generalist_results = [p for p in suitable if p[0].id == "generalist"]
        assert len(generalist_results) == 0

    def test_update_persona_from_feedback_success(self, persona_manager):
        """Test updating persona from feedback."""
        persona = Persona(
            id="updatable",
            name="Updatable",
            description="Can be updated",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        persona_manager.repository.save(persona)

        feedback = Feedback(
            persona_id="updatable",
            success_rating=4.5,
            user_comments="Great work",
            suggested_improvements={
                "add_expertise": ["performance"],
                "communication_style": "accessible",
            },
        )

        persona_manager.update_persona_from_feedback("updatable", feedback)

        # Reload and verify updates
        updated = persona_manager.load_persona("updatable")
        assert "performance" in updated.expertise
        assert updated.communication_style == CommunicationStyle.ACCESSIBLE
        assert updated.version == 2

    def test_update_persona_from_feedback_nonexistent(self, persona_manager):
        """Test updating non-existent persona."""
        feedback = Feedback(
            persona_id="nonexistent",
            success_rating=3.0,
        )

        with pytest.raises(ValueError, match="Persona not found"):
            persona_manager.update_persona_from_feedback("nonexistent", feedback)

    def test_export_persona_success(self, persona_manager, sample_persona):
        """Test exporting persona."""
        persona_manager.repository.save(sample_persona)

        exported = persona_manager.export_persona("test_persona")

        assert exported["id"] == "test_persona"
        assert exported["name"] == "Test Persona"
        assert exported["personality"] == "pragmatic"
        assert exported["communication_style"] == "technical"

    def test_export_persona_not_found(self, persona_manager):
        """Test exporting non-existent persona."""
        with pytest.raises(ValueError, match="Persona not found"):
            persona_manager.export_persona("nonexistent")

    def test_import_persona_success(self, persona_manager):
        """Test importing persona."""
        definition = {
            "id": "imported",
            "name": "Imported Persona",
            "description": "Imported from dict",
            "personality": "creative",
            "communication_style": "casual",
            "expertise": ["innovation"],
            "version": 1,
        }

        persona = persona_manager.import_persona(definition)

        assert persona.id == "imported"
        assert persona.name == "Imported Persona"
        assert persona.personality == PersonalityType.CREATIVE

    def test_import_persona_missing_fields(self, persona_manager):
        """Test importing persona with missing fields."""
        definition = {
            "id": "incomplete",
            # Missing name, description, etc.
            "personality": "pragmatic",
        }

        with pytest.raises(ValueError, match="Missing required field"):
            persona_manager.import_persona(definition)


class TestPersonaDataclass:
    """Test Persona dataclass."""

    def test_persona_matches_expertise_perfect(self):
        """Test expertise matching with perfect match."""
        persona = Persona(
            id="expert",
            name="Expert",
            description="Expert",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["security", "coding", "testing"],
        )

        score = persona.matches_expertise({"security", "coding", "testing"})
        assert score == 1.0

    def test_persona_matches_expertise_partial(self):
        """Test expertise matching with partial match."""
        persona = Persona(
            id="expert",
            name="Expert",
            description="Expert",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["security", "coding"],
        )

        score = persona.matches_expertise({"security", "performance", "testing"})
        # 1 out of 3 matches
        assert score == 1.0 / 3.0

    def test_persona_matches_expertise_none(self):
        """Test expertise matching with no required expertise."""
        persona = Persona(
            id="expert",
            name="Expert",
            description="Expert",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        score = persona.matches_expertise(set())
        # Neutral score when no requirements
        assert score == 0.5

    def test_persona_to_dict(self):
        """Test converting persona to dictionary."""
        persona = Persona(
            id="test",
            name="Test",
            description="Test persona",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            constraints=PersonaConstraints(max_tool_calls=50),
        )

        data = persona.to_dict()

        assert data["id"] == "test"
        assert data["personality"] == "pragmatic"
        assert data["communication_style"] == "technical"
        assert data["constraints"]["max_tool_calls"] == 50


class TestPersonaConstraints:
    """Test PersonaConstraints dataclass."""

    def test_constraints_defaults(self):
        """Test default constraint values."""
        constraints = PersonaConstraints()

        assert constraints.max_tool_calls is None
        assert constraints.preferred_tools == set()
        assert constraints.forbidden_tools == set()
        assert constraints.response_length == "medium"
        assert constraints.explanation_depth == "standard"

    def test_constraints_custom(self):
        """Test custom constraints."""
        constraints = PersonaConstraints(
            max_tool_calls=100,
            preferred_tools={"read", "write"},
            forbidden_tools={"delete"},
            response_length="long",
            explanation_depth="detailed",
        )

        assert constraints.max_tool_calls == 100
        assert "read" in constraints.preferred_tools
        assert "delete" in constraints.forbidden_tools
        assert constraints.response_length == "long"
        assert constraints.explanation_depth == "detailed"

    def test_constraints_invalid_response_length(self):
        """Test invalid response length."""
        with pytest.raises(ValueError, match="response_length must be one of"):
            PersonaConstraints(response_length="invalid")

    def test_constraints_invalid_explanation_depth(self):
        """Test invalid explanation depth."""
        with pytest.raises(ValueError, match="explanation_depth must be one of"):
            PersonaConstraints(explanation_depth="invalid")


class TestAdaptedPersona:
    """Test AdaptedPersona dataclass."""

    def test_adapted_persona_property_inheritance(self):
        """Test that adapted persona inherits base properties."""
        base = Persona(
            id="base",
            name="Base",
            description="Base persona",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        adapted = AdaptedPersona(base_persona=base)

        # Should inherit from base when no adjustments
        assert adapted.personality == PersonalityType.PRAGMATIC
        assert adapted.communication_style == CommunicationStyle.TECHNICAL
        assert adapted.expertise == ["coding"]

    def test_adapted_persona_personality_override(self):
        """Test personality override in adjustments."""
        base = Persona(
            id="base",
            name="Base",
            description="Base persona",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        adjustment = ContextAdjustment(
            task_type="security",
            personality_override=PersonalityType.CAUTIOUS,
        )

        adapted = AdaptedPersona(base_persona=base, context_adjustments=[adjustment])

        assert adapted.personality == PersonalityType.CAUTIOUS

    def test_adapted_persona_expertise_addition(self):
        """Test expertise addition in adjustments."""
        base = Persona(
            id="base",
            name="Base",
            description="Base persona",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        adjustment = ContextAdjustment(
            task_type="security",
            additional_expertise=["security", "auditing"],
        )

        adapted = AdaptedPersona(base_persona=base, context_adjustments=[adjustment])

        assert "coding" in adapted.expertise
        assert "security" in adapted.expertise
        assert "auditing" in adapted.expertise

    def test_adapted_persona_generate_system_prompt(self):
        """Test system prompt generation."""
        base = Persona(
            id="base",
            name="Test Assistant",
            description="Test",
            personality=PersonalityType.SUPPORTIVE,
            communication_style=CommunicationStyle.EDUCATIONAL,
            expertise=["teaching"],
            backstory="Helpful mentor",
        )

        trait = DynamicTrait(
            name="focus", value="beginner_friendly", confidence=0.9, reason="Context"
        )

        adapted = AdaptedPersona(
            base_persona=base,
            dynamic_traits=[trait],
            adaptation_reason="Teaching mode",
        )

        prompt = adapted.generate_system_prompt()

        assert "Test Assistant" in prompt
        assert "supportive" in prompt
        assert "educational" in prompt
        assert "teaching" in prompt
        assert "Helpful mentor" in prompt
        assert "focus: beginner_friendly" in prompt
        assert "Teaching mode" in prompt


class TestFeedback:
    """Test Feedback dataclass."""

    def test_feedback_valid_rating(self):
        """Test feedback with valid rating."""
        feedback = Feedback(persona_id="test", success_rating=4.5, user_comments="Great job")

        assert feedback.persona_id == "test"
        assert feedback.success_rating == 4.5
        assert feedback.user_comments == "Great job"

    def test_feedback_rating_too_low(self):
        """Test feedback with rating below minimum."""
        with pytest.raises(ValueError, match="success_rating must be between"):
            Feedback(persona_id="test", success_rating=0.5)

    def test_feedback_rating_too_high(self):
        """Test feedback with rating above maximum."""
        with pytest.raises(ValueError, match="success_rating must be between"):
            Feedback(persona_id="test", success_rating=5.5)

    def test_feedback_boundary_values(self):
        """Test feedback with boundary rating values."""
        feedback1 = Feedback(persona_id="test", success_rating=1.0)
        assert feedback1.success_rating == 1.0

        feedback2 = Feedback(persona_id="test", success_rating=5.0)
        assert feedback2.success_rating == 5.0


class TestPersonaManagerCaching:
    """Test persona manager caching behavior."""

    def test_adaptation_caching(self, persona_manager):
        """Test that adapted personas are cached."""
        persona = Persona(
            id="cached",
            name="Cached",
            description="Cache test",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        context = {"task_type": "debugging", "urgency": "normal"}

        # Adapt twice with same context
        adapted1 = persona_manager.adapt_persona(persona, context)
        adapted2 = persona_manager.adapt_persona(persona, context)

        # Should return same cached instance
        assert adapted1 is adapted2

    def test_cache_invalidation_on_feedback(self, persona_manager):
        """Test that feedback invalidates cache."""
        persona = Persona(
            id="cached",
            name="Cached",
            description="Cache test",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        persona_manager.repository.save(persona)

        context = {"task_type": "debugging"}
        persona_manager.adapt_persona(persona, context)

        # Submit feedback that updates the persona
        feedback = Feedback(
            persona_id="cached",
            success_rating=4.0,
            suggested_improvements={"add_expertise": ["performance"]},
        )

        persona_manager.update_persona_from_feedback("cached", feedback)

        # Cache should be cleared
        assert len(persona_manager._adaptation_cache) == 0


class TestPersonaManagerEventPublishing:
    """Test persona manager event publishing."""

    def test_load_persona_publishes_event(self, persona_manager, sample_persona):
        """Test that loading persona publishes event."""
        from unittest.mock import Mock

        mock_event_bus = Mock()
        mock_event_bus.publish = Mock()
        persona_manager._event_bus = mock_event_bus

        persona_manager.repository.save(sample_persona)
        persona_manager.load_persona("test_persona")

        # Event should be published (fire-and-forget, so we check the call was made)
        assert mock_event_bus.publish.called

    def test_adapt_persona_publishes_event(self, persona_manager, sample_persona):
        """Test that adapting persona publishes event."""
        from unittest.mock import Mock

        mock_event_bus = Mock()
        mock_event_bus.publish = Mock()
        persona_manager._event_bus = mock_event_bus

        context = {"task_type": "security_review"}
        persona_manager.adapt_persona(sample_persona, context)

        # Event should be published
        assert mock_event_bus.publish.called

    def test_create_custom_persona_publishes_event(self, persona_manager):
        """Test that creating custom persona publishes event."""
        from unittest.mock import Mock

        mock_event_bus = Mock()
        mock_event_bus.publish = Mock()
        persona_manager._event_bus = mock_event_bus

        traits = {
            "personality": "creative",
            "communication_style": "casual",
            "description": "Test persona",
        }

        persona_manager.create_custom_persona("Test", traits)

        # Event should be published
        assert mock_event_bus.publish.called


class TestPersonaManagerAdvanced:
    """Test advanced persona manager functionality."""

    def test_adapt_persona_with_multiple_contexts(self, persona_manager):
        """Test persona adaptation with multiple context factors."""
        persona = Persona(
            id="multi",
            name="Multi",
            description="Multi",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        context = {
            "task_type": "security_review",
            "urgency": "high",
            "user_preference": "thorough",
            "complexity": "high",
        }

        adapted = persona_manager.adapt_persona(persona, context)

        # Should apply multiple adjustments
        assert len(adapted.context_adjustments) >= 2
        assert len(adapted.dynamic_traits) >= 2

    def test_get_suitable_personas_empty_repository(self, persona_manager):
        """Test getting suitable personas from empty repository."""
        suitable = persona_manager.get_suitable_personas("coding task")

        assert len(suitable) == 0

    def test_get_suitable_personas_ranking(self, persona_manager):
        """Test that suitable personas are ranked correctly."""
        security_persona = Persona(
            id="security",
            name="Security",
            description="Security expert",
            personality=PersonalityType.CAUTIOUS,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["security", "vulnerabilities", "auditing"],
        )

        partial_persona = Persona(
            id="partial",
            name="Partial",
            description="Partial match",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["security"],
        )

        persona_manager.repository.save(security_persona)
        persona_manager.repository.save(partial_persona)

        suitable = persona_manager.get_suitable_personas("security and vulnerabilities")

        # Should be ranked by score
        assert len(suitable) >= 1
        # Security persona should rank higher (more expertise matches)
        persona_ids = [p[0].id for p in suitable]
        security_index = persona_ids.index("security")
        partial_index = persona_ids.index("partial")
        assert security_index < partial_index

    def test_update_persona_without_improvements(self, persona_manager):
        """Test updating persona without suggested improvements."""
        persona = Persona(
            id="no_improvement",
            name="No Improvement",
            description="Test",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        persona_manager.repository.save(persona)
        original_version = persona.version

        feedback = Feedback(
            persona_id="no_improvement",
            success_rating=5.0,
            user_comments="Perfect!",
            suggested_improvements=None,
        )

        persona_manager.update_persona_from_feedback("no_improvement", feedback)

        # Version should not increment without improvements
        updated = persona_manager.load_persona("no_improvement")
        assert updated.version == original_version

    def test_import_persona_with_prompt_templates(self, persona_manager):
        """Test importing persona with custom prompt templates."""
        definition = {
            "id": "templated",
            "name": "Templated Persona",
            "description": "Has custom templates",
            "personality": "supportive",
            "communication_style": "educational",
            "prompt_templates": {
                "system_prompt": "You are {name}, a helpful assistant.",
                "greeting": "Hello! I'm {name}.",
                "farewell": "Goodbye!",
            },
        }

        persona = persona_manager.import_persona(definition)

        assert persona.prompt_templates is not None
        assert "You are {name}" in persona.prompt_templates.system_prompt
        assert persona.prompt_templates.greeting == "Hello! I'm {name}."
        assert persona.prompt_templates.farewell == "Goodbye!"

    def test_create_custom_persona_with_prompt_templates(self, persona_manager):
        """Test creating custom persona with prompt templates."""
        traits = {
            "personality": "supportive",
            "communication_style": "educational",
            "description": "Educational assistant",
            "prompt_templates": {
                "system_prompt": "You are a teacher",
                "greeting": "Hello, student!",
            },
        }

        persona = persona_manager.create_custom_persona("Teacher", traits)

        assert persona.prompt_templates is not None
        assert "teacher" in persona.prompt_templates.system_prompt.lower()

    def test_adapted_persona_constraints_merging(self, persona_manager):
        """Test that adapted persona merges constraints correctly."""
        persona = Persona(
            id="constrained",
            name="Constrained",
            description="Has constraints",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            constraints=PersonaConstraints(
                max_tool_calls=50,
                response_length="medium",
                explanation_depth="standard",
            ),
        )

        context = {"urgency": "high"}
        adapted = persona_manager.adapt_persona(persona, context)

        # Constraints should be merged from modifications
        assert adapted.constraints.response_length == "short"  # Changed by urgency

    def test_multiple_personalities_for_different_tasks(self, persona_manager):
        """Test personality matching for different task types."""
        creative_persona = Persona(
            id="creative",
            name="Creative",
            description="Creative",
            personality=PersonalityType.CREATIVE,
            communication_style=CommunicationStyle.CASUAL,
            expertise=["design"],
        )

        methodical_persona = Persona(
            id="methodical",
            name="Methodical",
            description="Methodical",
            personality=PersonalityType.METHODICAL,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["debugging"],
        )

        persona_manager.repository.save(creative_persona)
        persona_manager.repository.save(methodical_persona)

        # Creative task
        creative_results = persona_manager.get_suitable_personas("Creative design work")
        creative_ids = [p[0].id for p in creative_results]
        assert "creative" in creative_ids

        # Debugging task
        debug_results = persona_manager.get_suitable_personas("Debug this issue")
        debug_ids = [p[0].id for p in debug_results]
        assert "methodical" in debug_ids

    def test_persona_repository_integration(self, persona_manager):
        """Test persona manager integration with repository."""
        persona = Persona(
            id="integration",
            name="Integration",
            description="Test",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        # Save via manager
        persona_manager.repository.save(persona)

        # Load via manager
        loaded = persona_manager.load_persona("integration")

        assert loaded.id == "integration"
        assert loaded.name == "Integration"

        # Export via manager
        exported = persona_manager.export_persona("integration")
        assert exported["id"] == "integration"

    def test_adapt_persona_caching_with_different_contexts(self, persona_manager):
        """Test that different contexts create different cache entries."""
        persona = Persona(
            id="cache_test",
            name="Cache Test",
            description="Test",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        context1 = {"task_type": "security_review"}
        context2 = {"task_type": "debugging"}

        adapted1 = persona_manager.adapt_persona(persona, context1)
        adapted2 = persona_manager.adapt_persona(persona, context2)

        # Different contexts should create different adaptations
        assert adapted1 is not adapted2

    def test_dynamic_traits_calculation(self, persona_manager):
        """Test dynamic traits are calculated correctly."""
        persona = Persona(
            id="dynamic",
            name="Dynamic",
            description="Test",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["security", "coding"],
        )

        context = {
            "task_type": "security_review",
            "urgency": "high",
            "complexity": "high",
        }

        adapted = persona_manager.adapt_persona(persona, context)

        # Should have dynamic traits
        assert len(adapted.dynamic_traits) >= 2

        # Check for expected traits
        trait_names = [trait.name for trait in adapted.dynamic_traits]
        assert "task_complexity_handling" in trait_names
        assert "efficiency_focus" in trait_names

    def test_expertise_extraction_from_task(self, persona_manager):
        """Test expertise extraction from task description."""
        # Test various task descriptions
        security_tasks = [
            "Security audit of the codebase",
            "Check for vulnerabilities",
            "Perform security review",
        ]

        for task in security_tasks:
            suitable = persona_manager.get_suitable_personas(task)
            # Should match security-related personas
            assert isinstance(suitable, list)

    def test_feedback_with_constraint_updates(self, persona_manager):
        """Test feedback with constraint updates."""
        persona = Persona(
            id="constraint_update",
            name="Constraint Update",
            description="Test",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            constraints=PersonaConstraints(max_tool_calls=50),
        )

        persona_manager.repository.save(persona)

        feedback = Feedback(
            persona_id="constraint_update",
            success_rating=3.5,
            suggested_improvements={
                "constraints": {
                    "preferred_tools": ["read", "write"],
                    "forbidden_tools": ["execute"],
                }
            },
        )

        persona_manager.update_persona_from_feedback("constraint_update", feedback)

        updated = persona_manager.load_persona("constraint_update")
        # Constraints should be merged (preferred_tools and forbidden_tools)
        assert "read" in updated.constraints.preferred_tools
        assert "write" in updated.constraints.preferred_tools
        assert "execute" in updated.constraints.forbidden_tools

    def test_export_import_roundtrip(self, persona_manager, sample_persona):
        """Test exporting and importing persona maintains data."""
        persona_manager.repository.save(sample_persona)

        # Export
        exported = persona_manager.export_persona("test_persona")

        # Create new manager and import
        new_manager = PersonaManager()
        imported = new_manager.import_persona(exported)

        assert imported.id == sample_persona.id
        assert imported.name == sample_persona.name
        assert imported.personality == sample_persona.personality
        assert imported.communication_style == sample_persona.communication_style
        assert imported.expertise == sample_persona.expertise


class TestPersonaManagerMerge:
    """Test persona manager merge functionality."""

    def test_merge_personas_basic(self, persona_manager):
        """Test basic persona merging."""
        persona1 = Persona(
            id="dev",
            name="Developer",
            description="Developer",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding", "testing"],
            constraints=PersonaConstraints(max_tool_calls=50),
        )

        persona2 = Persona(
            id="security",
            name="Security",
            description="Security",
            personality=PersonalityType.CAUTIOUS,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["security", "auditing"],
            constraints=PersonaConstraints(max_tool_calls=100),
        )

        merged = persona_manager.merge_personas([persona1, persona2], "DevSec Expert")

        assert merged.name == "DevSec Expert"
        assert "coding" in merged.expertise
        assert "security" in merged.expertise
        assert merged.personality == PersonalityType.PRAGMATIC  # Uses first persona's
        assert merged.constraints.max_tool_calls == 100  # Max of both

    def test_merge_personas_constraints(self, persona_manager):
        """Test that constraints are properly merged."""
        persona1 = Persona(
            id="p1",
            name="P1",
            description="P1",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            constraints=PersonaConstraints(
                max_tool_calls=50,
                preferred_tools={"read", "write"},
                forbidden_tools={"delete"},
            ),
        )

        persona2 = Persona(
            id="p2",
            name="P2",
            description="P2",
            personality=PersonalityType.METHODICAL,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["testing"],
            constraints=PersonaConstraints(
                max_tool_calls=100,
                preferred_tools={"test"},
                forbidden_tools={"execute"},
            ),
        )

        merged = persona_manager.merge_personas([persona1, persona2], "Merged")

        # Should use max of max_tool_calls
        assert merged.constraints.max_tool_calls == 100
        # Should union preferred_tools
        assert "read" in merged.constraints.preferred_tools
        assert "test" in merged.constraints.preferred_tools
        # Should union forbidden_tools
        assert "delete" in merged.constraints.forbidden_tools
        assert "execute" in merged.constraints.forbidden_tools

    def test_merge_personas_insufficient_count(self, persona_manager):
        """Test merging with insufficient personas."""
        persona = Persona(
            id="single",
            name="Single",
            description="Single",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        with pytest.raises(ValueError, match="Must provide at least 2 personas"):
            persona_manager.merge_personas([persona], "Should Fail")

    def test_merge_personas_with_backstories(self, persona_manager):
        """Test that backstories are combined."""
        persona1 = Persona(
            id="p1",
            name="P1",
            description="P1",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            backstory="Developer backstory",
            constraints=PersonaConstraints(max_tool_calls=50),
        )

        persona2 = Persona(
            id="p2",
            name="P2",
            description="P2",
            personality=PersonalityType.CAUTIOUS,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["security"],
            backstory="Security backstory",
            constraints=PersonaConstraints(max_tool_calls=100),
        )

        merged = persona_manager.merge_personas([persona1, persona2], "Combined")

        assert "Developer backstory" in merged.backstory
        assert "Security backstory" in merged.backstory

    def test_merge_personas_with_prompt_templates(self, persona_manager):
        """Test that prompt templates are merged."""
        persona1 = Persona(
            id="p1",
            name="P1",
            description="P1",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            prompt_templates=PromptTemplates(
                system_prompt="You are a developer",
                greeting="Hello dev",
            ),
            constraints=PersonaConstraints(max_tool_calls=50),
        )

        persona2 = Persona(
            id="p2",
            name="P2",
            description="P2",
            personality=PersonalityType.CAUTIOUS,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["security"],
            prompt_templates=PromptTemplates(
                system_prompt="You are security expert",
                farewell="Goodbye securely",
            ),
            constraints=PersonaConstraints(max_tool_calls=100),
        )

        merged = persona_manager.merge_personas([persona1, persona2], "Combined")

        # Should have combined prompt templates
        assert merged.prompt_templates is not None
        assert "developer" in merged.prompt_templates.system_prompt.lower()

    def test_merge_personas_communication_style_voting(self, persona_manager):
        """Test that most common communication style is selected."""
        persona1 = Persona(
            id="p1",
            name="P1",
            description="P1",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            constraints=PersonaConstraints(max_tool_calls=50),
        )

        persona2 = Persona(
            id="p2",
            name="P2",
            description="P2",
            personality=PersonalityType.CAUTIOUS,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["security"],
            constraints=PersonaConstraints(max_tool_calls=75),
        )

        persona3 = Persona(
            id="p3",
            name="P3",
            description="P3",
            personality=PersonalityType.METHODICAL,
            communication_style=CommunicationStyle.FORMAL,
            expertise=["testing"],
            constraints=PersonaConstraints(max_tool_calls=100),
        )

        merged = persona_manager.merge_personas([persona1, persona2, persona3], "Voted")

        # TECHNICAL appears twice, FORMAL once
        assert merged.communication_style == CommunicationStyle.TECHNICAL


class TestPersonaManagerValidation:
    """Test persona manager validation functionality."""

    def test_validate_persona_success(self, persona_manager):
        """Test validating a valid persona."""
        persona = Persona(
            id="valid",
            name="Valid Persona",
            description="A valid persona",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        # Should not raise
        persona_manager.validate_persona(persona)

    def test_validate_persona_missing_id(self, persona_manager):
        """Test validating persona without ID."""
        persona = Persona(
            id="",  # Empty ID
            name="No ID",
            description="Missing ID",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        with pytest.raises(ValueError, match="Persona ID is required"):
            persona_manager.validate_persona(persona)

    def test_validate_persona_missing_name(self, persona_manager):
        """Test validating persona without name."""
        persona = Persona(
            id="no_name",
            name="",  # Empty name
            description="Missing name",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
        )

        with pytest.raises(ValueError, match="Persona name is required"):
            persona_manager.validate_persona(persona)

    def test_validate_persona_conflicting_constraints(self, persona_manager):
        """Test validating persona with conflicting constraints."""
        persona = Persona(
            id="conflicted",
            name="Conflicted",
            description="Has conflicting constraints",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            constraints=PersonaConstraints(
                preferred_tools={"read"},
                forbidden_tools={"read"},  # Same tool in both
            ),
        )

        with pytest.raises(ValueError, match="Tools cannot be both preferred and forbidden"):
            persona_manager.validate_persona(persona)

    def test_validate_persona_multiple_conflicting_tools(self, persona_manager):
        """Test validating persona with multiple conflicting tools."""
        persona = Persona(
            id="multi_conflict",
            name="Multi Conflict",
            description="Multiple conflicts",
            personality=PersonalityType.PRAGMATIC,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise=["coding"],
            constraints=PersonaConstraints(
                preferred_tools={"read", "write", "execute"},
                forbidden_tools={"execute", "delete", "write"},
            ),
        )

        with pytest.raises(ValueError, match="Tools cannot be both preferred and forbidden"):
            persona_manager.validate_persona(persona)
