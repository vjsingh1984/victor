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

"""Tests for Data Analysis personas."""

import pytest

from victor.dataanalysis.teams.personas import (
    DATA_ANALYSIS_PERSONAS,
    DecisionStyle,
    DataAnalysisPersona,
    DataAnalysisPersonaTraits,
    ExpertiseCategory,
    CommunicationStyle,
    apply_persona_to_spec,
    get_persona,
    get_persona_by_expertise,
    get_personas_for_role,
    list_personas,
    register_data_analysis_personas,
)
from victor.framework.multi_agent import ExpertiseLevel, PersonaTraits as FrameworkPersonaTraits
from victor.framework.teams import TeamMemberSpec


class TestDataAnalysisPersonas:
    """Test Data Analysis persona definitions."""

    def test_all_personas_defined(self):
        """Test that all required personas are defined."""
        required_personas = [
            "data_engineer",
            "statistician",
            "ml_engineer",
            "visualization_specialist",
            "data_quality_analyst",
            "business_analyst",
        ]

        for persona_name in required_personas:
            assert persona_name in DATA_ANALYSIS_PERSONAS, f"Missing persona: {persona_name}"

    def test_persona_structure(self):
        """Test that personas have required attributes."""
        for persona in DATA_ANALYSIS_PERSONAS.values():
            assert isinstance(persona, DataAnalysisPersona)
            assert persona.name
            assert persona.role
            assert persona.expertise
            assert isinstance(persona.traits, DataAnalysisPersonaTraits)

    def test_data_engineer_persona(self):
        """Test data engineer persona has correct expertise."""
        persona = DATA_ANALYSIS_PERSONAS["data_engineer"]

        assert persona.name == "Data Engineer"
        assert persona.role == "data_engineer"
        assert ExpertiseCategory.DATA_ENGINEERING in persona.expertise
        assert ExpertiseCategory.ETL in persona.expertise
        assert persona.traits.communication_style == CommunicationStyle.METHODOLOGICAL
        assert persona.traits.decision_style == DecisionStyle.CONSERVATIVE

    def test_statistician_persona(self):
        """Test statistician persona has correct expertise."""
        persona = DATA_ANALYSIS_PERSONAS["statistician"]

        assert persona.name == "Statistician"
        assert persona.role == "statistician"
        assert ExpertiseCategory.STATISTICAL_ANALYSIS in persona.expertise
        assert ExpertiseCategory.HYPOTHESIS_TESTING in persona.expertise
        assert persona.traits.decision_style == DecisionStyle.RIGOROUS
        assert persona.traits.quantitative_focus > 0.9

    def test_ml_engineer_persona(self):
        """Test ML engineer persona has correct expertise."""
        persona = DATA_ANALYSIS_PERSONAS["ml_engineer"]

        assert persona.name == "ML Engineer"
        assert persona.role == "ml_engineer"
        assert ExpertiseCategory.MACHINE_LEARNING in persona.expertise
        assert ExpertiseCategory.FEATURE_ENGINEERING in persona.expertise
        assert persona.traits.decision_style == DecisionStyle.EXPERIMENTAL

    def test_visualization_specialist_persona(self):
        """Test visualization specialist persona has correct expertise."""
        persona = DATA_ANALYSIS_PERSONAS["visualization_specialist"]

        assert persona.name == "Visualization Specialist"
        assert persona.role == "visualization_specialist"
        assert ExpertiseCategory.VISUALIZATION in persona.expertise
        assert persona.traits.visualization_preference > 0.9

    def test_data_quality_analyst_persona(self):
        """Test data quality analyst persona has correct expertise."""
        persona = DATA_ANALYSIS_PERSONAS["data_quality_analyst"]

        assert persona.name == "Data Quality Analyst"
        assert persona.role == "data_quality_analyst"
        assert ExpertiseCategory.DATA_CLEANING in persona.expertise
        assert ExpertiseCategory.DATA_QUALITY in persona.expertise

    def test_business_analyst_persona(self):
        """Test business analyst persona has correct expertise."""
        persona = DATA_ANALYSIS_PERSONAS["business_analyst"]

        assert persona.name == "Business Analyst"
        assert persona.role == "business_analyst"
        assert ExpertiseCategory.BUSINESS_ANALYSIS in persona.expertise
        assert ExpertiseCategory.KPI_DEFINITION in persona.expertise
        assert persona.traits.communication_style == CommunicationStyle.EXECUTIVE


class TestPersonaHelpers:
    """Test persona helper functions."""

    def test_get_persona(self):
        """Test getting persona by name."""
        persona = get_persona("data_engineer")
        assert persona is not None
        assert persona.name == "Data Engineer"

    def test_get_persona_not_found(self):
        """Test getting non-existent persona returns None."""
        persona = get_persona("nonexistent")
        assert persona is None

    def test_list_personas(self):
        """Test listing all personas."""
        personas = list_personas()
        assert len(personas) == 6
        assert "data_engineer" in personas
        assert "statistician" in personas

    def test_get_personas_for_role(self):
        """Test getting personas by role."""
        personas = get_personas_for_role("data_engineer")
        assert len(personas) == 1
        assert personas[0].role == "data_engineer"

    def test_get_persona_by_expertise(self):
        """Test getting personas by expertise."""
        personas = get_persona_by_expertise(ExpertiseCategory.VISUALIZATION)
        assert len(personas) >= 1
        assert any(p.role == "visualization_specialist" for p in personas)

    def test_apply_persona_to_spec(self):
        """Test applying persona to team member spec."""
        spec = TeamMemberSpec(
            role="executor",
            goal="Train ML model",
            name="Model Trainer",
        )

        enhanced_spec = apply_persona_to_spec(spec, "ml_engineer")

        assert enhanced_spec.expertise
        assert enhanced_spec.backstory
        assert enhanced_spec.personality
        assert "machine_learning" in enhanced_spec.expertise

    def test_apply_nonexistent_persona(self):
        """Test applying non-existent persona doesn't modify spec."""
        spec = TeamMemberSpec(
            role="executor",
            goal="Train ML model",
            name="Model Trainer",
        )

        original_expertise = spec.expertise
        enhanced_spec = apply_persona_to_spec(spec, "nonexistent")

        assert enhanced_spec.expertise == original_expertise


class TestPersonaRegistration:
    """Test persona registration with framework."""

    def test_register_personas(self):
        """Test registering personas with framework."""
        count = register_data_analysis_personas()
        assert count == 6

    def test_personas_in_provider(self):
        """Test personas are available in framework provider."""
        from victor.framework.multi_agent.persona_provider import get_persona_provider

        register_data_analysis_personas()

        provider = get_persona_provider()

        # Check that personas are registered
        for persona_name in DATA_ANALYSIS_PERSONAS:
            persona = provider.get_persona(persona_name)
            assert persona is not None
            assert isinstance(persona, FrameworkPersonaTraits)

    def test_persona_metadata(self):
        """Test persona metadata is correct."""
        from victor.framework.multi_agent.persona_provider import get_persona_provider

        register_data_analysis_personas()
        provider = get_persona_provider()

        # Check data engineer metadata
        metadata = provider.get_persona_metadata("data_engineer")
        assert metadata is not None
        assert metadata.vertical == "dataanalysis"
        assert metadata.version == "1.0.0"
        assert metadata.category == "execution"

        # Check statistician metadata
        metadata = provider.get_persona_metadata("statistician")
        assert metadata.category == "research"

        # Check business analyst metadata
        metadata = provider.get_persona_metadata("business_analyst")
        assert metadata.category == "planning"

        # Check data quality analyst metadata
        metadata = provider.get_persona_metadata("data_quality_analyst")
        assert metadata.category == "review"


class TestPersonaTraits:
    """Test DataAnalysisPersonaTraits."""

    def test_to_prompt_hints(self):
        """Test generating prompt hints from traits."""
        traits = DataAnalysisPersonaTraits(
            communication_style=CommunicationStyle.VISUAL,
            decision_style=DecisionStyle.RIGOROUS,
            visualization_preference=0.8,
        )

        hints = traits.to_prompt_hints()
        assert "visual" in hints.lower()
        assert "statistical" in hints.lower()

    def test_to_framework_traits(self):
        """Test converting to framework traits."""
        traits = DataAnalysisPersonaTraits(
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_style=DecisionStyle.DATA_DRIVEN,
            quantitative_focus=0.9,
        )

        framework_traits = traits.to_framework_traits(
            name="Test Persona",
            role="tester",
            description="Test description",
        )

        assert isinstance(framework_traits, FrameworkPersonaTraits)
        assert framework_traits.name == "Test Persona"
        assert framework_traits.role == "tester"
        assert framework_traits.expertise_level == ExpertiseLevel.EXPERT


class TestPersonaBackstory:
    """Test persona backstory generation."""

    def test_generate_backstory(self):
        """Test generating backstory from persona."""
        persona = DATA_ANALYSIS_PERSONAS["data_engineer"]
        backstory = persona.generate_backstory()

        assert persona.name in backstory
        assert persona.role in backstory
        assert "pipeline" in backstory.lower()

    def test_backstory_includes_traits(self):
        """Test backstory includes trait hints."""
        persona = DATA_ANALYSIS_PERSONAS["visualization_specialist"]
        backstory = persona.generate_backstory()

        # Should include visualization preferences
        assert "visual" in backstory.lower()
