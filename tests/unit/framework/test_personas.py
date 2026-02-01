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

"""Tests for victor.framework.personas module.

These tests verify the Persona system that provides consistent
personality and communication styles for agents.
"""



# =============================================================================
# Persona Dataclass Tests
# =============================================================================


class TestPersonaCreation:
    """Tests for Persona dataclass creation."""

    def test_persona_required_fields(self):
        """Persona should have required name and background fields."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Test Agent",
            background="A helpful assistant for testing.",
        )

        assert persona.name == "Test Agent"
        assert persona.background == "A helpful assistant for testing."

    def test_persona_default_communication_style(self):
        """Persona should have 'professional' communication style by default."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Test Agent",
            background="Testing background.",
        )

        assert persona.communication_style == "professional"

    def test_persona_custom_communication_style(self):
        """Persona should accept custom communication style."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Friendly Bot",
            background="A casual chat companion.",
            communication_style="casual",
        )

        assert persona.communication_style == "casual"

    def test_persona_default_expertise_areas(self):
        """Persona should have empty expertise areas by default."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Test Agent",
            background="Testing background.",
        )

        assert persona.expertise_areas == ()

    def test_persona_with_expertise_areas(self):
        """Persona should store expertise areas as tuple."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Code Expert",
            background="Expert in Python programming.",
            expertise_areas=("python", "testing", "async"),
        )

        assert persona.expertise_areas == ("python", "testing", "async")
        assert "python" in persona.expertise_areas

    def test_persona_default_quirks(self):
        """Persona should have empty quirks by default."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Test Agent",
            background="Testing background.",
        )

        assert persona.quirks == ()

    def test_persona_with_quirks(self):
        """Persona should store personality quirks."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Quirky Bot",
            background="A bot with personality.",
            quirks=("uses metaphors", "asks clarifying questions"),
        )

        assert persona.quirks == ("uses metaphors", "asks clarifying questions")

    def test_persona_all_fields(self):
        """Persona should accept all fields together."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Full Agent",
            background="Complete background story.",
            communication_style="formal",
            expertise_areas=("security", "architecture"),
            quirks=("thorough", "methodical"),
        )

        assert persona.name == "Full Agent"
        assert persona.background == "Complete background story."
        assert persona.communication_style == "formal"
        assert persona.expertise_areas == ("security", "architecture")
        assert persona.quirks == ("thorough", "methodical")


# =============================================================================
# Persona format_message Tests
# =============================================================================


class TestPersonaFormatMessage:
    """Tests for Persona.format_message() method."""

    def test_format_message_formal_capitalizes(self):
        """format_message with formal style should capitalize sentences."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Formal Agent",
            background="Formal background.",
            communication_style="formal",
        )

        result = persona.format_message("hello there")

        assert result[0].isupper()  # First character is capitalized

    def test_format_message_formal_adds_punctuation(self):
        """format_message with formal style should ensure punctuation."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Formal Agent",
            background="Formal background.",
            communication_style="formal",
        )

        result = persona.format_message("hello there")

        assert result.endswith((".", "!", "?"))

    def test_format_message_casual_lowercases(self):
        """format_message with casual style should lowercase."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Casual Agent",
            background="Casual background.",
            communication_style="casual",
        )

        result = persona.format_message("HELLO THERE")

        assert result == result.lower()

    def test_format_message_professional_preserves_case(self):
        """format_message with professional style should preserve original case."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Professional Agent",
            background="Professional background.",
            communication_style="professional",
        )

        result = persona.format_message("Hello There")

        assert result == "Hello There"

    def test_format_message_empty_string(self):
        """format_message should handle empty strings."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Test Agent",
            background="Background.",
            communication_style="formal",
        )

        result = persona.format_message("")

        assert result == ""

    def test_format_message_already_punctuated(self):
        """format_message should not double punctuation."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Formal Agent",
            background="Background.",
            communication_style="formal",
        )

        result = persona.format_message("Hello there!")

        assert not result.endswith("!.")
        assert result.endswith("!")


# =============================================================================
# Persona get_system_prompt_section Tests
# =============================================================================


class TestPersonaSystemPrompt:
    """Tests for Persona.get_system_prompt_section() method."""

    def test_system_prompt_includes_name(self):
        """get_system_prompt_section should include persona name."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Code Helper",
            background="Helps with code.",
        )

        prompt = persona.get_system_prompt_section()

        assert "Code Helper" in prompt

    def test_system_prompt_includes_background(self):
        """get_system_prompt_section should include background."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Test Agent",
            background="You are an expert in software testing and quality assurance.",
        )

        prompt = persona.get_system_prompt_section()

        assert "expert in software testing" in prompt

    def test_system_prompt_includes_communication_style(self):
        """get_system_prompt_section should include communication style."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Test Agent",
            background="Background.",
            communication_style="friendly and approachable",
        )

        prompt = persona.get_system_prompt_section()

        assert "friendly and approachable" in prompt

    def test_system_prompt_includes_expertise(self):
        """get_system_prompt_section should include expertise areas."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Expert Agent",
            background="Background.",
            expertise_areas=("Python", "Machine Learning"),
        )

        prompt = persona.get_system_prompt_section()

        assert "Python" in prompt
        assert "Machine Learning" in prompt

    def test_system_prompt_includes_quirks(self):
        """get_system_prompt_section should include quirks."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Quirky Agent",
            background="Background.",
            quirks=("uses analogies", "asks follow-up questions"),
        )

        prompt = persona.get_system_prompt_section()

        assert "uses analogies" in prompt or "analogies" in prompt
        assert "follow-up questions" in prompt or "questions" in prompt

    def test_system_prompt_minimal_persona(self):
        """get_system_prompt_section should work with minimal persona."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Simple Agent",
            background="A simple helper.",
        )

        prompt = persona.get_system_prompt_section()

        assert "Simple Agent" in prompt
        assert "simple helper" in prompt


# =============================================================================
# Persona Registry Tests
# =============================================================================


class TestPersonaRegistry:
    """Tests for persona registry functions."""

    def test_get_persona_returns_registered(self):
        """get_persona should return registered persona."""
        from victor.framework.personas import get_persona, Persona

        persona = get_persona("friendly_assistant")

        assert persona is not None
        assert isinstance(persona, Persona)
        assert persona.name is not None

    def test_get_persona_friendly_assistant(self):
        """friendly_assistant persona should be available."""
        from victor.framework.personas import get_persona

        persona = get_persona("friendly_assistant")

        assert persona is not None
        assert (
            "friendly" in persona.name.lower() or "friendly" in persona.communication_style.lower()
        )

    def test_get_persona_senior_developer(self):
        """senior_developer persona should be available."""
        from victor.framework.personas import get_persona

        persona = get_persona("senior_developer")

        assert persona is not None
        assert "senior" in persona.name.lower() or "developer" in persona.name.lower()

    def test_get_persona_code_reviewer(self):
        """code_reviewer persona should be available."""
        from victor.framework.personas import get_persona

        persona = get_persona("code_reviewer")

        assert persona is not None
        assert "review" in persona.name.lower() or "reviewer" in persona.background.lower()

    def test_get_persona_mentor(self):
        """mentor persona should be available."""
        from victor.framework.personas import get_persona

        persona = get_persona("mentor")

        assert persona is not None
        assert "mentor" in persona.name.lower() or "mentor" in persona.background.lower()

    def test_get_persona_unknown_returns_none(self):
        """get_persona should return None for unknown persona."""
        from victor.framework.personas import get_persona

        persona = get_persona("nonexistent_persona_xyz")

        assert persona is None

    def test_register_persona_adds_to_registry(self):
        """register_persona should add new persona to registry."""
        from victor.framework.personas import (
            register_persona,
            get_persona,
            Persona,
        )

        custom = Persona(
            name="Custom Agent",
            background="A custom registered agent.",
        )
        register_persona("custom_test_agent", custom)

        retrieved = get_persona("custom_test_agent")

        assert retrieved is not None
        assert retrieved.name == "Custom Agent"

    def test_register_persona_overwrites_existing(self):
        """register_persona should overwrite existing persona."""
        from victor.framework.personas import (
            register_persona,
            get_persona,
            Persona,
        )

        first = Persona(
            name="First Version",
            background="Original.",
        )
        second = Persona(
            name="Second Version",
            background="Updated.",
        )

        register_persona("overwrite_test", first)
        register_persona("overwrite_test", second)

        retrieved = get_persona("overwrite_test")

        assert retrieved.name == "Second Version"


# =============================================================================
# Persona List Registry Tests
# =============================================================================


class TestPersonaRegistryListing:
    """Tests for listing registered personas."""

    def test_list_personas_returns_names(self):
        """list_personas should return all registered persona names."""
        from victor.framework.personas import list_personas

        names = list_personas()

        assert isinstance(names, list)
        assert "friendly_assistant" in names
        assert "senior_developer" in names
        assert "code_reviewer" in names
        assert "mentor" in names

    def test_list_personas_includes_custom(self):
        """list_personas should include custom registered personas."""
        from victor.framework.personas import (
            list_personas,
            register_persona,
            Persona,
        )

        custom = Persona(
            name="Listed Custom",
            background="Should appear in list.",
        )
        register_persona("listed_custom_test", custom)

        names = list_personas()

        assert "listed_custom_test" in names


# =============================================================================
# Persona to_dict and Serialization Tests
# =============================================================================


class TestPersonaSerialization:
    """Tests for Persona serialization."""

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all persona fields."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Serialized Agent",
            background="Background for serialization.",
            communication_style="formal",
            expertise_areas=("python", "testing"),
            quirks=("verbose", "thorough"),
        )

        data = persona.to_dict()

        assert data["name"] == "Serialized Agent"
        assert data["background"] == "Background for serialization."
        assert data["communication_style"] == "formal"
        assert data["expertise_areas"] == ("python", "testing")
        assert data["quirks"] == ("verbose", "thorough")

    def test_to_dict_minimal_persona(self):
        """to_dict should work with minimal persona."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Minimal",
            background="Minimal background.",
        )

        data = persona.to_dict()

        assert data["name"] == "Minimal"
        assert data["background"] == "Minimal background."
        assert data["communication_style"] == "professional"
        assert data["expertise_areas"] == ()
        assert data["quirks"] == ()


# =============================================================================
# Framework Export Tests
# =============================================================================


class TestPersonaExports:
    """Tests for Persona exports from framework."""

    def test_persona_exported_from_module(self):
        """Persona types should be exported from victor.framework.personas."""
        from victor.framework.personas import (
            Persona,
            get_persona,
            register_persona,
            list_personas,
            PERSONA_REGISTRY,
        )

        assert Persona is not None
        assert get_persona is not None
        assert register_persona is not None
        assert list_personas is not None
        assert PERSONA_REGISTRY is not None
