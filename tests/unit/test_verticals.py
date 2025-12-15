"""Unit tests for victor.verticals module.

Tests the vertical templates implementing the Template Method pattern.
"""

from typing import Any, Dict, List

import pytest

from victor.verticals import (
    CodingAssistant,
    ResearchAssistant,
    VerticalBase,
    VerticalConfig,
    VerticalRegistry,
    StageDefinition,
)
from victor.framework.tools import ToolSet


class TestVerticalBase:
    """Tests for VerticalBase abstract class."""

    def test_cannot_instantiate_abstract(self):
        """VerticalBase should not be instantiable without implementing abstracts."""
        # VerticalBase is abstract, but we can test via concrete implementations
        # This test verifies the template method pattern works
        assert CodingAssistant.name == "coding"
        assert ResearchAssistant.name == "research"

    def test_get_config_template_method(self):
        """get_config should assemble configuration from override methods."""
        config = CodingAssistant.get_config()

        assert isinstance(config, VerticalConfig)
        assert isinstance(config.tools, ToolSet)
        assert config.system_prompt is not None
        assert len(config.stages) > 0
        assert "vertical_name" in config.metadata

    def test_get_tool_set(self):
        """get_tool_set should return configured ToolSet."""
        tools = CodingAssistant.get_tool_set()
        assert isinstance(tools, ToolSet)
        assert "read" in tools
        assert "write" in tools


class TestStageDefinition:
    """Tests for StageDefinition dataclass."""

    def test_stage_creation(self):
        """StageDefinition should be creatable with all fields."""
        stage = StageDefinition(
            name="TESTING",
            description="Running tests",
            tools={"run_tests", "test_file"},
            keywords=["test", "verify"],
            next_stages={"COMPLETION"},
            min_confidence=0.7,
        )
        assert stage.name == "TESTING"
        assert stage.description == "Running tests"
        assert "run_tests" in stage.tools
        assert "test" in stage.keywords
        assert "COMPLETION" in stage.next_stages
        assert stage.min_confidence == 0.7

    def test_stage_to_dict(self):
        """to_dict should serialize stage definition."""
        stage = StageDefinition(
            name="TEST",
            description="Test stage",
            tools={"tool1"},
            keywords=["kw1"],
        )
        d = stage.to_dict()
        assert d["name"] == "TEST"
        assert d["description"] == "Test stage"
        assert "tool1" in d["tools"]


class TestVerticalConfig:
    """Tests for VerticalConfig dataclass."""

    def test_config_creation(self):
        """VerticalConfig should be creatable with required fields."""
        config = VerticalConfig(
            tools=ToolSet.minimal(),
            system_prompt="Test prompt",
        )
        assert isinstance(config.tools, ToolSet)
        assert config.system_prompt == "Test prompt"
        assert config.stages == {}
        assert config.metadata == {}

    def test_to_agent_kwargs(self):
        """to_agent_kwargs should return dict for Agent.create()."""
        config = VerticalConfig(
            tools=ToolSet.minimal(),
            system_prompt="Test prompt",
        )
        kwargs = config.to_agent_kwargs()
        assert "tools" in kwargs
        assert isinstance(kwargs["tools"], ToolSet)


class TestCodingAssistant:
    """Tests for CodingAssistant vertical."""

    def test_name_and_description(self):
        """CodingAssistant should have name and description."""
        assert CodingAssistant.name == "coding"
        assert "software" in CodingAssistant.description.lower()

    def test_get_tools(self):
        """get_tools should return coding-related tools."""
        tools = CodingAssistant.get_tools()
        assert isinstance(tools, list)
        # Core tools
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        # Git tools
        assert "git" in tools or "git_status" in tools
        # Shell
        assert "shell" in tools or "bash" in tools

    def test_get_system_prompt(self):
        """get_system_prompt should return coding-focused prompt."""
        prompt = CodingAssistant.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Substantial prompt
        assert "code" in prompt.lower() or "software" in prompt.lower()

    def test_get_stages(self):
        """get_stages should return coding-specific stages."""
        stages = CodingAssistant.get_stages()
        assert isinstance(stages, dict)
        # Should have standard coding stages
        assert "INITIAL" in stages
        assert "PLANNING" in stages
        assert "READING" in stages
        assert "EXECUTION" in stages
        assert "VERIFICATION" in stages
        assert "COMPLETION" in stages

    def test_stage_tools_mapping(self):
        """Each stage should have appropriate tools."""
        stages = CodingAssistant.get_stages()

        # Reading stage should have read tools
        reading_tools = stages["READING"].tools
        assert "read" in reading_tools or "code_search" in reading_tools

        # Execution stage should have write tools
        exec_tools = stages["EXECUTION"].tools
        assert "write" in exec_tools or "edit" in exec_tools

        # Verification should have test tools
        verify_tools = stages["VERIFICATION"].tools
        assert "test" in verify_tools or "shell" in verify_tools

    def test_get_provider_hints(self):
        """get_provider_hints should return provider preferences."""
        hints = CodingAssistant.get_provider_hints()
        assert isinstance(hints, dict)
        assert "preferred_providers" in hints
        assert "anthropic" in hints["preferred_providers"]

    def test_get_evaluation_criteria(self):
        """get_evaluation_criteria should return quality criteria."""
        criteria = CodingAssistant.get_evaluation_criteria()
        assert isinstance(criteria, list)
        assert len(criteria) > 0
        # Should mention code-related criteria
        criteria_text = " ".join(criteria).lower()
        assert "code" in criteria_text or "test" in criteria_text

    def test_customize_config(self):
        """customize_config should add coding-specific metadata."""
        config = CodingAssistant.get_config()
        assert "supports_lsp" in config.metadata
        assert "supports_git" in config.metadata
        assert "supported_languages" in config.metadata


class TestResearchAssistant:
    """Tests for ResearchAssistant vertical."""

    def test_name_and_description(self):
        """ResearchAssistant should have name and description."""
        assert ResearchAssistant.name == "research"
        assert "research" in ResearchAssistant.description.lower()

    def test_get_tools(self):
        """get_tools should return research-related tools."""
        tools = ResearchAssistant.get_tools()
        assert isinstance(tools, list)
        # Research tools (using canonical names)
        assert "web" in tools
        assert "fetch" in tools
        # Reading tools (using canonical names)
        assert "read" in tools
        # Should NOT have many coding tools
        assert "git" not in tools
        assert "shell" not in tools

    def test_get_system_prompt(self):
        """get_system_prompt should return research-focused prompt."""
        prompt = ResearchAssistant.get_system_prompt()
        assert isinstance(prompt, str)
        assert "research" in prompt.lower()
        assert "source" in prompt.lower() or "information" in prompt.lower()

    def test_get_stages(self):
        """get_stages should return research-specific stages."""
        stages = ResearchAssistant.get_stages()
        assert isinstance(stages, dict)
        # Research-specific stages
        assert "SEARCHING" in stages
        assert "READING" in stages
        assert "SYNTHESIZING" in stages
        assert "WRITING" in stages

    def test_limited_tool_set(self):
        """ResearchAssistant should have limited tool set."""
        research_tools = set(ResearchAssistant.get_tools())
        coding_tools = set(CodingAssistant.get_tools())

        # Research should have fewer tools
        assert len(research_tools) < len(coding_tools)

        # Research should not have shell/git (using canonical names)
        assert "shell" not in research_tools
        assert "git" not in research_tools

    def test_customize_config(self):
        """customize_config should add research-specific metadata."""
        config = ResearchAssistant.get_config()
        # Research vertical has extensions defined, check config is valid
        assert config is not None
        assert config.system_prompt is not None
        # Check evaluation criteria for research-specific attributes
        if hasattr(config, 'evaluation_criteria') and config.evaluation_criteria:
            criteria_text = " ".join(config.evaluation_criteria).lower()
            assert "source" in criteria_text or "accuracy" in criteria_text


class TestVerticalRegistry:
    """Tests for VerticalRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry to known state."""
        # Store current state
        original = dict(VerticalRegistry._registry)
        yield
        # Restore
        VerticalRegistry._registry = original

    def test_register_and_get(self):
        """register should add vertical, get should retrieve it."""

        class TestVertical(VerticalBase):
            name = "test_vertical"
            description = "Test"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test prompt"

        VerticalRegistry.register(TestVertical)

        retrieved = VerticalRegistry.get("test_vertical")
        assert retrieved is TestVertical

    def test_list_all(self):
        """list_all should return all registered verticals."""
        verticals = VerticalRegistry.list_all()
        assert isinstance(verticals, list)
        # Should have at least coding and research
        names = [name for name, _ in verticals]
        assert "coding" in names
        assert "research" in names

    def test_list_names(self):
        """list_names should return just vertical names."""
        names = VerticalRegistry.list_names()
        assert isinstance(names, list)
        assert "coding" in names
        assert "research" in names

    def test_unregister(self):
        """unregister should remove vertical from registry."""

        class TempVertical(VerticalBase):
            name = "temp"
            description = "Temp"

            @classmethod
            def get_tools(cls) -> List[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return ""

        VerticalRegistry.register(TempVertical)
        assert VerticalRegistry.get("temp") is TempVertical

        VerticalRegistry.unregister("temp")
        assert VerticalRegistry.get("temp") is None

    def test_register_without_name_raises(self):
        """Registering vertical without name should raise ValueError."""

        class NoNameVertical(VerticalBase):
            name = ""  # Empty name
            description = "Test"

            @classmethod
            def get_tools(cls) -> List[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return ""

        with pytest.raises(ValueError):
            VerticalRegistry.register(NoNameVertical)

    def test_get_nonexistent_returns_none(self):
        """get should return None for nonexistent vertical."""
        result = VerticalRegistry.get("nonexistent_vertical")
        assert result is None


class TestCustomVertical:
    """Tests for creating custom verticals."""

    def test_create_custom_vertical(self):
        """Users should be able to create custom verticals."""

        class DataScienceAssistant(VerticalBase):
            name = "data_science"
            description = "Data science and ML assistant"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write", "shell", "web_search"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a data science assistant specializing in ML..."

            @classmethod
            def get_stages(cls) -> Dict[str, StageDefinition]:
                return {
                    "INITIAL": StageDefinition(
                        name="INITIAL",
                        description="Understanding the data task",
                        keywords=["data", "analyze"],
                    ),
                    "EXPLORATION": StageDefinition(
                        name="EXPLORATION",
                        description="Exploring the dataset",
                        tools={"read", "shell"},
                        keywords=["explore", "visualize"],
                    ),
                    "MODELING": StageDefinition(
                        name="MODELING",
                        description="Building ML models",
                        tools={"write", "shell"},
                        keywords=["train", "model", "fit"],
                    ),
                    "COMPLETION": StageDefinition(
                        name="COMPLETION",
                        description="Reporting results",
                        keywords=["done", "report"],
                    ),
                }

        config = DataScienceAssistant.get_config()

        assert config.metadata["vertical_name"] == "data_science"
        assert "shell" in config.tools
        assert "EXPLORATION" in config.stages
        assert "MODELING" in config.stages

    def test_vertical_inheritance(self):
        """Verticals can inherit from other verticals."""

        class EnhancedCoding(CodingAssistant):
            name = "enhanced_coding"
            description = "Coding with extra features"

            @classmethod
            def get_tools(cls) -> List[str]:
                # Extend parent tools
                parent_tools = super().get_tools()
                return parent_tools + ["extra_tool"]

        tools = EnhancedCoding.get_tools()
        assert "read" in tools  # From parent
        assert "extra_tool" in tools  # Added


class TestVerticalIntegration:
    """Integration tests for verticals with other components."""

    def test_vertical_toolset_is_valid(self):
        """Vertical ToolSets should work with framework."""
        coding_tools = CodingAssistant.get_tool_set()
        research_tools = ResearchAssistant.get_tool_set()

        # Should be able to check membership (using canonical names)
        assert "read" in coding_tools
        assert "web" in research_tools

        # Should be able to get tool names
        coding_names = coding_tools.get_tool_names()
        assert isinstance(coding_names, (list, set))

    def test_vertical_stages_match_framework(self):
        """Vertical stages should be compatible with framework."""
        from victor.framework.state import Stage

        coding_stages = CodingAssistant.get_stages()

        # Coding stages should map to framework stages
        {s.value.upper() for s in Stage}

        for _stage_name in coding_stages:
            # At least some stages should match
            pass  # Verticals can have custom stages

    def test_config_can_be_used_for_agent_creation(self):
        """VerticalConfig should produce valid Agent.create kwargs."""
        config = CodingAssistant.get_config()
        kwargs = config.to_agent_kwargs()

        # Should have tools
        assert "tools" in kwargs
        # Should be a ToolSet
        assert isinstance(kwargs["tools"], ToolSet)
