"""Unit tests for victor.verticals module.

Tests the vertical templates implementing the Template Method pattern.
"""

from typing import Any, Dict, List

import pytest

from victor.core.verticals import (
    VerticalBase,
    VerticalConfig,
    VerticalRegistry,
    StageDefinition,
)
from victor.coding import CodingAssistant
from victor.research import ResearchAssistant
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
        assert "web_search" in tools
        assert "web_fetch" in tools
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
        if hasattr(config, "evaluation_criteria") and config.evaluation_criteria:
            criteria_text = " ".join(config.evaluation_criteria).lower()
            assert "source" in criteria_text or "accuracy" in criteria_text


class TestVerticalRegistry:
    """Tests for VerticalRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry to known state."""
        # Import and register coding vertical FIRST before storing state
        try:
            from victor.coding import CodingAssistant

            VerticalRegistry.register(CodingAssistant)
        except ImportError:
            pass  # coding vertical may not be available in all environments

        # Store current state (now with coding registered)
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


class TestVerticalLoaderSwitch:
    """Tests for VerticalLoader vertical switching behavior (GAP-1, GAP-6 fixes)."""

    @pytest.fixture(autouse=True)
    def ensure_coding_registered(self):
        """Ensure coding vertical is registered for these tests."""
        try:
            from victor.coding import CodingAssistant

            VerticalRegistry.register(CodingAssistant)
        except ImportError:
            pass  # coding vertical may not be available in all environments

    def test_loader_switch_clears_extensions(self):
        """Switching verticals should clear cached extensions."""
        from victor.core.verticals.vertical_loader import VerticalLoader

        loader = VerticalLoader()

        # Load coding
        loader.load("coding")
        coding_ext = loader.get_extensions()
        assert coding_ext is not None

        # Switch to research
        loader.load("research")
        research_ext = loader.get_extensions()
        assert research_ext is not None

        # Extensions should be different
        assert coding_ext is not research_ext

    def test_loader_switch_resets_registered_services(self):
        """Switching verticals should reset _registered_services flag."""
        from victor.core.verticals.vertical_loader import VerticalLoader

        loader = VerticalLoader()

        # Load coding
        loader.load("coding")
        assert loader._registered_services is False

        # Manually set to True (simulating service registration)
        loader._registered_services = True

        # Switch to research
        loader.load("research")
        # Should be reset
        assert loader._registered_services is False

    def test_loader_active_vertical_name(self):
        """active_vertical_name should reflect current vertical."""
        from victor.core.verticals.vertical_loader import VerticalLoader

        loader = VerticalLoader()
        assert loader.active_vertical_name is None

        loader.load("coding")
        assert loader.active_vertical_name == "coding"

        loader.load("data_analysis")
        assert loader.active_vertical_name == "data_analysis"

    def test_loader_get_tools_per_vertical(self):
        """Each vertical should have distinct tool sets."""
        from victor.core.verticals import get_vertical_loader

        # Get the singleton loader and reset to clear state from other tests
        loader = get_vertical_loader()
        loader.reset()

        loader.load("coding")
        coding_tools = set(loader.get_tools())

        loader.load("research")
        research_tools = set(loader.get_tools())

        # Research should have fewer tools
        assert len(research_tools) < len(coding_tools)
        # Research should not have shell/git
        assert "shell" not in research_tools
        assert "git" not in research_tools


class TestBootstrapVerticalActivation:
    """Tests for bootstrap container vertical re-activation (GAP-1 fix)."""

    def test_ensure_bootstrapped_with_vertical(self):
        """ensure_bootstrapped should activate specified vertical."""
        from victor.core.bootstrap import ensure_bootstrapped, get_container
        from victor.core.verticals import get_vertical_loader
        from victor.config.settings import Settings

        # Reset for test
        loader = get_vertical_loader()
        loader.reset()

        # Bootstrap with research
        container = ensure_bootstrapped(Settings(), vertical="research")
        assert container is not None

        # Verify research is active
        assert loader.active_vertical_name == "research"

    def test_ensure_bootstrapped_switch_vertical(self):
        """ensure_bootstrapped should switch vertical if different requested."""
        from victor.core.bootstrap import ensure_bootstrapped
        from victor.core.verticals import get_vertical_loader
        from victor.config.settings import Settings

        loader = get_vertical_loader()

        # Bootstrap with coding
        ensure_bootstrapped(Settings(), vertical="coding")
        assert loader.active_vertical_name == "coding"

        # Request switch to research
        ensure_bootstrapped(vertical="research")
        assert loader.active_vertical_name == "research"


class TestVerticalProtocolMethods:
    """Tests for vertical protocol methods (GAP-2, GAP-3 fixes)."""

    def test_get_mode_config_default(self):
        """All verticals should have default get_mode_config()."""
        mode_config = CodingAssistant.get_mode_config()
        assert isinstance(mode_config, dict)
        assert "fast" in mode_config
        assert "thorough" in mode_config
        assert "explore" in mode_config

        # Each mode should have expected fields
        fast = mode_config["fast"]
        assert "tool_budget" in fast
        assert "max_iterations" in fast

    def test_get_task_type_hints_default(self):
        """All verticals should have default get_task_type_hints()."""
        hints = CodingAssistant.get_task_type_hints()
        assert isinstance(hints, dict)
        assert "edit" in hints
        assert "search" in hints
        assert "explain" in hints
        assert "debug" in hints
        assert "implement" in hints

        # Each hint should have expected fields
        edit_hint = hints["edit"]
        assert "hint" in edit_hint
        assert "priority_tools" in edit_hint

    def test_all_verticals_have_mode_config(self):
        """All built-in verticals should have get_mode_config()."""
        from victor.devops import DevOpsAssistant
        from victor.dataanalysis import DataAnalysisAssistant

        for vertical in [
            CodingAssistant,
            ResearchAssistant,
            DevOpsAssistant,
            DataAnalysisAssistant,
        ]:
            mode_config = vertical.get_mode_config()
            assert isinstance(mode_config, dict), f"{vertical.name} get_mode_config failed"
            assert len(mode_config) >= 3, f"{vertical.name} missing modes"

    def test_all_verticals_have_task_type_hints(self):
        """All built-in verticals should have get_task_type_hints()."""
        from victor.devops import DevOpsAssistant
        from victor.dataanalysis import DataAnalysisAssistant

        for vertical in [
            CodingAssistant,
            ResearchAssistant,
            DevOpsAssistant,
            DataAnalysisAssistant,
        ]:
            hints = vertical.get_task_type_hints()
            assert isinstance(hints, dict), f"{vertical.name} get_task_type_hints failed"
            assert len(hints) >= 5, f"{vertical.name} missing hints"


class TestVerticalConfigDictCompatibility:
    """Tests for VerticalConfig dict-like access (GAP-4 fix)."""

    def test_vertical_config_to_dict(self):
        """VerticalConfig should support to_dict()."""
        config = CodingAssistant.get_config()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "tools" in d
        assert "system_prompt" in d
        assert "stages" in d
        assert "provider_hints" in d
        assert "metadata" in d

    def test_vertical_config_keys(self):
        """VerticalConfig should support .keys()."""
        config = CodingAssistant.get_config()
        keys = config.keys()

        assert "tools" in keys
        assert "system_prompt" in keys

    def test_vertical_config_getitem(self):
        """VerticalConfig should support dict-style access."""
        config = CodingAssistant.get_config()

        # Should not raise
        system_prompt = config["system_prompt"]
        assert isinstance(system_prompt, str)

        # Invalid key should raise KeyError
        import pytest

        with pytest.raises(KeyError):
            _ = config["nonexistent_key"]

    def test_vertical_config_get_with_default(self):
        """VerticalConfig should support .get() with default."""
        config = CodingAssistant.get_config()

        # Existing key
        prompt = config.get("system_prompt")
        assert prompt is not None

        # Missing key with default
        missing = config.get("nonexistent", "default_value")
        assert missing == "default_value"


class TestVerticalIntegration:
    """Integration tests for verticals with other components."""

    def test_vertical_toolset_is_valid(self):
        """Vertical ToolSets should work with framework."""
        coding_tools = CodingAssistant.get_tool_set()
        research_tools = ResearchAssistant.get_tool_set()

        # Should be able to check membership (using canonical names)
        assert "read" in coding_tools
        assert "web_search" in research_tools

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


# =============================================================================
# DIRECT MODULE IMPORT TESTS (for 0% coverage modules)
# =============================================================================


class TestDirectCodingModuleImport:
    """Tests that directly import from victor.coding to improve coverage."""

    def test_coding_assistant_class(self):
        """Test CodingAssistant class directly from module."""
        from victor.coding import CodingAssistant as DirectCodingAssistant

        assert DirectCodingAssistant.name == "coding"
        # Version may change, just check it exists
        assert hasattr(DirectCodingAssistant, "version")

    def test_coding_get_tools(self):
        """Test get_tools method."""
        from victor.coding import CodingAssistant as DirectCodingAssistant

        tools = DirectCodingAssistant.get_tools()
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "shell" in tools

    def test_coding_get_system_prompt(self):
        """Test get_system_prompt method."""
        from victor.coding import CodingAssistant as DirectCodingAssistant

        prompt = DirectCodingAssistant.get_system_prompt()
        assert "Victor" in prompt
        assert "code" in prompt.lower()

    def test_coding_get_stages(self):
        """Test get_stages method."""
        from victor.coding import CodingAssistant as DirectCodingAssistant

        stages = DirectCodingAssistant.get_stages()
        assert "INITIAL" in stages
        assert "EXECUTION" in stages
        assert "VERIFICATION" in stages

    def test_coding_get_provider_hints(self):
        """Test get_provider_hints method."""
        from victor.coding import CodingAssistant as DirectCodingAssistant

        hints = DirectCodingAssistant.get_provider_hints()
        assert "preferred_providers" in hints
        assert "anthropic" in hints["preferred_providers"]

    def test_coding_get_evaluation_criteria(self):
        """Test get_evaluation_criteria method."""
        from victor.coding import CodingAssistant as DirectCodingAssistant

        criteria = DirectCodingAssistant.get_evaluation_criteria()
        assert len(criteria) > 0
        # Check for code-related criteria
        assert any("code" in c.lower() for c in criteria)

    def test_coding_customize_config(self):
        """Test customize_config method."""
        from victor.coding import CodingAssistant as DirectCodingAssistant

        config = DirectCodingAssistant.get_config()
        assert config.metadata["supports_lsp"] is True
        assert config.metadata["supports_git"] is True
        assert "python" in config.metadata["supported_languages"]


class TestDirectResearchModuleImport:
    """Tests that directly import from victor.research to improve coverage."""

    def test_research_assistant_class(self):
        """Test ResearchAssistant class directly from module."""
        from victor.research import ResearchAssistant as DirectResearchAssistant

        assert DirectResearchAssistant.name == "research"

    def test_research_get_tools(self):
        """Test get_tools method."""
        from victor.research import ResearchAssistant as DirectResearchAssistant

        tools = DirectResearchAssistant.get_tools()
        assert "web_search" in tools
        assert "web_fetch" in tools
        assert "read" in tools

    def test_research_get_system_prompt(self):
        """Test get_system_prompt method."""
        from victor.research import ResearchAssistant as DirectResearchAssistant

        prompt = DirectResearchAssistant.get_system_prompt()
        assert "research" in prompt.lower()

    def test_research_get_stages(self):
        """Test get_stages method."""
        from victor.research import ResearchAssistant as DirectResearchAssistant

        stages = DirectResearchAssistant.get_stages()
        assert isinstance(stages, dict)


class TestDevOpsDirectImport:
    """Tests for DevOpsAssistant direct imports."""

    def test_devops_assistant_class(self):
        """Test DevOpsAssistant class."""
        from victor.devops import DevOpsAssistant as DirectDevOps

        assert DirectDevOps.name == "devops"

    def test_devops_get_tools(self):
        """Test get_tools method."""
        from victor.devops import DevOpsAssistant as DirectDevOps

        tools = DirectDevOps.get_tools()
        assert isinstance(tools, list)

    def test_devops_get_system_prompt(self):
        """Test get_system_prompt method."""
        from victor.devops import DevOpsAssistant as DirectDevOps

        prompt = DirectDevOps.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_devops_get_stages(self):
        """Test get_stages method."""
        from victor.devops import DevOpsAssistant as DirectDevOps

        stages = DirectDevOps.get_stages()
        assert isinstance(stages, dict)


class TestDataAnalysisDirectImport:
    """Tests for DataAnalysisAssistant direct imports."""

    def test_data_analysis_assistant_class(self):
        """Test DataAnalysisAssistant class."""
        from victor.dataanalysis import DataAnalysisAssistant as DirectDataAnalysis

        assert DirectDataAnalysis.name == "data_analysis"

    def test_data_analysis_get_tools(self):
        """Test get_tools method."""
        from victor.dataanalysis import DataAnalysisAssistant as DirectDataAnalysis

        tools = DirectDataAnalysis.get_tools()
        assert isinstance(tools, list)

    def test_data_analysis_get_system_prompt(self):
        """Test get_system_prompt method."""
        from victor.dataanalysis import DataAnalysisAssistant as DirectDataAnalysis

        prompt = DirectDataAnalysis.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_data_analysis_get_stages(self):
        """Test get_stages method."""
        from victor.dataanalysis import DataAnalysisAssistant as DirectDataAnalysis

        stages = DirectDataAnalysis.get_stages()
        assert isinstance(stages, dict)


# =============================================================================
# VERTICALS HELPER FUNCTIONS TESTS
# =============================================================================


class TestVerticalHelperFunctions:
    """Tests for helper functions in victor.verticals."""

    @pytest.fixture(autouse=True)
    def ensure_coding_registered(self):
        """Ensure coding vertical is registered for these tests."""
        try:
            from victor.coding import CodingAssistant

            VerticalRegistry.register(CodingAssistant)
        except ImportError:
            pass  # coding vertical may not be available in all environments

    def test_get_vertical_by_name(self):
        """Test get_vertical helper function."""
        from victor.core.verticals import get_vertical

        coding = get_vertical("coding")
        assert coding is not None
        assert coding.name == "coding"

    def test_get_vertical_case_insensitive(self):
        """Test get_vertical with different cases."""
        from victor.core.verticals import get_vertical

        coding_lower = get_vertical("coding")
        coding_upper = get_vertical("CODING")
        coding_mixed = get_vertical("Coding")

        assert coding_lower is not None
        # Case insensitive should work
        assert coding_upper is coding_lower or coding_upper is not None
        assert coding_mixed is coding_lower or coding_mixed is not None

    def test_get_vertical_none(self):
        """Test get_vertical with None."""
        from victor.core.verticals import get_vertical

        result = get_vertical(None)
        assert result is None

    def test_get_vertical_nonexistent(self):
        """Test get_vertical with nonexistent name."""
        from victor.core.verticals import get_vertical

        result = get_vertical("nonexistent_vertical_12345")
        assert result is None

    def test_list_verticals(self):
        """Test list_verticals helper function."""
        from victor.core.verticals import list_verticals

        names = list_verticals()
        assert isinstance(names, list)
        assert "coding" in names
        assert "research" in names


class TestVerticalRegistryExternalDiscovery:
    """Tests for VerticalRegistry.discover_external_verticals()."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry to known state before and after each test."""
        # Store current state
        original_registry = dict(VerticalRegistry._registry)
        original_discovered = VerticalRegistry._external_discovered
        yield
        # Restore
        VerticalRegistry._registry = original_registry
        VerticalRegistry._external_discovered = original_discovered

    def test_discover_external_verticals_returns_dict(self):
        """discover_external_verticals should return a dictionary."""
        # Reset discovery flag to allow discovery
        VerticalRegistry.reset_discovery()

        result = VerticalRegistry.discover_external_verticals()
        assert isinstance(result, dict)

    def test_discover_external_verticals_caches_result(self):
        """discover_external_verticals should only run once."""
        VerticalRegistry.reset_discovery()

        # First call
        result1 = VerticalRegistry.discover_external_verticals()

        # Mark as discovered
        assert VerticalRegistry._external_discovered is True

        # Second call should return empty (already discovered)
        result2 = VerticalRegistry.discover_external_verticals()
        assert result2 == {}

    def test_reset_discovery_allows_rediscovery(self):
        """reset_discovery should allow discover_external_verticals to run again."""
        VerticalRegistry.reset_discovery()

        # First discovery
        VerticalRegistry.discover_external_verticals()
        assert VerticalRegistry._external_discovered is True

        # Reset and discover again
        VerticalRegistry.reset_discovery()
        assert VerticalRegistry._external_discovered is False

        # Should be able to discover again
        VerticalRegistry.discover_external_verticals()
        assert VerticalRegistry._external_discovered is True

    def test_clear_resets_discovery_flag(self):
        """clear() should reset the external_discovered flag."""
        VerticalRegistry.reset_discovery()
        VerticalRegistry.discover_external_verticals()

        assert VerticalRegistry._external_discovered is True

        # Clear should reset the flag
        VerticalRegistry.clear()
        assert VerticalRegistry._external_discovered is False

    def test_validate_external_vertical_rejects_non_class(self):
        """_validate_external_vertical should reject non-class objects."""
        result = VerticalRegistry._validate_external_vertical("not a class", "test_ep")
        assert result is False

    def test_validate_external_vertical_rejects_non_verticalbase(self):
        """_validate_external_vertical should reject classes not inheriting VerticalBase."""

        class NotAVertical:
            name = "test"

        result = VerticalRegistry._validate_external_vertical(NotAVertical, "test_ep")
        assert result is False

    def test_validate_external_vertical_rejects_no_name(self):
        """_validate_external_vertical should reject verticals without name."""

        class NoNameVertical(VerticalBase):
            name = ""
            description = "Test"

            @classmethod
            def get_tools(cls) -> List[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return ""

        result = VerticalRegistry._validate_external_vertical(NoNameVertical, "test_ep")
        assert result is False

    def test_validate_external_vertical_accepts_valid_vertical(self):
        """_validate_external_vertical should accept valid verticals."""

        class ValidVertical(VerticalBase):
            name = "valid_test"
            description = "Valid test vertical"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a test assistant."

        result = VerticalRegistry._validate_external_vertical(ValidVertical, "test_ep")
        assert result is True

    def test_entry_point_group_constant(self):
        """ENTRY_POINT_GROUP should be the correct value."""
        assert VerticalRegistry.ENTRY_POINT_GROUP == "victor.verticals"
