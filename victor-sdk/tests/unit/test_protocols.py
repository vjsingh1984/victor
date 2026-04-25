"""Unit tests for victor-sdk protocols."""

import pytest

from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.verticals.protocols import (
    ToolProvider,
    SafetyProvider,
    PromptProvider,
    WorkflowProvider,
)
from victor_sdk.core.types import (
    CapabilityRequirement,
    VerticalConfig,
    VerticalDefinition,
    StageDefinition,
    TieredToolConfig,
    ToolSet,
    Tier,
)
from victor_sdk.core.exceptions import (
    VerticalException,
    VerticalConfigurationError,
    VerticalProtocolError,
)


class TestVerticalBase:
    """Tests for VerticalBase abstract class."""

    def test_cannot_instantiate_base(self):
        """VerticalBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VerticalBase()

    def test_concrete_implementation(self):
        """A concrete implementation can be created."""

        class ConcreteVertical(VerticalBase):
            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test vertical"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a test assistant."

        assert ConcreteVertical.get_name() == "test"
        assert ConcreteVertical.get_description() == "Test vertical"
        assert ConcreteVertical.get_tools() == ["read", "write"]
        assert ConcreteVertical.get_system_prompt() == "You are a test assistant."

    def test_get_config_default(self):
        """get_config() returns default VerticalConfig."""

        class TestVertical(VerticalBase):
            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test vertical"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test prompt"

        config = TestVertical.get_config()
        assert isinstance(config, VerticalConfig)
        assert config.name == "test"
        assert config.description == "Test vertical"
        assert config.system_prompt == "Test prompt"

    def test_get_definition_default(self):
        """get_definition() returns a serializable VerticalDefinition."""

        class TestVertical(VerticalBase):
            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test vertical"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test prompt"

        definition = TestVertical.get_definition()
        assert isinstance(definition, VerticalDefinition)
        assert definition.name == "test"
        assert definition.tools == ["read"]
        assert definition.system_prompt == "Test prompt"

    def test_get_definition_wraps_invalid_definition_errors(self):
        """Invalid hook output should surface as VerticalConfigurationError."""

        class TestVertical(VerticalBase):
            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test vertical"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test prompt"

            @classmethod
            def get_workflow_spec(cls) -> dict[str, object]:
                return {"stage_order": ["missing"]}

        with pytest.raises(VerticalConfigurationError, match="initial_stage|stage_order"):
            TestVertical.get_definition()

    def test_get_stages_default(self):
        """get_stages() returns default 3-stage workflow."""

        class TestVertical(VerticalBase):
            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test"

        stages = TestVertical.get_stages()
        assert "planning" in stages
        assert "execution" in stages
        assert "verification" in stages

    def test_custom_stages(self):
        """Subclass can override get_stages()."""

        class TestVertical(VerticalBase):
            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test"

            @classmethod
            def get_stages(cls) -> dict[str, StageDefinition]:
                return {
                    "custom": StageDefinition(
                        name="custom",
                        description="Custom stage",
                    )
                }

        stages = TestVertical.get_stages()
        assert "custom" in stages
        assert len(stages) == 1


class TestCoreTypes:
    """Tests for core type definitions."""

    def test_stage_definition(self):
        """StageDefinition stores stage configuration."""
        stage = StageDefinition(
            name="test",
            description="Test stage",
            required_tools=["read"],
            optional_tools=["write"],
            allow_custom_tools=True,
            keywords=["search"],
            next_stages={"verify"},
            min_confidence=0.8,
        )

        assert stage.name == "test"
        assert stage.required_tools == ["read"]
        assert stage.optional_tools == ["write"]
        assert stage.allow_custom_tools is True
        assert stage.tools == {"read", "write"}
        assert stage.keywords == ["search"]
        assert stage.next_stages == {"verify"}
        assert stage.min_confidence == 0.8
        assert stage.to_dict()["tools"] == ["read", "write"]

    def test_stage_definition_accepts_legacy_tools_alias(self):
        """Legacy ``tools=`` construction should remain supported during migration."""
        stage = StageDefinition(
            name="test",
            tools={"write", "read"},
        )

        assert stage.description == ""
        assert stage.required_tools == []
        assert stage.optional_tools == ["read", "write"]
        assert stage.tools == {"read", "write"}

    def test_stage_definition_get_effective_tools(self):
        """get_effective_tools() returns correct tool list."""
        stage = StageDefinition(
            name="test",
            description="Test",
            required_tools=["read"],
            optional_tools=["write", "shell"],
        )

        # All tools available (note: result is sorted)
        assert set(stage.get_effective_tools(["read", "write", "shell"])) == {
            "read",
            "write",
            "shell",
        }

        # Only some tools available
        assert stage.get_effective_tools(["read"]) == ["read"]

        # Required tools always included
        assert set(stage.get_effective_tools(["write"])) == {"read", "write"}

    def test_tiered_tool_config(self):
        """TieredToolConfig stores tiered tool lists."""
        config = TieredToolConfig(
            basic_tools=["read"],
            standard_tools=["write"],
            advanced_tools=["shell"],
        )

        assert config.get_tools_for_tier(Tier.BASIC) == ["read"]
        assert config.get_tools_for_tier(Tier.STANDARD) == ["read", "write"]
        assert config.get_tools_for_tier(Tier.ADVANCED) == ["read", "write", "shell"]

    def test_tiered_tool_config_get_max_tier(self):
        """get_max_tier_for_tools() detects available tier."""
        config = TieredToolConfig(
            basic_tools=["read"],
            standard_tools=["write"],
            advanced_tools=["shell"],
        )

        assert config.get_max_tier_for_tools(["read"]) == Tier.BASIC
        assert config.get_max_tier_for_tools(["read", "write"]) == Tier.STANDARD
        assert config.get_max_tier_for_tools(["read", "write", "shell"]) == Tier.ADVANCED

    def test_tiered_tool_config_runtime_compatibility_fields(self):
        """TieredToolConfig should expose runtime-compatible tier aliases."""
        config = TieredToolConfig(
            mandatory={"read"},
            vertical_core={"rag_query"},
            semantic_pool={"rag_search"},
            stage_tools={"QUERYING": {"rag_query"}},
            readonly_only_for_analysis=False,
        )

        assert config.basic_tools == ["read"]
        assert config.standard_tools == ["rag_query"]
        assert config.advanced_tools == ["rag_search"]
        assert config.mandatory == {"read"}
        assert config.vertical_core == {"rag_query"}
        assert config.get_base_tools() == {"read", "rag_query"}
        assert config.get_tools_for_stage("QUERYING") == {"read", "rag_query"}
        assert config.get_effective_semantic_pool() == {"rag_search"}
        assert config.readonly_only_for_analysis is False

    def test_tool_set(self):
        """ToolSet stores tool collection."""
        toolset = ToolSet(
            names=["read", "write"],
            description="File tools",
            tier=Tier.STANDARD,
        )

        assert "read" in toolset
        assert "write" in toolset
        assert "shell" not in toolset
        assert len(toolset) == 2
        assert list(toolset) == ["read", "write"]

    def test_vertical_config(self):
        """VerticalConfig stores vertical configuration."""
        config = VerticalConfig(
            name="test",
            description="Test vertical",
            tools=["read", "write"],
            system_prompt="Test prompt",
            tier=Tier.STANDARD,
            metadata={"key": "value"},
        )

        assert config.name == "test"
        assert config.get_tool_names() == ["read", "write"]
        assert config.metadata["key"] == "value"

    def test_vertical_config_with_metadata(self):
        """with_metadata() returns new config with added metadata."""
        config = VerticalConfig(
            name="test",
            description="Test",
            tools=[],
            system_prompt="Test",
        )

        new_config = config.with_metadata(key="value")
        assert new_config.metadata["key"] == "value"
        assert config.metadata == {}  # Original unchanged

    def test_vertical_config_with_extension(self):
        """with_extension() returns new config with added extension."""
        config = VerticalConfig(
            name="test",
            description="Test",
            tools=[],
            system_prompt="Test",
        )

        new_config = config.with_extension("safety", {"rules": []})
        assert new_config.extensions["safety"] == {"rules": []}
        assert config.extensions == {}  # Original unchanged

    def test_capability_requirement_serialization(self):
        """CapabilityRequirement stores structured requirement metadata."""

        requirement = CapabilityRequirement(
            capability_id="file_ops",
            min_version="1.2",
            optional=True,
            purpose="workspace exploration",
            metadata={"scope": "local"},
        )

        assert requirement.as_legacy_string() == "file_ops"
        assert requirement.to_dict() == {
            "capability_id": "file_ops",
            "optional": True,
            "min_version": "1.2",
            "purpose": "workspace exploration",
            "metadata": {"scope": "local"},
        }


class TestExceptions:
    """Tests for exception classes."""

    def test_vertical_exception(self):
        """VerticalException formats message with context."""
        exc = VerticalException(
            message="Test error",
            vertical_name="test-vertical",
            details={"key": "value"},
        )

        assert "Test error" in str(exc)
        assert "test-vertical" in str(exc)
        assert "key=value" in str(exc)

    def test_vertical_configuration_error(self):
        """VerticalConfigurationError can be raised."""
        with pytest.raises(VerticalConfigurationError):
            raise VerticalConfigurationError("Invalid config")

    def test_vertical_protocol_error(self):
        """VerticalProtocolError can be raised."""
        with pytest.raises(VerticalProtocolError):
            raise VerticalProtocolError("Protocol violation")


class TestProtocols:
    """Tests for protocol definitions."""

    def test_tool_provider_protocol(self):
        """ToolProvider protocol can be implemented."""

        class MyToolProvider:
            def get_tools(self) -> list[str]:
                return ["read", "write"]

        provider = MyToolProvider()
        assert isinstance(provider, ToolProvider)
        assert provider.get_tools() == ["read", "write"]

    def test_safety_provider_protocol(self):
        """SafetyProvider protocol can be implemented."""

        class MySafetyProvider:
            def get_safety_rules(self) -> dict:
                return {"rules": []}

            def validate_tool_call(self, tool_name: str, arguments: dict) -> bool:
                return True

            def validate_prompt(self, prompt: str) -> bool:
                return True

        provider = MySafetyProvider()
        assert isinstance(provider, SafetyProvider)
        assert provider.validate_tool_call("read", {}) is True

    def test_prompt_provider_protocol(self):
        """PromptProvider protocol can be implemented."""

        class MyPromptProvider:
            def get_base_prompt(self) -> str:
                return "Base prompt"

            def get_prompt_template(self, task_type: str) -> str:
                return f"Template for {task_type}"

            def format_prompt(self, template: str, context: dict) -> str:
                return template.format(**context)

        provider = MyPromptProvider()
        assert isinstance(provider, PromptProvider)
        assert provider.get_base_prompt() == "Base prompt"

    def test_workflow_provider_protocol(self):
        """WorkflowProvider protocol can be implemented."""

        class MyWorkflowProvider:
            def get_workflow_spec(self) -> dict:
                return {"stages": []}

            def get_stage_definitions(self) -> dict:
                return {}

            def get_initial_stage(self) -> str:
                return "start"

        provider = MyWorkflowProvider()
        assert isinstance(provider, WorkflowProvider)
        assert provider.get_initial_stage() == "start"
