"""TDD tests for VerticalBase mixin extraction.

Verifies that methods extracted into mixins remain accessible on
VerticalBase (backward compat) and can be used independently.
"""

from __future__ import annotations

from typing import List

import pytest

from victor_sdk.verticals.protocols.base import VerticalBase


# A minimal concrete vertical for testing
class _TestVertical(VerticalBase):
    name = "test"
    description = "test vertical"

    @classmethod
    def get_name(cls) -> str:
        return "test"

    @classmethod
    def get_description(cls) -> str:
        return "test vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a test."


class TestVerticalBaseBackwardCompat:
    """All 50 methods must remain accessible on VerticalBase after mixin extraction."""

    # Core methods (always on VerticalBase)
    CORE_METHODS = [
        "get_name",
        "get_description",
        "get_tools",
        "get_system_prompt",
        "get_config",
        "get_definition",
        "get_stages",
        "get_version",
        "get_tier",
        "get_metadata",
        "get_extensions",
        "get_manifest",
        "get_skills",
        "get_capability_configs",
        "clear_config_cache",
        "clear_extension_cache",
    ]

    # RL methods (extracted to RLMixin)
    RL_METHODS = ["get_rl_config_provider", "get_rl_hooks"]

    # Team methods (extracted to TeamMixin)
    TEAM_METHODS = [
        "get_team_spec_provider",
        "get_team_specs",
        "get_team_declarations",
        "get_default_team",
        "get_team_metadata",
    ]

    # Workflow metadata methods (extracted to WorkflowMetadataMixin)
    WORKFLOW_METADATA_METHODS = [
        "get_initial_stage",
        "get_workflow_spec",
        "get_provider_hints",
        "get_evaluation_criteria",
        "get_workflow_metadata",
    ]

    # Prompt metadata methods (extracted to PromptMetadataMixin)
    PROMPT_METADATA_METHODS = [
        "get_prompt_templates",
        "get_task_type_hints",
        "get_prompt_metadata",
    ]

    # Extension provider methods (extracted to ExtensionProviderMixin)
    EXTENSION_PROVIDER_METHODS = [
        "get_middleware",
        "get_safety_extension",
        "get_prompt_contributor",
        "get_mode_config_provider",
        "get_tool_dependency_provider",
        "get_workflow_provider",
        "get_service_provider",
        "get_enrichment_strategy",
        "get_tool_selection_strategy",
        "get_tiered_tool_config",
        "get_capability_provider",
        "get_handlers",
        "get_tool_graph",
        "get_mode_config",
        "get_tool_requirements",
        "get_capability_requirements",
    ]

    ALL_METHODS = (
        CORE_METHODS
        + RL_METHODS
        + TEAM_METHODS
        + WORKFLOW_METADATA_METHODS
        + PROMPT_METADATA_METHODS
        + EXTENSION_PROVIDER_METHODS
    )

    @pytest.mark.parametrize("method_name", ALL_METHODS)
    def test_method_accessible_on_vertical_base(self, method_name: str):
        """Every method must be accessible on VerticalBase subclasses."""
        assert hasattr(_TestVertical, method_name), (
            f"VerticalBase is missing method '{method_name}' — "
            f"mixin extraction must not remove methods from the public API"
        )
        # Verify it's callable
        assert callable(getattr(_TestVertical, method_name))

    def test_isinstance_check(self):
        """isinstance(v, VerticalBase) must still work."""
        assert isinstance(_TestVertical(), VerticalBase)
        assert issubclass(_TestVertical, VerticalBase)


class TestMixinStandaloneUsage:
    """Mixins should be importable independently."""

    def test_rl_mixin_importable(self):
        """RLMixin should be importable from the mixins package."""
        from victor_sdk.verticals.mixins import RLMixin

        assert hasattr(RLMixin, "get_rl_config_provider")
        assert hasattr(RLMixin, "get_rl_hooks")

    def test_team_mixin_importable(self):
        """TeamMixin should be importable from the mixins package."""
        from victor_sdk.verticals.mixins import TeamMixin

        assert hasattr(TeamMixin, "get_team_spec_provider")
        assert hasattr(TeamMixin, "get_team_specs")
        assert hasattr(TeamMixin, "get_team_declarations")
        assert hasattr(TeamMixin, "get_default_team")
        assert hasattr(TeamMixin, "get_team_metadata")

    def test_workflow_metadata_mixin_importable(self):
        """WorkflowMetadataMixin should be importable from the mixins package."""
        from victor_sdk.verticals.mixins import WorkflowMetadataMixin

        assert hasattr(WorkflowMetadataMixin, "get_initial_stage")
        assert hasattr(WorkflowMetadataMixin, "get_workflow_spec")
        assert hasattr(WorkflowMetadataMixin, "get_workflow_metadata")

    def test_prompt_metadata_mixin_importable(self):
        """PromptMetadataMixin should be importable from the mixins package."""
        from victor_sdk.verticals.mixins import PromptMetadataMixin

        assert hasattr(PromptMetadataMixin, "get_prompt_templates")
        assert hasattr(PromptMetadataMixin, "get_task_type_hints")
        assert hasattr(PromptMetadataMixin, "get_prompt_metadata")

    def test_extension_provider_mixin_importable(self):
        """ExtensionProviderMixin should be importable from the mixins package."""
        from victor_sdk.verticals.mixins import ExtensionProviderMixin

        assert hasattr(ExtensionProviderMixin, "get_middleware")
        assert hasattr(ExtensionProviderMixin, "get_safety_extension")
        assert hasattr(ExtensionProviderMixin, "get_tiered_tool_config")


class TestMixinDefaultBehavior:
    """Mixin default implementations should return safe defaults."""

    def test_rl_defaults(self):
        """RL methods should return None/empty by default."""
        assert _TestVertical.get_rl_config_provider() is None
        assert _TestVertical.get_rl_hooks() == []

    def test_team_defaults(self):
        """Team methods should return None/empty by default."""
        assert _TestVertical.get_team_spec_provider() is None
        assert _TestVertical.get_team_specs() == {}
        assert _TestVertical.get_team_declarations() == {}
        assert _TestVertical.get_default_team() is None

    def test_workflow_metadata_defaults(self):
        """Workflow metadata methods should return safe defaults."""
        assert _TestVertical.get_workflow_spec() is not None
        assert _TestVertical.get_provider_hints() == {}
        assert _TestVertical.get_evaluation_criteria() == []

    def test_extension_provider_defaults(self):
        """Extension provider methods should return None/empty by default."""
        assert _TestVertical.get_middleware() == []
        assert _TestVertical.get_safety_extension() is None
        assert _TestVertical.get_prompt_contributor() is None
        assert _TestVertical.get_tool_dependency_provider() is None

    def test_prompt_metadata_defaults(self):
        """Prompt metadata methods should return empty by default."""
        assert _TestVertical.get_prompt_templates() == {}
        assert _TestVertical.get_task_type_hints() == {}


class TestMRONoConflicts:
    """Mixin composition should not cause MRO errors."""

    def test_vertical_base_mro_resolves(self):
        """VerticalBase MRO should resolve without errors."""
        # This would raise TypeError if MRO fails
        mro = _TestVertical.__mro__
        assert VerticalBase in mro

    def test_get_extensions_works(self):
        """get_extensions() should correctly resolve methods from mixins."""
        extensions = _TestVertical.get_extensions()
        # Should not raise, should return VerticalExtensions
        assert extensions is not None
        assert extensions.middleware == []
        assert extensions.safety_extensions == [] or extensions.safety_extensions is None

    def test_get_manifest_works(self):
        """get_manifest() should detect capabilities from mixins."""
        manifest = _TestVertical.get_manifest()
        assert manifest is not None
        assert manifest.name == "test"
