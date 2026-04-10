"""TDD tests for mode config type promotion to victor-sdk."""


class TestModeConfigTypesInSDK:
    """Verify mode config data types are importable from SDK."""

    def test_import_mode_level(self):
        from victor_sdk.verticals.mode_config import ModeLevel

        assert ModeLevel.QUICK.value == "quick"
        assert ModeLevel.STANDARD.value == "standard"

    def test_import_mode_definition(self):
        from victor_sdk.verticals.mode_config import ModeDefinition

        mode = ModeDefinition(
            name="test", tool_budget=10, max_iterations=5
        )
        assert mode.name == "test"
        assert mode.tool_budget == 10

    def test_import_vertical_mode_config(self):
        from victor_sdk.verticals.mode_config import (
            ModeDefinition,
            VerticalModeConfig,
        )

        vmc = VerticalModeConfig(
            vertical_name="test",
            modes={"quick": ModeDefinition(name="quick", tool_budget=5, max_iterations=3)},
        )
        assert vmc.vertical_name == "test"
        assert "quick" in vmc.modes

    def test_import_mode_config(self):
        from victor_sdk.verticals.mode_config import ModeConfig

        config = ModeConfig(tool_budget=10, max_iterations=5)
        assert config.tool_budget == 10

    def test_backward_compat_core_import(self):
        """Core import path should still work."""
        from victor.core.mode_config import ModeDefinition, ModeLevel

        assert ModeLevel.QUICK.value == "quick"
        assert ModeDefinition is not None

    def test_registry_available_via_framework_extensions(self):
        """ModeConfigRegistry stays in core, accessible via framework.extensions."""
        from victor.framework.extensions import ModeConfigRegistry

        assert hasattr(ModeConfigRegistry, "get_instance")
