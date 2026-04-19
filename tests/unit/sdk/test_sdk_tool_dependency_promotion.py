"""TDD tests for tool dependency promotion to victor-sdk.

These tests verify that base tool dependency implementations are
importable from victor_sdk, enabling external verticals to stop
importing from victor.core.
"""

from victor_sdk.verticals.protocols import (
    ToolDependency,
    ToolDependencyProviderProtocol,
)


class TestToolDependencyBaseInSDK:
    """Verify BaseToolDependencyProvider and ToolDependencyConfig are in SDK."""

    def test_import_base_provider(self):
        from victor_sdk.verticals.tool_dependencies import (
            BaseToolDependencyProvider,
        )

        assert hasattr(BaseToolDependencyProvider, "get_dependencies")
        assert hasattr(BaseToolDependencyProvider, "get_transition_weight")
        assert hasattr(BaseToolDependencyProvider, "suggest_next_tool")

    def test_import_config(self):
        from victor_sdk.verticals.tool_dependencies import (
            ToolDependencyConfig,
        )

        config = ToolDependencyConfig()
        assert config.dependencies == []
        assert config.transitions == {}
        assert config.clusters == {}
        assert config.sequences == {}
        assert config.required_tools == set()
        assert config.optional_tools == set()

    def test_import_factory(self):
        from victor_sdk.verticals.tool_dependencies import (
            create_vertical_tool_dependency_provider,
        )

        assert callable(create_vertical_tool_dependency_provider)

    def test_yaml_provider_stays_in_core(self):
        """YAMLToolDependencyProvider depends on core YAML utils, stays in core."""
        from victor.core.tool_dependency_loader import (
            YAMLToolDependencyProvider,
        )

        assert hasattr(YAMLToolDependencyProvider, "get_dependencies")

    def test_import_empty_provider(self):
        from victor_sdk.verticals.tool_dependencies import (
            EmptyToolDependencyProvider,
        )

        provider = EmptyToolDependencyProvider()
        assert provider.get_dependencies() == []

    def test_import_error_class(self):
        from victor_sdk.verticals.tool_dependencies import (
            ToolDependencyLoadError,
        )

        assert issubclass(ToolDependencyLoadError, Exception)

    def test_base_provider_implements_protocol(self):
        from victor_sdk.verticals.tool_dependencies import (
            BaseToolDependencyProvider,
            ToolDependencyConfig,
        )

        config = ToolDependencyConfig(
            dependencies=[
                ToolDependency(
                    tool_name="edit",
                    depends_on={"read"},
                    enables={"test"},
                )
            ],
            transitions={"read": [("edit", 0.8)]},
            required_tools={"read", "edit"},
        )
        provider = BaseToolDependencyProvider(config)
        assert isinstance(provider, ToolDependencyProviderProtocol)

        deps = provider.get_dependencies()
        assert len(deps) == 1
        assert deps[0].tool_name == "edit"

    def test_backward_compat_core_import(self):
        """victor.core.tool_dependency_base should re-export from SDK."""
        from victor.core.tool_dependency_base import (
            BaseToolDependencyProvider,
            ToolDependencyConfig,
        )

        assert BaseToolDependencyProvider is not None
        assert ToolDependencyConfig is not None


class TestToolDependencySDKExports:
    """Verify top-level SDK exports include tool dependency symbols."""

    def test_exports_in_verticals_init(self):
        from victor_sdk.verticals import tool_dependencies

        assert hasattr(tool_dependencies, "BaseToolDependencyProvider")
        assert hasattr(tool_dependencies, "ToolDependencyConfig")
        assert hasattr(tool_dependencies, "create_vertical_tool_dependency_provider")
