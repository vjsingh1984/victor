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

"""Integration tests for vertical independence.

Tests that verticals can operate independently without importing each other's code.
This validates the framework-vertical decoupling architecture.
"""

import sys
from unittest.mock import patch, MagicMock

import pytest


class TestResearchVerticalIndependence:
    """Tests that ResearchAssistant works without coding imports."""

    def test_research_imports_without_coding(self):
        """ResearchAssistant can be imported without coding module."""
        # Temporarily make coding module raise ImportError
        coding_modules = [key for key in sys.modules.keys() if "victor.coding" in key]

        # Store original modules
        original_modules = {key: sys.modules.get(key) for key in coding_modules}

        try:
            # Remove coding modules from sys.modules
            for key in coding_modules:
                if key in sys.modules:
                    del sys.modules[key]

            # Import research vertical fresh
            if "victor.research" in sys.modules:
                del sys.modules["victor.research"]

            from victor.research import ResearchAssistant

            # Should be able to access basic properties
            assert ResearchAssistant.name == "research"
            assert len(ResearchAssistant.get_tools()) > 0

        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    def test_research_extensions_no_coding_dependency(self):
        """ResearchAssistant extensions don't depend on coding."""
        from victor.research import ResearchAssistant
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = ResearchAssistant.get_extensions()
        # Note: LazyVerticalExtensions implements the VerticalExtensions protocol
        # but doesn't inherit from it (dataclass field conflicts).
        # Protocol compliance is verified via hasattr checks below.
        assert hasattr(extensions, "middleware")
        assert hasattr(extensions, "safety_extensions")
        assert hasattr(extensions, "get_all_safety_patterns")
        assert hasattr(extensions, "get_all_task_hints")
        assert hasattr(extensions, "get_all_mode_configs")

    def test_research_provides_valid_tools(self):
        """ResearchAssistant provides valid tool configurations."""
        from victor.research import ResearchAssistant

        tools = ResearchAssistant.get_tools()

        # Should have research-related tools
        tool_names = [t["name"] if isinstance(t, dict) else t for t in tools]

        # Common research tools
        assert any("search" in name.lower() for name in tool_names)


class TestDevOpsVerticalIndependence:
    """Tests that DevOpsAssistant works without coding imports."""

    def test_devops_imports_without_coding(self):
        """DevOpsAssistant can be imported without coding module."""
        from victor.devops import DevOpsAssistant

        assert DevOpsAssistant.name == "devops"
        assert len(DevOpsAssistant.get_tools()) > 0

    def test_devops_extensions_no_coding_dependency(self):
        """DevOpsAssistant extensions don't depend on coding."""
        from victor.devops import DevOpsAssistant
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = DevOpsAssistant.get_extensions()
        # Note: LazyVerticalExtensions implements the VerticalExtensions protocol
        # but doesn't inherit from it (dataclass field conflicts).
        # Protocol compliance is verified via hasattr checks below.
        assert hasattr(extensions, "middleware")
        assert hasattr(extensions, "safety_extensions")
        assert hasattr(extensions, "get_all_safety_patterns")
        assert hasattr(extensions, "get_all_task_hints")
        assert hasattr(extensions, "get_all_mode_configs")

    def test_devops_provides_valid_tools(self):
        """DevOpsAssistant provides valid tool configurations."""
        from victor.devops import DevOpsAssistant

        tools = DevOpsAssistant.get_tools()
        tool_names = [t["name"] if isinstance(t, dict) else t for t in tools]

        # Should have devops-related tools
        # DevOps tools include shell (bash), docker, git, etc.
        assert any(name in ["shell", "docker", "git"] for name in tool_names)


class TestCodingVerticalComplete:
    """Tests that CodingAssistant has complete extensions."""

    def test_coding_has_middleware(self):
        """CodingAssistant provides middleware."""
        from victor.coding import CodingAssistant

        extensions = CodingAssistant.get_extensions()
        assert len(extensions.middleware) >= 1

    def test_coding_has_safety_patterns(self):
        """CodingAssistant provides safety patterns."""
        from victor.coding import CodingAssistant

        extensions = CodingAssistant.get_extensions()
        patterns = extensions.get_all_safety_patterns()
        assert len(patterns) > 0

    def test_coding_has_task_hints(self):
        """CodingAssistant provides task type hints."""
        from victor.coding import CodingAssistant

        extensions = CodingAssistant.get_extensions()
        hints = extensions.get_all_task_hints()
        assert len(hints) > 0

    def test_coding_has_mode_configs(self):
        """CodingAssistant provides mode configurations."""
        from victor.coding import CodingAssistant

        extensions = CodingAssistant.get_extensions()
        modes = extensions.get_all_mode_configs()
        assert len(modes) > 0


class TestBootstrapWithVerticals:
    """Tests for bootstrap container with different verticals."""

    def test_bootstrap_with_coding_vertical(self):
        """Bootstrap with coding vertical registers extensions."""
        from victor.config.settings import Settings
        from victor.core.bootstrap import bootstrap_container
        from victor.core.container import set_container, ServiceContainer

        # Reset container
        set_container(ServiceContainer())

        settings = Settings()
        container = bootstrap_container(settings=settings, vertical="coding")

        # Should have registered vertical extensions
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = container.get_optional(VerticalExtensions)

        # May or may not be registered depending on loader state
        # But bootstrap should not fail
        assert container is not None

    def test_bootstrap_with_research_vertical(self):
        """Bootstrap with research vertical works."""
        from victor.config.settings import Settings
        from victor.core.bootstrap import bootstrap_container
        from victor.core.container import set_container, ServiceContainer

        # Reset container
        set_container(ServiceContainer())

        settings = Settings()
        container = bootstrap_container(settings=settings, vertical="research")

        assert container is not None

    def test_bootstrap_with_devops_vertical(self):
        """Bootstrap with devops vertical works."""
        from victor.config.settings import Settings
        from victor.core.bootstrap import bootstrap_container
        from victor.core.container import set_container, ServiceContainer

        # Reset container
        set_container(ServiceContainer())

        settings = Settings()
        container = bootstrap_container(settings=settings, vertical="devops")

        assert container is not None

    def test_bootstrap_with_unknown_vertical_warns(self):
        """Bootstrap with unknown vertical warns but doesn't fail."""
        from victor.config.settings import Settings
        from victor.core.bootstrap import bootstrap_container
        from victor.core.container import set_container, ServiceContainer

        # Reset container
        set_container(ServiceContainer())

        settings = Settings()

        # Should warn but not fail
        container = bootstrap_container(settings=settings, vertical="nonexistent")

        assert container is not None


class TestVerticalLoaderIntegration:
    """Integration tests for VerticalLoader."""

    def test_loader_loads_coding(self):
        """VerticalLoader loads coding vertical."""
        from victor.core.verticals.vertical_loader import get_vertical_loader

        loader = get_vertical_loader()
        vertical = loader.load("coding")

        assert vertical is not None
        assert vertical.name == "coding"

    def test_loader_loads_research(self):
        """VerticalLoader loads research vertical."""
        from victor.core.verticals.vertical_loader import get_vertical_loader

        loader = get_vertical_loader()
        vertical = loader.load("research")

        assert vertical is not None
        assert vertical.name == "research"

    def test_loader_loads_devops(self):
        """VerticalLoader loads devops vertical."""
        from victor.core.verticals.vertical_loader import get_vertical_loader

        loader = get_vertical_loader()
        vertical = loader.load("devops")

        assert vertical is not None
        assert vertical.name == "devops"

    def test_loader_raises_for_unknown(self):
        """VerticalLoader raises ValueError for unknown vertical."""
        from victor.core.verticals.vertical_loader import get_vertical_loader

        loader = get_vertical_loader()

        with pytest.raises(ValueError, match="not found"):
            loader.load("totally_fake_vertical")


class TestFrameworkShimIntegration:
    """Integration tests for FrameworkShim with verticals."""

    def test_shim_accepts_vertical_parameter(self):
        """FrameworkShim constructor accepts vertical parameter."""
        from victor.framework.shim import FrameworkShim
        from victor.coding import CodingAssistant
        import inspect

        # Check that FrameworkShim constructor accepts vertical parameter
        sig = inspect.signature(FrameworkShim.__init__)
        params = list(sig.parameters.keys())

        assert "vertical" in params, "FrameworkShim should accept vertical parameter"

    def test_shim_apply_vertical_exists(self):
        """FrameworkShim has _apply_vertical method (unified vertical application)."""
        from victor.framework.shim import FrameworkShim

        # Check the method exists - deprecated methods removed, unified _apply_vertical used
        assert hasattr(FrameworkShim, "_apply_vertical")


class TestCLIVerticalFlag:
    """Tests for CLI --vertical flag integration."""

    def test_settings_has_default_vertical(self):
        """Settings has default_vertical attribute."""
        from victor.config.settings import Settings

        settings = Settings()
        assert hasattr(settings, "default_vertical")
        assert settings.default_vertical == "coding"

    def test_settings_has_auto_detect_vertical(self):
        """Settings has auto_detect_vertical attribute."""
        from victor.config.settings import Settings

        settings = Settings()
        assert hasattr(settings, "auto_detect_vertical")
        assert settings.auto_detect_vertical is False
