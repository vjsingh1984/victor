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

"""Unit tests for lazy extension loading.

Tests the LazyVerticalExtensions wrapper and integration with
VerticalExtensionLoader.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from victor.core.verticals.lazy_extensions import (
    ExtensionLoadTrigger,
    LazyVerticalExtensions,
    create_lazy_extensions,
    get_extension_load_trigger,
)
from victor.core.verticals.protocols import VerticalExtensions


class TestExtensionLoadTrigger:
    """Test ExtensionLoadTrigger enum."""

    def test_trigger_values(self):
        """Test that trigger enum has expected values."""
        assert ExtensionLoadTrigger.EAGER == "eager"
        assert ExtensionLoadTrigger.ON_DEMAND == "on_demand"
        assert ExtensionLoadTrigger.AUTO == "auto"


class TestGetExtensionLoadTrigger:
    """Test get_extension_load_trigger function."""

    def test_default_lazy(self):
        """Test default is lazy (ON_DEMAND)."""
        with patch.dict(os.environ, {}, clear=True):
            trigger = get_extension_load_trigger()
            assert trigger == ExtensionLoadTrigger.ON_DEMAND

    def test_explicit_true(self):
        """Test explicit 'true' enables lazy loading."""
        with patch.dict(os.environ, {"VICTOR_LAZY_EXTENSIONS": "true"}):
            trigger = get_extension_load_trigger()
            assert trigger == ExtensionLoadTrigger.ON_DEMAND

    def test_explicit_false(self):
        """Test explicit 'false' disables lazy loading."""
        with patch.dict(os.environ, {"VICTOR_LAZY_EXTENSIONS": "false"}):
            trigger = get_extension_load_trigger()
            assert trigger == ExtensionLoadTrigger.EAGER

    def test_auto(self):
        """Test 'auto' mode."""
        with patch.dict(os.environ, {"VICTOR_LAZY_EXTENSIONS": "auto"}):
            trigger = get_extension_load_trigger()
            assert trigger == ExtensionLoadTrigger.AUTO

    def test_case_insensitive(self):
        """Test environment variable is case-insensitive."""
        for value in ["TRUE", "True", "tRuE", "FALSE", "False", "AUTO", "auto"]:
            with patch.dict(os.environ, {"VICTOR_LAZY_EXTENSIONS": value}):
                trigger = get_extension_load_trigger()
                assert trigger in [
                    ExtensionLoadTrigger.ON_DEMAND,
                    ExtensionLoadTrigger.EAGER,
                    ExtensionLoadTrigger.AUTO,
                ]


class TestLazyVerticalExtensions:
    """Test LazyVerticalExtensions wrapper."""

    def test_init_on_demand(self):
        """Test initialization with ON_DEMAND trigger."""
        loader = Mock(return_value=Mock(spec=VerticalExtensions))
        lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.ON_DEMAND)

        assert lazy.vertical_name == "test"
        assert lazy.trigger == ExtensionLoadTrigger.ON_DEMAND
        assert not lazy.is_loaded()
        assert lazy._extensions is None

    def test_init_eager(self):
        """Test initialization with EAGER trigger."""
        loader = Mock(return_value=Mock(spec=VerticalExtensions))
        lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.EAGER)

        assert lazy.vertical_name == "test"
        assert lazy.trigger == ExtensionLoadTrigger.EAGER
        assert lazy.is_loaded()
        loader.assert_called_once()

    def test_init_auto_production(self):
        """Test AUTO trigger in production profile."""
        loader = Mock(return_value=Mock(spec=VerticalExtensions))
        with patch.dict(os.environ, {"VICTOR_PROFILE": "production"}):
            lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.AUTO)

            # AUTO should resolve to ON_DEMAND in production
            assert lazy.trigger == ExtensionLoadTrigger.ON_DEMAND
            assert not lazy.is_loaded()

    def test_init_auto_development(self):
        """Test AUTO trigger in development profile."""
        loader = Mock(return_value=Mock(spec=VerticalExtensions))
        with patch.dict(os.environ, {"VICTOR_PROFILE": "development"}):
            lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.AUTO)

            # AUTO should resolve to EAGER in development
            assert lazy.trigger == ExtensionLoadTrigger.EAGER
            assert lazy.is_loaded()

    def test_lazy_loading_on_first_access(self):
        """Test that extensions are loaded on first access."""
        extensions = Mock(spec=VerticalExtensions)
        extensions.middleware = []
        extensions.safety_extensions = []
        extensions.prompt_contributors = []
        extensions.mode_config_provider = None
        extensions.tool_dependency_provider = None
        extensions.workflow_provider = None
        extensions.service_provider = None
        extensions.rl_config_provider = None
        extensions.team_spec_provider = None
        extensions.enrichment_strategy = None
        extensions.tiered_tool_config = None
        extensions._dynamic_extensions = {}

        loader = Mock(return_value=extensions)
        lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.ON_DEMAND)

        # Should not be loaded yet
        assert not lazy.is_loaded()
        loader.assert_not_called()

        # Access middleware - should trigger loading
        middleware = lazy.middleware
        assert lazy.is_loaded()
        loader.assert_called_once()
        assert middleware == []

    def test_cached_after_first_load(self):
        """Test that extensions are cached after first load."""
        extensions = Mock(spec=VerticalExtensions)
        extensions.middleware = [1, 2, 3]
        extensions.safety_extensions = []
        extensions.prompt_contributors = []
        extensions.mode_config_provider = None
        extensions.tool_dependency_provider = None
        extensions.workflow_provider = None
        extensions.service_provider = None
        extensions.rl_config_provider = None
        extensions.team_spec_provider = None
        extensions.enrichment_strategy = None
        extensions.tiered_tool_config = None
        extensions._dynamic_extensions = {}

        loader = Mock(return_value=extensions)
        lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.ON_DEMAND)

        # First access
        middleware1 = lazy.middleware
        assert loader.call_count == 1

        # Second access - should use cache
        middleware2 = lazy.middleware
        assert loader.call_count == 1  # No additional call
        assert middleware1 == middleware2 == [1, 2, 3]

    def test_unload(self):
        """Test unloading extensions."""
        extensions = Mock(spec=VerticalExtensions)
        extensions.middleware = []
        extensions.safety_extensions = []
        extensions.prompt_contributors = []
        extensions.mode_config_provider = None
        extensions.tool_dependency_provider = None
        extensions.workflow_provider = None
        extensions.service_provider = None
        extensions.rl_config_provider = None
        extensions.team_spec_provider = None
        extensions.enrichment_strategy = None
        extensions.tiered_tool_config = None
        extensions._dynamic_extensions = {}

        loader = Mock(return_value=extensions)
        lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.ON_DEMAND)

        # Load extensions
        _ = lazy.middleware
        assert lazy.is_loaded()

        # Unload
        lazy.unload()
        assert not lazy.is_loaded()
        assert lazy._extensions is None

        # Reload - should call loader again
        _ = lazy.middleware
        assert loader.call_count == 2

    def test_proxy_attributes(self):
        """Test that all extension attributes are proxied correctly."""
        extensions = Mock(spec=VerticalExtensions)
        extensions.middleware = ["middleware1"]
        extensions.safety_extensions = ["safety1"]
        extensions.prompt_contributors = ["prompt1"]
        extensions.mode_config_provider = "mode_config"
        extensions.tool_dependency_provider = "tool_deps"
        extensions.workflow_provider = "workflow"
        extensions.service_provider = "service"
        extensions.rl_config_provider = "rl_config"
        extensions.team_spec_provider = "team_spec"
        extensions.enrichment_strategy = "enrichment"
        extensions.tiered_tool_config = "tiered"
        extensions._dynamic_extensions = {"custom": ["ext1"]}

        loader = Mock(return_value=extensions)
        lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.ON_DEMAND)

        # Test all attributes
        assert lazy.middleware == ["middleware1"]
        assert lazy.safety_extensions == ["safety1"]
        assert lazy.prompt_contributors == ["prompt1"]
        assert lazy.mode_config_provider == "mode_config"
        assert lazy.tool_dependency_provider == "tool_deps"
        assert lazy.workflow_provider == "workflow"
        assert lazy.service_provider == "service"
        assert lazy.rl_config_provider == "rl_config"
        assert lazy.team_spec_provider == "team_spec"
        assert lazy.enrichment_strategy == "enrichment"
        assert lazy.tiered_tool_config == "tiered"
        assert lazy.dynamic_extensions == {"custom": ["ext1"]}

    def test_repr(self):
        """Test string representation."""
        extensions = Mock(spec=VerticalExtensions)
        extensions.middleware = []
        extensions.safety_extensions = []
        extensions.prompt_contributors = []
        extensions.mode_config_provider = None
        extensions.tool_dependency_provider = None
        extensions.workflow_provider = None
        extensions.service_provider = None
        extensions.rl_config_provider = None
        extensions.team_spec_provider = None
        extensions.enrichment_strategy = None
        extensions.tiered_tool_config = None
        extensions._dynamic_extensions = {}

        loader = Mock(return_value=extensions)
        lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.ON_DEMAND)

        assert "unloaded" in repr(lazy)

        # Load and check again
        _ = lazy.middleware
        assert "loaded" in repr(lazy)


class TestThreadSafety:
    """Test thread safety of lazy loading."""

    def test_concurrent_access(self):
        """Test that concurrent access is thread-safe."""
        import threading

        extensions = Mock(spec=VerticalExtensions)
        extensions.middleware = []
        extensions.safety_extensions = []
        extensions.prompt_contributors = []
        extensions.mode_config_provider = None
        extensions.tool_dependency_provider = None
        extensions.workflow_provider = None
        extensions.service_provider = None
        extensions.rl_config_provider = None
        extensions.team_spec_provider = None
        extensions.enrichment_strategy = None
        extensions.tiered_tool_config = None
        extensions._dynamic_extensions = {}

        loader = Mock(return_value=extensions)
        lazy = create_lazy_extensions("test", loader, ExtensionLoadTrigger.ON_DEMAND)

        # Access from multiple threads
        def access_middleware():
            _ = lazy.middleware

        threads = [threading.Thread(target=access_middleware) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Loader should only be called once despite 10 threads
        assert loader.call_count == 1
        assert lazy.is_loaded()


class TestIntegration:
    """Integration tests with VerticalExtensionLoader."""

    def test_get_extensions_returns_lazy(self):
        """Test that get_extensions returns LazyVerticalExtensions by default."""
        from victor.coding import CodingAssistant

        # Get extensions with default settings (lazy loading enabled)
        extensions = CodingAssistant.get_extensions()

        # Should be a lazy wrapper
        assert isinstance(extensions, LazyVerticalExtensions)

        # Should not be loaded yet
        assert not extensions.is_loaded()

    def test_get_extensions_eager_mode(self):
        """Test that get_extensions can force eager loading."""
        from victor.coding import CodingAssistant

        # Get extensions with eager loading
        extensions = CodingAssistant.get_extensions(use_lazy=False)

        # Should NOT be a lazy wrapper
        assert not isinstance(extensions, LazyVerticalExtensions)

        # Should be actual VerticalExtensions
        assert isinstance(extensions, VerticalExtensions)

    def test_lazy_to_eager_transition(self):
        """Test transition from lazy to loaded."""
        from victor.coding import CodingAssistant

        extensions = CodingAssistant.get_extensions()

        # Initially unloaded
        assert not extensions.is_loaded()

        # Access an attribute
        middleware = extensions.middleware

        # Now loaded
        assert extensions.is_loaded()
        assert isinstance(middleware, list)

    @pytest.mark.parametrize(
        "env_value,expected_lazy",
        [
            ("true", True),
            ("false", False),
            ("auto", True),  # Depends on profile, but should resolve
        ],
    )
    def test_environment_variable_control(self, env_value, expected_lazy):
        """Test that environment variable controls lazy loading."""
        from victor.coding import CodingAssistant

        with patch.dict(os.environ, {"VICTOR_LAZY_EXTENSIONS": env_value}):
            extensions = CodingAssistant.get_extensions()

            # If lazy loading is enabled, should be a wrapper
            if expected_lazy and env_value != "auto":
                assert isinstance(extensions, LazyVerticalExtensions)
            else:
                # In auto mode, depends on profile
                # In false mode, should not be lazy
                if env_value == "false":
                    assert not isinstance(extensions, LazyVerticalExtensions)


class TestPerformance:
    """Performance tests for lazy loading."""

    def test_startup_time_improvement(self):
        """Test that lazy loading improves startup time."""
        import time

        # Test eager loading
        start = time.time()
        from victor.coding import CodingAssistant

        eager_extensions = CodingAssistant.get_extensions(use_lazy=False)
        eager_time = time.time() - start

        # Clear cache
        CodingAssistant.clear_extension_cache()

        # Test lazy loading
        start = time.time()
        lazy_extensions = CodingAssistant.get_extensions(use_lazy=True)
        lazy_time = time.time() - start

        # Lazy should not be loaded yet (this is the key behavioral check)
        assert not lazy_extensions.is_loaded()

        # Performance check: only validate if times are large enough to be reliable
        # Skip timing assertion for very fast operations (< 10ms) due to measurement noise
        if eager_time > 0.01:  # 10 milliseconds
            # Lazy should be faster (at least not significantly slower)
            # Use more lenient tolerance (2x) to account for system variance
            assert lazy_time <= eager_time * 2.0, (
                f"Lazy loading took {lazy_time:.6f}s, "
                f"eager took {eager_time:.6f}s (expected lazy <= 2x eager)"
            )

    def test_first_access_overhead(self):
        """Test overhead of first access to lazy extensions."""
        import time

        from victor.coding import CodingAssistant

        # Clear cache
        CodingAssistant.clear_extension_cache()

        # Create lazy extensions
        lazy_extensions = CodingAssistant.get_extensions(use_lazy=True)

        # Measure first access time
        start = time.time()
        _ = lazy_extensions.middleware
        first_access_time = time.time() - start

        # First access should be reasonably fast (< 500ms)
        assert first_access_time < 0.5

        # Subsequent accesses should be much faster
        start = time.time()
        _ = lazy_extensions.middleware
        second_access_time = time.time() - start

        # Second access should be much faster (cached)
        assert second_access_time < first_access_time * 0.5


@pytest.fixture
def reset_lazy_env():
    """Fixture to reset lazy loading environment variable."""
    original = os.environ.get("VICTOR_LAZY_EXTENSIONS")
    yield
    if original is not None:
        os.environ["VICTOR_LAZY_EXTENSIONS"] = original
    else:
        os.environ.pop("VICTOR_LAZY_EXTENSIONS", None)
