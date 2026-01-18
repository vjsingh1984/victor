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

"""Unit tests for lazy loading functionality."""

import os
import sys
from unittest import mock

import pytest

from victor.core.verticals.base import VerticalRegistry


class TestLazyLoading:
    """Test suite for lazy loading of verticals."""

    def test_lazy_import_registered(self):
        """Test that lazy imports are registered correctly."""
        # Clear registry
        VerticalRegistry.clear(reregister_builtins=False)

        # Register a lazy import
        VerticalRegistry.register_lazy_import("test_vertical", "victor.coding:CodingAssistant")

        # Verify it's in the lazy imports dict
        assert "test_vertical" in VerticalRegistry._lazy_imports
        assert VerticalRegistry._lazy_imports["test_vertical"] == "victor.coding:CodingAssistant"

        # Clean up
        VerticalRegistry.clear(reregister_builtins=True)

    def test_lazy_import_triggers_on_get(self):
        """Test that accessing a lazy import triggers the actual import."""
        # Clear registry
        VerticalRegistry.clear(reregister_builtins=False)

        # Register a lazy import
        VerticalRegistry.register_lazy_import("coding", "victor.coding:CodingAssistant")

        # Verify it's not yet in the regular registry
        assert "coding" not in VerticalRegistry._registry

        # Access it - this should trigger the import
        coding = VerticalRegistry.get("coding")

        # Verify it's now in the regular registry
        assert coding is not None
        assert "coding" in VerticalRegistry._registry
        assert coding.name == "coding"

        # Clean up
        VerticalRegistry.clear(reregister_builtins=True)

    def test_lazy_import_failure_handling(self):
        """Test that failed lazy imports are removed from the registry."""
        # Clear registry
        VerticalRegistry.clear(reregister_builtins=False)

        # Register an invalid lazy import
        VerticalRegistry.register_lazy_import("invalid", "nonexistent.module:InvalidClass")

        # Try to access it
        result = VerticalRegistry.get("invalid")

        # Should return None and remove the lazy import
        assert result is None
        assert "invalid" not in VerticalRegistry._lazy_imports

        # Clean up
        VerticalRegistry.clear(reregister_builtins=True)

    def test_list_names_includes_lazy_imports(self):
        """Test that list_names includes both registered and lazy imports."""
        # Clear registry
        VerticalRegistry.clear(reregister_builtins=False)

        # Register a real vertical
        from victor.coding import CodingAssistant

        VerticalRegistry.register(CodingAssistant)

        # Register a lazy import
        VerticalRegistry.register_lazy_import("research", "victor.research:ResearchAssistant")

        # List names should include both
        names = VerticalRegistry.list_names()
        assert "coding" in names
        assert "research" in names

        # Clean up
        VerticalRegistry.clear(reregister_builtins=True)

    def test_builtin_lazy_loading_enabled_by_default(self):
        """Test that built-in verticals use lazy loading by default."""
        # Clear and reset registry
        VerticalRegistry.clear(reregister_builtins=True)

        # With lazy loading enabled (default), built-ins should be in lazy imports
        # but not in the regular registry until accessed
        lazy_imports = VerticalRegistry._lazy_imports

        # Check that all built-ins are registered as lazy imports
        # Note: "dataanalysis" is normalized to "data_analysis"
        expected_verticals = {
            "coding",
            "research",
            "devops",
            "data_analysis",  # Normalized from "dataanalysis"
            "rag",
            "benchmark",
        }

        actual_verticals = set(lazy_imports.keys())

        # Check that expected verticals are present
        for vertical in expected_verticals:
            assert vertical in actual_verticals, f"{vertical} should be registered as lazy import"

    def test_can_access_lazy_loaded_verticals(self):
        """Test that lazy-loaded verticals work correctly after access."""
        # Clear and reset registry
        VerticalRegistry.clear(reregister_builtins=True)

        # Access coding vertical (triggers lazy load)
        coding = VerticalRegistry.get("coding")

        # Should be able to use it normally
        assert coding is not None
        assert coding.name == "coding"

        # Should be able to call methods
        config = coding.get_config()
        assert config is not None
        assert hasattr(config, "tools")

        tools = coding.get_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_eager_loading_via_env_var(self):
        """Test that VICTOR_LAZY_LOADING=false disables lazy loading."""
        # Set environment variable
        original_value = os.environ.get("VICTOR_LAZY_LOADING")
        os.environ["VICTOR_LAZY_LOADING"] = "false"

        try:
            # Clear and reset registry (will use eager loading)
            VerticalRegistry.clear(reregister_builtins=True)

            # With eager loading, built-ins should be in the regular registry
            # (because they're imported immediately)
            registry = VerticalRegistry._registry

            # At least coding should be loaded
            assert "coding" in registry or len(registry) > 0

        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop("VICTOR_LAZY_LOADING", None)
            else:
                os.environ["VICTOR_LAZY_LOADING"] = original_value

            # Clear registry again to reset state for subsequent tests
            # This ensures the next test starts with a clean slate
            VerticalRegistry.clear(reregister_builtins=True)

    def test_lazy_loading_performance_improvement(self):
        """Test that lazy loading improves startup performance."""
        import time
        import subprocess

        # Test eager loading by running in a subprocess
        eager_result = subprocess.run(
            [sys.executable, "-c", """
import os
os.environ['VICTOR_LAZY_LOADING'] = 'false'
import time
start = time.perf_counter()
from victor.core.verticals import VerticalRegistry
eager_time = time.perf_counter() - start
print(f"{eager_time:.6f}")
"""],
            capture_output=True,
            text=True
        )
        eager_time = float(eager_result.stdout.strip())

        # Test lazy loading by running in a subprocess
        lazy_result = subprocess.run(
            [sys.executable, "-c", """
import os
os.environ['VICTOR_LAZY_LOADING'] = 'true'
import time
start = time.perf_counter()
from victor.core.verticals import VerticalRegistry
lazy_time = time.perf_counter() - start
print(f"{lazy_time:.6f}")
"""],
            capture_output=True,
            text=True
        )
        lazy_time = float(lazy_result.stdout.strip())

        # Lazy loading should be faster
        # (allowing some tolerance for variance)
        assert lazy_time < eager_time, f"Lazy loading ({lazy_time*1000:.1f}ms) should be faster than eager loading ({eager_time*1000:.1f}ms)"

    def test_lazy_import_thread_safety(self):
        """Test that lazy loading is thread-safe."""
        import threading

        # Clear registry
        VerticalRegistry.clear(reregister_builtins=False)

        # Register a lazy import
        VerticalRegistry.register_lazy_import("coding", "victor.coding:CodingAssistant")

        results = []
        errors = []

        def access_vertical():
            try:
                vertical = VerticalRegistry.get("coding")
                results.append(vertical is not None)
            except Exception as e:
                errors.append(e)

        # Create multiple threads that access the same vertical simultaneously
        threads = [threading.Thread(target=access_vertical) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All threads should succeed
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(results), "Not all threads successfully accessed the vertical"

        # Clean up
        VerticalRegistry.clear(reregister_builtins=True)


class TestLazyLoadingIntegration:
    """Integration tests for lazy loading with real verticals."""

    def setup_method(self):
        """Ensure lazy loading is enabled for integration tests."""
        # Set environment variable to enable lazy loading
        os.environ["VICTOR_LAZY_LOADING"] = "true"

    def teardown_method(self):
        """Restore default environment."""
        os.environ["VICTOR_LAZY_LOADING"] = "true"

    def test_coding_vertical_lazy_load(self):
        """Test lazy loading of coding vertical."""
        VerticalRegistry.clear(reregister_builtins=True)

        # Access coding vertical
        coding = VerticalRegistry.get("coding")

        assert coding is not None
        assert coding.name == "coding"

        # Test methods work
        tools = coding.get_tools()
        assert isinstance(tools, list)
        assert "read" in tools or "write" in tools

    def test_research_vertical_lazy_load(self):
        """Test lazy loading of research vertical."""
        VerticalRegistry.clear(reregister_builtins=True)

        # Access research vertical
        research = VerticalRegistry.get("research")

        assert research is not None
        assert research.name == "research"

        # Test methods work
        tools = research.get_tools()
        assert isinstance(tools, list)

    def test_all_verticals_accessible(self):
        """Test that all built-in verticals are accessible via lazy loading."""
        VerticalRegistry.clear(reregister_builtins=True)

        # Note: "dataanalysis" is normalized to "data_analysis"
        expected_verticals = ["coding", "research", "devops", "data_analysis", "rag", "benchmark"]

        for vertical_name in expected_verticals:
            vertical = VerticalRegistry.get(vertical_name)
            assert vertical is not None, f"Could not load {vertical_name}"
            # Vertical name should match (or be normalized version)
            assert vertical.name == vertical_name or vertical_name in vertical.name

    def test_lazy_loaded_vertical_extensions(self):
        """Test that lazy-loaded verticals can provide extensions."""
        VerticalRegistry.clear(reregister_builtins=True)

        # Get coding vertical
        coding = VerticalRegistry.get("coding")

        # Get extensions
        extensions = coding.get_extensions()

        assert extensions is not None
        # Should have some extensions (middleware, safety, etc.)
        assert hasattr(extensions, "middleware")
        assert hasattr(extensions, "safety_extensions")
