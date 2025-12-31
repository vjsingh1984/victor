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

"""Tests for VerticalBase class - LSP compliance and core functionality."""

import pytest
from typing import List
from unittest.mock import patch, MagicMock

from victor.core.verticals.base import VerticalBase, VerticalConfig


class ConcreteVertical(VerticalBase):
    """Concrete implementation of VerticalBase for testing."""

    name = "test_vertical"
    description = "A test vertical for unit testing"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "grep"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a test assistant."


class TestGetExtensionsLSPCompliance:
    """Test suite for get_extensions() LSP compliance.

    LSP (Liskov Substitution Principle) requires that:
    1. The return type must be consistent (never None if declared as VerticalExtensions)
    2. Subclasses must be substitutable for the base class
    3. All code paths must return valid objects
    """

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_get_extensions_never_returns_none(self):
        """Verify that get_extensions() NEVER returns None.

        This is the primary LSP compliance test. The method must return
        a valid VerticalExtensions object in all scenarios.
        """
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = ConcreteVertical.get_extensions(use_cache=False)

        # Primary assertion: must not be None
        assert extensions is not None, "get_extensions() must never return None"

        # Type assertion: must be VerticalExtensions
        assert isinstance(
            extensions, VerticalExtensions
        ), f"Expected VerticalExtensions, got {type(extensions)}"

    def test_get_extensions_returns_valid_object_on_exception(self):
        """Verify get_extensions() returns valid VerticalExtensions even on exceptions.

        This tests the exception handler path to ensure LSP compliance
        when extension getter methods fail.
        """
        from victor.core.verticals.protocols import VerticalExtensions

        class FailingVertical(VerticalBase):
            """Vertical whose extension getter raises an exception."""

            name = "failing_vertical"
            description = "A vertical that fails during extension loading"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Failing prompt"

            @classmethod
            def get_safety_extension(cls):
                # Simulate an error during extension loading
                raise RuntimeError("Simulated extension loading failure")

        # Force cache bypass to trigger fresh extension loading
        FailingVertical.clear_config_cache(clear_all=True)

        extensions = FailingVertical.get_extensions(use_cache=False)

        # Must still return a valid object, not None (LSP compliance)
        assert extensions is not None, "get_extensions() returned None on exception - LSP violation"
        assert isinstance(
            extensions, VerticalExtensions
        ), f"Expected VerticalExtensions, got {type(extensions)}"

        # Verify the returned extensions have proper default values
        assert extensions.middleware == []
        assert extensions.safety_extensions == []
        assert extensions.prompt_contributors == []

    def test_get_extensions_has_proper_default_values(self):
        """Verify that returned VerticalExtensions has proper default values.

        All fields should have sensible defaults (empty lists, None for optionals).
        """
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = ConcreteVertical.get_extensions(use_cache=False)

        # Verify default values for list fields
        assert isinstance(extensions.middleware, list), "middleware should be a list"
        assert isinstance(extensions.safety_extensions, list), "safety_extensions should be a list"
        assert isinstance(
            extensions.prompt_contributors, list
        ), "prompt_contributors should be a list"

        # Verify optional fields can be None (this is allowed)
        # But the container itself must exist
        assert hasattr(extensions, "mode_config_provider")
        assert hasattr(extensions, "tool_dependency_provider")
        assert hasattr(extensions, "workflow_provider")
        assert hasattr(extensions, "service_provider")
        assert hasattr(extensions, "rl_config_provider")
        assert hasattr(extensions, "team_spec_provider")
        assert hasattr(extensions, "enrichment_strategy")

    def test_get_extensions_caching_works(self):
        """Verify that caching returns the same object."""
        # First call - should cache
        extensions1 = ConcreteVertical.get_extensions(use_cache=True)
        # Second call - should return cached
        extensions2 = ConcreteVertical.get_extensions(use_cache=True)

        assert extensions1 is extensions2, "Caching should return same object"

    def test_get_extensions_cache_bypass_creates_new_object(self):
        """Verify that use_cache=False creates a new object."""
        # First call
        extensions1 = ConcreteVertical.get_extensions(use_cache=True)
        # Second call with cache bypass
        extensions2 = ConcreteVertical.get_extensions(use_cache=False)

        # Should be different objects (though may have same content)
        assert extensions1 is not extensions2, "Cache bypass should create new object"

    def test_get_extensions_consistent_type_across_subclasses(self):
        """Verify LSP: subclasses return the same type as base class would."""
        from victor.core.verticals.protocols import VerticalExtensions

        class AnotherVertical(VerticalBase):
            name = "another_vertical"
            description = "Another test vertical"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Another prompt"

        extensions1 = ConcreteVertical.get_extensions(use_cache=False)
        extensions2 = AnotherVertical.get_extensions(use_cache=False)

        # Both must return VerticalExtensions (LSP)
        assert type(extensions1) == type(
            extensions2
        ), "LSP violation: different subclasses return different types"
        assert isinstance(extensions1, VerticalExtensions)
        assert isinstance(extensions2, VerticalExtensions)


class TestVerticalBaseConfig:
    """Tests for VerticalBase.get_config() method."""

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_get_config_returns_vertical_config(self):
        """Verify get_config returns a VerticalConfig object."""
        config = ConcreteVertical.get_config()

        assert isinstance(config, VerticalConfig)
        assert config.system_prompt == "You are a test assistant."

    def test_get_config_caching(self):
        """Verify config caching works correctly."""
        config1 = ConcreteVertical.get_config(use_cache=True)
        config2 = ConcreteVertical.get_config(use_cache=True)

        assert config1 is config2, "Cached configs should be same object"
