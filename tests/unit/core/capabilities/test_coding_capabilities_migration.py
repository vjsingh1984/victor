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

"""Tests for coding vertical capability migration (Phase 5.2).

These tests verify that the migration from the old CodingCapabilityProvider
to the new CapabilityRegistry maintains backward compatibility.
"""

import pytest

from victor.core.capabilities import (
    CapabilityRegistry,
    CapabilityType,
    register_vertical_capabilities,
)


@pytest.fixture
def fresh_registry():
    """Create a fresh registry for testing."""
    return CapabilityRegistry()


class TestCodingCapabilityMigration:
    """Tests for coding capability migration to new registry."""

    def test_coding_capabilities_registered(self, fresh_registry):
        """Test that coding capabilities can be registered from YAML."""
        count = register_vertical_capabilities("coding", registry=fresh_registry)
        assert count >= 5, f"Expected at least 5 capabilities, got {count}"

    def test_git_safety_capability_exists(self, fresh_registry):
        """Test that git_safety capability exists with correct type."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        cap = fresh_registry.get("coding_git_safety")
        assert cap is not None
        assert cap.capability_type == CapabilityType.SAFETY
        assert "block_force_push" in str(cap.default_config)

    def test_code_style_capability_exists(self, fresh_registry):
        """Test that code_style capability exists with correct type."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        cap = fresh_registry.get("coding_code_style")
        assert cap is not None
        assert cap.capability_type == CapabilityType.MODE
        assert cap.default_config.get("formatter") == "black"
        assert cap.default_config.get("linter") == "ruff"

    def test_test_requirements_capability_exists(self, fresh_registry):
        """Test that test_requirements capability exists with correct type."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        cap = fresh_registry.get("coding_test_requirements")
        assert cap is not None
        assert cap.capability_type == CapabilityType.MODE
        assert cap.default_config.get("framework") == "pytest"

    def test_language_server_capability_exists(self, fresh_registry):
        """Test that language_server capability exists with correct type."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        cap = fresh_registry.get("coding_language_server")
        assert cap is not None
        assert cap.capability_type == CapabilityType.TOOL
        assert "python" in cap.default_config.get("languages", [])

    def test_refactoring_capability_exists(self, fresh_registry):
        """Test that refactoring capability exists with correct type and dependencies."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        cap = fresh_registry.get("coding_refactoring")
        assert cap is not None
        assert cap.capability_type == CapabilityType.TOOL
        assert "coding_language_server" in cap.dependencies

    def test_capability_handler_integration(self, fresh_registry):
        """Test that capability handlers can be created from registered capabilities."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        handler = fresh_registry.get_handler("coding_code_style")
        assert handler is not None
        assert handler.definition.name == "coding_code_style"

    def test_list_capabilities_by_type(self, fresh_registry):
        """Test listing capabilities by type."""
        register_vertical_capabilities("coding", registry=fresh_registry)

        # Safety capabilities - list_by_type returns names (strings)
        safety_caps = fresh_registry.list_by_type(CapabilityType.SAFETY)
        assert "coding_git_safety" in safety_caps

        # Mode capabilities
        mode_caps = fresh_registry.list_by_type(CapabilityType.MODE)
        assert "coding_code_style" in mode_caps
        assert "coding_test_requirements" in mode_caps

        # Tool capabilities
        tool_caps = fresh_registry.list_by_type(CapabilityType.TOOL)
        assert "coding_language_server" in tool_caps
        assert "coding_refactoring" in tool_caps


class TestCodingCapabilityProviderBackwardCompatibility:
    """Tests for backward compatibility with existing CodingCapabilityProvider."""

    def test_provider_import(self):
        """Test that CodingCapabilityProvider can still be imported."""
        from victor.coding.capabilities import CodingCapabilityProvider

        provider = CodingCapabilityProvider()
        assert provider is not None
        # Provider uses _vertical_name internally
        assert provider._vertical_name == "coding"

    def test_provider_list_capabilities(self):
        """Test that provider can list capabilities."""
        from victor.coding.capabilities import CodingCapabilityProvider

        provider = CodingCapabilityProvider()
        capabilities = provider.list_capabilities()
        assert len(capabilities) >= 5
        assert "git_safety" in capabilities
        assert "code_style" in capabilities

    def test_capabilities_list_export(self):
        """Test that CAPABILITIES list is still exported."""
        from victor.coding.capabilities import CAPABILITIES

        assert len(CAPABILITIES) >= 5

    def test_get_coding_capabilities_function(self):
        """Test that get_coding_capabilities function still works."""
        from victor.coding.capabilities import get_coding_capabilities

        caps = get_coding_capabilities()
        assert len(caps) >= 5

    def test_configure_functions_still_exist(self):
        """Test that configure_* functions are still available."""
        from victor.coding.capabilities import (
            configure_git_safety,
            configure_code_style,
            configure_test_requirements,
            configure_language_server,
            configure_refactoring,
        )

        # Just test they exist and are callable
        assert callable(configure_git_safety)
        assert callable(configure_code_style)
        assert callable(configure_test_requirements)
        assert callable(configure_language_server)
        assert callable(configure_refactoring)


class TestCodingCapabilitySchemaValidation:
    """Tests for config schema validation in coding capabilities."""

    def test_code_style_schema_validation(self, fresh_registry):
        """Test that code_style has valid config schema."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        cap = fresh_registry.get("coding_code_style")
        assert cap.config_schema is not None
        assert "properties" in cap.config_schema
        assert "formatter" in cap.config_schema.get("properties", {})
        assert "linter" in cap.config_schema.get("properties", {})

    def test_test_requirements_schema_validation(self, fresh_registry):
        """Test that test_requirements has valid config schema."""
        register_vertical_capabilities("coding", registry=fresh_registry)
        cap = fresh_registry.get("coding_test_requirements")
        assert cap.config_schema is not None
        assert "properties" in cap.config_schema
