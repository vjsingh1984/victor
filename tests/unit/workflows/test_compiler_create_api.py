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

"""Unit tests for workflow compiler creation API.

Tests for the plugin creation API that allows creating workflow
compilers from various sources (YAML, JSON, S3, custom plugins, etc.).
"""

from __future__ import annotations

import pytest

from victor.workflows.create import (
    create_compiler,
    _parse_scheme,
    register_scheme_alias,
    list_supported_schemes,
    is_scheme_supported,
    get_default_scheme,
    set_default_scheme,
    DEFAULT_SCHEME,
)
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin
from victor.workflows.compiler_registry import WorkflowCompilerRegistry


# =============================================================================
# Test Fixtures
# =============================================================================


class MockCompilerPlugin(WorkflowCompilerPlugin):
    """Mock plugin for testing."""

    def __init__(self, enable_caching: bool = True, cache_ttl: int = 3600, **options):
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.options = options

    def compile(self, source, *, workflow_name=None, validate=True):
        return {"source": source, "name": workflow_name}


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    registry = WorkflowCompilerRegistry.get_instance()
    registry.reset_instance()
    yield
    registry.reset_instance()


# =============================================================================
# Scheme Parsing Tests
# =============================================================================


class TestParseScheme:
    """Test scheme parsing from source URIs."""

    def test_explicit_yaml_scheme(self):
        """Test parsing explicit YAML scheme."""
        assert _parse_scheme("yaml://file.yaml") == "yaml"

    def test_explicit_json_scheme(self):
        """Test parsing explicit JSON scheme."""
        assert _parse_scheme("json://file.json") == "json"

    def test_explicit_s3_scheme(self):
        """Test parsing S3 scheme."""
        assert _parse_scheme("s3://bucket/key") == "s3"

    def test_compound_scheme(self):
        """Test parsing compound scheme (s3+https)."""
        assert _parse_scheme("s3+https://bucket/key") == "s3"

    def test_no_scheme_yaml_extension(self):
        """Test default scheme for YAML files."""
        assert _parse_scheme("workflow.yaml") == "yaml"

    def test_no_scheme_yml_extension(self):
        """Test default scheme for YML files."""
        assert _parse_scheme("workflow.yml") == "yaml"

    def test_no_scheme_json_extension(self):
        """Test default scheme for JSON files."""
        assert _parse_scheme("workflow.json") == "json"

    def test_no_scheme_no_extension(self):
        """Test default scheme when no extension."""
        assert _parse_scheme("workflow") == DEFAULT_SCHEME

    def test_case_insensitive_scheme(self):
        """Test that schemes are case-insensitive."""
        assert _parse_scheme("YAML://file.yaml") == "yaml"
        assert _parse_scheme("S3://bucket/key") == "s3"


# =============================================================================
# Compiler Creation Tests
# =============================================================================


class TestCreateCompiler:
    """Test create_compiler function."""

    def test_create_yaml_compiler(self):
        """Test creating YAML compiler with explicit scheme."""
        compiler = create_compiler("yaml://")

        assert isinstance(compiler, UnifiedWorkflowCompiler)

    def test_create_json_compiler(self):
        """Test creating JSON compiler with explicit scheme."""
        compiler = create_compiler("json://")

        assert isinstance(compiler, UnifiedWorkflowCompiler)

    def test_create_file_compiler(self):
        """Test creating file compiler with explicit scheme."""
        compiler = create_compiler("file://")

        assert isinstance(compiler, UnifiedWorkflowCompiler)

    def test_create_default_compiler_from_file_extension(self):
        """Test creating default compiler from file extension."""
        compiler = create_compiler("workflow.yaml")

        assert isinstance(compiler, UnifiedWorkflowCompiler)

    def test_create_with_custom_options(self):
        """Test creating compiler with custom options."""
        compiler = create_compiler(
            "yaml://",
            enable_caching=False,
            cache_ttl=7200,
        )

        assert isinstance(compiler, UnifiedWorkflowCompiler)
        assert compiler._config.enable_caching is False
        assert compiler._config.cache_ttl == 7200

    def test_create_unknown_scheme_raises_error(self):
        """Test that unknown scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compiler scheme.*unknown"):
            create_compiler("unknown://source")

    def test_create_with_registered_plugin(self):
        """Test creating compiler with registered plugin."""
        registry = WorkflowCompilerRegistry.get_instance()
        registry.register_compiler("mock", MockCompilerPlugin)

        compiler = create_compiler("mock://")

        assert isinstance(compiler, MockCompilerPlugin)

    def test_create_plugin_with_options(self):
        """Test creating plugin with custom options."""
        registry = WorkflowCompilerRegistry.get_instance()
        registry.register_compiler("mock", MockCompilerPlugin)

        compiler = create_compiler(
            "mock://",
            enable_caching=True,
            cache_ttl=5000,
        )

        # Plugin should receive options
        assert isinstance(compiler, MockCompilerPlugin)


# =============================================================================
# Scheme Management Tests
# =============================================================================


class TestSchemeManagement:
    """Test scheme management functions."""

    def test_list_supported_schemes_default(self):
        """Test listing supported schemes (default)."""
        schemes = list_supported_schemes()

        # Should include built-in schemes
        assert "yaml" in schemes
        assert "yml" in schemes
        assert "file" in schemes
        assert "json" in schemes

    def test_list_supported_schemes_with_plugin(self):
        """Test listing supported schemes with registered plugin."""
        registry = WorkflowCompilerRegistry.get_instance()
        registry.register_compiler("test", MockCompilerPlugin)

        schemes = list_supported_schemes()

        # Should include plugin
        assert "test" in schemes

    def test_is_scheme_supported_builtin(self):
        """Test checking if built-in scheme is supported."""
        assert is_scheme_supported("yaml") is True
        assert is_scheme_supported("json") is True
        assert is_scheme_supported("file") is True

    def test_is_scheme_supported_plugin(self):
        """Test checking if plugin scheme is supported."""
        registry = WorkflowCompilerRegistry.get_instance()
        registry.register_compiler("custom", MockCompilerPlugin)

        assert is_scheme_supported("custom") is True

    def test_is_scheme_supported_unknown(self):
        """Test checking if unknown scheme is supported."""
        assert is_scheme_supported("unknown") is False

    def test_get_default_scheme(self):
        """Test getting default scheme."""
        scheme = get_default_scheme()
        assert scheme == DEFAULT_SCHEME

    def test_set_default_scheme_valid(self):
        """Test setting default scheme to valid value."""
        # Reset to original after test
        original = get_default_scheme()

        set_default_scheme("json")
        assert get_default_scheme() == "json"

        # Restore
        set_default_scheme(original)

    def test_set_default_scheme_invalid(self):
        """Test setting default scheme to invalid value."""
        with pytest.raises(ValueError, match="is not supported"):
            set_default_scheme("unknown_scheme")

    def test_register_scheme_alias(self):
        """Test registering scheme alias."""
        # Register alias
        register_scheme_alias("yaml2", "yaml")

        # Should be supported now
        assert is_scheme_supported("yaml2") is True

        # Should use YAML compiler
        compiler = create_compiler("yaml2://")
        assert isinstance(compiler, UnifiedWorkflowCompiler)

    def test_register_scheme_alias_invalid_target(self):
        """Test registering alias to invalid target scheme."""
        with pytest.raises(ValueError, match="not a built-in scheme"):
            register_scheme_alias("alias", "unknown_target")


# =============================================================================
# Integration Tests
# =============================================================================


class TestPluginCreationIntegration:
    """Integration tests for plugin creation."""

    def test_full_plugin_workflow(self):
        """Test complete plugin registration and usage workflow."""
        # Register plugin
        registry = WorkflowCompilerRegistry.get_instance()
        registry.register_compiler("test", MockCompilerPlugin)

        # Create compiler
        compiler = create_compiler("test://")

        # Compile workflow
        result = compiler.compile("test_source", workflow_name="test_workflow")

        # Verify result
        assert result["source"] == "test_source"
        assert result["name"] == "test_workflow"

    def test_multiple_plugins_coexistence(self):
        """Test that multiple plugins can coexist."""
        registry = WorkflowCompilerRegistry.get_instance()

        # Register multiple plugins
        registry.register_compiler("test1", MockCompilerPlugin)
        registry.register_compiler("test2", MockCompilerPlugin)

        # Both should be accessible
        assert is_scheme_supported("test1") is True
        assert is_scheme_supported("test2") is True

        # List should include both
        schemes = list_supported_schemes()
        assert "test1" in schemes
        assert "test2" in schemes

        # Both should create correct compiler
        compiler1 = create_compiler("test1://")
        compiler2 = create_compiler("test2://")
        assert isinstance(compiler1, MockCompilerPlugin)
        assert isinstance(compiler2, MockCompilerPlugin)

    def test_builtin_schemes_always_available(self):
        """Test that built-in schemes are always available."""
        # Even without any plugins registered
        schemes = list_supported_schemes()

        # Built-in schemes should be present
        assert "yaml" in schemes
        assert "yml" in schemes
        assert "file" in schemes
        assert "json" in schemes

    def test_plugin_priority_over_builtin(self):
        """Test that registered plugins take priority over built-ins."""
        registry = WorkflowCompilerRegistry.get_instance()

        # Register plugin for "yaml" scheme
        registry.register_compiler("yaml", MockCompilerPlugin)

        # Plugin should take priority
        compiler = create_compiler("yaml://")
        assert isinstance(compiler, MockCompilerPlugin)
        assert not isinstance(compiler, UnifiedWorkflowCompiler)
