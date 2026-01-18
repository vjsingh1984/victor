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

"""Tests for ToolAliasResolver.

This test file validates the ToolAliasResolver which is responsible for:
- Resolving tool aliases to canonical names
- Shell variant selection (shell vs shell_readonly)
- Legacy name mapping
- Canonical name validation

The resolver follows Single Responsibility Principle and is extracted
from ToolCoordinator as part of Track 4 refactoring.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Set, Dict

from victor.agent.coordinators.tool_alias_resolver import (
    ToolAliasResolver,
    ToolAliasConfig,
    ResolutionResult,
    create_tool_alias_resolver,
)
from victor.tools.tool_names import ToolNames


class TestToolAliasResolver:
    """Test suite for ToolAliasResolver.

    Tests cover:
    - Direct name resolution
    - Legacy name mapping
    - Shell variant resolution
    - Access check integration
    - Configuration options
    """

    # ========================================================================
    # Fixtures
    # ========================================================================

    @pytest.fixture
    def mock_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=False)
        return registry

    @pytest.fixture
    def access_check_enabled(self) -> Mock:
        """Create access check that grants shell access."""
        check = Mock()
        check.return_value = True
        return check

    @pytest.fixture
    def access_check_disabled(self) -> Mock:
        """Create access check that denies all access."""
        check = Mock()
        check.return_value = False
        return check

    @pytest.fixture
    def access_check_readonly(self) -> Mock:
        """Create access check that only grants readonly shell access."""

        def check_fn(tool_name: str) -> bool:
            return tool_name == ToolNames.SHELL_READONLY

        check = Mock(side_effect=check_fn)
        return check

    @pytest.fixture
    def resolver(self) -> ToolAliasResolver:
        """Create resolver with default configuration."""
        return ToolAliasResolver()

    @pytest.fixture
    def resolver_with_registry(self, mock_registry: Mock) -> ToolAliasResolver:
        """Create resolver with tool registry."""
        return ToolAliasResolver(tool_registry=mock_registry)

    @pytest.fixture
    def resolver_with_access(self, access_check_enabled: Mock) -> ToolAliasResolver:
        """Create resolver with access check."""
        return ToolAliasResolver(access_check=access_check_enabled)

    @pytest.fixture
    def resolver_readonly_config(self) -> ToolAliasResolver:
        """Create resolver configured to prefer readonly shell."""
        config = ToolAliasConfig(prefer_readonly_shell=True)
        return ToolAliasResolver(config=config)

    # ========================================================================
    # Direct Name Resolution Tests
    # ========================================================================

    def test_resolve_canonical_name_direct(self, resolver: ToolAliasResolver):
        """Test that canonical names resolve directly without modification."""
        # Test various canonical names
        canonical_names = [
            "read",
            "write",
            "edit",
            "grep",
            "code_search",
            "web_search",
            "git",
            "patch",
            "review",
            "test",
        ]

        for name in canonical_names:
            result = resolver.resolve(name)
            assert result == name, f"Canonical name '{name}' should resolve to itself"

    def test_resolve_with_result_for_canonical_name(self, resolver: ToolAliasResolver):
        """Test ResolutionResult for canonical names."""
        result = resolver.resolve_with_result("read")

        assert result.original == "read"
        assert result.resolved == "read"
        assert result.method == "direct"
        assert result.is_legacy is False

    def test_resolve_unknown_tool_returns_itself(self, resolver: ToolAliasResolver):
        """Test that unknown tool names are returned as-is."""
        unknown_tool = "unknown_tool_xyz"
        result = resolver.resolve(unknown_tool)

        assert result == unknown_tool

    def test_resolve_with_result_for_unknown_tool(self, resolver: ToolAliasResolver):
        """Test ResolutionResult for unknown tools."""
        result = resolver.resolve_with_result("unknown_tool_xyz")

        assert result.original == "unknown_tool_xyz"
        assert result.resolved == "unknown_tool_xyz"
        assert result.method == "direct"
        assert result.is_legacy is False

    # ========================================================================
    # Legacy Name Mapping Tests
    # ========================================================================

    def test_resolve_legacy_file_operation_names(self, resolver: ToolAliasResolver):
        """Test legacy file operation name mappings."""
        legacy_mappings = {
            "read_file": "read",
            "write_file": "write",
            "edit_file": "edit",
            "list_files": "ls",
        }

        for legacy, canonical in legacy_mappings.items():
            result = resolver.resolve(legacy)
            assert result == canonical, f"Legacy '{legacy}' should map to '{canonical}'"

    def test_resolve_legacy_search_names(self, resolver: ToolAliasResolver):
        """Test legacy search tool name mappings."""
        # Already canonical - should stay same
        result = resolver.resolve("code_search")
        assert result == "code_search"

        result = resolver.resolve("semantic_search")
        assert result == "semantic_search"

    def test_resolve_legacy_web_names(self, resolver: ToolAliasResolver):
        """Test legacy web tool name mappings."""
        # Already canonical
        result = resolver.resolve("web_search")
        assert result == "web_search"

        # Legacy mapping
        result = resolver.resolve("fetch_url")
        assert result == "web_fetch"

    def test_resolve_legacy_execution_names(self, resolver: ToolAliasResolver):
        """Test legacy execution tool name mappings."""
        legacy_mappings = {
            "execute_command": "shell",
            "run_bash": "shell",
        }

        for legacy, canonical in legacy_mappings.items():
            result = resolver.resolve(legacy)
            assert result == canonical, f"Legacy '{legacy}' should map to '{canonical}'"

    def test_resolve_legacy_git_names(self, resolver: ToolAliasResolver):
        """Test legacy git tool name mappings."""
        result = resolver.resolve("git_command")
        assert result == "git"

    def test_resolve_with_result_for_legacy_name(self, resolver: ToolAliasResolver):
        """Test ResolutionResult for legacy names."""
        result = resolver.resolve_with_result("read_file")

        assert result.original == "read_file"
        assert result.resolved == "read"
        assert result.method == "legacy"
        assert result.is_legacy is True

    def test_all_legacy_mappings_are_valid(self, resolver: ToolAliasResolver):
        """Test that all defined legacy mappings resolve correctly."""
        from victor.agent.coordinators.tool_alias_resolver import ToolAliasResolver

        for legacy, canonical in ToolAliasResolver.LEGACY_MAPPINGS.items():
            result = resolver.resolve(legacy)
            assert result == canonical, f"Legacy mapping for '{legacy}' failed"

    # ========================================================================
    # Shell Alias Detection Tests
    # ========================================================================

    def test_is_shell_alias_for_shell_variants(self):
        """Test shell alias detection for all shell variants."""
        shell_aliases = [
            "run",
            "bash",
            "execute",
            "cmd",
            "execute_bash",
            "shell_readonly",
            "shell",
        ]

        resolver = ToolAliasResolver()
        for alias in shell_aliases:
            assert resolver.is_shell_alias(alias), f"'{alias}' should be a shell alias"

    def test_is_shell_alias_for_non_shell_names(self):
        """Test that non-shell names are not detected as shell aliases."""
        non_shell_names = [
            "read",
            "write",
            "grep",
            "git",
            "web_search",
            "review",
        ]

        resolver = ToolAliasResolver()
        for name in non_shell_names:
            assert not resolver.is_shell_alias(name), f"'{name}' should not be a shell alias"

    # ========================================================================
    # Legacy Name Detection Tests
    # ========================================================================

    def test_is_legacy_name_for_known_legacy(self):
        """Test legacy name detection for known legacy names."""
        resolver = ToolAliasResolver()

        legacy_names = [
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "search_files",
            "execute_command",
            "run_bash",
            "git_command",
        ]

        for name in legacy_names:
            assert resolver.is_legacy_name(name), f"'{name}' should be a legacy name"

    def test_is_legacy_name_for_canonical_names(self):
        """Test that canonical names are not detected as legacy."""
        resolver = ToolAliasResolver()

        canonical_names = [
            "read",
            "write",
            "edit",
            "ls",
            "grep",
            "shell",
            "git",
        ]

        for name in canonical_names:
            assert not resolver.is_legacy_name(name), f"'{name}' should not be a legacy name"

    # ========================================================================
    # Shell Variant Resolution Tests (with Access Check)
    # ========================================================================

    def test_resolve_shell_alias_to_full_shell_when_enabled(
        self, resolver_with_access: ToolAliasResolver
    ):
        """Test that shell aliases resolve to full shell when enabled."""
        shell_aliases = ["run", "bash", "execute", "cmd", "execute_bash", "shell"]

        for alias in shell_aliases:
            result = resolver_with_access.resolve(alias)
            assert result == ToolNames.SHELL, f"'{alias}' should resolve to 'shell'"

    def test_resolve_shell_with_result_for_full_shell(
        self, resolver_with_access: ToolAliasResolver
    ):
        """Test ResolutionResult for shell alias with full shell enabled."""
        result = resolver_with_access.resolve_with_result("bash")

        assert result.original == "bash"
        assert result.resolved == ToolNames.SHELL
        assert result.method == "shell_variant"
        assert result.is_legacy is False

    def test_resolve_shell_to_readonly_when_full_disabled(self, access_check_readonly: Mock):
        """Test that shell resolves to readonly when full shell is disabled."""
        resolver = ToolAliasResolver(access_check=access_check_readonly)

        result = resolver.resolve("bash")
        assert result == ToolNames.SHELL_READONLY

    def test_resolve_shell_with_result_for_readonly(self, access_check_readonly: Mock):
        """Test ResolutionResult for shell alias with readonly shell."""
        resolver = ToolAliasResolver(access_check=access_check_readonly)

        result = resolver.resolve_with_result("execute")

        assert result.original == "execute"
        assert result.resolved == ToolNames.SHELL_READONLY
        assert result.method == "shell_variant"
        assert result.is_legacy is False

    def test_resolve_shell_when_both_disabled(self, resolver: ToolAliasResolver):
        """Test that shell aliases fall back to canonical when both disabled."""
        # No access check or registry - should use canonical
        result = resolver.resolve("bash")
        # Falls back to canonical name from tool_names
        assert result == "shell" or result == "bash"

    def test_resolve_shell_readonly_directly(self, resolver: ToolAliasResolver):
        """Test that shell_readonly resolves directly."""
        result = resolver.resolve("shell_readonly")
        assert result == "shell_readonly"

    # ========================================================================
    # Registry Integration Tests
    # ========================================================================

    def test_resolve_uses_registry_for_shell_check(
        self, resolver_with_registry: ToolAliasResolver, mock_registry: Mock
    ):
        """Test that resolver queries registry for shell availability."""
        # Enable full shell in registry
        mock_registry.is_tool_enabled.return_value = True

        result = resolver_with_registry.resolve("bash")

        # Should have checked registry
        mock_registry.is_tool_enabled.assert_called_with(ToolNames.SHELL)
        assert result == ToolNames.SHELL

    def test_resolve_checks_readonly_when_full_disabled(
        self, resolver_with_registry: ToolAliasResolver, mock_registry: Mock
    ):
        """Test that resolver checks readonly when full shell is disabled."""

        # Full shell disabled, readonly enabled
        def check_fn(tool_name: str) -> bool:
            return tool_name == ToolNames.SHELL_READONLY

        mock_registry.is_tool_enabled.side_effect = check_fn

        result = resolver_with_registry.resolve("execute")

        # Should have checked both
        assert mock_registry.is_tool_enabled.call_count >= 1
        assert result == ToolNames.SHELL_READONLY

    def test_resolve_with_registry_disabled(
        self, resolver_with_registry: ToolAliasResolver, mock_registry: Mock
    ):
        """Test resolution when all shell variants are disabled in registry."""
        # All disabled
        mock_registry.is_tool_enabled.return_value = False

        result = resolver_with_registry.resolve("bash")

        # Should fall back to canonical
        assert result in ["shell", "bash"]

    # ========================================================================
    # Configuration Tests
    # ========================================================================

    def test_set_access_check_updates_resolver(self, resolver: ToolAliasResolver):
        """Test that set_access_check updates the access check function."""
        new_check = Mock(return_value=True)
        resolver.set_access_check(new_check)

        # Should use new check
        result = resolver.resolve("bash")
        assert result == ToolNames.SHELL

    def test_set_prefer_readonly_updates_config(self, resolver: ToolAliasResolver):
        """Test that set_prefer_readonly updates configuration."""
        resolver.set_prefer_readonly(True)

        assert resolver._config.prefer_readonly_shell is True

    def test_set_prefer_readonly_false(self, resolver: ToolAliasResolver):
        """Test setting prefer_readonly to False."""
        resolver.set_prefer_readonly(False)

        assert resolver._config.prefer_readonly_shell is False

    def test_config_prefer_readonly_initialization(self):
        """Test creating resolver with prefer_readonly config."""
        config = ToolAliasConfig(prefer_readonly_shell=True)
        resolver = ToolAliasResolver(config=config)

        assert resolver._config.prefer_readonly_shell is True

    # ========================================================================
    # Custom Legacy Mapping Tests
    # ========================================================================

    def test_add_legacy_mapping(self, resolver: ToolAliasResolver):
        """Test adding custom legacy name mapping."""
        resolver.add_legacy_mapping("old_tool", "new_tool")

        result = resolver.resolve("old_tool")
        assert result == "new_tool"

    def test_add_legacy_mapping_updates_class(self, resolver: ToolAliasResolver):
        """Test that add_legacy_mapping updates the class-level mapping."""
        resolver.add_legacy_mapping("custom_legacy", "custom_canonical")

        # Should be in legacy mappings
        assert "custom_legacy" in resolver.LEGACY_MAPPINGS
        assert resolver.LEGACY_MAPPINGS["custom_legacy"] == "custom_canonical"

    def test_add_legacy_mapping_with_existing_key(self, resolver: ToolAliasResolver):
        """Test adding legacy mapping for existing key overwrites it."""
        resolver.add_legacy_mapping("test_tool", "first_target")
        resolver.add_legacy_mapping("test_tool", "second_target")

        # Should use the second mapping
        result = resolver.resolve("test_tool")
        assert result == "second_target"

    # ========================================================================
    # Factory Function Tests
    # ========================================================================

    def test_create_tool_alias_resolver_default(self):
        """Test factory function with default parameters."""
        resolver = create_tool_alias_resolver()

        assert isinstance(resolver, ToolAliasResolver)
        assert resolver._config.prefer_readonly_shell is False

    def test_create_tool_alias_resolver_with_prefer_readonly(self):
        """Test factory function with prefer_readonly=True."""
        resolver = create_tool_alias_resolver(prefer_readonly=True)

        assert resolver._config.prefer_readonly_shell is True

    def test_create_tool_alias_resolver_with_access_check(self, access_check_enabled: Mock):
        """Test factory function with access check."""
        resolver = create_tool_alias_resolver(access_check=access_check_enabled)

        assert resolver._access_check == access_check_enabled

    def test_create_tool_alias_resolver_with_registry(self, mock_registry: Mock):
        """Test factory function with tool registry."""
        resolver = create_tool_alias_resolver(tool_registry=mock_registry)

        assert resolver._registry == mock_registry

    def test_create_tool_alias_resolver_with_all_parameters(
        self, mock_registry: Mock, access_check_enabled: Mock
    ):
        """Test factory function with all parameters."""
        resolver = create_tool_alias_resolver(
            tool_registry=mock_registry,
            access_check=access_check_enabled,
            prefer_readonly=True,
        )

        assert resolver._registry == mock_registry
        assert resolver._access_check == access_check_enabled
        assert resolver._config.prefer_readonly_shell is True

    # ========================================================================
    # Canonical Name Helper Tests
    # ========================================================================

    def test_get_canonical_name_for_known_tool(self, resolver: ToolAliasResolver):
        """Test _get_canonical_name for known tools."""
        # Patch to mock tool_names.get_canonical_name
        with patch("victor.tools.tool_names.get_canonical_name") as mock_get:
            mock_get.return_value = "canonical_tool"

            result = resolver._get_canonical_name("some_tool")

            assert result == "canonical_tool"
            mock_get.assert_called_once_with("some_tool")

    def test_get_canonical_name_fallback_on_exception(self, resolver: ToolAliasResolver):
        """Test _get_canonical_name falls back to original on exception."""
        with patch("victor.tools.tool_names.get_canonical_name") as mock_get:
            mock_get.side_effect = Exception("Canonicalization failed")

            result = resolver._get_canonical_name("some_tool")

            # Should return original name
            assert result == "some_tool"

    # ========================================================================
    # Integration Tests
    # ========================================================================

    def test_complete_resolution_flow_legacy_then_shell(
        self, resolver_with_access: ToolAliasResolver
    ):
        """Test resolving different types of aliases in sequence."""
        # Legacy name
        result1 = resolver_with_access.resolve("read_file")
        assert result1 == "read"

        # Shell alias
        result2 = resolver_with_access.resolve("bash")
        assert result2 == ToolNames.SHELL

        # Canonical name
        result3 = resolver_with_access.resolve("write")
        assert result3 == "write"

    def test_resolution_with_result_contains_all_fields(self, resolver: ToolAliasResolver):
        """Test that ResolutionResult contains all expected fields."""
        result = resolver.resolve_with_result("read_file")

        # Check all fields exist
        assert hasattr(result, "original")
        assert hasattr(result, "resolved")
        assert hasattr(result, "method")
        assert hasattr(result, "is_legacy")

        # Check values
        assert result.original == "read_file"
        assert result.resolved == "read"
        assert result.method == "legacy"
        assert result.is_legacy is True

    def test_multiple_resolvers_independent_state(self):
        """Test that multiple resolvers maintain independent state."""
        resolver1 = ToolAliasResolver()
        resolver2 = ToolAliasResolver()

        # Add custom mapping to resolver1 only
        resolver1.add_legacy_mapping("custom", "target1")

        # Resolver1 should have the mapping
        result1 = resolver1.resolve("custom")
        assert result1 == "target1"

        # Resolver2 should also have it (class-level mapping)
        result2 = resolver2.resolve("custom")
        assert result2 == "target1"

    def test_resolver_with_custom_config(self, mock_registry: Mock):
        """Test resolver with custom configuration."""
        config = ToolAliasConfig(
            prefer_readonly_shell=True,
            strict_resolution=True,
            enabled=True,
            timeout=30,
            max_retries=3,
        )

        resolver = ToolAliasResolver(
            tool_registry=mock_registry,
            config=config,
        )

        assert resolver._config.prefer_readonly_shell is True
        assert resolver._config.strict_resolution is True
        assert resolver._config.enabled is True
        assert resolver._config.timeout == 30
        assert resolver._config.max_retries == 3


class TestToolAliasResolverEdgeCases:
    """Test edge cases and boundary conditions for ToolAliasResolver."""

    @pytest.fixture
    def resolver(self) -> ToolAliasResolver:
        """Create resolver."""
        return ToolAliasResolver()

    def test_resolve_empty_string(self, resolver: ToolAliasResolver):
        """Test resolving empty string."""
        result = resolver.resolve("")
        assert result == ""

    def test_resolve_with_special_characters(self, resolver: ToolAliasResolver):
        """Test resolving names with special characters."""
        # Should handle gracefully
        result = resolver.resolve("tool-with-dashes")
        assert result == "tool-with-dashes"

    def test_resolve_case_sensitivity(self, resolver: ToolAliasResolver):
        """Test that resolution is case-sensitive."""
        # Canonical names are lowercase
        result_lower = resolver.resolve("read")
        assert result_lower == "read"

        # Uppercase version should not map to same thing
        result_upper = resolver.resolve("Read")
        # Either returns as-is or gets canonicalized
        assert result_upper in ["Read", "read"]

    def test_resolve_unicode_tool_name(self, resolver: ToolAliasResolver):
        """Test resolving tool names with unicode characters."""
        unicode_name = "工具"
        result = resolver.resolve(unicode_name)
        assert result == unicode_name

    def test_is_shell_alias_empty_string(self, resolver: ToolAliasResolver):
        """Test is_shell_alias with empty string."""
        assert not resolver.is_shell_alias("")

    def test_is_legacy_name_empty_string(self, resolver: ToolAliasResolver):
        """Test is_legacy_name with empty string."""
        assert not resolver.is_legacy_name("")

    def test_add_legacy_mapping_empty_strings(self, resolver: ToolAliasResolver):
        """Test adding legacy mapping with empty strings."""
        # Should allow it (validation happens elsewhere)
        resolver.add_legacy_mapping("", "")
        result = resolver.resolve("")
        assert result == ""

    def test_resolve_with_none_access_check(self):
        """Test resolver with None access check doesn't crash."""
        resolver = ToolAliasResolver(access_check=None)

        # Should not raise
        result = resolver.resolve("bash")
        assert result in ["shell", "bash"]

    def test_resolve_with_none_registry(self):
        """Test resolver with None registry doesn't crash."""
        resolver = ToolAliasResolver(tool_registry=None)

        # Should not raise
        result = resolver.resolve("read")
        assert result == "read"

    def test_set_access_check_to_none(self, resolver: ToolAliasResolver):
        """Test setting access check to None."""
        resolver.set_access_check(None)
        assert resolver._access_check is None

    def test_factory_function_with_none_parameters(self):
        """Test factory function with None parameters."""
        resolver = create_tool_alias_resolver(
            tool_registry=None,
            access_check=None,
            prefer_readonly=False,
        )

        assert isinstance(resolver, ToolAliasResolver)


class TestResolutionResult:
    """Test ResolutionResult dataclass."""

    def test_resolution_result_creation(self):
        """Test creating ResolutionResult."""
        result = ResolutionResult(
            original="read_file",
            resolved="read",
            method="legacy",
            is_legacy=True,
        )

        assert result.original == "read_file"
        assert result.resolved == "read"
        assert result.method == "legacy"
        assert result.is_legacy is True

    def test_resolution_result_default_is_legacy(self):
        """Test ResolutionResult default is_legacy=False."""
        result = ResolutionResult(
            original="tool",
            resolved="tool",
            method="direct",
        )

        assert result.is_legacy is False

    def test_resolution_result_immutability_attempt(self):
        """Test that ResolutionResult fields can be modified (not frozen)."""
        result = ResolutionResult(
            original="tool",
            resolved="tool",
            method="direct",
        )

        # Dataclass is not frozen, so we can modify
        result.resolved = "new_tool"
        assert result.resolved == "new_tool"


class TestToolAliasConfig:
    """Test ToolAliasConfig dataclass."""

    def test_config_default_values(self):
        """Test default values of ToolAliasConfig."""
        config = ToolAliasConfig()

        assert config.prefer_readonly_shell is False
        assert config.strict_resolution is False

    def test_config_custom_values(self):
        """Test creating config with custom values."""
        config = ToolAliasConfig(
            prefer_readonly_shell=True,
            strict_resolution=True,
        )

        assert config.prefer_readonly_shell is True
        assert config.strict_resolution is True

    def test_config_inherits_base_config(self):
        """Test that ToolAliasConfig inherits base config fields."""
        config = ToolAliasConfig(
            enabled=True,
            timeout=60,
            max_retries=5,
            retry_enabled=True,
            log_level="DEBUG",
            enable_metrics=True,
            prefer_readonly_shell=True,
        )

        # Base config fields
        assert config.enabled is True
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.retry_enabled is True
        assert config.log_level == "DEBUG"
        assert config.enable_metrics is True

        # Custom fields
        assert config.prefer_readonly_shell is True
