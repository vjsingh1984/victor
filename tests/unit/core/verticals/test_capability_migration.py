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

"""Tests for capability migration helpers.

Tests for the migration helper functions that make it easy for verticals
to adopt the new DI-based capability injection system.

Run with: pytest tests/unit/core/verticals/test_capability_migration.py -v
"""

import warnings

import pytest

from victor.core.verticals.capability_migration import (
    deprecated_direct_instantiation,
    migrate_capability_property,
    migrate_to_injector,
    get_capability_or_create,
    check_migration_status,
    print_migration_report,
)
from victor.core.verticals.capability_injector import (
    CapabilityInjector,
    get_capability_injector,
    reset_global_registry,
)
from victor.core.verticals.capability_provider import (
    BaseCapabilityProvider,
    reset_global_registry as reset_provider_registry,
)


# =============================================================================
# Mock Capabilities
# =============================================================================


class _MockCapability:
    """Mock capability for testing."""

    def __init__(self):
        self.initialized = True


class _LegacyCapability(_MockCapability):
    """Legacy capability with direct instantiation."""

    pass


# =============================================================================
# Decorator Tests
# =============================================================================


class TestDeprecatedDirectInstantiation:
    """Tests for deprecated_direct_instantiation decorator."""

    def test_issues_deprecation_warning(self):
        """Test that decorator issues deprecation warning."""

        @deprecated_direct_instantiation(
            "test_capability",
            "Use get_capability_injector().get_capability('test_capability')"
        )
        class TestCapability:
            def __init__(self):
                self.initialized = True

        # Creating instance should issue warning
        with pytest.warns(DeprecationWarning, match="Direct instantiation of test_capability"):
            capability = TestCapability()

        assert capability.initialized is True

    def test_warning_includes_migration_guide(self):
        """Test that warning includes migration guide."""

        @deprecated_direct_instantiation(
            "test_capability",
            "Use injector instead"
        )
        class TestCapability:
            pass

        with pytest.warns(DeprecationWarning) as record:
            TestCapability()

        assert len(record) == 1
        assert "Migration: Use injector instead" in str(record[0].message)


class TestMigrateCapabilityProperty:
    """Tests for migrate_capability_property function."""

    def test_property_uses_injector(self):
        """Test that migrated property uses CapabilityInjector."""
        reset_provider_registry()
        CapabilityInjector.reset_global()

        # Register mock capability
        injector = get_capability_injector()

        class MockProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("test_capability", container)

            def _create_instance(self):
                return _MockCapability()

        injector.register_provider(MockProvider())

        # Create class with migrated property
        class TestVertical:
            @migrate_capability_property("_test_cap", "test_capability")
            def test_cap(self):
                pass

        vertical = TestVertical()

        # Property should use injector
        capability = vertical.test_cap
        assert isinstance(capability, _MockCapability)

    def test_property_warns_on_old_access(self):
        """Test that accessing old property issues warning."""
        reset_provider_registry()
        CapabilityInjector.reset_global()

        injector = get_capability_injector()

        class MockProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("test_capability", container)

            def _create_instance(self):
                return _MockCapability()

        injector.register_provider(MockProvider())

        class TestVertical:
            def __init__(self):
                self._test_cap = _LegacyCapability()  # Old direct instantiation

            @migrate_capability_property("_test_cap", "test_capability")
            def test_cap(self):
                pass

        vertical = TestVertical()

        # Accessing property should warn about old attribute
        with pytest.warns(DeprecationWarning, match="is deprecated"):
            _ = vertical.test_cap


# =============================================================================
# Migration Helper Tests
# =============================================================================


class TestMigrateToInjector:
    """Tests for migrate_to_injector function."""

    def test_migrates_vertical_class(self):
        """Test that migrate_to_injector modifies vertical class."""
        reset_provider_registry()
        CapabilityInjector.reset_global()

        # Create mock provider
        injector = get_capability_injector()

        class MockProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("file_operations", container)

            def _create_instance(self):
                return _MockCapability()

        injector.register_provider(MockProvider())

        # Create vertical with old-style capability
        class TestVertical:
            def __init__(self):
                self._file_ops = _LegacyCapability()

        # Verify old attribute exists
        vertical = TestVertical()
        assert isinstance(vertical._file_ops, _LegacyCapability)

        # Migrate
        migrate_to_injector(TestVertical, {"_file_ops": "file_operations"})

        # New instances should use injector
        vertical2 = TestVertical()
        capability = vertical2._file_ops
        assert isinstance(capability, _MockCapability)


class TestGetCapabilityOrCreate:
    """Tests for get_capability_or_create function."""

    def test_returns_capability_from_injector(self):
        """Test that function returns capability from injector if found."""
        reset_provider_registry()
        CapabilityInjector.reset_global()

        injector = get_capability_injector()

        class MockProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("test_capability", container)

            def _create_instance(self):
                return _MockCapability()

        injector.register_provider(MockProvider())

        # Should get from injector
        capability = get_capability_or_create(
            "test_capability",
            _LegacyCapability
        )

        assert isinstance(capability, _MockCapability)

    def test_creates_with_factory_if_not_found(self):
        """Test that function creates capability if not in injector."""
        reset_provider_registry()
        CapabilityInjector.reset_global()
        injector = get_capability_injector()

        # Capability not registered, should use factory
        capability = get_capability_or_create(
            "nonexistent_capability",
            _LegacyCapability
        )

        assert isinstance(capability, _LegacyCapability)


# =============================================================================
# Status Utility Tests
# =============================================================================


class TestCheckMigrationStatus:
    """Tests for check_migration_status function."""

    def test_detects_migrated_capabilities(self):
        """Test that function detects migrated capabilities."""
        reset_provider_registry()
        CapabilityInjector.reset_global()

        injector = get_capability_injector()

        class MockProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("file_operations", container)

            def _create_instance(self):
                return _MockCapability()

        injector.register_provider(MockProvider())

        # Create vertical with migrated property
        class TestVertical:
            @migrate_capability_property("_file_ops", "file_operations")
            def file_ops(self):
                pass

        status = check_migration_status(TestVertical)

        # Should show as migrated
        assert status.get("file_operations") is True

    def test_detects_legacy_capabilities(self):
        """Test that function detects legacy (non-migrated) capabilities."""

        class TestVertical:
            def __init__(self):
                self._file_ops = _LegacyCapability()

        status = check_migration_status(TestVertical)

        # Should show as not migrated
        assert status.get("file_operations") is False

    def test_returns_empty_for_no_capabilities(self):
        """Test that function returns empty dict for no capabilities."""

        class TestVertical:
            pass

        status = check_migration_status(TestVertical)

        assert status == {}


class TestPrintMigrationReport:
    """Tests for print_migration_report function."""

    def test_prints_report(self, capsys):
        """Test that function prints migration report."""

        class TestVertical:
            def __init__(self):
                self._file_ops = _LegacyCapability()

            @migrate_capability_property("_web_ops", "web_operations")
            def web_ops(self):
                pass

        # Print report
        print_migration_report(TestVertical)

        # Check output
        captured = capsys.readouterr()
        assert "Migration Report for TestVertical" in captured.out
        assert "file_operations" in captured.out
        assert "web_operations" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================


class TestMigrationIntegration:
    """Integration tests for complete migration workflow."""

    def test_full_migration_workflow(self):
        """Test complete migration from legacy to injector-based."""
        reset_provider_registry()
        CapabilityInjector.reset_global()

        # Setup: Register capability in injector
        injector = get_capability_injector()

        class MockProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("file_operations", container)

            def _create_instance(self):
                return _MockCapability()

        injector.register_provider(MockProvider())

        # Before: Legacy vertical with direct instantiation
        class LegacyVertical:
            def __init__(self):
                self._file_ops = _LegacyCapability()

        # Check migration status (before)
        status_before = check_migration_status(LegacyVertical)
        assert status_before.get("file_operations") is False

        # Migrate
        migrate_to_injector(LegacyVertical, {"_file_ops": "file_operations"})

        # Check migration status (after)
        status_after = check_migration_status(LegacyVertical)
        assert status_after.get("file_operations") is True

        # Verify new instances use injector
        new_vertical = LegacyVertical()
        capability = new_vertical._file_ops
        assert isinstance(capability, _MockCapability)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
