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
    migrate_to_injector,
    get_capability_or_create,
    check_migration_status,
    print_migration_report,
)
from victor.core.verticals.capability_injector import (
    CapabilityInjector,
    get_capability_injector,
)
from victor.core.verticals.capability_provider import (
    BaseCapabilityProvider,
    reset_global_registry,
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
            "test_capability", "Use get_capability_injector().get_capability('test_capability')"
        )
        class TestCapability:
            def __init__(self):
                self.initialized = True

        # Creating instance should issue warning
        with pytest.warns(DeprecationWarning, match="Direct instantiation"):
            capability = TestCapability()

        assert capability.initialized is True

    def test_warning_includes_migration_guide(self):
        """Test that warning includes migration guide."""

        @deprecated_direct_instantiation("test_capability", "Use injector instead")
        class TestCapability:
            pass

        with pytest.warns(DeprecationWarning) as record:
            TestCapability()

        assert len(record) == 1
        assert "Migration: Use injector instead" in str(record[0].message)


# =============================================================================
# Migration Helper Tests
# =============================================================================


class TestGetCapabilityOrCreate:
    """Tests for get_capability_or_create function."""

    def test_returns_capability_from_injector(self):
        """Test that function returns capability from injector if found."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = get_capability_injector()

        class MockProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("test_capability", container)

            def _create_instance(self):
                return _MockCapability()

        injector.register_provider(MockProvider())

        # Should get from injector
        capability = get_capability_or_create("test_capability", _LegacyCapability)

        assert isinstance(capability, _MockCapability)

    def test_creates_with_factory_if_not_found(self):
        """Test that function creates capability if not in injector."""
        reset_global_registry()
        CapabilityInjector.reset_global()
        injector = get_capability_injector()

        # Capability not registered, should use factory
        capability = get_capability_or_create("nonexistent_capability", _LegacyCapability)

        assert isinstance(capability, _LegacyCapability)


# =============================================================================
# Status Utility Tests
# =============================================================================


class TestCheckMigrationStatus:
    """Tests for check_migration_status function."""

    def test_returns_empty_for_no_capabilities(self):
        """Test that function returns empty dict for no capabilities."""

        class TestVertical:
            pass

        status = check_migration_status(TestVertical)

        assert status == {}


class TestPrintMigrationReport:
    """Tests for print_migration_report function."""

    def test_prints_report_for_no_capabilities(self, capsys):
        """Test that function prints report even with no capabilities."""

        class TestVertical:
            pass

        # Print report
        print_migration_report(TestVertical)

        # Check output
        captured = capsys.readouterr()
        assert "Migration Report for TestVertical" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================


class TestMigrationIntegration:
    """Integration tests for complete migration workflow."""

    def test_full_migration_workflow(self):
        """Test complete migration from legacy to injector-based."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        # Setup: Register capability in injector without auto-registration
        injector = CapabilityInjector(auto_register_builtins=False)

        class MockProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("custom_capability", container)

            def _create_instance(self):
                return _MockCapability()

        injector.register_provider(MockProvider())

        # Before: Legacy vertical with direct instantiation
        class LegacyVertical:
            def __init__(self):
                self._custom_cap = _LegacyCapability()

        # Check migration status (before)
        status_before = check_migration_status(LegacyVertical)
        # No file_operations capability, so status is empty
        assert isinstance(status_before, dict)

        # Migrate
        migrate_to_injector(LegacyVertical, {"_custom_cap": "custom_capability"})

        # Verify new instances use injector
        new_vertical = LegacyVertical()
        capability = new_vertical._custom_cap
        assert isinstance(capability, _MockCapability)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
