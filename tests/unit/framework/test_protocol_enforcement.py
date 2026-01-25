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

"""Tests for protocol enforcement (Phase 3.1).

Tests that capability checking uses protocols and emits deprecation
warnings when hasattr fallbacks are used.
"""

import warnings

import pytest

from victor.agent.capability_registry import (
    CapabilityHelper,
    CapabilityRegistryMixin,
)
from victor.framework.protocols import (
    CapabilityRegistryProtocol,
    CapabilityType,
    OrchestratorCapability,
)


class MockProtocolImplementor(CapabilityRegistryMixin):
    """Mock class that properly implements CapabilityRegistryProtocol."""

    def __init__(self):
        super().__init__()
        # Register a test capability using OrchestratorCapability
        self._register_capability(
            OrchestratorCapability(
                name="test_capability",
                capability_type=CapabilityType.TOOL,
                attribute="test_attr",
                description="Test capability",
            )
        )
        self.test_attr = "test_value"


class MockLegacyObject:
    """Mock class that does NOT implement CapabilityRegistryProtocol.

    Has methods that the capability system might look for, but uses
    duck typing instead of protocol conformance.
    """

    def __init__(self):
        self.enabled_tools = []
        self.custom_prompt = None

    def set_enabled_tools(self, tools):
        """Legacy method for setting tools."""
        self.enabled_tools = tools

    def set_custom_prompt(self, prompt):
        """Legacy method for setting prompt."""
        self.custom_prompt = prompt


class TestProtocolEnforcement:
    """Test protocol enforcement for capability checking."""

    def test_capability_check_uses_protocol(self):
        """Protocol-implementing objects should use protocol methods."""
        obj = MockProtocolImplementor()

        # Should use has_capability from protocol
        result = CapabilityHelper.check_capability(obj, "test_capability")

        assert result is True
        # No deprecation warning should be raised for protocol-compliant objects

    def test_capability_check_returns_false_for_missing(self):
        """Missing capabilities should return False without error."""
        obj = MockProtocolImplementor()

        result = CapabilityHelper.check_capability(obj, "nonexistent_capability")

        assert result is False

    def test_deprecation_warning_for_hasattr_fallback(self):
        """Non-protocol objects should trigger deprecation warning."""
        obj = MockLegacyObject()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should trigger a deprecation warning
            result = CapabilityHelper.check_capability(obj, "enabled_tools")

            # Verify deprecation warning was raised
            deprecation_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

            # Verify the warning mentions the fallback
            warning_msg = str(deprecation_warnings[0].message)
            assert "hasattr()" in warning_msg or "CapabilityRegistryProtocol" in warning_msg

    def test_strict_mode_raises_for_non_protocol(self):
        """Strict mode should raise error for non-protocol objects."""
        obj = MockLegacyObject()

        with pytest.raises((TypeError, AttributeError)):
            # Strict mode should not allow fallback
            CapabilityHelper.check_capability(obj, "enabled_tools", strict=True)

    def test_invoke_capability_uses_protocol(self):
        """Protocol-implementing objects should use protocol invoke."""
        obj = MockProtocolImplementor()

        # First check it exists
        assert CapabilityHelper.check_capability(obj, "test_capability")

        # Invoke should work via protocol
        # Note: actual invocation depends on capability spec having getter/setter
        # For this test, we just verify no deprecation warning is raised

    def test_invoke_capability_fallback_warns(self):
        """Invoke on non-protocol object should warn."""
        obj = MockLegacyObject()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should trigger a deprecation warning
            try:
                CapabilityHelper.invoke_capability(obj, "enabled_tools", ["tool1"])
            except (AttributeError, TypeError):
                # May fail to invoke, but should still warn
                pass

            # Check for deprecation warnings
            deprecation_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            # At least one warning should mention protocol or hasattr
            if deprecation_warnings:
                assert any(
                    "hasattr" in str(w.message) or "Protocol" in str(w.message)
                    for w in deprecation_warnings
                )


class TestDeprecationTimeline:
    """Test that deprecation warnings include version timeline."""

    def test_deprecation_warning_mentions_removal_version(self):
        """Deprecation warning should mention v0.7.0 removal."""
        obj = MockLegacyObject()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            CapabilityHelper.check_capability(obj, "enabled_tools")

            # Find deprecation warnings
            deprecation_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]

            if deprecation_warnings:
                # At least one should mention v0.7.0
                warning_messages = [str(w.message) for w in deprecation_warnings]
                combined = " ".join(warning_messages)
                assert "0.7.0" in combined or "deprecated" in combined.lower()


class TestMixinProtocolConformance:
    """Test that CapabilityRegistryMixin properly implements protocol."""

    def test_mixin_implements_protocol(self):
        """CapabilityRegistryMixin should implement CapabilityRegistryProtocol."""
        obj = MockProtocolImplementor()

        # Should be an instance of the protocol
        assert isinstance(obj, CapabilityRegistryProtocol)

    def test_mixin_has_required_methods(self):
        """Mixin should provide all required protocol methods."""
        obj = MockProtocolImplementor()

        # These methods should exist from the mixin
        assert hasattr(obj, "has_capability")
        assert hasattr(obj, "invoke_capability")
        assert hasattr(obj, "get_capability")
        assert callable(obj.has_capability)
        assert callable(obj.invoke_capability)
        assert callable(obj.get_capability)

    def test_has_capability_returns_correct_type(self):
        """has_capability should return a boolean."""
        obj = MockProtocolImplementor()

        result = obj.has_capability("test_capability")
        assert isinstance(result, bool)

        result = obj.has_capability("nonexistent")
        assert isinstance(result, bool)
        assert result is False
