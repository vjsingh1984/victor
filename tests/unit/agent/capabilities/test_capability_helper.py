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

"""Tests for CapabilityHelper (Phase 1.2: Capability Helpers Consolidation).

Tests that CapabilityHelper:
1. Validates protocol conformance
2. Invokes capabilities correctly
3. Maintains backward compatibility
4. Issues deprecation warnings
"""

import warnings
from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock

import pytest

from victor.framework.protocols import CapabilityRegistryProtocol


class MockOrchestrator(CapabilityRegistryProtocol):
    """Mock orchestrator implementing CapabilityRegistryProtocol."""

    def __init__(self):
        self._capabilities: Dict[str, bool] = {
            "enabled_tools": True,
            "prompt_builder": True,
        }
        self._invoked: Dict[str, Any] = {}
        self._versions: Dict[str, str] = {
            "enabled_tools": "1.0",
            "prompt_builder": "2.0",
        }

    def has_capability(self, name: str, min_version: Optional[str] = None) -> bool:
        if name not in self._capabilities:
            return False
        if min_version:
            actual = self._versions.get(name, "1.0")
            return actual >= min_version
        return self._capabilities.get(name, False)

    def invoke_capability(
        self, name: str, *args: Any, min_version: Optional[str] = None, **kwargs: Any
    ) -> Any:
        self._invoked[name] = (args, kwargs)
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return dict(self._capabilities)


class NonProtocolObject:
    """Object that doesn't implement CapabilityRegistryProtocol."""

    def set_enabled_tools(self, tools: Set[str]) -> None:
        self.enabled_tools = tools


class TestCapabilityHelper:
    """Test CapabilityHelper class."""

    def test_check_capability_validates_protocol(self):
        """check_capability should use protocol when available."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = MockOrchestrator()
        assert CapabilityHelper.check_capability(obj, "enabled_tools") is True
        assert CapabilityHelper.check_capability(obj, "unknown_capability") is False

    def test_check_capability_with_min_version(self):
        """check_capability should validate version requirements."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = MockOrchestrator()
        # enabled_tools is version 1.0
        assert CapabilityHelper.check_capability(obj, "enabled_tools", min_version="1.0") is True
        assert CapabilityHelper.check_capability(obj, "enabled_tools", min_version="2.0") is False
        # prompt_builder is version 2.0
        assert CapabilityHelper.check_capability(obj, "prompt_builder", min_version="2.0") is True

    def test_invoke_capability_calls_correct_method(self):
        """invoke_capability should call the correct method."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = MockOrchestrator()
        result = CapabilityHelper.invoke_capability(obj, "enabled_tools", {"read", "write"})
        assert result is True
        assert "enabled_tools" in obj._invoked
        assert obj._invoked["enabled_tools"][0] == ({"read", "write"},)

    def test_invoke_capability_with_kwargs(self):
        """invoke_capability should pass kwargs correctly."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = MockOrchestrator()
        result = CapabilityHelper.invoke_capability(
            obj, "enabled_tools", {"read"}, extra="value"
        )
        assert result is True
        assert obj._invoked["enabled_tools"][1] == {"extra": "value"}

    def test_backward_compatibility_with_existing_calls(self):
        """CapabilityHelper should be usable as a drop-in replacement."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = MockOrchestrator()

        # Pattern 1: Simple check and invoke
        if CapabilityHelper.check_capability(obj, "enabled_tools"):
            CapabilityHelper.invoke_capability(obj, "enabled_tools", {"read"})

        assert "enabled_tools" in obj._invoked

    def test_deprecation_warning_for_non_protocol_object(self):
        """Deprecation warning should be issued for non-protocol objects."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = NonProtocolObject()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This should issue a deprecation warning
            CapabilityHelper.check_capability(obj, "enabled_tools")

            # Check if deprecation warning was issued
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0

    def test_strict_mode_raises_for_non_protocol(self):
        """Strict mode should raise TypeError for non-protocol objects."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = NonProtocolObject()

        with pytest.raises(TypeError) as exc_info:
            CapabilityHelper.check_capability(obj, "enabled_tools", strict=True)

        assert "CapabilityRegistryProtocol" in str(exc_info.value)


class TestCapabilityHelperEdgeCases:
    """Test edge cases for CapabilityHelper."""

    def test_check_empty_capability_name(self):
        """check_capability should handle empty capability names."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = MockOrchestrator()
        assert CapabilityHelper.check_capability(obj, "") is False

    def test_invoke_nonexistent_capability(self):
        """invoke_capability should handle non-existent capabilities gracefully."""
        from victor.agent.capability_registry import CapabilityHelper

        obj = MockOrchestrator()

        # Should return None or raise KeyError
        with pytest.raises((KeyError, TypeError, AttributeError)):
            CapabilityHelper.invoke_capability(
                obj, "nonexistent_capability", "value", strict=True
            )

    def test_helper_methods_are_static(self):
        """Helper methods should be static (no instance required)."""
        from victor.agent.capability_registry import CapabilityHelper

        # Should be callable without instantiation
        assert callable(CapabilityHelper.check_capability)
        assert callable(CapabilityHelper.invoke_capability)

        # Should work with staticmethod
        assert isinstance(
            CapabilityHelper.__dict__["check_capability"], staticmethod
        )
        assert isinstance(
            CapabilityHelper.__dict__["invoke_capability"], staticmethod
        )
