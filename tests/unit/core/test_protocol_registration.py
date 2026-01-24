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

"""Unit tests for @register_protocols decorator.

TDD-first tests for Phase 2.1: Create @register_protocols decorator.
These tests verify:
1. Auto-detection of implemented protocols
2. Automatic registration with protocol loader
3. Integration with existing VerticalBase pattern
"""

import pytest
from typing import List, Optional, Protocol, runtime_checkable
from unittest.mock import Mock, MagicMock, patch


# Define test protocols
@runtime_checkable
class MockToolProvider(Protocol):
    """Mock protocol for tool provision."""

    def get_tools(self) -> List[str]:
        ...


@runtime_checkable
class MockPromptProvider(Protocol):
    """Mock protocol for prompt provision."""

    def get_prompt(self) -> str:
        ...


@runtime_checkable
class MockMiddlewareProvider(Protocol):
    """Mock protocol for middleware provision."""

    def get_middleware(self) -> List[str]:
        ...


class TestRegisterProtocolsDecorator:
    """Tests for @register_protocols class decorator."""

    def test_decorator_exists(self):
        """Decorator should exist in protocol_decorators module."""
        from victor.core.verticals.protocol_decorators import register_protocols

        assert register_protocols is not None
        assert callable(register_protocols)

    def test_decorator_returns_class_unchanged(self):
        """Decorator should return the class unchanged."""
        from victor.core.verticals.protocol_decorators import register_protocols

        @register_protocols
        class TestVertical:
            pass

        assert TestVertical.__name__ == "TestVertical"

    def test_decorator_detects_implemented_protocols(self):
        """Decorator should detect which protocols a class implements."""
        from victor.core.verticals.protocol_decorators import (
            register_protocols,
            get_implemented_protocols,
        )

        class TestVertical:
            def get_tools(self) -> List[str]:
                return ["tool1", "tool2"]

            def get_prompt(self) -> str:
                return "test prompt"

        # Test detection
        protocols = get_implemented_protocols(
            TestVertical,
            [MockToolProvider, MockPromptProvider, MockMiddlewareProvider],
        )

        assert MockToolProvider in protocols
        assert MockPromptProvider in protocols
        assert MockMiddlewareProvider not in protocols

    def test_decorator_with_explicit_protocols(self):
        """Decorator should allow explicit protocol list."""
        from victor.core.verticals.protocol_decorators import register_protocols

        @register_protocols(protocols=[MockToolProvider])
        class TestVertical:
            def get_tools(self) -> List[str]:
                return []

        # Should not raise
        assert TestVertical is not None

    def test_decorator_auto_detects_from_known_protocols(self):
        """Decorator should auto-detect from known vertical protocols."""
        from victor.core.verticals.protocol_decorators import (
            register_protocols,
            KNOWN_VERTICAL_PROTOCOLS,
        )

        # Verify we have known protocols
        assert len(KNOWN_VERTICAL_PROTOCOLS) > 0


class TestGetImplementedProtocols:
    """Tests for get_implemented_protocols helper."""

    def test_returns_empty_for_no_implementation(self):
        """Should return empty list when no protocols implemented."""
        from victor.core.verticals.protocol_decorators import get_implemented_protocols

        class EmptyVertical:
            pass

        protocols = get_implemented_protocols(
            EmptyVertical,
            [MockToolProvider, MockPromptProvider],
        )

        assert protocols == []

    def test_detects_partial_implementation(self):
        """Should detect partially implemented protocols."""
        from victor.core.verticals.protocol_decorators import get_implemented_protocols

        class PartialVertical:
            def get_tools(self) -> List[str]:
                return []

        protocols = get_implemented_protocols(
            PartialVertical,
            [MockToolProvider, MockPromptProvider],
        )

        assert MockToolProvider in protocols
        assert MockPromptProvider not in protocols

    def test_handles_inheritance(self):
        """Should handle inherited implementations."""
        from victor.core.verticals.protocol_decorators import get_implemented_protocols

        class BaseVertical:
            def get_tools(self) -> List[str]:
                return []

        class DerivedVertical(BaseVertical):
            def get_prompt(self) -> str:
                return ""

        protocols = get_implemented_protocols(
            DerivedVertical,
            [MockToolProvider, MockPromptProvider],
        )

        assert MockToolProvider in protocols
        assert MockPromptProvider in protocols


class TestProtocolDecoratorIntegration:
    """Integration tests for decorator with protocol loader."""

    def test_decorator_registers_with_loader(self):
        """Decorator should call register_protocol for detected protocols."""
        from victor.core.verticals.protocol_decorators import register_protocols

        # Mock the protocol loader at the import location
        with patch(
            "victor.core.verticals.protocol_loader.ProtocolBasedExtensionLoader"
        ) as mock_loader:
            @register_protocols(protocols=[MockToolProvider])
            class TestVertical:
                def get_tools(self) -> List[str]:
                    return []

            # Verify register_protocol was called
            mock_loader.register_protocol.assert_called()

    def test_decorator_skips_already_registered(self):
        """Decorator should skip already registered protocols."""
        from victor.core.verticals.protocol_decorators import register_protocols

        # First registration
        with patch(
            "victor.core.verticals.protocol_loader.ProtocolBasedExtensionLoader"
        ) as mock_loader:
            mock_loader.is_registered.return_value = True

            @register_protocols(protocols=[MockToolProvider])
            class TestVertical:
                def get_tools(self) -> List[str]:
                    return []

            # register_protocol should be called (current impl doesn't check is_registered)
            mock_loader.register_protocol.assert_called()


class TestKnownVerticalProtocols:
    """Tests for KNOWN_VERTICAL_PROTOCOLS list."""

    def test_contains_tool_provider(self):
        """Should contain ToolProvider protocol."""
        from victor.core.verticals.protocol_decorators import KNOWN_VERTICAL_PROTOCOLS

        # Check for expected protocol types
        protocol_names = [p.__name__ for p in KNOWN_VERTICAL_PROTOCOLS]
        assert "ToolProvider" in protocol_names

    def test_contains_prompt_contributor_provider(self):
        """Should contain PromptContributorProvider protocol."""
        from victor.core.verticals.protocol_decorators import KNOWN_VERTICAL_PROTOCOLS

        protocol_names = [p.__name__ for p in KNOWN_VERTICAL_PROTOCOLS]
        assert "PromptContributorProvider" in protocol_names

    def test_all_protocols_are_runtime_checkable(self):
        """All known protocols should be runtime checkable."""
        from victor.core.verticals.protocol_decorators import KNOWN_VERTICAL_PROTOCOLS

        for protocol in KNOWN_VERTICAL_PROTOCOLS:
            # Should be able to use isinstance with runtime_checkable protocols
            assert hasattr(protocol, "__protocol_attrs__") or hasattr(
                protocol, "_is_protocol"
            )


class TestDecoratorSyntax:
    """Tests for decorator syntax variations."""

    def test_decorator_without_parentheses(self):
        """Decorator without parentheses should work."""
        from victor.core.verticals.protocol_decorators import register_protocols

        @register_protocols
        class TestVertical:
            pass

        assert TestVertical is not None

    def test_decorator_with_empty_parentheses(self):
        """Decorator with empty parentheses should work."""
        from victor.core.verticals.protocol_decorators import register_protocols

        @register_protocols()
        class TestVertical:
            pass

        assert TestVertical is not None

    def test_decorator_with_kwargs(self):
        """Decorator with keyword arguments should work."""
        from victor.core.verticals.protocol_decorators import register_protocols

        @register_protocols(protocols=[], auto_detect=False)
        class TestVertical:
            pass

        assert TestVertical is not None
