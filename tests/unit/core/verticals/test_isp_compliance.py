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

"""Tests for ISP (Interface Segregation Principle) compliance in verticals.

This test suite verifies that verticals can implement only the protocols
they need, without being forced to implement all possible capabilities.
"""

import pytest
from typing import List, Dict, Any, Optional

from victor.core.verticals.base import VerticalBase
from victor.core.verticals.protocols.providers import (
    ToolProvider,
    PromptContributorProvider,
    MiddlewareProvider,
    SafetyProvider,
    WorkflowProvider,
    TeamProvider,
)
from victor.core.verticals.protocol_loader import ProtocolBasedExtensionLoader


# =============================================================================
# Test Fixtures
# =============================================================================


class MinimalVertical(VerticalBase):
    """Minimal vertical implementing only ToolProvider protocol."""

    name = "minimal_test"
    description = "Minimal test vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a minimal test assistant..."


class MultiProtocolVertical(VerticalBase):
    """Vertical implementing multiple protocols."""

    name = "multi_protocol_test"
    description = "Multi-protocol test vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "grep"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a multi-protocol test assistant..."

    @classmethod
    def get_middleware(cls) -> List[Any]:
        return []  # Empty middleware list

    @classmethod
    def get_safety_extension(cls) -> Optional[Any]:
        return None  # No safety extension


class ComprehensiveVertical(VerticalBase):
    """Vertical implementing many protocols."""

    name = "comprehensive_test"
    description = "Comprehensive test vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "grep", "web_search"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a comprehensive test assistant..."

    @classmethod
    def get_middleware(cls) -> List[Any]:
        return []

    @classmethod
    def get_safety_extension(cls) -> Optional[Any]:
        return None

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        return None

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        return {
            "edit": {"hint": "Edit files", "priority_tools": ["write"]},
        }


# =============================================================================
# Protocol Registration Tests
# =============================================================================


class TestProtocolRegistration:
    """Tests for protocol registration functionality."""

    def test_register_single_protocol(self):
        """Test registering a single protocol."""
        # Register protocol
        MinimalVertical.register_protocol(ToolProvider)

        # Verify registration via loader
        assert ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, ToolProvider)

        # Verify registration via VerticalBase method
        assert MinimalVertical.implements_protocol(ToolProvider)

    def test_register_multiple_protocols(self):
        """Test registering multiple protocols."""
        # Register multiple protocols
        MultiProtocolVertical.register_protocol(ToolProvider)
        MultiProtocolVertical.register_protocol(MiddlewareProvider)
        MultiProtocolVertical.register_protocol(SafetyProvider)

        # Verify all are registered via loader
        assert ProtocolBasedExtensionLoader.implements_protocol(MultiProtocolVertical, ToolProvider)
        assert ProtocolBasedExtensionLoader.implements_protocol(
            MultiProtocolVertical, MiddlewareProvider
        )
        assert ProtocolBasedExtensionLoader.implements_protocol(
            MultiProtocolVertical, SafetyProvider
        )

        # Verify all are registered via VerticalBase method
        assert MultiProtocolVertical.implements_protocol(ToolProvider)
        assert MultiProtocolVertical.implements_protocol(MiddlewareProvider)
        assert MultiProtocolVertical.implements_protocol(SafetyProvider)

    def test_register_comprehensive_protocols(self):
        """Test registering comprehensive set of protocols."""
        # Register all implemented protocols
        ComprehensiveVertical.register_protocol(ToolProvider)
        ComprehensiveVertical.register_protocol(MiddlewareProvider)
        ComprehensiveVertical.register_protocol(SafetyProvider)
        ComprehensiveVertical.register_protocol(WorkflowProvider)
        ComprehensiveVertical.register_protocol(PromptContributorProvider)

        # Verify all are registered
        protocols = [
            ToolProvider,
            MiddlewareProvider,
            SafetyProvider,
            WorkflowProvider,
            PromptContributorProvider,
        ]

        for protocol in protocols:
            assert ProtocolBasedExtensionLoader.implements_protocol(ComprehensiveVertical, protocol)
            assert ComprehensiveVertical.implements_protocol(protocol)

    def test_list_implemented_protocols(self):
        """Test listing all implemented protocols."""
        # Register protocols
        MinimalVertical.register_protocol(ToolProvider)

        # List protocols
        protocols = MinimalVertical.list_implemented_protocols()

        # Verify ToolProvider is in the list
        assert ToolProvider in protocols

    def test_unregistered_protocol_returns_false(self):
        """Test that unregistered protocols return False."""
        # Don't register TeamProvider
        # (MinimalVertical doesn't implement it)

        # Verify it's not implemented
        assert not ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, TeamProvider)
        assert not MinimalVertical.implements_protocol(TeamProvider)

    def test_protocol_registration_persistence(self):
        """Test that protocol registration persists across checks."""
        # Register protocol
        MinimalVertical.register_protocol(ToolProvider)

        # Check multiple times
        for _ in range(3):
            assert ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, ToolProvider)


# =============================================================================
# Protocol Method Tests
# =============================================================================


class TestProtocolMethods:
    """Tests for protocol method access and execution."""

    def test_tool_provider_methods(self):
        """Test ToolProvider protocol methods."""
        MinimalVertical.register_protocol(ToolProvider)

        # Verify method exists and is callable
        assert hasattr(MinimalVertical, "get_tools")
        assert callable(MinimalVertical.get_tools)

        # Call the method
        tools = MinimalVertical.get_tools()
        assert "read" in tools
        assert "write" in tools

    def test_middleware_provider_methods(self):
        """Test MiddlewareProvider protocol methods."""
        MultiProtocolVertical.register_protocol(MiddlewareProvider)

        # Verify method exists
        assert hasattr(MultiProtocolVertical, "get_middleware")
        assert callable(MultiProtocolVertical.get_middleware)

        # Call the method
        middleware = MultiProtocolVertical.get_middleware()
        assert isinstance(middleware, list)

    def test_prompt_contributor_methods(self):
        """Test PromptContributorProvider protocol methods."""
        ComprehensiveVertical.register_protocol(PromptContributorProvider)

        # Verify method exists
        assert hasattr(ComprehensiveVertical, "get_task_type_hints")
        assert callable(ComprehensiveVertical.get_task_type_hints)

        # Call the method
        hints = ComprehensiveVertical.get_task_type_hints()
        assert isinstance(hints, dict)
        assert "edit" in hints


# =============================================================================
# Protocol Filtering Tests
# =============================================================================


class TestProtocolFiltering:
    """Tests for protocol-based capability filtering."""

    def test_filter_verticals_by_protocol(self):
        """Test filtering verticals by protocol implementation."""
        # Register protocols
        MinimalVertical.register_protocol(ToolProvider)
        MultiProtocolVertical.register_protocol(ToolProvider)
        MultiProtocolVertical.register_protocol(MiddlewareProvider)

        # Get all ToolProvider implementations
        tool_providers = ProtocolBasedExtensionLoader.list_verticals(ToolProvider)

        # Verify both are in the list
        assert MinimalVertical in tool_providers
        assert MultiProtocolVertical in tool_providers

        # Get all MiddlewareProvider implementations
        middleware_providers = ProtocolBasedExtensionLoader.list_verticals(MiddlewareProvider)

        # Verify only MultiProtocolVertical is in the list
        assert MultiProtocolVertical in middleware_providers
        assert MinimalVertical not in middleware_providers

    def test_conditional_protocol_usage(self):
        """Test conditional usage based on protocol support."""
        # Register protocols
        MinimalVertical.register_protocol(ToolProvider)
        MultiProtocolVertical.register_protocol(ToolProvider)
        MultiProtocolVertical.register_protocol(MiddlewareProvider)

        # Simulate framework code that conditionally uses protocols
        def process_vertical(vertical_class):
            """Process a vertical based on its protocol support."""
            result = {"tools": None, "middleware": None}

            # All verticals have ToolProvider
            if vertical_class.implements_protocol(ToolProvider):
                result["tools"] = vertical_class.get_tools()

            # Only some have MiddlewareProvider
            if vertical_class.implements_protocol(MiddlewareProvider):
                result["middleware"] = vertical_class.get_middleware()

            return result

        # Process MinimalVertical (only ToolProvider)
        minimal_result = process_vertical(MinimalVertical)
        assert minimal_result["tools"] == ["read", "write"]
        assert minimal_result["middleware"] is None

        # Process MultiProtocolVertical (ToolProvider + MiddlewareProvider)
        multi_result = process_vertical(MultiProtocolVertical)
        assert multi_result["tools"] == ["read", "write", "grep"]
        assert multi_result["middleware"] == []


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing VerticalBase usage."""

    def test_vertical_without_protocol_registration(self):
        """Test that verticals without protocol registration still work."""

        # Create a vertical without explicit protocol registration
        class LegacyVertical(VerticalBase):
            name = "legacy_test"
            description = "Legacy test vertical"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Legacy assistant..."

        # Should still be able to call methods
        tools = LegacyVertical.get_tools()
        assert tools == ["read"]

        prompt = LegacyVertical.get_system_prompt()
        assert "Legacy" in prompt

    def test_base_methods_still_work(self):
        """Test that base VerticalBase methods still work with protocols."""
        MinimalVertical.register_protocol(ToolProvider)

        # Should still be able to use base methods
        config = MinimalVertical.get_config()
        assert config is not None
        assert config.tools is not None

        stages = MinimalVertical.get_stages()
        assert isinstance(stages, dict)


# =============================================================================
# Protocol Cache Tests
# =============================================================================


class TestProtocolCaching:
    """Tests for protocol caching and invalidation."""

    def test_protocol_conformance_cache(self):
        """Test that protocol conformance is cached."""
        MinimalVertical.register_protocol(ToolProvider)

        # First check - caches the result
        result1 = ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, ToolProvider)

        # Second check - uses cache
        result2 = ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, ToolProvider)

        # Should return same result
        assert result1 == result2 == True

    def test_clear_protocol_cache(self):
        """Test clearing protocol cache."""
        MinimalVertical.register_protocol(ToolProvider)

        # Check and cache
        assert ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, ToolProvider)

        # Clear cache
        ProtocolBasedExtensionLoader.clear_cache(
            vertical_class=MinimalVertical,
            protocol_type=ToolProvider,
        )

        # Check again after cache clear
        assert ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, ToolProvider)

    def test_unregister_protocol(self):
        """Test unregistering a protocol."""
        MinimalVertical.register_protocol(ToolProvider)

        # Verify registered
        assert ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, ToolProvider)

        # Unregister
        ProtocolBasedExtensionLoader.unregister_protocol(ToolProvider, MinimalVertical)

        # Verify no longer registered
        assert not ProtocolBasedExtensionLoader.implements_protocol(MinimalVertical, ToolProvider)


# =============================================================================
# Integration Tests
# =============================================================================


class TestISPIntegration:
    """Integration tests for ISP compliance."""

    def test_real_world_vertical_scenario(self):
        """Test a real-world vertical scenario."""

        # Simulate a research-like vertical
        class ResearchLikeVertical(VerticalBase):
            name = "research_like_test"
            description = "Research-like test vertical"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["web_search", "web_fetch", "read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Research assistant..."

            @classmethod
            def get_task_type_hints(cls) -> Dict[str, Any]:
                return {
                    "search": {"hint": "Use web search"},
                    "synthesize": {"hint": "Combine sources"},
                }

        # Register protocols
        ResearchLikeVertical.register_protocol(ToolProvider)
        ResearchLikeVertical.register_protocol(PromptContributorProvider)

        # Verify protocol support
        assert ResearchLikeVertical.implements_protocol(ToolProvider)
        assert ResearchLikeVertical.implements_protocol(PromptContributorProvider)
        assert not ResearchLikeVertical.implements_protocol(TeamProvider)

        # Use protocols
        tools = ResearchLikeVertical.get_tools()
        assert "web_search" in tools

        hints = ResearchLikeVertical.get_task_type_hints()
        assert "search" in hints

    def test_framework_protocol_detection(self):
        """Test how framework code would detect protocol support."""
        # Register different protocols
        MinimalVertical.register_protocol(ToolProvider)
        MultiProtocolVertical.register_protocol(ToolProvider)
        MultiProtocolVertical.register_protocol(MiddlewareProvider)

        # Framework code: collect all tool providers
        tool_providers = [
            v
            for v in [MinimalVertical, MultiProtocolVertical]
            if v.implements_protocol(ToolProvider)
        ]
        assert len(tool_providers) == 2

        # Framework code: collect all middleware providers
        middleware_providers = [
            v
            for v in [MinimalVertical, MultiProtocolVertical]
            if v.implements_protocol(MiddlewareProvider)
        ]
        assert len(middleware_providers) == 1
        assert MultiProtocolVertical in middleware_providers
