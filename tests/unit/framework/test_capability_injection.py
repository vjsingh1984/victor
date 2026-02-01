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

"""Unit tests for capability injection via DI.

TDD-first tests for Phase 1.4: Auto-inject FileOperationsCapability via DI.
These tests verify:
1. FileOperationsCapability protocol compliance
2. Capability provider registration in DI
3. Singleton scope for capability instances
4. Vertical access to injected capabilities
"""

from unittest.mock import Mock

from victor.framework.capabilities import FileOperationsCapability


class TestFileOperationsCapabilityProtocol:
    """Tests for FileOperationsCapability protocol compliance."""

    def test_protocol_exists(self):
        """CapabilityProviderProtocol should exist."""
        from victor.protocols.capability_provider import FileOperationsCapabilityProtocol

        assert FileOperationsCapabilityProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        from victor.protocols.capability_provider import FileOperationsCapabilityProtocol

        assert hasattr(FileOperationsCapabilityProtocol, "__protocol_attrs__") or hasattr(
            FileOperationsCapabilityProtocol, "_is_protocol"
        )

    def test_capability_implements_protocol(self):
        """FileOperationsCapability should implement protocol."""

        capability = FileOperationsCapability()

        # Verify required methods exist
        assert hasattr(capability, "get_tools")
        assert hasattr(capability, "get_tool_list")

    def test_protocol_methods_match_implementation(self):
        """Protocol methods should match implementation."""
        capability = FileOperationsCapability()

        # Test get_tools
        tools = capability.get_tools()
        assert isinstance(tools, set)
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "grep" in tools

        # Test get_tool_list
        tool_list = capability.get_tool_list()
        assert isinstance(tool_list, list)
        assert len(tool_list) == 4


class TestCapabilityInjector:
    """Tests for CapabilityInjector DI integration."""

    def test_injector_provides_file_operations(self):
        """Injector should provide FileOperationsCapability."""
        from victor.core.verticals.capability_injector import CapabilityInjector

        container = Mock()
        injector = CapabilityInjector(container)

        capability = injector.get_file_operations_capability()

        assert capability is not None
        assert isinstance(capability, FileOperationsCapability)

    def test_injector_returns_singleton(self):
        """Injector should return same instance each time."""
        from victor.core.verticals.capability_injector import CapabilityInjector

        container = Mock()
        injector = CapabilityInjector(container)

        cap1 = injector.get_file_operations_capability()
        cap2 = injector.get_file_operations_capability()

        assert cap1 is cap2

    def test_injector_allows_custom_capability(self):
        """Injector should allow custom capability override."""
        from victor.core.verticals.capability_injector import CapabilityInjector

        custom_capability = FileOperationsCapability()
        container = Mock()

        injector = CapabilityInjector(
            container,
            file_operations=custom_capability,
        )

        assert injector.get_file_operations_capability() is custom_capability


class TestCapabilityDIRegistration:
    """Tests for DI container registration."""

    def test_file_operations_registered_in_container(self):
        """FileOperationsCapability should be registered in DI container."""
        from victor.protocols.capability_provider import FileOperationsCapabilityProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        assert container.is_registered(FileOperationsCapabilityProtocol)

    def test_resolve_capability_from_container(self):
        """Should be able to resolve FileOperationsCapability from container."""
        from victor.protocols.capability_provider import FileOperationsCapabilityProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        capability = container.get(FileOperationsCapabilityProtocol)

        assert capability is not None
        assert hasattr(capability, "get_tools")
        assert hasattr(capability, "get_tool_list")

    def test_capability_singleton_scope(self):
        """Capability should be registered as singleton."""
        from victor.protocols.capability_provider import FileOperationsCapabilityProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        cap1 = container.get(FileOperationsCapabilityProtocol)
        cap2 = container.get(FileOperationsCapabilityProtocol)

        assert cap1 is cap2


class TestCapabilityIntegration:
    """Integration tests for capability injection."""

    def test_file_operations_tools_match_expected(self):
        """FileOperationsCapability tools should match expected defaults."""
        capability = FileOperationsCapability()

        tools = capability.get_tool_list()

        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "grep" in tools
        assert len(tools) == 4

    def test_capability_injector_global_instance(self):
        """Should be able to get global CapabilityInjector instance."""
        from victor.core.verticals.capability_injector import get_capability_injector

        injector = get_capability_injector()

        assert injector is not None
        capability = injector.get_file_operations_capability()
        assert capability is not None

    def test_capability_can_be_used_in_vertical(self):
        """Capability should work correctly when used in vertical context."""
        from victor.core.verticals.capability_injector import get_capability_injector

        injector = get_capability_injector()
        capability = injector.get_file_operations_capability()

        # Simulate vertical tool collection
        tools = list(capability.get_tools())

        assert len(tools) == 4
        assert all(isinstance(t, str) for t in tools)


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    def test_direct_instantiation_still_works(self):
        """Direct FileOperationsCapability() should still work."""
        capability = FileOperationsCapability()

        assert capability.get_tool_list() == ["read", "write", "edit", "grep"]

    def test_custom_operations_still_works(self):
        """Custom operations list should still work."""
        from victor.framework.capabilities.file_operations import (
            FileOperation,
            FileOperationType,
        )

        custom_ops = [
            FileOperation(FileOperationType.READ, "custom_read", required=True),
        ]

        capability = FileOperationsCapability(operations=custom_ops)

        assert capability.get_tool_list() == ["custom_read"]
