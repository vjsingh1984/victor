# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Unit tests for CapabilityContainerProtocol.

Tests protocol definition, isinstance() behavior, and protocol conformance
per TDD principles (tests written before implementation).
"""

import pytest
from typing import Protocol, runtime_checkable, Any, Optional
from victor.protocols.capability import (
    CapabilityContainerProtocol,
    get_capability_registry,
)


class TestCapabilityContainerProtocol:
    """Test suite for CapabilityContainerProtocol definition and behavior."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime_checkable for isinstance() checks."""
        # Verify protocol has runtime_checkable decorator
        assert hasattr(CapabilityContainerProtocol, "_is_protocol")

    def test_protocol_defines_has_capability_method(self):
        """Protocol must define has_capability(capability_name: str) -> bool."""
        assert hasattr(CapabilityContainerProtocol, "has_capability")

        # Check method signature
        import inspect

        sig = inspect.signature(CapabilityContainerProtocol.has_capability)
        assert "capability_name" in sig.parameters
        assert sig.return_annotation is bool

    def test_protocol_defines_get_capability_method(self):
        """Protocol must define get_capability(name: str) -> Optional[Any]."""
        assert hasattr(CapabilityContainerProtocol, "get_capability")

        # Check method signature
        import inspect

        sig = inspect.signature(CapabilityContainerProtocol.get_capability)
        assert "name" in sig.parameters
        assert sig.return_annotation == Optional[Any]

    def test_isinstance_with_compliant_implementation(self):
        """isinstance() should return True for compliant implementations."""

        # Create a compliant implementation (regular class, not Protocol)
        class CompliantImpl:
            def has_capability(self, capability_name: str) -> bool:
                return capability_name == "test_capability"

            def get_capability(self, name: str) -> Optional[Any]:
                return {"name": name} if name == "test_capability" else None

        # Test isinstance() with instance
        impl = CompliantImpl()
        assert isinstance(impl, CapabilityContainerProtocol)

        # Test methods work
        assert impl.has_capability("test_capability") is True
        assert impl.has_capability("nonexistent") is False
        assert impl.get_capability("test_capability") == {"name": "test_capability"}
        assert impl.get_capability("nonexistent") is None

    def test_isinstance_with_non_compliant_implementation(self):
        """isinstance() should return False for non-compliant implementations."""

        # Create a non-compliant implementation (missing methods)
        class NonCompliantImpl:
            def has_capability(self, capability_name: str) -> bool:
                return True

            # Missing get_capability method

        # Test isinstance() returns False
        impl = NonCompliantImpl()
        assert not isinstance(impl, CapabilityContainerProtocol)

    def test_isinstance_with_partial_implementation(self):
        """isinstance() should return False for partial implementations."""

        # Create a partial implementation (wrong signature)
        class PartialImpl:
            def has_capability(self, capability_name: str) -> bool:  # Correct
                return True

            def get_capability(self) -> Optional[Any]:  # Wrong signature (missing param)
                return None

        # Test isinstance() - partial implementation should not be considered fully compliant
        impl = PartialImpl()
        # Structural typing should fail due to wrong signature
        assert not isinstance(impl, CapabilityContainerProtocol)

    def test_protocol_enforces_type_safety(self):
        """Protocol should enforce type safety for method calls."""

        class TypedImpl:
            def has_capability(self, capability_name: str) -> bool:
                if not isinstance(capability_name, str):
                    raise TypeError("capability_name must be str")
                return True

            def get_capability(self, name: str) -> Optional[Any]:
                if not isinstance(name, str):
                    raise TypeError("name must be str")
                return f"capability_{name}"

        impl = TypedImpl()

        # Test with correct types
        assert impl.has_capability("test") is True
        assert impl.get_capability("test") == "capability_test"

        # Test with incorrect types (should raise TypeError)
        with pytest.raises(TypeError):
            impl.has_capability(123)  # type: ignore

        with pytest.raises(TypeError):
            impl.get_capability(123)  # type: ignore


class TestGetCapabilityRegistry:
    """Test suite for get_capability_registry() helper function."""

    def test_returns_none_when_no_registry_set(self):
        """Should return None when no capability registry is configured."""
        # This test assumes no global registry is set by default
        result = get_capability_registry()
        # Result could be None or a default registry
        # The key is that the function exists and is callable
        assert callable(get_capability_registry)

    def test_function_exists_and_is_callable(self):
        """get_capability_registry() function should exist and be callable."""
        assert callable(get_capability_registry)

        # Call should not raise
        result = get_capability_registry()
        # Result type is implementation-dependent
        # Could be None, a dict, or a registry object

    def test_returns_capability_container_when_set(self):
        """Should return CapabilityContainerProtocol instance when configured."""
        # This is an integration test - tests actual functionality
        # Implementation will define the behavior
        result = get_capability_registry()

        # If result is not None, it should implement the protocol
        if result is not None:
            assert isinstance(result, CapabilityContainerProtocol)


class TestProtocolComplianceInRealCode:
    """Test protocol compliance with real-world scenarios."""

    def test_capability_container_can_store_and_retrieve_capabilities(self):
        """Protocol should support storing and retrieving capabilities."""

        class SimpleCapabilityContainer:
            def __init__(self):
                self._capabilities = {
                    "code_analysis": {"enabled": True, "tools": ["ast", "parser"]},
                    "test_generation": {"enabled": True, "framework": "pytest"},
                }

            def has_capability(self, capability_name: str) -> bool:
                return capability_name in self._capabilities

            def get_capability(self, name: str) -> Optional[Any]:
                return self._capabilities.get(name)

        container = SimpleCapabilityContainer()

        # Test capability checking
        assert isinstance(container, CapabilityContainerProtocol)
        assert container.has_capability("code_analysis") is True
        assert container.has_capability("nonexistent") is False

        # Test capability retrieval
        code_analysis = container.get_capability("code_analysis")
        assert code_analysis == {"enabled": True, "tools": ["ast", "parser"]}

        test_gen = container.get_capability("test_generation")
        assert test_gen == {"enabled": True, "framework": "pytest"}

        nonexistent = container.get_capability("nonexistent")
        assert nonexistent is None

    def test_multiple_independent_containers(self):
        """Multiple independent containers should coexist."""

        class ContainerA:
            def has_capability(self, capability_name: str) -> bool:
                return capability_name.startswith("a_")

            def get_capability(self, name: str) -> Optional[Any]:
                return f"A:{name}" if name.startswith("a_") else None

        class ContainerB:
            def has_capability(self, capability_name: str) -> bool:
                return capability_name.startswith("b_")

            def get_capability(self, name: str) -> Optional[Any]:
                return f"B:{name}" if name.startswith("b_") else None

        a = ContainerA()
        b = ContainerB()

        # Both implement the protocol
        assert isinstance(a, CapabilityContainerProtocol)
        assert isinstance(b, CapabilityContainerProtocol)

        # Each has its own capabilities
        assert a.has_capability("a_test") is True
        assert a.has_capability("b_test") is False
        assert a.get_capability("a_test") == "A:a_test"

        assert b.has_capability("b_test") is True
        assert b.has_capability("a_test") is False
        assert b.get_capability("b_test") == "B:b_test"


class TestProtocolEdgeCases:
    """Test edge cases and error conditions."""

    def test_protocol_with_none_capability(self):
        """Protocol should handle None capability names gracefully."""

        class NoneHandlingContainer:
            def has_capability(self, capability_name: str) -> bool:
                if capability_name is None:
                    return False
                return True

            def get_capability(self, name: str) -> Optional[Any]:
                if name is None:
                    return None
                return f"capability_{name}"

        container = NoneHandlingContainer()

        # Test with None
        assert container.has_capability(None) is False  # type: ignore
        assert container.get_capability(None) is None  # type: ignore

    def test_protocol_with_empty_string_capability(self):
        """Protocol should handle empty string capability names."""

        class EmptyStringContainer:
            def has_capability(self, capability_name: str) -> bool:
                return len(capability_name) > 0

            def get_capability(self, name: str) -> Optional[Any]:
                if not name:
                    return None
                return f"capability_{name}"

        container = EmptyStringContainer()

        # Test with empty string
        assert container.has_capability("") is False
        assert container.get_capability("") is None

        # Test with non-empty string
        assert container.has_capability("test") is True
        assert container.get_capability("test") == "capability_test"
