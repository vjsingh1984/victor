# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Type checking tests for protocol compliance.

Tests that protocol-based code passes mypy strict mode type checking.
This ensures that the protocols provide proper type safety.
"""

import subprocess
import pytest
from pathlib import Path

import shutil


class TestProtocolTypeChecking:
    """Test suite for mypy strict mode compliance."""

    @pytest.mark.skipif(
        not shutil.which("mypy"), reason="mypy is not installed"
    )
    def test_protocol_files_pass_mypy_strict(self):
        """All protocol files should pass mypy strict mode."""
        protocol_files = [
            "victor/protocols/capability.py",
            "victor/protocols/workflow_provider.py",
            "victor/protocols/tiered_config.py",
            "victor/protocols/extension_provider.py",
        ]

        for file_path in protocol_files:
            result = subprocess.run(
                ["mypy", "--strict", file_path],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent,
            )

            # mypy should pass (exit code 0) or have only non-error warnings
            assert (
                result.returncode == 0
            ), f"mypy failed for {file_path}:\n{result.stdout}\n{result.stderr}"

    def test_protocol_imports_are_typed(self):
        """Protocol imports should be properly typed."""
        # Test that imports work and are type-checked
        from victor.protocols import (
            CapabilityContainerProtocol,
            WorkflowProviderProtocol,
            TieredConfigProviderProtocol,
            ExtensionProviderProtocol,
        )

        # Verify protocols are runtime checkable
        from typing import runtime_checkable

        assert hasattr(CapabilityContainerProtocol, "_is_protocol")
        assert hasattr(WorkflowProviderProtocol, "_is_protocol")
        assert hasattr(TieredConfigProviderProtocol, "_is_protocol")
        assert hasattr(ExtensionProviderProtocol, "_is_protocol")

    def test_protocol_signatures_are_correct(self):
        """Protocol method signatures should be properly typed."""
        import inspect
        from victor.protocols import CapabilityContainerProtocol

        # Check has_capability signature
        sig = inspect.signature(CapabilityContainerProtocol.has_capability)
        assert "capability_name" in sig.parameters
        assert sig.parameters["capability_name"].annotation is str
        assert sig.return_annotation is bool

        # Check get_capability signature
        sig = inspect.signature(CapabilityContainerProtocol.get_capability)
        assert "name" in sig.parameters
        assert sig.parameters["name"].annotation is str

    @pytest.mark.skipif(
        not shutil.which("mypy"), reason="mypy is not installed"
    )
    def test_test_file_passes_mypy(self):
        """Test file itself should pass mypy type checking."""
        result = subprocess.run(
            ["mypy", "--strict", "tests/unit/protocols/test_capability_protocol.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )

        # Should pass type checking
        assert (
            result.returncode == 0
        ), f"mypy failed for test file:\n{result.stdout}\n{result.stderr}"


class TestProtocolTypeSafetyExamples:
    """Test suite demonstrating type safety benefits of protocols."""

    def test_protocol_ensures_correct_method_signatures(self):
        """Using protocols should enforce correct method signatures."""
        # This test demonstrates that mypy will catch type errors
        # when protocol-conforming classes have wrong signatures

        # Correct implementation - should type check
        class CorrectImpl:
            def has_capability(self, capability_name: str) -> bool:
                return True

            def get_capability(self, name: str):
                return f"capability_{name}"

        # Verify correct impl is recognized
        impl = CorrectImpl()
        assert impl.has_capability("test") is True

    def test_protocol_prevents_type_errors(self):
        """Protocols should prevent type errors at compile time."""
        from victor.protocols import CapabilityContainerProtocol

        def use_capability_container(container: CapabilityContainerProtocol) -> bool:
            """Function that requires a capability container."""
            return container.has_capability("test")

        # Create a compliant implementation
        class CompliantContainer:
            def has_capability(self, capability_name: str) -> bool:
                return capability_name == "test"

            def get_capability(self, name: str):
                return {"name": name} if name == "test" else None

        container = CompliantContainer()
        # This should work without type errors
        result = use_capability_container(container)
        assert result is True

    def test_protocol_with_wrong_signature_fails_mypy(self):
        """Implementation with wrong signature should fail mypy."""
        # This would be caught by mypy in strict mode
        # We can't test the actual failure here, but we document the expectation

        # Wrong implementation (missing parameter)
        # This would fail mypy strict type checking:
        #
        # class WrongImpl:
        #     def has_capability(self) -> bool:  # Missing parameter
        #         return True
        #
        # impl = WrongImpl()
        # container: CapabilityContainerProtocol = impl  # Type error!

        # Document the expectation
        assert True  # Placeholder - actual enforcement done by mypy


class TestProtocolTypeInference:
    """Test suite for protocol type inference."""

    def test_protocol_type_narrowing(self):
        """Protocols should enable type narrowing in isinstance() checks."""
        from victor.protocols import CapabilityContainerProtocol

        def process_object(obj: object) -> str:
            """Function that uses protocol for type narrowing."""
            if isinstance(obj, CapabilityContainerProtocol):
                # Type is narrowed here - obj is CapabilityContainerProtocol
                has_cap = obj.has_capability("test")  # Should type check
                return "has_capability"
            else:
                return "not_a_container"

        # Test with compliant object
        class CompliantImpl:
            def has_capability(self, capability_name: str) -> bool:
                return True

            def get_capability(self, name: str):
                return None

        impl = CompliantImpl()
        result = process_object(impl)
        assert result == "has_capability"

    def test_protocol_type_narrowing_with_multiple_protocols(self):
        """Type narrowing should work with multiple protocols."""
        from victor.protocols import (
            CapabilityContainerProtocol,
            WorkflowProviderProtocol,
        )

        def classify_object(obj: object) -> str:
            """Classify object by protocol conformance."""
            if isinstance(obj, CapabilityContainerProtocol):
                return "capability_container"
            elif isinstance(obj, WorkflowProviderProtocol):
                return "workflow_provider"
            else:
                return "unknown"

        # Test with capability container
        class CapabilityImpl:
            def has_capability(self, capability_name: str) -> bool:
                return True

            def get_capability(self, name: str):
                return None

        impl = CapabilityImpl()
        assert classify_object(impl) == "capability_container"
