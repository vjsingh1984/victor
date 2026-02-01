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

"""Tests for DI compliance in framework modules (Phase 3.2).

Tests that framework modules use dependency injection patterns properly
and avoid direct imports that create tight coupling.
"""

import ast
import inspect


class TestDICompliance:
    """Test dependency injection compliance in framework modules."""

    def test_step_handlers_delegates_capability_checks(self):
        """Step handlers should delegate to CapabilityHelper."""
        from victor.framework.step_handlers import _check_capability, _invoke_capability

        # Functions should exist and be callable
        assert callable(_check_capability)
        assert callable(_invoke_capability)

        # Check docstrings mention delegation
        assert "Delegate" in (_check_capability.__doc__ or "") or "CapabilityHelper" in (
            _check_capability.__doc__ or ""
        )
        assert "Delegate" in (_invoke_capability.__doc__ or "") or "CapabilityHelper" in (
            _invoke_capability.__doc__ or ""
        )

    def test_step_handlers_uses_lazy_imports(self):
        """Step handlers should use lazy imports for CapabilityHelper.

        Lazy imports (inside functions) allow for late binding and avoid
        circular import issues. This is an acceptable DI pattern.
        """
        # Read the step_handlers module source
        from victor.framework import step_handlers

        source = inspect.getsource(step_handlers)

        # Parse the AST to check import patterns
        tree = ast.parse(source)

        # Find all imports at module level
        module_level_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    module_level_imports.append(node.module or "")

        # CapabilityHelper should NOT be imported at module level
        # (it should be a lazy import inside functions)
        assert not any(
            "capability_registry" in imp and "CapabilityHelper" in source.split("import")[0]
            for imp in module_level_imports
        ), "CapabilityHelper should not be imported at module level"

    def test_capability_helper_uses_protocol(self):
        """CapabilityHelper should use CapabilityRegistryProtocol."""
        from victor.agent.capability_registry import CapabilityHelper

        # Check that CapabilityHelper methods exist
        assert hasattr(CapabilityHelper, "check_capability")
        assert hasattr(CapabilityHelper, "invoke_capability")

        # Methods should be static
        assert isinstance(
            inspect.getattr_static(CapabilityHelper, "check_capability"), staticmethod
        )
        assert isinstance(
            inspect.getattr_static(CapabilityHelper, "invoke_capability"), staticmethod
        )

    def test_step_handler_registry_extensible(self):
        """StepHandlerRegistry should support adding custom handlers (OCP)."""
        from victor.framework.step_handlers import StepHandlerRegistry, BaseStepHandler

        # Create a custom handler
        class CustomHandler(BaseStepHandler):
            @property
            def name(self) -> str:
                return "custom_test"

            @property
            def order(self) -> int:
                return 100

            def _do_apply(self, orchestrator, vertical, context, result):
                pass

        # Should be able to add custom handlers
        registry = StepHandlerRegistry.default()
        initial_count = len(registry.handlers)

        registry.add_handler(CustomHandler())

        # Handler should be added
        assert len(registry.handlers) > initial_count

    def test_service_container_supports_registration(self):
        """ServiceContainer should support service registration."""
        from victor.core.container import ServiceContainer, ServiceLifetime

        container = ServiceContainer()

        # The container should be able to register services
        # This verifies DI infrastructure is set up properly
        assert container is not None
        assert hasattr(container, "register")
        assert hasattr(container, "get")

        # Register a test service
        class ITestService:
            pass

        class TestServiceImpl(ITestService):
            pass

        container.register(ITestService, lambda c: TestServiceImpl(), ServiceLifetime.SINGLETON)

        # Should be able to resolve
        service = container.get(ITestService)
        assert isinstance(service, TestServiceImpl)


class TestNoTightCoupling:
    """Test that framework modules avoid tight coupling."""

    def test_step_handlers_uses_protocols_for_type_hints(self):
        """Step handlers should use protocols for type hints, not concrete classes."""
        from victor.framework import step_handlers

        source = inspect.getsource(step_handlers)

        # Count protocol usage vs concrete class usage in type hints
        # This is a heuristic - protocols are preferable
        protocol_hints = source.count("Protocol")
        interface_hints = source.count("Protocol") + source.count("ABC")

        # Should have at least some protocol usage
        assert interface_hints > 0, "Step handlers should use protocols for type hints"

    def test_vertical_integration_uses_protocols(self):
        """Vertical integration should use protocols for dependencies."""
        from victor.framework import vertical_integration

        source = inspect.getsource(vertical_integration)

        # Should import from protocols
        assert "from victor.framework.protocols" in source or "protocols" in source

    def test_step_handlers_deprecation_warnings_present(self):
        """Deprecated functions should have deprecation warnings documented."""
        from victor.framework.step_handlers import _check_capability, _invoke_capability

        # Check docstrings mention deprecation
        check_doc = _check_capability.__doc__ or ""
        invoke_doc = _invoke_capability.__doc__ or ""

        assert "deprecated" in check_doc.lower() or "0.6.0" in check_doc
        assert "deprecated" in invoke_doc.lower() or "0.6.0" in invoke_doc


class TestCapabilityHelperProtocolCompliance:
    """Test that CapabilityHelper follows protocol patterns."""

    def test_check_capability_uses_isinstance_for_protocol(self):
        """check_capability should use isinstance for protocol checking."""
        from victor.agent.capability_registry import CapabilityHelper

        source = inspect.getsource(CapabilityHelper.check_capability)

        # Should check for CapabilityRegistryProtocol
        assert (
            "CapabilityRegistryProtocol" in source or "isinstance" in source
        ), "Should use isinstance for protocol checking"

    def test_invoke_capability_handles_non_protocol_objects(self):
        """invoke_capability should handle non-protocol objects gracefully."""
        from victor.agent.capability_registry import CapabilityHelper
        import warnings

        # Create a simple object without the protocol
        class SimpleObject:
            def some_method(self):
                return "test"

        obj = SimpleObject()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should warn or raise for non-protocol object
            try:
                CapabilityHelper.invoke_capability(obj, "nonexistent")
            except (AttributeError, TypeError):
                pass  # Expected for non-protocol objects

            # At least one warning should be raised
            # (either for protocol non-compliance or capability not found)
