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

"""Tests for orchestrator accessor protocols (Phase 11.2).

Tests the accessor protocols that replace direct private attribute access.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestCapabilityLoaderAccessProtocol:
    """Tests for CapabilityLoaderAccessProtocol."""

    def test_protocol_is_defined(self):
        """Test that protocol is defined and importable."""
        from victor.agent.protocols import CapabilityLoaderAccessProtocol

        assert CapabilityLoaderAccessProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol can be used with isinstance."""
        from victor.agent.protocols import CapabilityLoaderAccessProtocol

        class FakeOrchestrator:
            def get_capability_loader(self):
                return None

            def set_capability_loader(self, loader):
                pass

        orchestrator = FakeOrchestrator()
        assert isinstance(orchestrator, CapabilityLoaderAccessProtocol)

    def test_non_conforming_class_fails_isinstance(self):
        """Test that non-conforming class fails isinstance."""
        from victor.agent.protocols import CapabilityLoaderAccessProtocol

        class NotAnOrchestrator:
            pass

        obj = NotAnOrchestrator()
        assert not isinstance(obj, CapabilityLoaderAccessProtocol)


class TestMiddlewareChainAccessProtocol:
    """Tests for MiddlewareChainAccessProtocol."""

    def test_protocol_is_defined(self):
        """Test that protocol is defined and importable."""
        from victor.agent.protocols import MiddlewareChainAccessProtocol

        assert MiddlewareChainAccessProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol can be used with isinstance."""
        from victor.agent.protocols import MiddlewareChainAccessProtocol

        class FakeOrchestrator:
            def get_middleware_chain(self):
                return None

            def set_middleware_chain(self, chain):
                pass

        orchestrator = FakeOrchestrator()
        assert isinstance(orchestrator, MiddlewareChainAccessProtocol)


class TestRLHooksAccessProtocol:
    """Tests for RLHooksAccessProtocol."""

    def test_protocol_is_defined(self):
        """Test that protocol is defined and importable."""
        from victor.agent.protocols import RLHooksAccessProtocol

        assert RLHooksAccessProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol can be used with isinstance."""
        from victor.agent.protocols import RLHooksAccessProtocol

        class FakeOrchestrator:
            def get_rl_hooks(self):
                return None

            def set_rl_hooks(self, hooks):
                pass

        orchestrator = FakeOrchestrator()
        assert isinstance(orchestrator, RLHooksAccessProtocol)


class TestAccessorProtocolIntegration:
    """Integration tests for accessor protocols."""

    def test_check_protocol_works_with_accessor_protocols(self):
        """Test that check_protocol works with accessor protocols."""
        from victor.agent.protocols import CapabilityLoaderAccessProtocol
        from victor.core.verticals.protocols.utils import check_protocol

        class FakeOrchestrator:
            def get_capability_loader(self):
                return MagicMock()

            def set_capability_loader(self, loader):
                pass

        orchestrator = FakeOrchestrator()
        result = check_protocol(orchestrator, CapabilityLoaderAccessProtocol)

        assert result is orchestrator

    def test_accessor_protocol_usage_pattern(self):
        """Test the expected usage pattern for accessor protocols."""
        from victor.agent.protocols import CapabilityLoaderAccessProtocol
        from victor.core.verticals.protocols.utils import check_protocol

        class MockCapabilityLoader:
            capabilities = ["cap1", "cap2"]

        class FakeOrchestrator:
            def __init__(self):
                self._loader = None

            def get_capability_loader(self):
                return self._loader

            def set_capability_loader(self, loader):
                self._loader = loader

        orchestrator = FakeOrchestrator()

        # Check protocol conformance
        accessor = check_protocol(orchestrator, CapabilityLoaderAccessProtocol)
        assert accessor is not None

        # Set loader through accessor
        loader = MockCapabilityLoader()
        accessor.set_capability_loader(loader)

        # Get loader through accessor
        retrieved = accessor.get_capability_loader()
        assert retrieved is loader
        assert retrieved.capabilities == ["cap1", "cap2"]


class TestAccessorProtocolCompliance:
    """Tests for replacing private attribute access with accessor protocols."""

    def test_decomposed_handlers_should_use_protocol(self):
        """Test that decomposed handlers use protocol instead of private access."""
        from victor.agent.protocols import CapabilityLoaderAccessProtocol
        from victor.core.verticals.protocols.utils import check_protocol

        # This test documents the expected usage pattern
        class OrchestratorWithProtocol:
            def __init__(self):
                self._capability_loader = None

            def get_capability_loader(self):
                return self._capability_loader

            def set_capability_loader(self, loader):
                self._capability_loader = loader

        orchestrator = OrchestratorWithProtocol()

        # Recommended: Use protocol accessor
        accessor = check_protocol(orchestrator, CapabilityLoaderAccessProtocol)
        if accessor:
            loader = accessor.get_capability_loader()
        else:
            loader = None

        assert loader is None  # Initially None

        # Not recommended: Direct private access
        # loader = orchestrator._capability_loader  # Don't do this!
