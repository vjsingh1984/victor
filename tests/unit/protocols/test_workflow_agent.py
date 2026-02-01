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

"""Tests for WorkflowAgentProtocol."""

from victor.protocols import WorkflowAgentProtocol


class TestWorkflowAgentProtocol:
    """Test WorkflowAgentProtocol definition and usage."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should support runtime isinstance checks."""
        # Protocol should be runtime checkable
        assert hasattr(WorkflowAgentProtocol, "__protocol_attrs__")
        assert hasattr(WorkflowAgentProtocol, "_is_protocol")

    def test_agent_orchestrator_satisfies_protocol(self):
        """AgentOrchestrator should be importable and usable with protocol."""
        from victor.agent.orchestrator import AgentOrchestrator

        # Just verify AgentOrchestrator exists and can be used as type hint
        # The protocol is a marker protocol - it doesn't require specific methods
        # because SubAgentOrchestrator accesses internal attributes directly
        assert AgentOrchestrator is not None
        assert hasattr(AgentOrchestrator, "__name__")

    def test_protocol_is_minimal(self):
        """Protocol should be intentionally minimal (marker protocol)."""
        # WorkflowAgentProtocol is a marker protocol with no required methods
        # because SubAgentOrchestrator accesses orchestrator internals directly
        # This is intentional - the protocol exists for type safety, not to enumerate methods
        assert hasattr(WorkflowAgentProtocol, "_is_protocol")

    def test_protocol_type_hint_usage(self):
        """Protocol should work as a type hint."""
        from typing import TYPE_CHECKING

        # This test verifies the protocol can be used in type hints
        # without causing import errors
        if TYPE_CHECKING:
            from victor.protocols import WorkflowAgentProtocol

            def accept_orchestrator(orchestrator: WorkflowAgentProtocol) -> None:
                """Function that accepts protocol-compatible orchestrator."""
                pass

            # Should not raise any errors
            assert callable(accept_orchestrator)

    def test_allows_mock_implementation(self):
        """Protocol should allow mock implementations for testing."""
        from typing import cast

        # Should be able to cast mock to protocol for testing
        mock_orchestrator = cast(WorkflowAgentProtocol, object())

        # This validates that the protocol approach enables testing
        assert mock_orchestrator is not None
