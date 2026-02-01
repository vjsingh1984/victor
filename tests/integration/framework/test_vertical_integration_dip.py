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

"""Integration tests for DIP-compliant vertical integration."""



from victor.core.verticals.mutable_context import MutableVerticalContext


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self):
        self._vertical_context = None
        self._allowed_tools = None
        self._system_prompt = None

    def set_vertical_context(self, context: MutableVerticalContext) -> None:
        """Set vertical context."""
        self._vertical_context = context

    def get_vertical_context(self):
        """Get vertical context."""
        return self._vertical_context

    def set_enabled_tools(self, tools):
        """Set enabled tools."""
        self._allowed_tools = tools

    def apply_vertical_middleware(self, middleware):
        """Apply middleware."""
        pass

    def apply_vertical_safety_patterns(self, patterns):
        """Apply safety patterns."""
        pass


class MockVertical:
    """Mock vertical for testing."""

    name = "test_vertical"

    @staticmethod
    def get_config():
        """Get vertical config."""
        return {
            "capabilities": {
                "test_cap": {"value": 1},
                "system_prompt": {"prompt": "Test prompt"},
            }
        }


class TestVerticalIntegrationDIP:
    """Test suite for DIP-compliant vertical integration."""

    def test_context_is_mutable_vertical_context(self):
        """Test that vertical integration creates MutableVerticalContext."""
        context = MutableVerticalContext("test", {})

        assert isinstance(context, MutableVerticalContext)
        assert context.name == "test"

    def test_apply_capability_through_context(self):
        """Test applying capabilities through context (DIP)."""
        context = MutableVerticalContext("test", {})
        orchestrator = MockOrchestrator()

        # Apply capabilities through context
        context.apply_capability("allowed_tools", tools=["read", "write"])
        context.apply_capability("system_prompt", prompt="Test prompt")

        # Attach context to orchestrator
        orchestrator.set_vertical_context(context)

        # Verify context has capabilities
        assert context.has_capability("allowed_tools")
        assert context.has_capability("system_prompt")

        # Verify orchestrator has context
        assert orchestrator.get_vertical_context() is context

    def test_no_direct_orchestrator_mutation(self):
        """Test that orchestrator is not mutated directly."""
        context = MutableVerticalContext("test", {})
        orchestrator = MockOrchestrator()

        # Apply capabilities through context only
        context.apply_capability("test_cap", value=1)

        # Attach context
        orchestrator.set_vertical_context(context)

        # Orchestrator should have context, not direct mutations
        assert orchestrator.get_vertical_context() is context
        assert not hasattr(orchestrator, "_directly_mutated")

    def test_mutation_tracking(self):
        """Test that mutations are tracked."""
        context = MutableVerticalContext("test", {})

        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)

        # Check mutation history
        history = context.get_mutation_history()
        assert len(history) == 2
        assert history[0].capability == "cap1"
        assert history[1].capability == "cap2"

    def test_rollback_support(self):
        """Test that mutations can be rolled back."""
        context = MutableVerticalContext("test", {})

        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)

        # Rollback last mutation
        success = context.rollback_last()
        assert success is True
        assert not context.has_capability("cap2")
        assert context.has_capability("cap1")

    def test_capability_query_methods(self):
        """Test capability query methods."""
        context = MutableVerticalContext("test", {})

        context.apply_capability("cap1", value=1)

        # Test has_capability
        assert context.has_capability("cap1")
        assert not context.has_capability("cap2")

        # Test get_capability
        assert context.get_capability("cap1") == {"value": 1}
        assert context.get_capability("cap2") is None

        # Test get_all_applied_capabilities
        all_caps = context.get_all_applied_capabilities()
        assert "cap1" in all_caps

    def test_export_import_state(self):
        """Test state export and import."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)

        # Export state
        state = context.export_state()
        assert "mutations" in state
        assert "capability_values" in state

        # Import to new context
        new_context = MutableVerticalContext("test2", {})
        new_context.import_state(state)

        # Verify state was imported
        assert new_context.get_mutation_count() == 1
        assert new_context.has_capability("cap1")

    def test_integration_with_orchestrator(self):
        """Test full integration with orchestrator."""
        orchestrator = MockOrchestrator()
        context = MutableVerticalContext("test", {})

        # Apply capabilities
        context.apply_capability("allowed_tools", tools=["read", "write"])
        context.apply_capability("system_prompt", prompt="Test")

        # Attach to orchestrator
        orchestrator.set_vertical_context(context)

        # Verify integration
        assert orchestrator.get_vertical_context() is context
        assert context.get_mutation_count() == 2

    def test_config_contains_applied_capabilities(self):
        """Test that applied capabilities are tracked via get_all_applied_capabilities()."""
        context = MutableVerticalContext("test", None)

        context.apply_capability("cap1", value=1)

        # Check applied capabilities via the proper API (not direct config access)
        applied = context.get_all_applied_capabilities()
        assert "cap1" in applied
        assert applied["cap1"] == {"value": 1}

    def test_clear_mutations(self):
        """Test clearing mutations."""
        context = MutableVerticalContext("test", {})

        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)

        # Clear mutations
        context.clear_mutations()

        # Verify cleared
        assert context.get_mutation_count() == 0
        assert not context.has_capability("cap1")
        assert not context.has_capability("cap2")

    def test_get_recent_mutations(self):
        """Test getting recent mutations."""
        context = MutableVerticalContext("test", {})

        context.apply_capability("old_cap", value=1)
        # Simulate old mutation
        context._mutations[0].timestamp = 0

        context.apply_capability("new_cap", value=2)

        # Get recent mutations (last 5 minutes)
        recent = context.get_recent_mutations(seconds=300)
        assert len(recent) == 1
        assert recent[0].capability == "new_cap"

    def test_rollback_to_specific_index(self):
        """Test rollback to specific mutation index."""
        context = MutableVerticalContext("test", {})

        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)
        context.apply_capability("cap3", value=3)

        # Rollback to index 1 (keep cap1, cap2; remove cap3)
        context.rollback_to(1)

        assert context.has_capability("cap1")
        assert context.has_capability("cap2")
        assert not context.has_capability("cap3")
        assert context.get_mutation_count() == 2
