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

"""Tests for MutableVerticalContext."""

import pytest

from victor.core.verticals.mutable_context import MutableVerticalContext


class TestMutableVerticalContext:
    """Test suite for MutableVerticalContext."""

    def test_initialization(self):
        """Test context initialization."""
        context = MutableVerticalContext("test", {})
        assert context.name == "test"
        assert context.get_mutation_count() == 0
        assert not context.has_capability("any_capability")

    def test_apply_capability(self):
        """Test applying a capability."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("test_cap", key="value")

        assert context.has_capability("test_cap")
        assert context.get_capability("test_cap") == {"key": "value"}
        assert context.get_mutation_count() == 1

    def test_apply_multiple_capabilities(self):
        """Test applying multiple capabilities."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)
        context.apply_capability("cap3", value=3)

        assert context.get_mutation_count() == 3
        assert context.get_capability("cap1") == {"value": 1}
        assert context.get_capability("cap2") == {"value": 2}
        assert context.get_capability("cap3") == {"value": 3}

    def test_get_capability_nonexistent(self):
        """Test getting nonexistent capability returns None."""
        context = MutableVerticalContext("test", {})
        assert context.get_capability("nonexistent") is None

    def test_has_capability_true(self):
        """Test has_capability returns True for applied capability."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("test_cap", value=1)
        assert context.has_capability("test_cap")

    def test_has_capability_false(self):
        """Test has_capability returns False for missing capability."""
        context = MutableVerticalContext("test", {})
        assert not context.has_capability("missing_cap")

    def test_get_mutation_history(self):
        """Test getting mutation history."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)

        history = context.get_mutation_history()
        assert len(history) == 2
        assert history[0].capability == "cap1"
        assert history[1].capability == "cap2"

    def test_get_mutation_count(self):
        """Test getting mutation count."""
        context = MutableVerticalContext("test", {})
        assert context.get_mutation_count() == 0

        context.apply_capability("cap1", value=1)
        assert context.get_mutation_count() == 1

        context.apply_capability("cap2", value=2)
        assert context.get_mutation_count() == 2

    def test_get_recent_mutations(self):
        """Test getting recent mutations."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("old_cap", value=1)
        # Simulate old mutation by modifying timestamp
        context._mutations[0].timestamp = 0

        context.apply_capability("recent_cap", value=2)

        recent = context.get_recent_mutations(seconds=300)
        assert len(recent) == 1
        assert recent[0].capability == "recent_cap"

    def test_rollback_last(self):
        """Test rolling back last mutation."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)

        assert context.rollback_last() is True
        assert not context.has_capability("cap2")
        assert context.has_capability("cap1")
        assert context.get_mutation_count() == 1

    def test_rollback_last_empty(self):
        """Test rolling back when no mutations exist."""
        context = MutableVerticalContext("test", {})
        assert context.rollback_last() is False

    def test_rollback_to_index(self):
        """Test rolling back to specific index."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)
        context.apply_capability("cap3", value=3)

        context.rollback_to(1)  # Keep cap1 and cap2, remove cap3

        assert context.has_capability("cap1")
        assert context.has_capability("cap2")
        assert not context.has_capability("cap3")
        assert context.get_mutation_count() == 2

    def test_rollback_to_index_invalid(self):
        """Test rollback_to with invalid index."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)

        with pytest.raises(IndexError):
            context.rollback_to(5)  # Index out of range

        with pytest.raises(IndexError):
            context.rollback_to(-1)  # Negative index

    def test_clear_mutations(self):
        """Test clearing all mutations."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)

        context.clear_mutations()

        assert context.get_mutation_count() == 0
        assert not context.has_capability("cap1")
        assert not context.has_capability("cap2")

    def test_get_mutations_by_capability(self):
        """Test getting mutations for specific capability."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)
        context.apply_capability("cap1", value=3)  # Update cap1

        cap1_mutations = context.get_mutations_by_capability("cap1")
        assert len(cap1_mutations) == 2

        cap2_mutations = context.get_mutations_by_capability("cap2")
        assert len(cap2_mutations) == 1

    def test_get_all_applied_capabilities(self):
        """Test getting all applied capabilities."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)
        context.apply_capability("cap2", value=2)

        all_caps = context.get_all_applied_capabilities()
        assert len(all_caps) == 2
        assert all_caps["cap1"] == {"value": 1}
        assert all_caps["cap2"] == {"value": 2}

    def test_export_state(self):
        """Test exporting state to dict."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)

        state = context.export_state()
        assert "name" in state
        assert "config" in state
        assert "mutations" in state
        assert "capability_values" in state
        assert state["name"] == "test"
        assert len(state["mutations"]) == 1
        assert state["capability_values"]["cap1"] == {"value": 1}

    def test_import_state(self):
        """Test importing state from dict."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)

        state = context.export_state()

        # Import to new context
        new_context = MutableVerticalContext("test2", {})
        new_context.import_state(state)

        assert new_context.get_mutation_count() == 1
        assert new_context.has_capability("cap1")
        assert new_context.get_capability("cap1") == {"value": 1}
        assert new_context.name == "test"

    def test_apply_stages_tracks_mutation(self):
        """Test that apply_stages tracks mutation."""
        context = MutableVerticalContext("test", {})
        stages = {"stage1": {"tools": ["read"]}}

        context.apply_stages(stages)

        assert context.has_capability("stages")
        assert context.get_capability("stages") == {"stages": stages}

    def test_apply_middleware_tracks_mutation(self):
        """Test that apply_middleware tracks mutation."""
        context = MutableVerticalContext("test", {})
        middleware = ["mw1", "mw2"]

        context.apply_middleware(middleware)

        assert context.has_capability("middleware")
        assert context.get_capability("middleware") == {"middleware": middleware}

    def test_apply_safety_patterns_tracks_mutation(self):
        """Test that apply_safety_patterns tracks mutation."""
        context = MutableVerticalContext("test", {})
        patterns = ["pattern1", "pattern2"]

        context.apply_safety_patterns(patterns)

        assert context.has_capability("safety_patterns")
        assert context.get_capability("safety_patterns") == {"patterns": patterns}

    def test_apply_system_prompt_tracks_mutation(self):
        """Test that apply_system_prompt tracks mutation."""
        context = MutableVerticalContext("test", {})
        prompt = "Custom prompt"

        context.apply_system_prompt(prompt)

        assert context.has_capability("system_prompt")
        assert context.get_capability("system_prompt") == {"prompt": prompt}

    def test_config_updated_with_applied_capabilities(self):
        """Test that config is updated with applied capabilities."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)

        # Check that capability was stored in _capability_values
        assert "cap1" in context._capability_values
        assert context._capability_values["cap1"] == {"value": 1}

        # Check mutation was recorded
        assert len(context._mutations) == 1
        assert context._mutations[0].capability == "cap1"

    def test_multiple_apply_same_capability(self):
        """Test applying same capability multiple times."""
        context = MutableVerticalContext("test", {})
        context.apply_capability("cap1", value=1)
        context.apply_capability("cap1", value=2)

        # Should have 2 mutations
        assert context.get_mutation_count() == 2

        # Latest value should be returned
        assert context.get_capability("cap1") == {"value": 2}
