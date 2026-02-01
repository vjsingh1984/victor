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

"""Property-based tests for HandlerRegistry.

Phase 3.3: Property-based tests for handler registration logic.

Uses Hypothesis to test invariants across many iterations:
1. Registration idempotence with replace=True
2. Get consistency after registration
3. List completeness
4. Vertical filtering accuracy
"""

import pytest
from hypothesis import given, strategies as st, settings, Phase
from unittest.mock import Mock

from victor.framework.handler_registry import (
    get_handler_registry,
)


# Strategies
handler_name_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
    min_size=1,
    max_size=20,
)
vertical_name_strategy = st.sampled_from(
    ["coding", "research", "devops", "rag", "dataanalysis", "benchmark"]
)
description_strategy = st.text(min_size=0, max_size=100)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before each test."""
    registry = get_handler_registry()
    registry.clear()
    yield
    registry.clear()


class TestRegistrationProperties:
    """Property-based tests for handler registration invariants."""

    @given(name=handler_name_strategy, vertical=vertical_name_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_registration_is_retrievable(self, name: str, vertical: str):
        """After registration, handler should be retrievable by name."""
        registry = get_handler_registry()
        handler = Mock()

        # Use replace=True since Hypothesis generates multiple examples
        registry.register(name, handler, vertical=vertical, replace=True)

        retrieved = registry.get(name)
        assert retrieved is handler

    @given(name=handler_name_strategy, vertical=vertical_name_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_unregister_removes_handler(self, name: str, vertical: str):
        """Unregistered handlers should not be retrievable."""
        registry = get_handler_registry()
        handler = Mock()

        # Use replace=True since Hypothesis generates multiple examples
        registry.register(name, handler, vertical=vertical, replace=True)
        registry.unregister(name)

        assert registry.get(name) is None

    @given(
        names=st.lists(handler_name_strategy, min_size=1, max_size=10, unique=True),
        vertical=vertical_name_strategy,
    )
    @settings(max_examples=30, phases=[Phase.generate])
    def test_list_handlers_contains_all_registered(self, names: list[str], vertical: str):
        """list_handlers() should return all registered handler names."""
        registry = get_handler_registry()

        for name in names:
            # Use replace=True since Hypothesis generates multiple examples
            registry.register(name, Mock(), vertical=vertical, replace=True)

        listed = registry.list_handlers()
        for name in names:
            assert name in listed

    @given(
        names=st.lists(handler_name_strategy, min_size=1, max_size=10, unique=True),
        verticals=st.lists(vertical_name_strategy, min_size=1, max_size=10),
    )
    @settings(max_examples=30, phases=[Phase.generate])
    def test_list_by_vertical_filters_correctly(self, names: list[str], verticals: list[str]):
        """list_by_vertical() should only return handlers for that vertical."""
        registry = get_handler_registry()

        # Register handlers with corresponding verticals (use replace=True)
        for i, name in enumerate(names):
            vertical = verticals[i % len(verticals)]
            registry.register(name, Mock(), vertical=vertical, replace=True)

        # Check each vertical filter
        for vertical in set(verticals):
            filtered = registry.list_by_vertical(vertical)
            for name in filtered:
                entry = registry.get_entry(name)
                assert entry is not None
                assert entry.vertical == vertical


class TestReplaceProperties:
    """Property-based tests for replace behavior."""

    @given(name=handler_name_strategy, vertical=vertical_name_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_replace_updates_handler(self, name: str, vertical: str):
        """With replace=True, registration should update the handler."""
        registry = get_handler_registry()
        handler1 = Mock()
        handler2 = Mock()

        # First register with replace=True to handle existing handlers
        registry.register(name, handler1, vertical=vertical, replace=True)
        registry.register(name, handler2, vertical=vertical, replace=True)

        retrieved = registry.get(name)
        assert retrieved is handler2

    @given(name=handler_name_strategy, vertical=vertical_name_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_no_replace_raises_on_duplicate(self, name: str, vertical: str):
        """Without replace=True, duplicate registration should raise."""
        registry = get_handler_registry()
        handler1 = Mock()
        handler2 = Mock()

        # First unregister if already exists, then register fresh
        registry.unregister(name)
        registry.register(name, handler1, vertical=vertical)

        with pytest.raises(ValueError):
            registry.register(name, handler2, vertical=vertical, replace=False)


class TestClearProperties:
    """Property-based tests for clear behavior."""

    @given(
        names=st.lists(handler_name_strategy, min_size=1, max_size=10, unique=True),
        vertical=vertical_name_strategy,
    )
    @settings(max_examples=30, phases=[Phase.generate])
    def test_clear_removes_all_handlers(self, names: list[str], vertical: str):
        """clear() should remove all registered handlers."""
        registry = get_handler_registry()

        for name in names:
            registry.register(name, Mock(), vertical=vertical)

        registry.clear()

        for name in names:
            assert registry.get(name) is None


class TestEntryProperties:
    """Property-based tests for HandlerEntry properties."""

    @given(
        name=handler_name_strategy,
        vertical=vertical_name_strategy,
        description=description_strategy,
    )
    @settings(max_examples=50, phases=[Phase.generate])
    def test_entry_preserves_metadata(self, name: str, vertical: str, description: str):
        """HandlerEntry should preserve all metadata."""
        registry = get_handler_registry()
        handler = Mock()

        # Use replace=True since Hypothesis generates multiple examples
        registry.register(name, handler, vertical=vertical, description=description, replace=True)

        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.name == name
        assert entry.vertical == vertical
        assert entry.description == description
        assert entry.handler is handler
