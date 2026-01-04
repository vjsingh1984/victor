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

"""Unit tests for HandlerRegistry."""

import pytest
from victor.framework.handler_registry import (
    HandlerEntry,
    HandlerRegistry,
    get_handler_registry,
    register_handler,
    get_handler,
)


class TestHandlerEntry:
    """Tests for HandlerEntry dataclass."""

    def test_create_entry_minimal(self):
        """Test creating entry with minimal fields."""
        entry = HandlerEntry(name="test", handler=lambda x: x)
        assert entry.name == "test"
        assert entry.handler is not None
        assert entry.vertical is None
        assert entry.description is None

    def test_create_entry_full(self):
        """Test creating entry with all fields."""

        def handler(x):
            return x * 2

        entry = HandlerEntry(
            name="test_handler", handler=handler, vertical="coding", description="A test handler"
        )
        assert entry.name == "test_handler"
        assert entry.handler is handler
        assert entry.vertical == "coding"
        assert entry.description == "A test handler"


class TestHandlerRegistry:
    """Tests for HandlerRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        HandlerRegistry.reset_instance()
        yield
        HandlerRegistry.reset_instance()

    def test_singleton_instance(self):
        """Test registry is a singleton."""
        reg1 = HandlerRegistry.get_instance()
        reg2 = HandlerRegistry.get_instance()
        assert reg1 is reg2

    def test_reset_instance(self):
        """Test reset creates new instance."""
        reg1 = HandlerRegistry.get_instance()
        reg1.register("test", lambda: None)

        HandlerRegistry.reset_instance()
        reg2 = HandlerRegistry.get_instance()

        assert reg1 is not reg2
        assert not reg2.has("test")

    def test_register_handler(self):
        """Test basic handler registration."""
        registry = HandlerRegistry()

        def handler(x):
            return x

        registry.register("my_handler", handler)

        assert registry.has("my_handler")
        assert registry.get("my_handler") is handler

    def test_register_with_metadata(self):
        """Test registration with full metadata."""
        registry = HandlerRegistry()

        def handler(x):
            return x

        registry.register(
            "my_handler", handler, vertical="devops", description="Handles deployments"
        )

        entry = registry.get_entry("my_handler")
        assert entry.name == "my_handler"
        assert entry.vertical == "devops"
        assert entry.description == "Handles deployments"

    def test_register_duplicate_raises(self):
        """Test duplicate registration raises error."""
        registry = HandlerRegistry()
        registry.register("handler", lambda: None)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("handler", lambda: None)

    def test_register_duplicate_with_replace(self):
        """Test duplicate registration with replace flag."""
        registry = HandlerRegistry()

        def handler1():
            return 1

        def handler2():
            return 2

        registry.register("handler", handler1)
        registry.register("handler", handler2, replace=True)

        assert registry.get("handler") is handler2

    def test_unregister_existing(self):
        """Test unregistering existing handler."""
        registry = HandlerRegistry()
        registry.register("handler", lambda: None)

        result = registry.unregister("handler")

        assert result is True
        assert not registry.has("handler")

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent handler."""
        registry = HandlerRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_get_nonexistent(self):
        """Test get returns None for nonexistent handler."""
        registry = HandlerRegistry()

        assert registry.get("nonexistent") is None

    def test_get_entry_nonexistent(self):
        """Test get_entry returns None for nonexistent handler."""
        registry = HandlerRegistry()

        assert registry.get_entry("nonexistent") is None

    def test_list_handlers(self):
        """Test listing all handler names."""
        registry = HandlerRegistry()
        registry.register("handler1", lambda: None)
        registry.register("handler2", lambda: None)
        registry.register("handler3", lambda: None)

        names = registry.list_handlers()

        assert set(names) == {"handler1", "handler2", "handler3"}

    def test_list_entries(self):
        """Test listing all handler entries."""
        registry = HandlerRegistry()
        registry.register("h1", lambda: 1, vertical="v1")
        registry.register("h2", lambda: 2, vertical="v2")

        entries = registry.list_entries()

        assert len(entries) == 2
        assert all(isinstance(e, HandlerEntry) for e in entries)

    def test_list_by_vertical(self):
        """Test filtering handlers by vertical."""
        registry = HandlerRegistry()
        registry.register("coding_h1", lambda: None, vertical="coding")
        registry.register("coding_h2", lambda: None, vertical="coding")
        registry.register("devops_h1", lambda: None, vertical="devops")

        coding_handlers = registry.list_by_vertical("coding")

        assert set(coding_handlers) == {"coding_h1", "coding_h2"}

    def test_clear(self):
        """Test clearing all handlers."""
        registry = HandlerRegistry()
        registry.register("h1", lambda: None)
        registry.register("h2", lambda: None)

        registry.clear()

        assert len(registry.list_handlers()) == 0

    def test_register_from_vertical(self):
        """Test bulk registration from vertical."""
        registry = HandlerRegistry()
        handlers = {
            "handler1": lambda: 1,
            "handler2": lambda: 2,
            "handler3": lambda: 3,
        }

        count = registry.register_from_vertical("coding", handlers)

        assert count == 3
        assert all(registry.get_entry(h).vertical == "coding" for h in handlers)


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        HandlerRegistry.reset_instance()
        yield
        HandlerRegistry.reset_instance()

    def test_get_handler_registry(self):
        """Test get_handler_registry returns singleton."""
        reg1 = get_handler_registry()
        reg2 = get_handler_registry()
        assert reg1 is reg2

    def test_register_handler_function(self):
        """Test register_handler convenience function."""
        register_handler("test_handler", lambda: None, vertical="test")

        registry = get_handler_registry()
        assert registry.has("test_handler")
        assert registry.get_entry("test_handler").vertical == "test"

    def test_get_handler_function(self):
        """Test get_handler convenience function."""

        def handler():
            return "result"

        register_handler("my_handler", handler)

        result = get_handler("my_handler")

        assert result is handler

    def test_get_handler_nonexistent(self):
        """Test get_handler returns None for nonexistent."""
        result = get_handler("nonexistent")

        assert result is None
