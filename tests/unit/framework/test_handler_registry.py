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
from unittest.mock import MagicMock, patch

from victor.framework.handler_registry import (
    HandlerEntry,
    HandlerRegistry,
    get_handler_registry,
    register_handler,
    get_handler,
    sync_handlers_with_executor,
    discover_handlers_from_vertical,
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


class TestHandlerRegistryEnhancements:
    """Tests for enhanced HandlerRegistry methods."""

    @pytest.fixture(autouse=True)
    def reset_registries(self):
        """Reset singleton and executor handlers before each test."""
        HandlerRegistry.reset_instance()
        # Clear executor handlers
        from victor.workflows import executor

        executor._compute_handlers.clear()
        yield
        HandlerRegistry.reset_instance()
        executor._compute_handlers.clear()

    def test_list_verticals(self):
        """Test list_verticals returns unique vertical names."""
        registry = HandlerRegistry()

        registry.register("h1", lambda: None, vertical="coding")
        registry.register("h2", lambda: None, vertical="coding")
        registry.register("h3", lambda: None, vertical="research")
        registry.register("h4", lambda: None)  # No vertical

        verticals = registry.list_verticals()

        assert set(verticals) == {"coding", "research"}

    def test_list_verticals_empty(self):
        """Test list_verticals with no registered handlers."""
        registry = HandlerRegistry()

        verticals = registry.list_verticals()

        assert verticals == []

    def test_sync_with_executor_to_executor(self):
        """Test syncing handlers to executor."""
        from victor.workflows import executor

        registry = HandlerRegistry()

        def handler():
            return "test"

        registry.register("test_handler", handler, vertical="coding")

        pushed, pulled = registry.sync_with_executor(direction="to_executor")

        assert pushed == 1
        assert pulled == 0
        assert executor.get_compute_handler("test_handler") is handler

    def test_sync_with_executor_from_executor(self):
        """Test syncing handlers from executor."""
        from victor.workflows import executor

        registry = HandlerRegistry()

        def handler():
            return "from_executor"

        executor.register_compute_handler("executor_handler", handler)

        pushed, pulled = registry.sync_with_executor(direction="from_executor")

        assert pushed == 0
        assert pulled == 1
        assert registry.get("executor_handler") is handler

    def test_sync_with_executor_bidirectional(self):
        """Test bidirectional sync."""
        from victor.workflows import executor

        registry = HandlerRegistry()

        def reg_handler():
            return "registry"

        def exec_handler():
            return "executor"

        registry.register("reg_handler", reg_handler)
        executor.register_compute_handler("exec_handler", exec_handler)

        pushed, pulled = registry.sync_with_executor(direction="bidirectional")

        assert pushed == 1
        assert pulled == 1
        assert executor.get_compute_handler("reg_handler") is reg_handler
        assert registry.get("exec_handler") is exec_handler

    def test_sync_with_executor_no_replace_existing(self):
        """Test sync does not replace existing handlers by default."""
        from victor.workflows import executor

        registry = HandlerRegistry()

        def old_handler():
            return "old"

        def new_handler():
            return "new"

        # Register in both
        registry.register("shared", old_handler)
        executor.register_compute_handler("shared", old_handler)

        # Now add conflicting handler to registry
        registry.unregister("shared")
        registry.register("shared", new_handler)

        # Sync without replace
        pushed, pulled = registry.sync_with_executor(direction="to_executor")

        # Should not push because executor already has it
        assert pushed == 0

    def test_sync_with_executor_with_replace(self):
        """Test sync replaces existing handlers when replace=True."""
        from victor.workflows import executor

        registry = HandlerRegistry()

        def old_handler():
            return "old"

        def new_handler():
            return "new"

        executor.register_compute_handler("shared", old_handler)
        registry.register("shared", new_handler)

        pushed, pulled = registry.sync_with_executor(
            direction="to_executor",
            replace=True,
        )

        assert pushed == 1
        assert executor.get_compute_handler("shared") is new_handler

    def test_discover_from_vertical_coding(self):
        """Test discovering handlers from coding vertical."""
        registry = HandlerRegistry()

        count = registry.discover_from_vertical("coding", sync_to_executor=False)

        # coding/handlers.py has HANDLERS dict
        assert count > 0
        assert "code_validation" in registry.list_handlers()
        assert "test_runner" in registry.list_handlers()

    def test_discover_from_vertical_research(self):
        """Test discovering handlers from research vertical."""
        registry = HandlerRegistry()

        count = registry.discover_from_vertical("research", sync_to_executor=False)

        # research/handlers.py has HANDLERS dict
        assert count > 0
        assert "web_scraper" in registry.list_handlers()

    def test_discover_from_vertical_syncs_to_executor(self):
        """Test discover syncs to executor by default."""
        from victor.workflows import executor

        registry = HandlerRegistry()

        registry.discover_from_vertical("coding", sync_to_executor=True)

        # Should be in executor
        assert executor.get_compute_handler("code_validation") is not None

    def test_discover_from_vertical_invalid_raises(self):
        """Test discover raises for invalid vertical."""
        registry = HandlerRegistry()

        with pytest.raises(ImportError):
            registry.discover_from_vertical("nonexistent_vertical_xyz")

    def test_discover_from_vertical_sets_vertical_metadata(self):
        """Test discover sets vertical metadata on entries."""
        registry = HandlerRegistry()

        registry.discover_from_vertical("coding", sync_to_executor=False)

        entry = registry.get_entry("code_validation")
        assert entry.vertical == "coding"


class TestModuleLevelEnhancedFunctions:
    """Tests for module-level enhanced convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registries(self):
        """Reset singleton and executor before each test."""
        HandlerRegistry.reset_instance()
        from victor.workflows import executor

        executor._compute_handlers.clear()
        yield
        HandlerRegistry.reset_instance()
        executor._compute_handlers.clear()

    def test_sync_handlers_with_executor_function(self):
        """Test sync_handlers_with_executor convenience function."""
        from victor.workflows import executor

        register_handler("test_h", lambda: None)

        pushed, pulled = sync_handlers_with_executor(direction="to_executor")

        assert pushed == 1
        assert executor.get_compute_handler("test_h") is not None

    def test_discover_handlers_from_vertical_function(self):
        """Test discover_handlers_from_vertical convenience function."""
        count = discover_handlers_from_vertical("coding", sync_to_executor=False)

        assert count > 0
        assert get_handler("code_validation") is not None


# =============================================================================
# Phase 1.3: @handler_decorator Tests
# =============================================================================


class TestHandlerDecoratorClass:
    """Tests for @handler_decorator class decorator (Phase 1.3)."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        HandlerRegistry.reset_instance()
        yield
        HandlerRegistry.reset_instance()

    def test_decorator_registers_handler(self):
        """Decorator should register handler with registry."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass
        from typing import Any, Tuple

        @dataclass
        class MockHandler:
            async def execute(self, node, context, registry) -> Tuple[Any, int]:
                return {}, 0

        # Apply decorator
        handler_decorator("decorated_test_handler")(MockHandler)

        # Verify registration
        registry = get_handler_registry()
        assert registry.has("decorated_test_handler")

    def test_decorator_returns_class_unchanged(self):
        """Decorator should return the class unchanged."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass
        from typing import Any, Tuple

        @dataclass
        class MockHandler:
            async def execute(self, node, context, registry) -> Tuple[Any, int]:
                return {}, 0

        # Apply decorator
        decorated = handler_decorator("test_handler_unchanged")(MockHandler)

        # Class should be unchanged
        assert decorated is MockHandler

    def test_decorator_with_explicit_vertical(self):
        """Decorator should use explicit vertical when provided."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass

        @dataclass
        class MockHandler:
            pass

        # Apply decorator with explicit vertical
        handler_decorator("explicit_vertical_handler", vertical="custom_vertical")(MockHandler)

        # Verify vertical
        registry = get_handler_registry()
        entry = registry.get_entry("explicit_vertical_handler")
        assert entry is not None
        assert entry.vertical == "custom_vertical"

    def test_decorator_with_description(self):
        """Decorator should accept optional description."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass

        @dataclass
        class MockHandler:
            pass

        handler_decorator(
            "described_handler",
            vertical="test",
            description="Test handler description",
        )(MockHandler)

        registry = get_handler_registry()
        entry = registry.get_entry("described_handler")
        assert entry is not None
        assert entry.description == "Test handler description"

    def test_decorator_creates_instance(self):
        """Decorator should register a handler instance, not class."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass
        from typing import Any, Tuple

        @dataclass
        class MockHandler:
            async def execute(self, node, context, registry) -> Tuple[Any, int]:
                return {"result": "test"}, 1

        handler_decorator("instance_handler")(MockHandler)

        registry = get_handler_registry()
        handler = registry.get("instance_handler")

        # Should be an instance, not a class
        assert handler is not None
        assert isinstance(handler, MockHandler)

    def test_decorator_replace_option(self):
        """Decorator should support replace option."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass

        @dataclass
        class Handler1:
            name: str = "handler1"

        @dataclass
        class Handler2:
            name: str = "handler2"

        # Register first handler
        handler_decorator("replaceable_handler")(Handler1)

        # Try to register second without replace - should raise
        with pytest.raises(ValueError):
            handler_decorator("replaceable_handler")(Handler2)

        # With replace=True, should succeed
        handler_decorator("replaceable_handler", replace=True)(Handler2)

        # Verify second handler is registered
        registry = get_handler_registry()
        handler = registry.get("replaceable_handler")
        assert isinstance(handler, Handler2)

    def test_decorator_can_be_used_inline(self):
        """Decorator should work as inline decorator."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass
        from typing import Any, Tuple

        @handler_decorator("inline_handler", vertical="test")
        @dataclass
        class InlineHandler:
            async def execute(self, node, context, registry) -> Tuple[Any, int]:
                return {}, 0

        registry = get_handler_registry()
        assert registry.has("inline_handler")
        assert isinstance(registry.get("inline_handler"), InlineHandler)


class TestGetVerticalFromModule:
    """Tests for get_vertical_from_module helper (Phase 1.3)."""

    def test_extracts_vertical_from_victor_path(self):
        """Should extract vertical name from victor.X.handlers path."""
        from victor.framework.handler_registry import get_vertical_from_module

        assert get_vertical_from_module("victor.coding.handlers") == "coding"
        assert get_vertical_from_module("victor.coding.handlers.custom") == "coding"
        assert get_vertical_from_module("victor.research.handlers") == "research"
        assert get_vertical_from_module("victor.devops.handlers") == "devops"
        assert get_vertical_from_module("victor.dataanalysis.handlers") == "dataanalysis"
        assert get_vertical_from_module("victor.rag.handlers") == "rag"

    def test_handles_non_vertical_paths(self):
        """Should return None for non-vertical paths."""
        from victor.framework.handler_registry import get_vertical_from_module

        assert get_vertical_from_module("victor.framework.handlers") is None
        assert get_vertical_from_module("victor.core.handlers") is None
        assert get_vertical_from_module("victor.tools.handlers") is None

    def test_handles_short_paths(self):
        """Should return None for paths too short to be verticals."""
        from victor.framework.handler_registry import get_vertical_from_module

        assert get_vertical_from_module("victor") is None
        assert get_vertical_from_module("victor.coding") is None

    def test_handles_unknown_module(self):
        """Should return None for unknown module paths."""
        from victor.framework.handler_registry import get_vertical_from_module

        assert get_vertical_from_module("unknown.module") is None


class TestHandlerDecoratorIntegration:
    """Integration tests for decorator with registry (Phase 1.3)."""

    @pytest.fixture(autouse=True)
    def reset_registries(self):
        """Reset singleton and executor handlers before each test."""
        HandlerRegistry.reset_instance()
        from victor.workflows import executor
        executor._compute_handlers.clear()
        yield
        HandlerRegistry.reset_instance()
        executor._compute_handlers.clear()

    def test_multiple_handlers_different_verticals(self):
        """Should support multiple handlers from different verticals."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass

        @dataclass
        class CodingHandler:
            pass

        @dataclass
        class ResearchHandler:
            pass

        handler_decorator("coding_handler", vertical="coding")(CodingHandler)
        handler_decorator("research_handler", vertical="research")(ResearchHandler)

        registry = get_handler_registry()

        coding_handlers = registry.list_by_vertical("coding")
        research_handlers = registry.list_by_vertical("research")

        assert "coding_handler" in coding_handlers
        assert "research_handler" in research_handlers

    def test_decorator_works_with_existing_registry_handlers(self):
        """Decorator handlers should coexist with manual registrations."""
        from victor.framework.handler_registry import handler_decorator
        from dataclasses import dataclass

        registry = get_handler_registry()

        # Manual registration
        manual_handler = MagicMock()
        registry.register("manual_handler", manual_handler, vertical="test")

        # Decorator registration
        @dataclass
        class DecoratorHandler:
            pass

        handler_decorator("decorator_handler", vertical="test")(DecoratorHandler)

        # Both should be present
        assert registry.has("manual_handler")
        assert registry.has("decorator_handler")
        assert len(registry.list_by_vertical("test")) == 2
