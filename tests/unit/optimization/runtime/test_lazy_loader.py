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

"""Unit tests for LazyComponentLoader."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

from victor.optimization.runtime.lazy_loader import (
    ComponentDescriptor,
    LazyComponentLoader,
    LoadingStats,
    lazy_load,
    set_global_loader,
)


class TestLazyComponentLoader:
    """Test suite for LazyComponentLoader."""

    def test_register_component(self):
        """Test component registration."""
        loader = LazyComponentLoader()

        loader.register_component("test", lambda: "test_value")

        assert "test" in loader.list_registered()
        assert not loader.is_loaded("test")

    def test_register_component_with_dependencies(self):
        """Test component registration with dependencies."""
        loader = LazyComponentLoader()

        loader.register_component(
            "database",
            lambda: "db_connection",
            dependencies=["config"],
        )

        assert "database" in loader.list_registered()
        descriptor = loader._components["database"]
        assert descriptor.dependencies == ["config"]

    def test_register_component_duplicate_raises_error(self):
        """Test that registering duplicate component raises error."""
        loader = LazyComponentLoader()

        loader.register_component("test", lambda: "test_value")

        with pytest.raises(ValueError, match="already registered"):
            loader.register_component("test", lambda: "another_value")

    def test_register_component_circular_dependency_raises_error(self):
        """Test that circular dependencies are detected."""
        loader = LazyComponentLoader()

        # Register config first
        loader.register_component("config", lambda: {})

        # Try to register component with circular dependency
        with pytest.raises(ValueError, match="circular dependencies"):
            loader.register_component(
                "component",
                lambda: "value",
                dependencies=["component"],  # Self-dependency
            )

    def test_get_component_loads_on_first_access(self):
        """Test that component is loaded on first access."""
        loader = LazyComponentLoader()
        load_count = {"count": 0}

        def loader_func():
            load_count["count"] += 1
            return "loaded_value"

        loader.register_component("test", loader_func)

        assert not loader.is_loaded("test")
        assert load_count["count"] == 0

        result = loader.get_component("test")

        assert result == "loaded_value"
        assert loader.is_loaded("test")
        assert load_count["count"] == 1

    def test_get_component_returns_cached_instance(self):
        """Test that subsequent accesses return cached instance."""
        loader = LazyComponentLoader()
        load_count = {"count": 0}

        def loader_func():
            load_count["count"] += 1
            return {"data": "value"}

        loader.register_component("test", loader_func)

        result1 = loader.get_component("test")
        result2 = loader.get_component("test")

        assert result1 is result2  # Same instance
        assert load_count["count"] == 1  # Only loaded once

    def test_get_component_loads_dependencies_first(self):
        """Test that dependencies are loaded before component."""
        loader = LazyComponentLoader()
        load_order = []

        def load_config():
            load_order.append("config")
            return {"setting": "value"}

        def load_database():
            load_order.append("database")
            return {"connection": "active"}

        loader.register_component("config", load_config)
        loader.register_component("database", load_database, dependencies=["config"])

        loader.get_component("database")

        assert load_order == ["config", "database"]

    def test_get_component_nonexistent_raises_error(self):
        """Test that getting nonexistent component raises error."""
        loader = LazyComponentLoader()

        with pytest.raises(KeyError, match="not registered"):
            loader.get_component("nonexistent")

    def test_preload_components(self):
        """Test preloading multiple components."""
        loader = LazyComponentLoader()
        loaded = []

        def loader_func(name):
            def _load():
                loaded.append(name)
                return f"{name}_value"

            return _load

        loader.register_component("comp1", loader_func("comp1"))
        loader.register_component("comp2", loader_func("comp2"))
        loader.register_component("comp3", loader_func("comp3"))

        loader.preload_components(["comp1", "comp3"])

        assert loader.is_loaded("comp1")
        assert not loader.is_loaded("comp2")
        assert loader.is_loaded("comp3")
        assert "comp1" in loaded
        assert "comp3" in loaded

    def test_unload_component(self):
        """Test unloading a component."""
        loader = LazyComponentLoader()

        loader.register_component("test", lambda: "value")
        loader.get_component("test")

        assert loader.is_loaded("test")

        loader.unload_component("test")

        assert not loader.is_loaded("test")

    def test_unload_component_allows_reload(self):
        """Test that unloaded component can be reloaded."""
        loader = LazyComponentLoader()
        load_count = {"count": 0}

        def loader_func():
            load_count["count"] += 1
            return f"value_{load_count['count']}"

        loader.register_component("test", loader_func)

        # First load
        result1 = loader.get_component("test")
        assert result1 == "value_1"
        assert load_count["count"] == 1

        # Unload
        loader.unload_component("test")

        # Second load
        result2 = loader.get_component("test")
        assert result2 == "value_2"
        assert load_count["count"] == 2

    def test_get_loading_stats(self):
        """Test getting loading statistics."""
        loader = LazyComponentLoader()

        loader.register_component("fast", lambda: "fast_value")
        loader.register_component("slow", lambda: time.sleep(0.01) or "slow_value")

        # Access components
        loader.get_component("fast")
        loader.get_component("fast")  # Cache hit
        loader.get_component("slow")

        stats = loader.get_loading_stats()

        assert stats.hit_count == 1
        assert stats.miss_count == 2
        assert stats.hit_rate == 1.0 / 3.0
        assert "fast" in stats.component_load_times
        assert "slow" in stats.component_load_times
        assert stats.component_load_times["slow"] >= 10  # At least 10ms

    def test_reset_stats(self):
        """Test resetting statistics."""
        loader = LazyComponentLoader()

        loader.register_component("test", lambda: "value")
        loader.get_component("test")

        stats_before = loader.get_loading_stats()
        assert stats_before.miss_count > 0

        loader.reset_stats()

        stats_after = loader.get_loading_stats()
        assert stats_after.miss_count == 0
        assert stats_after.hit_count == 0

    def test_list_loaded(self):
        """Test listing loaded components."""
        loader = LazyComponentLoader()

        loader.register_component("comp1", lambda: "value1")
        loader.register_component("comp2", lambda: "value2")
        loader.register_component("comp3", lambda: "value3")

        loader.get_component("comp1")
        loader.get_component("comp3")

        loaded = loader.list_loaded()
        assert "comp1" in loaded
        assert "comp2" not in loaded
        assert "comp3" in loaded

    def test_list_registered(self):
        """Test listing all registered components."""
        loader = LazyComponentLoader()

        loader.register_component("comp1", lambda: "value1")
        loader.register_component("comp2", lambda: "value2")

        registered = loader.list_registered()
        assert "comp1" in registered
        assert "comp2" in registered
        assert len(registered) == 2

    def test_thread_safe_loading(self):
        """Test that loading is thread-safe."""
        loader = LazyComponentLoader()
        load_count = {"count": 0}

        def slow_loader():
            time.sleep(0.01)  # Simulate slow load
            load_count["count"] += 1
            return "value"

        loader.register_component("test", slow_loader)

        # Access from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(lambda: loader.get_component("test")) for _ in range(10)]
            results = [f.result() for f in futures]

        # All should get the same result
        assert all(r == "value" for r in results)
        # Component should only be loaded once
        assert load_count["count"] == 1

    def test_loading_stats_avg_load_time(self):
        """Test average load time calculation."""
        loader = LazyComponentLoader()

        loader.register_component("fast", lambda: "fast")
        loader.register_component("slow", lambda: time.sleep(0.01) or "slow")

        loader.get_component("fast")
        loader.get_component("slow")

        stats = loader.get_loading_stats()
        assert stats.avg_load_time_ms > 0

    def test_component_descriptor(self):
        """Test ComponentDescriptor dataclass."""
        descriptor = ComponentDescriptor(
            key="test",
            loader=lambda: "value",
            dependencies=["dep1", "dep2"],
        )

        assert descriptor.key == "test"
        assert descriptor.dependencies == ["dep1", "dep2"]
        assert not descriptor.loaded
        assert descriptor.instance is None


class TestLazyLoadDecorator:
    """Test suite for @lazy_load decorator."""

    def test_lazy_load_decorator_with_global_loader(self):
        """Test lazy_load decorator with global loader."""
        loader = LazyComponentLoader()
        loader.register_component("database", lambda: {"connection": "active"})
        set_global_loader(loader)

        @lazy_load("database")
        def get_user(database, user_id):
            return {"user_id": user_id, "db": database}

        result = get_user(user_id=123)

        assert result["user_id"] == 123
        assert result["db"]["connection"] == "active"

    def test_lazy_load_decorator_without_global_loader_raises_error(self):
        """Test that decorator raises error without global loader."""
        set_global_loader(None)

        @lazy_load("database")
        def func(database):
            return database

        with pytest.raises(RuntimeError, match="LazyComponentLoader not found"):
            func()
